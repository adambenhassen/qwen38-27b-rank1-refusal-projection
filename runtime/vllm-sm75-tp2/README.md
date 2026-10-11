# vLLM on Turing, TP=2 (2× RTX 2080 Ti 22 GB, SM75)

Port of the runtime rank-1 refusal dial to a Turing (SM75) build of vLLM
serving Swift-1.5 fine-tunes of Qwen3.8-27B
(INT4 W4A16 and INT8 W8A16) with native MTP k=3, FP8 KV and 262K context.

Same direction file (`hf/refusal_dirs_qwen38.safetensors`), same per-module
coefficients, same meaning of λ, same `cache_salt: "refusal:<x>"` seal.

## Source

Two commits on top of the SM75 vLLM source tree
([`1a2f871eb0`](https://github.com/adambenhassen/vLLM-2080Ti-Definitive/commit/1a2f871eb0) and
[`9c4357603a`](https://github.com/adambenhassen/vLLM-2080Ti-Definitive/commit/9c4357603a) on branch
[`sm75-fixes`](https://github.com/adambenhassen/vLLM-2080Ti-Definitive/tree/sm75-fixes)),
also in [`patches/`](patches/):

| patch | what |
|---|---|
| `0001` | the dial: projection module, decoder-layer hooks, V2 model runner state, scheduler field, worker RPCs, `/admin/refusal_lambda`, env registration, CPU tests |
| `0002` | `chat_template_kwargs.refusal_lambda` → `cache_salt`, for clients that cannot send `cache_salt` |

The tree is installed editable, so these are plain commits, not an
anchor-checked patcher over `dist-packages`.

## What differs from `runtime/vllm-0.27.1`

- **λ as module buffers.** The global λ tensor and the per-token buffer are
  registered as non-persistent buffers, not plain attributes. This fork
  serves from AOT-compiled artifacts (`Directly load AOT compilation`);
  buffers are module state the loaded artifact reads live, so the value can
  never be serialized into the artifact.
- **Compile cache key.** `VLLM_REFUSAL_DIRS` is registered in `envs.py`, so it
  enters the compile hash: an artifact compiled without the hooks is never
  reused with them, or the reverse. `VLLM_REFUSAL_LAMBDA_INIT` is excluded
  (runtime value, not graph structure).
- **Global λ and the prefix cache under TP=2.** Upstream adds the global λ to
  the block hash, but the hash is computed in the engine-core process, while
  `set_refusal_lambda` reaches only the worker processes. With TP=2 the key
  never changes, so after a global change unsalted requests would reuse KV
  computed at the old λ. Here `POST /admin/refusal_lambda` sets λ on every
  rank, then resets the prefix cache. If the cache is in use the reset fails,
  the old λ is restored, and the call returns **409**. Per-request λ is
  unaffected: the salt is already part of the block hash.
- **One per-token buffer.** The drafter is never projected
  (`VLLM_REFUSAL_MTP_MODE` is not ported), so there is no draft role.
- **Fail closed both ways.** Boot aborts on an unclaimed direction (as
  upstream) and on a direction claimed twice.
- **Adaptive verification guard.** When the CPU token layout is only an upper
  bound (DSpark adaptive verification), the batch falls back to the global λ
  with a one-time warning instead of misassigning rows. Not reached with MTP.
- **Hooks in `Qwen3NextDecoderLayer.forward`**, which Qwen3.5 inherits on this
  fork, instead of a copied forward. `Qwen3NextDecoderLayer` keeps both
  projections `None`.

## Use

```bash
# server env
VLLM_REFUSAL_DIRS=/path/to/refusal_dirs_qwen38.safetensors
VLLM_REFUSAL_LAMBDA_INIT=0          # unsalted requests: unmodified model

# per request (OpenAI chat API)
{"cache_salt": "refusal:1.0", ...}
# or, for clients without cache_salt (patch 0002)
{"chat_template_kwargs": {"refusal_lambda": 1}, ...}

# global, only while idle (409 otherwise)
curl -XPOST localhost:8010/admin/refusal_lambda \
  -H 'Content-Type: application/json' -d '{"lambda": 1}'
```

[`pi-models.example.json`](pi-models.example.json) is a pi coding agent
(`@earendil-works/pi-coding-agent`) model entry that sends the kwarg on every
request.

## Measured — 2026-10-10

Swift-1.5 Qwen3.8-27B INT4 (W4A16 g128), TP=2 over NVLink, MTP k=3,
`max_num_seqs=8`. 12 benign prompts, temperature 0, thinking off, 200 tokens,
top-5 logprobs.

| check | result |
|---|---|
| λ=0 vs the unpatched build | 9/12 identical, mean KL 4e-6, max 5e-3 |
| patched build, hooks off, vs the unpatched build | 11/12 identical, mean KL 5e-6 |
| `refusal:1.0` vs unsalted | 10/12 outputs differ, mean KL 9e-3 |
| `refusal:1.0` repeated | 12/12 identical |
| `refusal:0` vs unsalted | 12/12 identical |
| 8 concurrent, alternating `refusal:1.0` / none | each request matches its own λ (KL ≤ 2e-4), not the other (1e-3 to 7e-2) |
| global λ=1 vs per-request 1.0 | 12/12 identical |
| global λ back to 0 (prefix cache reset) | 12/12 identical to the original |
| `POST` while a request runs | 409, λ unchanged |
| kwarg `refusal_lambda: 1` vs `cache_salt: refusal:1.0` | identical |
| decode, single stream | 85.4 → 81.7 tok/s (−4%) |
| decode, 8 concurrent | 315 → 311 tok/s (−1 to 2%) |
| MTP mean acceptance length | 3.15 → 3.11 |

λ=0 is **not** bit-exact on this build. Any recompilation drifts by the
same amount: the patched source with the hooks off differs from the old
artifact as much as λ=0 does. For scale, INT8 vs BF16 on the same model is
mean KL ~9e-4.

**Not measured here:** refusal rate and quality on Swift-1.5. The directions
come from base Qwen3.8-27B vs Ektome; whether λ=1 is still the right operating
point for a fine-tune is open. `bench/` runs unchanged against this server.

## Tests

`tests/v1/worker/test_refusal_projection.py` in the fork (17 tests, CPU only):

- direction mapping for both runtime prefixes, MTP excluded, unclaimed and double-claimed directions;
- λ=0 identity and the weight-edit identity at λ=1;
- per-token rows and in-place λ, including under `torch.compile(fullgraph=True)`;
- `RefusalState` expansion and the layout guard;
- the chat-kwarg mapping.

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest --noconftest -q \
  tests/v1/worker/test_refusal_projection.py
```
