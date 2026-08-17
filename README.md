# qwen3.8-27b-on-5090

Qwen3.8-27B (NVFP4) served with **OSCAR int2 KV cache** + the model's **built-in MTP**
speculative decoding, on a single **RTX 5090** — full 262k context, no CPU offload.

```bash
./serve.sh          # downloads weights once, builds, serves on :30000
curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

## Expected speeds (single stream, cold cache, RTX 5090 32GB)

| | stock NVFP4 recipe | this repo |
|---|---|---|
| decode | ~74 tok/s | **~130–136 tok/s** |
| prefill (cold) | ~9.4k tok/s* | **~8.7–8.8k tok/s** |
| context | 262k | 262k (needle verified @ 201k) |

\* stock uses bf16 KV; int2 KV trades ~7% prefill for **8.4×** smaller KV and much
faster decode attention. Decode varies with draft acceptance (2.5–2.9 tokens/step on
prose; higher on code). Spec depth is tunable: `STEPS=3 DRAFT=4` ≈ 128 tok/s,
`STEPS=4 DRAFT=6 TOPK=2` (default) ≈ 130–136 tok/s.

## What's inside

- `docker-compose.yml` — fetch job (weights → named volume) + server (builds `Dockerfile`)
- `Dockerfile` — pins `lmsysorg/sglang:dev-cu13` + OSCAR/MTP env defaults
- `serve.sh` — one-shot launcher
- `oscar_port/` — vendored OSCAR int2 kernels: fused causal tile kernel (prefill from
  the packed pool, ~6× over the dequant path), int2 decode kernel with in-kernel
  rotation folds, pool/attention patches
- `patches/` — W4A16 draft lm_head (NVFP4 head via flashinfer, 3.4× faster), MTP
  embed/lm_head on meta device (−4.75GB), sitecustomize loader
- `rotations/` — calibrated K/V rotations for the target's full-attention layers

## Notes

- Quantizer is bit-exact vs [OSCAR](https://github.com/FutureMLS-Lab/OSCAR) upstream
  (verified across 8 configs incl. Lloyd-Max); attention kernels match the official
  dequant oracle to int2 noise floor (~0.004 rel-err).
- The int2→bf16 divergence is small: KL ≈ 0.08, top-1 agreement 94% (teacher-forced,
  aligned). Verified agentically in pi.dev: 5/5 multi-stage coding tasks.
- CUDA graphs stay on for decode/verify/draft (prefill graphs disabled by design).
