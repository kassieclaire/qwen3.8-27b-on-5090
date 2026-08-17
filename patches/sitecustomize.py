import os, sys

def _t(v): return str(v).strip().lower() in ("1","true","yes","on")

# 1. lm_head fp8 per-row scale fix (unsloth checkpoints)
if _t(os.environ.get("SGLANG_FIX_LM_HEAD_FP8","")):
    try:
        import lm_head_fp8_fix  # noqa
    except Exception as e:
        print("sitecustomize: lm_head fp8 fix failed:", e, file=sys.stderr)

# 1b. lm_head NVFP4 -> dense bf16 dequant (RadixArk checkpoints; needed so the
#     DSPARK draft can matmul the shared target lm_head directly)
if _t(os.environ.get("SGLANG_FIX_LM_HEAD_NVFP4","")):
    try:
        import lm_head_nvfp4_fix  # noqa
    except Exception as e:
        print("sitecustomize: lm_head nvfp4 fix failed:", e, file=sys.stderr)

# 1c. diagnostic KV pool tracer (no-op unless SGLANG_TRACE_KVPOOL=1)
if _t(os.environ.get("SGLANG_TRACE_KVPOOL","")):
    try:
        import kvpool_tracer  # noqa
    except Exception as e:
        print("sitecustomize: kvpool tracer failed:", e, file=sys.stderr)

# 1d. skip loading embed/lm_head into MTP draft (EAGLE worker replaces them;
#     saves ~6GB transient VRAM during draft load)
if _t(os.environ.get("SGLANG_SKIP_MTP_EMBED_LOAD","")):
    try:
        import mtp_embed_skip  # noqa
    except Exception as e:
        print("sitecustomize: mtp embed skip failed:", e, file=sys.stderr)

# 1e. quantize the embedded-MTP draft trunk to FP8 at load time
if _t(os.environ.get("SGLANG_MTP_FP8_DRAFT","")):
    try:
        import mtp_fp8_draft  # noqa
    except Exception as e:
        print("sitecustomize: mtp fp8 draft failed:", e, file=sys.stderr)

# 2. OSCAR INT2 KV cache
if _t(os.environ.get("SGLANG_OSCAR_ENABLE","")):
    try:
        sys.path.insert(0, "/models/oscar_port")
        import apply_patch
        apply_patch.apply()
    except Exception as e:
        print("sitecustomize: OSCAR int2 patch failed:", e, file=sys.stderr)
        raise

# 3. QKV dump hook (calibration only)
if _t(os.environ.get("SGLANG_QKV_DUMP_SHIM","")) and _t(os.environ.get("DUMP_KVCACHE","")):
    try:
        import qkv_dump_shim  # noqa
    except Exception as e:
        print("sitecustomize: qkv dump shim failed:", e, file=sys.stderr)
