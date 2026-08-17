# qwen3.8-27b on RTX 5090 — SGLang + OSCAR int2-KV + built-in MTP
# The base provides SGLang/flashinfer/CUDA13. This layer only pins env defaults;
# all custom code is mounted at runtime from ./oscar_port and ./patches (see compose).
FROM lmsysorg/sglang:dev-cu13

ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_OSCAR_ENABLE=1 \
    SGLANG_OSCAR_K_CLIP_RATIO=0.96 \
    SGLANG_OSCAR_V_CLIP_RATIO=0.92 \
    SGLANG_OSCAR_GROUP_SIZE=256 \
    SGLANG_LLOYD_MAX=1 \
    SGLANG_ENABLE_SPEC_V2=1 \
    SGLANG_SKIP_MTP_EMBED_LOAD=1 \
    SGLANG_FIX_LM_HEAD_NVFP4=1 \
    SGLANG_FIX_LM_HEAD_NVFP4_PATH=/models/Qwen3.8-27B-NVFP4 \
    PYTHONPATH=/models/patches:/models/oscar_port

EXPOSE 30000
