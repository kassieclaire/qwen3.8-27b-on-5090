"""
Fix: unsloth/Qwen3.8-27B-NVFP4 stores lm_head as FP8 (float8_e4m3fn) with a
per-row `lm_head.weight_scale` (channel strategy, config group_0
format=float-quantized, target `re:.*lm_head`).

SGLang's CompressedTensorsConfig.get_quant_method() only dispatches on
LinearBase / FusedMoE. ParallelLMHead is a VocabParallelEmbedding, NOT a
LinearBase, so it falls back to UnquantizedEmbeddingMethod and the
weight_scale is dropped ("Parameter lm_head.weight_scale not found in
params_dict"). The raw FP8 codes are then cast straight to bf16, so logits
are ~1300x too large AND mis-scaled per row by up to 12x relative to each
other -> garbage/degenerate sampling on anything but very short, very
confident completions.

This shim hooks the model loader and folds weight_scale into lm_head.weight.
Activated by SGLANG_FIX_LM_HEAD_FP8=1.
"""
import os, logging

logger = logging.getLogger(__name__)


def _install():
    import torch
    from safetensors import safe_open
    from sglang.srt.model_loader.loader import DefaultModelLoader

    model_path = os.environ.get("SGLANG_FIX_LM_HEAD_FP8_PATH", "")
    if not model_path:
        logger.warning("[lm_head_fix] SGLANG_FIX_LM_HEAD_FP8_PATH unset; skipping")
        return

    # locate lm_head.weight_scale across shards
    import glob, json
    scale = None
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            with safe_open(shard, "pt") as h:
                if "lm_head.weight_scale" in h.keys():
                    scale = h.get_tensor("lm_head.weight_scale")
                    logger.warning(f"[lm_head_fix] found lm_head.weight_scale in {shard}")
                    break
        except Exception as e:  # pragma: no cover
            logger.warning(f"[lm_head_fix] could not read {shard}: {e}")
    if scale is None:
        logger.warning("[lm_head_fix] no lm_head.weight_scale found; nothing to do")
        return

    orig = DefaultModelLoader.load_model

    def patched(self, *args, **kwargs):
        model = orig(self, *args, **kwargs)
        try:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                lm = getattr(model, "language_model", None)
                lm_head = getattr(lm, "lm_head", None) if lm is not None else None
            if lm_head is None:
                logger.warning("[lm_head_fix] no lm_head found on model")
                return model
            w = lm_head.weight
            dev = w.device
            s = scale.to(device=dev, dtype=torch.float32)
            before = w.detach().float().abs().max().item()
            with torch.no_grad():
                # w currently holds the *raw fp8 codes* cast to bf16
                fixed = (w.detach().float() * s).to(w.dtype)
                w.copy_(fixed)
            after = w.detach().float().abs().max().item()
            logger.warning(
                f"[lm_head_fix] APPLIED per-row fp8 scale to lm_head: "
                f"absmax {before:.4g} -> {after:.6g}"
            )
        except Exception as e:
            logger.error(f"[lm_head_fix] FAILED: {e}", exc_info=True)
        return model

    DefaultModelLoader.load_model = patched
    logger.warning("[lm_head_fix] loader hook installed")


if os.environ.get("SGLANG_FIX_LM_HEAD_FP8", "") == "1":
    _install()
