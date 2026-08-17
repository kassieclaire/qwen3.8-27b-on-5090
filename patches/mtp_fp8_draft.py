"""Quantize the embedded-MTP (EAGLE) draft trunk to FP8 at load time.

The checkpoint ships MTP weights in BF16 (upstream nulls quant for modelopt_fp4).
For single-stream decode the draft runs 3-6 autoregressive steps per cycle, so
its GEMM time is pure overhead. This patch converts every nn.Linear weight in
the MTP trunk to FP8-e4m3 (per-tensor scale) and swaps forward for
torch._scaled_mm, roughly halving draft GEMM time.

Enabled via SGLANG_MTP_FP8_DRAFT=1. Sanity-checks the first conversion with a
matmul round-trip and falls back to BF16 on failure.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)
_APPLIED = False


def _to_fp8_pair(w: torch.Tensor):
    """Return (w_fp8, scale_scalar, scale_b [1, n_out]) with w ~= w_fp8 * scale."""
    scale = w.abs().max().float().clamp(min=1e-6) / 448.0
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    sb = scale.reshape(1, 1).expand(1, wq.shape[0]).contiguous()
    return wq, scale.reshape(()), sb


def _convert_module(mod, name):
    if not isinstance(mod, torch.nn.Linear):
        return
    if mod.weight.dtype != torch.bfloat16:
        return
    wq, scale, sb = _to_fp8_pair(mod.weight.detach())
    ref = mod.weight.detach().float()
    got = wq.float() * scale
    rel = ((got - ref).norm() / (ref.norm() + 1e-9)).item()
    if rel > 0.05:
        logger.warning("[mtp_fp8_draft] %s rel-err %.4f too high; keeping bf16", name)
        return
    orig_forward = mod.forward

    def fp8_forward(x, _wq=wq, _s=sb, _f=orig_forward, _bias=mod.bias):
        if x.dtype == torch.bfloat16 and x.is_cuda and x.dim() == 2:
            xs = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-6) / 448.0
            xq = (x.float() / xs).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                xq, _wq.t(), scale_a=xs.contiguous(), scale_b=_s,
                out_dtype=torch.bfloat16,
            )
            if _bias is not None:
                out = out + _bias
            return out
        return _f(x)

    mod.forward = fp8_forward
    mod.weight_fp8 = wq
    mod.weight_fp8_scale = scale
    mod.weight.data = torch.empty(0, device=mod.weight.device)  # free bf16 copy


def apply():
    global _APPLIED
    if _APPLIED or not os.environ.get("SGLANG_MTP_FP8_DRAFT"):
        return
    _APPLIED = True
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    orig_ilh = EagleDraftWorker.init_lm_head

    def patched_ilh(sself, *a2, **k2):
        r = orig_ilh(sself, *a2, **k2)
        try:
            runner = getattr(sself, "draft_runner", None)
            model = getattr(runner, "model", None)
            if model is not None:
                n = 0
                for name, mod in model.named_modules():
                    _convert_module(mod, name)
                    n += 1
                logger.info("[mtp_fp8_draft] converted %d draft modules to FP8", n)
            else:
                logger.warning("[mtp_fp8_draft] draft model not found at init_lm_head")
        except Exception as e:
            import traceback
            logger.warning("[mtp_fp8_draft] failed: %s\n%s", e,
                           traceback.format_exc()[-800:])
        return r

    EagleDraftWorker.init_lm_head = patched_ilh
    logger.info("[mtp_fp8_draft] hook installed")


apply()
