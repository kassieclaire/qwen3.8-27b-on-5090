"""Standalone OSCAR INT2 KV-cache Triton kernels.

Every module in this package is pure Python + Triton + torch, with NO imports
from ``sglang`` itself. That is deliberate: it makes the kernels immune to the
upstream package reshuffles between the OSCAR fork (which used
``sglang.srt.mem_cache`` / ``sglang.QuantKernel`` / ``sglang.jit_kernel``) and
newer upstream (``sglang.kernels.ops.*``).

Modules
-------
kv_quant_kernels
    Verbatim copy of the OSCAR fork's ``srt/mem_cache/kv_quant_kernels.py``.
    Uniform int2 quantize/dequantize + packing, single-scale and grouped.
oscar_rotation_clip_int2_kv
    Verbatim copy of the OSCAR fork's
    ``QuantKernel/oscar_rotation_clip_int2_kv.py``, with its one intra-sglang
    import repointed at the sibling ``kv_quant_kernels``. Provides the fused
    per-row quantile-clip + int2 pack kernel used on the WRITE side.
int2_decode_attention
    INT2 decode-attention kernels lifted from the OSCAR fork's
    ``srt/layers/attention/triton_ops/decode_attention.py``. Needed because
    newer upstream has no int2 read path whatsoever.

NOT ported (mixed-KV / HP-window machinery, which is incompatible with
speculative decoding): ``gpu_flush_int2``, ``mla_latent_int2``,
``fused_hadamard_int2_kv``, ``unified_kv_pool``, ``unified_kv_allocator``.
"""

from .kv_quant_kernels import (
    dequantize_kv_int2_triton,
    quantized_set_kv_int2_triton,
)
from .oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_pretransformed_clip_triton,
)

__all__ = [
    "dequantize_kv_int2_triton",
    "quantized_set_kv_int2_triton",
    "quantized_set_kv_int2_pretransformed_clip_triton",
    "decode_attention_fwd_quantized",
]


def __getattr__(name):
    # Lazy: importing the decode kernels pulls in a large Triton module, and
    # some callers (e.g. capacity planning) only need the write-side helpers.
    if name == "decode_attention_fwd_quantized":
        from .int2_decode_attention import decode_attention_fwd_quantized

        return decode_attention_fwd_quantized
    raise AttributeError(name)
