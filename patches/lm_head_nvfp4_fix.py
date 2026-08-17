"""
Fix v2: RadixArk/Qwen3.8-27B-NVFP4 stores lm_head as NVFP4 (uint8 packed).
DSPARK's draft (`dspark.py:compute_base_logits`) borrows the TARGET lm_head
and does ``torch.matmul(hidden, lm_head.weight.T)`` — a shape error on the
packed weight.

v1 replaced the whole lm_head with a dense bf16 dequant (2.5 GB, 1.53 ms per
read), which also forced the TARGET's logits onto the slow dense path.

v2 hooks DSparkWorkerV2.__init__: the target keeps its native NVFP4 head
(served through flashinfer fp4_gemm), and the DRAFT gets a private W4A16
Tensor-subclass weight built from the same packed tensors:

  * flashinfer.mm_bf16_fp4 (cute-dsl, SM120): 0.45 ms vs 1.53 ms dense bf16
    per draft propose step
  * ~0.7 GB packed instead of 2.5 GB dense

The subclass carries ZERO storage (as_strided with 0 strides over an empty
base): logical shape [vocab, hidden] exists for validation code, but every
matmul is intercepted in ``__torch_function__`` and routed to the W4A16
kernel, so the phantom storage is never read.

The checkpoint's plain row-major per-block scales [vocab, hidden/16] are
converted to the 128x4-swizzled layout cute-dsl expects (validated 100%
byte-match against flashinfer.nvfp4_quantize).

Activated by SGLANG_FIX_LM_HEAD_NVFP4=1 + SGLANG_FIX_LM_HEAD_NVFP4_PATH.
"""
import os, glob, logging

logger = logging.getLogger(__name__)


def _swizzle_128x4(plain_u8, n: int, n_sf: int):
    """plain [n, n_sf] uint8 scales -> 128x4-swizzled flat bytes.

    offset(r, c) = rb*(n_sf*128) + (c//4)*512 + (r%32)*16 + ((r//32)%4)*4 + (c%4)
    where rb = r // 128.  Validated 100% byte-match against
    flashinfer.nvfp4_quantize's swizzled output for n in {128,256,384},
    k in {128,256,512}.
    """
    plain = plain_u8.view(n, n_sf).cpu()
    out = torch.zeros(n * n_sf, dtype=torch.uint8)
    r = torch.arange(n)
    rb = r // 128
    rr = r % 128
    off_r = rb * (n_sf * 128) + ((rr // 32) % 4) * 4 + (rr % 32) * 16
    for ci in range(n_sf):
        out[off_r + (ci // 4) * 512 + (ci % 4)] = plain[:, ci]
    return out.cuda()


def _load_packed_head(model_path):
    from safetensors import safe_open

    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            with safe_open(shard, "pt") as h:
                keys = set(h.keys())
                if "lm_head.weight" in keys and str(
                    h.get_slice("lm_head.weight").get_dtype()
                ).upper().startswith("U8"):
                    return (
                        h.get_tensor("lm_head.weight"),
                        h.get_tensor("lm_head.weight_scale")
                        if "lm_head.weight_scale" in keys else None,
                        h.get_tensor("lm_head.weight_scale_2")
                        if "lm_head.weight_scale_2" in keys else None,
                    )
        except Exception as e:
            logger.warning(f"[lm_head_nvfp4] could not read {shard}: {e}")
    return None, None, None


def _install():
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )

    orig_init = DSparkWorkerV2.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _swap_draft_head(self)
        except Exception as e:
            logger.error(f"[lm_head_nvfp4] draft head swap FAILED: {e}",
                         exc_info=True)

    DSparkWorkerV2.__init__ = patched_init
    logger.warning("[lm_head_nvfp4] DSparkWorkerV2.__init__ hook installed")


def _swap_draft_head(worker):
    import flashinfer
    import flashinfer.gemm as fg

    model_path = os.environ.get("SGLANG_FIX_LM_HEAD_NVFP4_PATH", "")
    packed, scale, scale2 = _load_packed_head(model_path)
    if packed is None or scale is None:
        logger.warning("[lm_head_nvfp4] no packed lm_head found; keeping dense")
        return

    draft = worker.draft_model
    dev = worker.device
    n, half = packed.shape
    n_sf = half * 2 // 16
    p = packed.to(dev, non_blocking=True)
    sc = scale.to(dev, non_blocking=True).view(torch.uint8)
    s2 = float(scale2) if scale2 is not None else 1.0
    sf_swz = _swizzle_128x4(sc, n, n_sf)
    b_p, sf_p, al_p = flashinfer.prepare_bf16_fp4_weights(
        p, sf_swz,
        torch.tensor([s2], device=dev, dtype=torch.float32),
        backend="cute-dsl",
    )

    # ---- sanity: compare against exact dense dequant BEFORE installing ----
    LUT = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                        -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], device=dev)
    lo, hi = (p & 0xF).long(), (p >> 4).long()
    codes = torch.empty(n, half * 2, dtype=torch.long, device=dev)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    W_ref = (LUT[codes].view(n, -1, 16)
             * sc.view(torch.float8_e4m3fn).float().view(n, -1, 1)).view(n, -1) * s2
    x_probe = torch.randn(4, half * 2, device=dev, dtype=torch.bfloat16)
    out = fg.mm_bf16_fp4(x_probe, b_p, sf_p, al_p, backend="cute-dsl")
    ref = x_probe.float() @ W_ref.t().float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    del W_ref, codes, lo, hi, ref, out
    if rel > 0.02:
        logger.error(f"[lm_head_nvfp4] W4A16 sanity FAILED rel={rel:.4f}; keeping dense")
        return
    logger.warning(f"[lm_head_nvfp4] W4A16 sanity OK rel={rel:.5f}")

    class _W4HeadModule(torch.nn.Module):
        """Quacks like the target lm_head: delegates attributes to the real
        head (org_vocab_size, gather metadata...) but serves a W4A16 weight."""
        def __init__(self, orig_head):
            super().__init__()
            object.__setattr__(self, "_orig", orig_head)
            self.weight = _W4WeightTensor.make(b_p, sf_p, al_p, (n, half * 2))

        def forward(self, x):
            # Should not be hit (the draft uses compute_base_logits), but keep
            # it correct if anything calls lm_head(hidden) directly.
            x2 = x.reshape(-1, x.shape[-1])
            o = fg.mm_bf16_fp4(x2, b_p, sf_p, al_p, backend="cute-dsl")
            return o.view(*x.shape[:-1], -1)

        def __getattr__(self, name):
            o = object.__getattribute__(self, "_orig")
            return getattr(o, name)

    draft.lm_head = _W4HeadModule(draft.lm_head)
    logger.warning("[lm_head_nvfp4] APPLIED W4A16 draft head")


import torch


class _W4WeightTensor(torch.Tensor):
    """Zero-storage Tensor subclass, logical shape [vocab, hidden].

    All metadata (shape/dtype/dim/T) is overridden at the PYTHON class level so
    it never enters torch's __torch_function__ dispatch (which is what made
    earlier versions recurse). Only actual torch FUNCTION calls on the
    instance (matmul etc.) dispatch, and those route to the W4A16 kernel.
    """

    _b_p = None
    _sf_p = None
    _al_p = None
    _logical_shape = None

    @property
    def shape(self):
        return self._logical_shape

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return self._b_p.device

    @property
    def T(self):
        base = torch.empty(1, dtype=torch.bfloat16, device=self._b_p.device)
        t = base.as_strided((self._logical_shape[1], self._logical_shape[0]), (0, 0))
        s = t.as_subclass(_W4WeightTensor)
        s._b_p, s._sf_p, s._al_p = self._b_p, self._sf_p, self._al_p
        s._logical_shape = (self._logical_shape[1], self._logical_shape[0])
        return s

    def dim(self):
        return len(self._logical_shape)

    def ndim(self):
        return len(self._logical_shape)

    def numel(self):
        n = 1
        for d in self._logical_shape:
            n *= d
        return n

    def is_contiguous(self):
        return True

    def t(self):
        return self.T

    def detach(self):
        return self

    def to(self, *a, **k):
        return self

    def contiguous(self):
        return self

    def float(self):
        return self

    @staticmethod
    def make(b_p, sf_p, al_p, shape):
        # 1-element dummy storage: as_strided with 0 strides never reads past
        # element 0 (all matmuls are intercepted), but passes storage-size
        # validation for any logical shape.
        base = torch.empty(1, dtype=torch.bfloat16, device=b_p.device)
        t = base.as_strided(shape, (0,) * len(shape))
        sub = t.as_subclass(_W4WeightTensor)
        sub._b_p = b_p
        sub._sf_p = sf_p
        sub._al_p = al_p
        sub._logical_shape = tuple(shape)
        return sub

    def _w4(self):
        base = torch.empty(1, dtype=torch.bfloat16, device=self._b_p.device)
        t = base.as_strided(self.shape, (0,) * self.dim())
        s = t.as_subclass(_W4WeightTensor)
        s._b_p, s._sf_p, s._al_p = self._b_p, self._sf_p, self._al_p
        return s

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        import flashinfer.gemm as fg
        fname = getattr(func, "__name__", str(func))

        if fname in ("matmul", "mm", "bmm", "addmm", "linear"):
            w = next((a for a in args if isinstance(a, cls)), None)
            other = next((a for a in args if isinstance(a, torch.Tensor)
                          and not isinstance(a, cls)), None)
            if w is not None and other is not None and other.dim() >= 1:
                x = other.reshape(-1, other.shape[-1])
                o = fg.mm_bf16_fp4(x.to(torch.bfloat16), w._b_p, w._sf_p,
                                   w._al_p, backend="cute-dsl")
                o = o.view(*other.shape[:-1], -1)
                if fname == "addmm":
                    bias = args[0] if len(args) > 0 and not isinstance(args[0], cls) else None
                    if bias is not None and isinstance(bias, torch.Tensor):
                        o = o + bias
                return o

        if fname in ("dim", "ndim", "size", "numel", "is_contiguous", "element_size", "device", "shape", "stride", "t", "T", "detach", "clone", "contiguous", "float", "half", "to"):
            # metadata/view ops: call the plain-tensor implementation with
            # torch-function dispatch disabled to avoid re-entry.
            with torch._C.DisableTorchFunctionSubclass():
                plain_args = [a.as_subclass(torch.Tensor) if isinstance(a, cls) else a
                              for a in args]
                out = func(*plain_args, **kwargs)
            src = next((a for a in args if isinstance(a, cls)), None)
            if isinstance(out, torch.Tensor) and src is not None and not isinstance(out, cls):
                out2 = out.as_subclass(cls)
                out2._b_p, out2._sf_p, out2._al_p = src._b_p, src._sf_p, src._al_p
                out2._logical_shape = tuple(out.shape)
                return out2
            return out

        raise NotImplementedError(
            f"[lm_head_nvfp4] op {fname} on the W4A16 draft head is not "
            f"supported (args: {[type(a).__name__ for a in args]})"
        )


if os.environ.get("SGLANG_FIX_LM_HEAD_NVFP4", "") == "1":
    _install()
