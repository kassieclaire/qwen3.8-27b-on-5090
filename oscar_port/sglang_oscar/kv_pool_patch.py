"""Monkeypatch: teach the NEW upstream ``MHATokenToKVPool`` the OSCAR INT2 path.

This is the port of OSCAR path (B) -- the PLAIN int2 path -- which is an
ordinary ``MHATokenToKVPool`` constructed with ``dtype == "int2"``. It is NOT
the mixed-KV / HP-window path (``UnifiedInt2HPKVPool``), which is hard-gated on
``speculative_algorithm is None`` and is deliberately absent from this port.

Why monkeypatch instead of copying ``memory_pool.py``?
------------------------------------------------------
The OSCAR fork's ``memory_pool.py`` is 2779 lines; upstream's is 5034 and has
diverged substantially (pluggable quant methods, VMM post-capture backing,
page-major / HND / vectorized_5d layouts, ``KVWriteLoc``, DCP masking, ...).
Copying the old file over the new tree would clobber all of that and break
DSPARK. Instead we wrap exactly four methods.

What is patched on ``MHATokenToKVPool``
---------------------------------------
  __init__          -> after the real init, if dtype is int2: rebuild buffers as
                       packed uint8 + scale/zero, load rotations, stash clip ratios
  _get_key_buffer   -> return the packed uint8 buffer as-is (no ``.view(dtype)``)
  _get_value_buffer -> ditto
  set_kv_buffer     -> route int2 writes through the OSCAR clip+pack triton kernel
  plus new accessors get_raw_{key,value}_buffer / get_{key,value}_scales_zeros
  used by the int2 attention read path.

KEY UPSTREAM DIFFERENCES THAT FORCED ADAPTATION (see README for the full list)
-----------------------------------------------------------------------------
1. ``KVCache.__init__`` no longer takes ``model_dtype``; it also refuses a
   non-torch dtype in its ``store_dtype`` branch. We therefore let upstream
   construct itself with a REAL torch dtype and flip the pool to int2
   afterwards, rather than passing the string "int2" down through upstream
   constructors that would choke on it.
2. ``MHATokenToKVPool.__init__`` lost ``model_dtype`` /
   ``kv_cache_quant_group_size`` / ``scale_dtype`` / ``oscar_rotation_layer_ids``.
   Those now arrive via env vars + a module-level context set by the pool
   configurator patch (``hybrid_patch.py``).
3. ``set_kv_buffer`` gained ``loc_info`` (a ``KVWriteLoc``, needs
   ``unwrap_write_loc``) and ``dcp_kv_mask``, and LOST the OSCAR-era
   ``already_hadamard_transformed`` / ``is_decode`` kwargs. We accept the old
   kwargs too so the ported prefill path can pass them.
4. ``_create_buffers`` split into ``_create_buffers_normal`` /
   ``_create_quantized_buffers`` and is followed by ``_build_kv_buffer_descs``
   + ``_init_data_ptrs_and_strides``. We rebuild buffers and then re-run those
   two so ``move_kv_cache`` (used by spec decoding!) stays consistent.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from .kernels.kv_quant_kernels import dequantize_kv_int2_triton
from .kernels.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from .rotations import (
    draft_oscar_enabled,
    env_bool,
    load_oscar_rotation_config,
    load_oscar_rotations,
    oscar_enabled,
    scale_dtype,
)

logger = logging.getLogger(__name__)

INT2 = "int2"

# Set by hybrid_patch.py right before the full-attention sub-pool is built, so
# __init__ knows which GLOBAL layer ids this pool's LOCAL 0..N-1 slots map to.
# (Upstream dropped the ``oscar_rotation_layer_ids`` constructor argument.)
_ROTATION_LAYER_IDS_CTX: Optional[List[int]] = None
_QUANT_GROUP_SIZE_CTX: Optional[int] = None
# True while constructing the DSPARK draft worker's KV pool (different
# geometry => different rotation checkpoints).
_IS_DRAFT_CTX = False


def set_rotation_context(
    layer_ids: Optional[List[int]],
    quant_group_size: Optional[int] = None,
    draft: Optional[bool] = None,
) -> None:
    """Set the layer-id / group-size context for the next pool construction.

    ``draft`` defaults to None meaning "leave the worker flag alone" -- callers
    that only know the layer ids (e.g. the HybridLinearKVPool wrapper) must not
    clobber which worker we are building for.
    """
    global _ROTATION_LAYER_IDS_CTX, _QUANT_GROUP_SIZE_CTX, _IS_DRAFT_CTX
    _ROTATION_LAYER_IDS_CTX = list(layer_ids) if layer_ids is not None else None
    _QUANT_GROUP_SIZE_CTX = quant_group_size
    if draft is not None:
        _IS_DRAFT_CTX = draft


def clear_rotation_context() -> None:
    """Clear the per-construction layer-id / group-size context.

    Deliberately PRESERVES ``_IS_DRAFT_CTX``: that flag identifies which WORKER
    we are building for (set once by the dtype resolver) and has a longer
    lifetime than a single pool construction. Resetting it here made the draft
    worker's inner pool load the TARGET rotations and fail with
    "missing layer 0". Use ``set_draft_context()`` to change it.
    """
    global _ROTATION_LAYER_IDS_CTX, _QUANT_GROUP_SIZE_CTX
    _ROTATION_LAYER_IDS_CTX = None
    _QUANT_GROUP_SIZE_CTX = None


def set_draft_context(draft: bool) -> None:
    """Mark subsequent pool construction as belonging to the DSPARK draft worker.

    The draft has its own KV geometry and therefore its own rotation
    checkpoints (``SGLANG_OSCAR_DRAFT_*``); see rotations.py for why.
    """
    global _IS_DRAFT_CTX
    _IS_DRAFT_CTX = draft


_ROT_HD_CACHE: dict = {}


def _rotation_head_dim(path: str) -> Optional[int]:
    """Return the per-head width of the rotations in ``path`` (None if unknown).

    Used to attribute a pool to the target vs the draft by geometry rather than
    by construction order. Cached because it deserializes a checkpoint.
    """
    if not path:
        return None
    if path in _ROT_HD_CACHE:
        return _ROT_HD_CACHE[path]
    hd = None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        layers = state.get("layers", state)
        for _lid, entry in layers.items():
            R = entry.get("rotation") if isinstance(entry, dict) else entry
            if R is not None:
                hd = int(R.shape[-1])
                break
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("[oscar] could not probe head_dim of %s: %s", path, exc)
    _ROT_HD_CACHE[path] = hd
    return hd


def _resolve_quant_grouping(head_dim: int, group_size: Optional[int], tag: str):
    """Return (group_size, num_groups). Verbatim logic from the OSCAR fork's
    ``MHATokenToKVPool._resolve_quant_grouping``."""
    gs = head_dim if group_size is None else group_size
    if gs <= 0:
        raise ValueError(f"{tag} kv_cache_quant_group_size must be positive, got {gs}")
    if head_dim % gs != 0:
        raise ValueError(
            f"{tag} head_dim ({head_dim}) must be divisible by "
            f"kv_cache_quant_group_size ({gs})"
        )
    num_groups = head_dim // gs
    # The packed-byte layout puts 4 int2 values per byte, so a group boundary
    # must coincide with a byte-slot boundary => num_groups % 4 == 0 (or 1).
    # VERIFIED on hardware: head_dim=256 with --kv-cache-quant-group-size 128
    # gives num_groups=2 and the triton kernel raises NotImplementedError.
    if num_groups != 1 and (num_groups % 4 != 0 or (num_groups & (num_groups - 1))):
        raise ValueError(
            f"{tag} int2 quant grouping unsupported: head_dim={head_dim}, "
            f"group_size={gs} -> num_groups={num_groups}. The Triton kernels "
            f"need num_groups == 1 or (power-of-two AND divisible by 4). "
            f"For head_dim=256 use no group size (per-head scale) or 64."
        )
    return gs, num_groups


def is_int2_pool(pool) -> bool:
    return getattr(pool, "dtype", None) == INT2


def _oscar_init(self) -> None:
    """Convert an already-constructed MHATokenToKVPool into an OSCAR int2 pool."""
    head_dim = self.head_dim
    v_head_dim = self.v_head_dim
    if head_dim % 4 != 0 or v_head_dim % 4 != 0:
        raise ValueError(
            f"int2 KV needs head_dim and v_head_dim divisible by 4, got "
            f"{head_dim} / {v_head_dim}"
        )

    gsz = _QUANT_GROUP_SIZE_CTX
    self.kv_cache_quant_group_size = gsz
    self.k_quant_group_size, self.k_num_scale_groups = _resolve_quant_grouping(
        head_dim, gsz, "K"
    )
    self.v_quant_group_size, self.v_num_scale_groups = _resolve_quant_grouping(
        v_head_dim, gsz, "V"
    )
    self.scale_dtype = scale_dtype()

    # ---- flip the pool to int2 and rebuild the buffers -------------------
    self.dtype = INT2
    self.store_dtype = torch.uint8

    rows = self.size + self.page_size
    dev = self.device
    # Drop the dense bf16 buffers upstream just allocated before allocating the
    # packed ones, so peak VRAM is not dense + packed simultaneously.
    self.k_buffer = None
    self.v_buffer = None
    torch.cuda.empty_cache()

    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

    from contextlib import nullcontext

    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            self.k_buffer = [
                torch.zeros(
                    (rows, self.head_num, head_dim // 4),
                    dtype=torch.uint8,
                    device=dev,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (rows, self.head_num, v_head_dim // 4),
                    dtype=torch.uint8,
                    device=dev,
                )
                for _ in range(self.layer_num)
            ]
            self.k_scales_zeros = [
                torch.zeros(
                    (rows, self.head_num, 2 * self.k_num_scale_groups),
                    dtype=self.scale_dtype,
                    device=dev,
                )
                for _ in range(self.layer_num)
            ]
            self.v_scales_zeros = [
                torch.zeros(
                    (rows, self.head_num, 2 * self.v_num_scale_groups),
                    dtype=self.scale_dtype,
                    device=dev,
                )
                for _ in range(self.layer_num)
            ]

    # *** LOAD-BEARING FOR SPECULATIVE DECODING ***
    # Upstream's ``_slot_move_pointer_buffers()`` (which feeds
    # ``data_ptrs``/``data_strides``, and therefore ``move_kv_cache``) only
    # picks up extra buffers when they are named ``k_scale_buffer`` /
    # ``v_scale_buffer``. DSPARK calls ``move_kv_cache`` to relocate accepted
    # draft KV; without these aliases the packed int2 bytes would be moved but
    # their scale/zero pairs would NOT, silently corrupting every relocated
    # token. Aliasing (not copying) keeps a single set of tensors.
    # Verified: with the alias, move_kv_cache relocates data AND scales.
    self.k_scale_buffer = self.k_scales_zeros
    self.v_scale_buffer = self.v_scales_zeros
    self.dq_k_buffer = None
    self.dq_v_buffer = None

    # ---- rotations -------------------------------------------------------
    self._R_k = None
    self._R_v = None
    self._k_clip_ratio = 0.0
    self._v_clip_ratio = 0.0
    self._lloyd_max = False
    # Which worker is this pool for? The _IS_DRAFT_CTX flag is set by the dtype
    # resolver, but resolver order does not match pool-construction order (the
    # draft's dtype is resolved BEFORE the target's pool is built), so the flag
    # alone mis-attributes pools. Decide from the pool's own geometry, which is
    # unambiguous here: the target's full-attention layers have head_dim 256,
    # the DSPARK draft's have 128. Fall back to the flag if a rotation set only
    # exists for one of them.
    is_draft = bool(_IS_DRAFT_CTX)
    if oscar_enabled():
        tgt_hd = _rotation_head_dim(load_oscar_rotation_config(False).k_rotation_path)
        drf_hd = (
            _rotation_head_dim(load_oscar_rotation_config(True).k_rotation_path)
            if draft_oscar_enabled()
            else None
        )
        # Geometry is authoritative whenever the TARGET rotations are known:
        # a pool whose head_dim matches the target checkpoint IS the target,
        # regardless of a stale _IS_DRAFT_CTX left over from the draft's dtype
        # resolution (the draft's dtype is resolved before the target's pool is
        # built). Only fall back to the flag when head_dim is ambiguous, i.e.
        # both models happen to share a head_dim.
        if tgt_hd is not None and head_dim != tgt_hd:
            is_draft = True
        elif tgt_hd is not None and (drf_hd is None or drf_hd != tgt_hd):
            is_draft = False
        logger.info(
            "[oscar] pool head_dim=%d -> %s rotations (target hd=%s, draft hd=%s)",
            head_dim,
            "DRAFT" if is_draft else "TARGET",
            tgt_hd,
            drf_hd,
        )
    if is_draft and not draft_oscar_enabled():
        raise RuntimeError(
            "The DSPARK draft KV pool was asked for int2 but no draft rotation "
            "checkpoints are set. The draft has a different KV geometry from "
            "the target (head_dim %d here vs the target's), so the target's "
            "rotations cannot be reused. Calibrate draft rotations and set "
            "SGLANG_OSCAR_DRAFT_K_ROTATION_PATH / "
            "SGLANG_OSCAR_DRAFT_V_ROTATION_PATH." % head_dim
        )
    if oscar_enabled() or is_draft:
        cfg = load_oscar_rotation_config(draft=is_draft)
        rot_dtype = getattr(self, "model_dtype", None) or torch.bfloat16
        # The global->local layer-id remap only applies to the TARGET's hybrid
        # pool (checkpoint keyed 3,7,...,63 vs 16 local slots). The draft's
        # 5 layers are keyed 0..4 already, so it must use the contiguous
        # default or the remap would ask for layer 3 in a 0..4 checkpoint.
        layer_ids = None if is_draft else _ROTATION_LAYER_IDS_CTX
        self._R_k = load_oscar_rotations(
            cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=head_dim,
            device=torch.device(dev),
            dtype=rot_dtype,
            layer_ids=layer_ids,
        )
        self._R_v = load_oscar_rotations(
            cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=v_head_dim,
            device=torch.device(dev),
            dtype=rot_dtype,
            layer_ids=layer_ids,
        )
        self._k_clip_ratio = cfg.k_clip_ratio
        self._v_clip_ratio = cfg.v_clip_ratio
        self._lloyd_max = env_bool("SGLANG_LLOYD_MAX", False)
        logger.info(
            "[oscar] %s MHATokenToKVPool INT2 enabled: layers=%s k_clip=%.4f "
            "v_clip=%.4f lloyd_max=%s groups(k/v)=%d/%d scale_dtype=%s",
            "DRAFT" if is_draft else "TARGET",
            layer_ids,
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
            self.k_num_scale_groups,
            self.v_num_scale_groups,
            self.scale_dtype,
        )
    else:
        raise RuntimeError(
            "kv-cache-dtype int2 was requested but neither "
            "SGLANG_OSCAR_K_ROTATION_PATH nor SGLANG_OSCAR_V_ROTATION_PATH is "
            "set. This port only implements the OSCAR rotated int2 path; the "
            "fork's un-rotated Hadamard int2 fallback is not included."
        )

    # ---- re-derive the pointer/stride tables used by move_kv_cache --------
    # move_kv_cache is what spec decoding uses to relocate accepted draft KV,
    # so these MUST match the new (packed) buffers or DSPARK corrupts the cache.
    self._kv_buffer_descs = self._build_kv_buffer_descs()
    self._init_data_ptrs_and_strides()
    if getattr(self, "_kv_copy_config", None) is not None:
        # Row stride changed (head_dim -> head_dim//4 bytes), so the tiling
        # heuristic must be recomputed against the new stride.
        self._init_kv_copy_and_warmup()
        # ``_init_kv_copy_and_warmup`` sizes the grid's byte-tile count from
        # data_strides[0] (a K buffer) alone. data_ptrs now also covers the
        # scale/zero buffers, whose row stride differs. If any buffer were
        # WIDER than the one used for sizing, its tail bytes would never be
        # copied. Re-derive byte_tiles from the widest buffer.
        max_stride = int(self.data_strides.max().item())
        bpt = self._kv_copy_config["bytes_per_tile"]
        needed = (max_stride + bpt - 1) // bpt
        if needed > self._kv_copy_config["byte_tiles"]:
            logger.info(
                "[oscar] widening move_kv_cache byte_tiles %d -> %d "
                "(max buffer stride %d B)",
                self._kv_copy_config["byte_tiles"], needed, max_stride,
            )
            self._kv_copy_config["byte_tiles"] = needed

    self.row_dim = self.head_num * head_dim
    self.v_row_dim = self.head_num * v_head_dim
    self._finalize_allocation_log(self.size)


def _install_on_mha_pool(MHATokenToKVPool, unwrap_write_loc):
    """Wrap MHATokenToKVPool's methods. Idempotent."""
    if getattr(MHATokenToKVPool, "_oscar_int2_patched", False):
        return
    orig_init = MHATokenToKVPool.__init__
    orig_get_k = MHATokenToKVPool._get_key_buffer
    orig_get_v = MHATokenToKVPool._get_value_buffer
    orig_set = MHATokenToKVPool.set_kv_buffer
    orig_set_prefix_valid = getattr(
        MHATokenToKVPool, "set_kv_buffer_prefix_valid", None
    )

    def __init__(self, *args, **kwargs):
        # Upstream's KVCache.__init__ only understands torch dtypes. Accept the
        # string "int2" from callers, construct with the model dtype, then
        # convert. ``model_dtype`` is remembered for the rotation dtype.
        want_int2 = False
        model_dtype = kwargs.pop("model_dtype", None)
        if kwargs.get("dtype", None) == INT2:
            want_int2 = True
            kwargs["dtype"] = model_dtype or torch.bfloat16
        elif len(args) >= 3 and args[2] == INT2:
            want_int2 = True
            args = list(args)
            args[2] = model_dtype or torch.bfloat16
            args = tuple(args)
        if not want_int2:
            orig_init(self, *args, **kwargs)
            self.model_dtype = model_dtype or self.dtype
            return

        # int2: suppress only the DENSE ALLOCATION that upstream's __init__
        # performs. _oscar_init() replaces those buffers with packed uint8 ones
        # anyway, but allocating them first costs a TRANSIENT peak of the FULL
        # dense size -- 16 GiB at 256k ctx, which does not fit on a 32 GB card
        # and silently caps max_total_num_tokens.
        #
        # We stub ``_create_buffers_normal`` (the innermost allocator) rather
        # than ``_create_buffers``, because the latter ALSO builds
        # ``_kv_buffer_descs`` and calls ``_init_data_ptrs_and_strides()``.
        # Skipping those leaves the pool without ``data_strides`` and blows up
        # later in the DSPARK/kv-copy path with:
        #   AttributeError: 'MHATokenToKVPool' object has no attribute 'data_strides'
        # Allocate 0-row placeholders so the descriptor/pointer bookkeeping still
        # sees real tensors of the right dtype and trailing shape.
        cls = type(self)
        sentinel = object()
        prev = cls.__dict__.get("_create_buffers_normal", sentinel)

        def _stub_create_buffers_normal(_self):
            dt = _self.dtype if _self.store_dtype is None else _self.store_dtype
            _self.k_buffer = [
                torch.zeros(
                    (0, _self.head_num, _self.head_dim),
                    dtype=dt, device=_self.device,
                )
                for _ in range(_self.layer_num)
            ]
            _self.v_buffer = [
                torch.zeros(
                    (0, _self.head_num, _self.v_head_dim),
                    dtype=dt, device=_self.device,
                )
                for _ in range(_self.layer_num)
            ]

        # ``_init_kv_copy_and_warmup`` (enabled whenever speculative decoding is
        # on, i.e. always under DSPARK) launches a warmup copy kernel against
        # k_buffer/v_buffer. With the 0-row placeholders above that kernel reads
        # out of bounds -> "Triton Error [CUDA]: an illegal memory access".
        # Defer it until _oscar_init() has installed the real packed buffers.
        prev_warm = cls.__dict__.get("_init_kv_copy_and_warmup", sentinel)
        _deferred = []

        def _defer_warmup(_self, *a, **k):
            _deferred.append((a, k))

        cls._create_buffers_normal = _stub_create_buffers_normal
        cls._init_kv_copy_and_warmup = _defer_warmup
        try:
            orig_init(self, *args, **kwargs)
        finally:
            if prev is sentinel:
                del cls._create_buffers_normal
            else:
                cls._create_buffers_normal = prev
            if prev_warm is sentinel:
                del cls._init_kv_copy_and_warmup
            else:
                cls._init_kv_copy_and_warmup = prev_warm
        self.model_dtype = model_dtype or self.dtype
        _oscar_init(self)
        # Now that the packed buffers exist, run the deferred warmup.
        for a, k in _deferred:
            self._init_kv_copy_and_warmup(*a, **k)

    def _get_key_buffer(self, layer_id: int):
        if is_int2_pool(self):
            # Packed uint8 [rows, H, hd//4]; consumers must use the int2 kernels.
            return self.k_buffer[layer_id - self.start_layer]
        return orig_get_k(self, layer_id)

    def _get_value_buffer(self, layer_id: int):
        if is_int2_pool(self):
            return self.v_buffer[layer_id - self.start_layer]
        return orig_get_v(self, layer_id)

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
        # OSCAR-era kwargs, kept so the ported prefill path can pass them.
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        if not is_int2_pool(self):
            return orig_set(
                self,
                layer,
                loc_info,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
                dcp_kv_mask,
            )
        if dcp_kv_mask is not None:
            raise NotImplementedError("DCP masking is not supported with int2 KV.")

        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        idx = layer_id - self.start_layer

        R_k = self._R_k[idx]
        R_v = self._R_v[idx]
        v_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))
        if already_hadamard_transformed:
            ck = cache_k.to(R_k.dtype)
            cv = cache_v.to(R_v.dtype)
        else:
            ck = cache_k  # fused kernel rotates K in-register
            if v_absorbed:
                cv = cache_v.to(R_v.dtype).contiguous()
            else:
                cv = _rotate(cache_v, R_v)

        # Fused rotate(K)+clip+quant+pack: official OSCAR gates this behind
        # SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT (default False) because the fused
        # kernel implements UNIFORM min-max quant only — no Lloyd-Max path.
        # Match official semantics exactly: only use it when lloyd_max is off
        # (and respect the official env gate when explicitly requested).
        use_fused_rot = (
            not already_hadamard_transformed
            and not self._lloyd_max
            and env_bool("SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT", False)
            and R_k.dim() == 2
        )
        if use_fused_rot:
            from .kernels.oscar_rotation_clip_int2_kv import (
                quantized_set_kv_int2_oscar_rotate_k_clip_triton as _fused_rot_q,
            )
            _fused_rot_q(
                ck.to(R_k.dtype),
                cv,
                R_k,
                loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                self._k_clip_ratio,
                self._v_clip_ratio,
            )
            return

        if not already_hadamard_transformed:
            ck = _rotate(ck, R_k)

        quantized_set_kv_int2_pretransformed_clip_triton(
            ck,
            cv,
            loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            lloyd_max=self._lloyd_max,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer,
        loc_2d,
        commit_lens,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
    ):
        """int2-aware version of DSPARK's target-hidden KV injection.

        ``DSparkDraftModel.write_target_hidden_kv`` calls this to push the
        target's tapped hidden states into the DRAFT's KV cache. Upstream's
        implementation does ``cache_k.to(self.dtype)``; with our ``"int2"``
        sentinel that is parsed as a DEVICE and raises
        ``RuntimeError: Invalid device string: 'int2'``.

        ``loc_2d`` is [bs, max_commit] with only the first ``commit_lens[i]``
        entries valid per row. We flatten to the valid subset and reuse the
        ordinary int2 write path (rotate -> clip -> pack), which is exactly what
        the equivalent dense path does modulo quantization.
        """
        if not is_int2_pool(self):
            return orig_set_prefix_valid(
                self,
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
            )

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got {tuple(loc_2d.shape)}")

        dev = self.k_buffer[0].device
        loc_2d = loc_2d.to(device=dev, dtype=torch.int64, non_blocking=True)
        commit_lens = commit_lens.to(device=dev, non_blocking=True)

        # Build a boolean mask of the valid entries: col < commit_lens[row].
        cols = torch.arange(loc_2d.shape[1], device=dev).unsqueeze(0)
        mask = cols < commit_lens.to(torch.int64).unsqueeze(1)
        flat_mask = mask.reshape(-1)
        loc = loc_2d.reshape(-1)[flat_mask]
        if loc.numel() == 0:
            return

        ck = cache_k.reshape(-1, *cache_k.shape[1:])[flat_mask]
        cv = cache_v.reshape(-1, *cache_v.shape[1:])[flat_mask]
        if k_scale is not None:
            ck = ck / k_scale
        if v_scale is not None:
            cv = cv / v_scale

        return self.set_kv_buffer(
            layer,
            loc,
            ck,
            cv,
            layer_id_override=layer_id_override,
        )

    # --- accessors the int2 attention path needs -------------------------
    def get_raw_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_buffer[layer_id - self.start_layer]

    def get_raw_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_buffer[layer_id - self.start_layer]

    def get_key_scales_zeros(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_scales_zeros[layer_id - self.start_layer]

    def get_value_scales_zeros(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_scales_zeros[layer_id - self.start_layer]

    def get_raw_kv_buffer(self, layer_id: int):
        i = layer_id - self.start_layer
        return {
            "k_buffer": self.k_buffer[i],
            "v_buffer": self.v_buffer[i],
            "k_scales_zeros": self.k_scales_zeros[i] if is_int2_pool(self) else None,
            "v_scales_zeros": self.v_scales_zeros[i] if is_int2_pool(self) else None,
            "dtype": self.dtype,
        }

    MHATokenToKVPool.__init__ = __init__
    MHATokenToKVPool._get_key_buffer = _get_key_buffer
    MHATokenToKVPool._get_value_buffer = _get_value_buffer
    MHATokenToKVPool.set_kv_buffer = set_kv_buffer
    if orig_set_prefix_valid is not None:
        # DSPARK's target-hidden injection path; see the method docstring.
        MHATokenToKVPool.set_kv_buffer_prefix_valid = set_kv_buffer_prefix_valid
    MHATokenToKVPool.get_raw_key_buffer = get_raw_key_buffer
    MHATokenToKVPool.get_raw_value_buffer = get_raw_value_buffer
    MHATokenToKVPool.get_key_scales_zeros = get_key_scales_zeros
    MHATokenToKVPool.get_value_scales_zeros = get_value_scales_zeros
    MHATokenToKVPool.get_raw_kv_buffer = get_raw_kv_buffer
    MHATokenToKVPool._oscar_int2_patched = True
    logger.info("[oscar] patched MHATokenToKVPool for INT2")


def _rotate(t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply the OSCAR rotation along the last dim.

    ``R`` is ``[hd, hd]`` (V1, one matrix shared by all heads) or
    ``[kv_heads, hd, hd]`` (V2, per-head). For Qwen3.8-27B we expect V1
    ``[256, 256]``, but V2 is handled so per-head RotationZoo checkpoints work.
    """
    x = t.to(R.dtype)
    if R.dim() == 2:
        return (x @ R).contiguous()
    return torch.einsum("thd,hde->the", x, R).contiguous()


def _pool_v_head_dim(pool) -> int:
    """LOGICAL value head dim (256 here), tolerant of upstream API drift.

    ``MHATokenToKVPool`` keeps a plain ``v_head_dim`` attribute, but the OUTER
    ``HybridLinearKVPool`` does NOT -- hybrid_patch deliberately does not
    install a ``v_head_dim`` property because a data descriptor on the class
    would shadow upstream's instance assignment.

    *** Do NOT fall back to ``pool.get_v_head_dim()`` here. *** Upstream
    implements it as::

        return self.full_kv_pool.get_value_buffer(start_layer).shape[-1]

    i.e. it reports the *buffer* width. Under int2 the buffer is PACKED four
    values per byte, so that returns head_dim // 4 == 64 instead of 256, and
    the prefill path then builds a 64-wide dense tensor and dies in
    ``torch.cat`` against the 256-wide extend tensor.

    So: prefer the explicit attribute, then recurse into the inner pool (which
    does carry the logical value), and only then give up.
    """
    v = getattr(pool, "v_head_dim", None)
    if v is not None:
        return v
    inner = getattr(pool, "full_kv_pool", None)
    if inner is not None:
        return _pool_v_head_dim(inner)
    # Last resort: for this model K and V share head_dim.
    return pool.head_dim


def dequantize_int2_slots(pool, layer_id: int, indices: torch.Tensor, dtype):
    """Dequantize the int2 slots at ``indices`` to dense ``[N, H, hd]``.

    Used by the prefill path (the prefix is stored int2 and must be
    materialized dense before a standard attention kernel can consume it).
    """
    v_head_dim = _pool_v_head_dim(pool)
    if indices.numel() == 0:
        return (
            torch.empty((0, pool.head_num, pool.head_dim), dtype=dtype,
                        device=indices.device),
            torch.empty((0, pool.head_num, v_head_dim), dtype=dtype,
                        device=indices.device),
        )
    idx = indices.to(torch.int64)
    k = dequantize_kv_int2_triton(
        pool.get_raw_key_buffer(layer_id)[idx],
        pool.get_key_scales_zeros(layer_id)[idx],
        pool.head_dim,
        dtype,
    )
    v = dequantize_kv_int2_triton(
        pool.get_raw_value_buffer(layer_id)[idx],
        pool.get_value_scales_zeros(layer_id)[idx],
        v_head_dim,
        dtype,
    )
    return k, v
