"""Monkeypatch: ``HybridLinearKVPool`` int2 support + global->local layer remap.

Qwen3.8-27B is hybrid: 64 layers, ``full_attention_interval=4``, so only layers
3, 7, 11, ..., 63 (16 of them) hold a real KV cache. ``HybridLinearKVPool``
wraps an inner ``MHATokenToKVPool`` that is dense over just those 16 layers and
indexes them LOCALLY 0..15, while everything above the pool speaks GLOBAL ids.

Two things have to happen here:

1. The inner pool must be constructed with ``dtype="int2"``. Upstream's
   ``HybridLinearKVPool.__init__`` passes its own ``dtype`` straight through, so
   we just need ``self.kv_cache_dtype`` to already be the string ``"int2"`` --
   that is arranged in ``server_args_patch.py``.

2. The OSCAR rotation checkpoint is keyed by GLOBAL layer id, but the inner pool
   stores rotations at LOCAL indices. We publish ``full_attention_layer_ids`` to
   ``kv_pool_patch`` via a context variable right before the inner pool is
   built, so ``load_oscar_rotations(layer_ids=[3, 7, ..., 63])`` fills local
   slot i from global id ``full_attention_layer_ids[i]``.

We also add the int2 accessor forwards (``get_raw_key_buffer`` etc.), each of
which remaps global -> local, because the int2 attention read path calls them
with a global ``layer.layer_id``.

NOT ported from the OSCAR fork's HybridLinearKVPool: ``mixed_kv_enabled()``,
the ``_OscarRotationProxy`` for ``_R_k``/``_R_v``, ``get_hp_*_buffer``, and the
catch-all ``__getattr__`` forwarding -- all of those exist to serve the
mixed-KV (HP window) pool, which this port does not include. We DO provide
plain ``_R_k`` / ``_R_v`` properties that remap global -> local, since the
attention path needs them.
"""

from __future__ import annotations

import logging

from . import kv_pool_patch

logger = logging.getLogger(__name__)


class _RotationView:
    """Indexable view over the inner pool's rotation stack that accepts a
    GLOBAL layer id (or an already-local index) and returns the right matrix.

    The OSCAR fork used a ``_OscarRotationProxy`` for the same purpose. This is
    the trimmed version: no HP/mixed-KV concerns, just the id translation.
    """

    __slots__ = ("_stack", "_mapping", "_start_layer")

    def __init__(self, stack, mapping, start_layer):
        self._stack = stack
        self._mapping = mapping
        self._start_layer = start_layer

    def __getitem__(self, layer_id: int):
        if self._stack is None:
            raise KeyError("no OSCAR rotations loaded on the inner pool")
        # Callers reach here with either a global id (3, 7, 11, ...) or an
        # already-local index (0..15). Prefer the global mapping; fall back to
        # treating it as local when it is not a known global id.
        if layer_id in self._mapping:
            return self._stack[self._mapping[layer_id]]
        shifted = layer_id + self._start_layer
        if shifted in self._mapping:
            return self._stack[self._mapping[shifted]]
        return self._stack[layer_id]

    def __len__(self):
        return 0 if self._stack is None else len(self._stack)


def install(HybridLinearKVPool):
    if getattr(HybridLinearKVPool, "_oscar_int2_patched", False):
        return
    orig_init = HybridLinearKVPool.__init__

    def __init__(self, *args, **kwargs):
        # Grab full_attention_layer_ids whether it came positionally or by kw.
        layer_ids = kwargs.get("full_attention_layer_ids")
        if layer_ids is None and len(args) >= 6:
            layer_ids = args[5]
        dtype = kwargs.get("dtype")
        if dtype is None and len(args) >= 2:
            dtype = args[1]

        is_int2 = dtype == kv_pool_patch.INT2
        if is_int2:
            kv_pool_patch.set_rotation_context(
                layer_ids, kwargs.pop("kv_cache_quant_group_size", None)
            )
            logger.info(
                "[oscar] HybridLinearKVPool: int2 inner pool for %d "
                "full-attention layers (global ids %s)",
                len(layer_ids or []),
                list(layer_ids or [])[:4] + (["..."] if layer_ids and len(layer_ids) > 4 else []),
            )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            if is_int2:
                kv_pool_patch.clear_rotation_context()

    # Upstream already provides ``_transfer_full_attention_id`` (global -> local);
    # reuse it so the id mapping lives in exactly one place.
    def get_raw_key_buffer(self, layer_id: int):
        return self.full_kv_pool.get_raw_key_buffer(
            self._transfer_full_attention_id(layer_id)
        )

    def get_raw_value_buffer(self, layer_id: int):
        return self.full_kv_pool.get_raw_value_buffer(
            self._transfer_full_attention_id(layer_id)
        )

    def get_key_scales_zeros(self, layer_id: int):
        return self.full_kv_pool.get_key_scales_zeros(
            self._transfer_full_attention_id(layer_id)
        )

    def get_value_scales_zeros(self, layer_id: int):
        return self.full_kv_pool.get_value_scales_zeros(
            self._transfer_full_attention_id(layer_id)
        )

    def get_raw_kv_buffer(self, layer_id: int):
        return self.full_kv_pool.get_raw_kv_buffer(
            self._transfer_full_attention_id(layer_id)
        )

    @property
    def _R_k(self):
        return _RotationView(
            getattr(self.full_kv_pool, "_R_k", None),
            self.full_attention_layer_id_mapping,
            self.start_layer,
        )

    @property
    def _R_v(self):
        return _RotationView(
            getattr(self.full_kv_pool, "_R_v", None),
            self.full_attention_layer_id_mapping,
            self.start_layer,
        )

    @property
    def _k_clip_ratio(self):
        return self.full_kv_pool._k_clip_ratio

    @property
    def _v_clip_ratio(self):
        return self.full_kv_pool._v_clip_ratio

    # ------------------------------------------------------------------
    # set_kv_buffer passthrough for the OSCAR-only kwargs.
    #
    # The int2 WRITE path (attention_patch.forward_extend) calls
    #     pool.set_kv_buffer(..., already_hadamard_transformed=True)
    # on whatever pool the backend holds. For a hybrid model that is the OUTER
    # HybridLinearKVPool, whose upstream signature is
    #     (layer, loc, cache_k, cache_v, k_scale=1.0, v_scale=1.0,
    #      dcp_kv_mask=None)
    # -- it knows nothing about the OSCAR kwargs and raises TypeError.
    #
    # Upstream's own body just remaps the layer id and delegates to
    # ``self.full_kv_pool.set_kv_buffer``. We wrap it so the extra kwargs are
    # stripped off and forwarded to the INNER pool (which kv_pool_patch taught
    # to accept them). When the extra kwargs are absent we call straight
    # through to upstream so non-int2 behaviour is byte-identical.
    orig_set_kv_buffer = HybridLinearKVPool.set_kv_buffer

    _OSCAR_KWARGS = ("already_hadamard_transformed", "is_decode")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        extra = {k: kwargs.pop(k) for k in _OSCAR_KWARGS if k in kwargs}
        if not extra:
            return orig_set_kv_buffer(self, layer, loc, cache_k, cache_v,
                                      *args, **kwargs)
        # Delegate to the inner pool directly, applying the same global->local
        # layer id remap upstream would have done.
        return self.full_kv_pool.set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            *args,
            layer_id_override=self._transfer_full_attention_id(layer.layer_id),
            **kwargs,
            **extra,
        )

    # ------------------------------------------------------------------
    # get_v_head_dim must report the LOGICAL dim under int2.
    #
    # Upstream implements it as
    #     self.full_kv_pool.get_value_buffer(start_layer).shape[-1]
    # i.e. it infers the head dim from the buffer width. That is correct for
    # dense dtypes but WRONG for int2, where the buffer is packed 4 values per
    # byte -- it would report 64 instead of 256. Upstream calls this during
    # attention-backend setup, so a wrong value silently propagates into
    # output-buffer shapes. Serve the logical value from the inner pool.
    orig_get_v_head_dim = getattr(HybridLinearKVPool, "get_v_head_dim", None)

    def get_v_head_dim(self):
        inner = getattr(self, "full_kv_pool", None)
        v = getattr(inner, "v_head_dim", None) if inner is not None else None
        if v is not None and getattr(inner, "dtype", None) == kv_pool_patch.INT2:
            return v
        if orig_get_v_head_dim is not None:
            return orig_get_v_head_dim(self)
        return v if v is not None else self.head_dim

    if orig_get_v_head_dim is not None:
        HybridLinearKVPool.get_v_head_dim = get_v_head_dim

    HybridLinearKVPool.set_kv_buffer = set_kv_buffer
    HybridLinearKVPool.__init__ = __init__
    HybridLinearKVPool.get_raw_key_buffer = get_raw_key_buffer
    HybridLinearKVPool.get_raw_value_buffer = get_raw_value_buffer
    HybridLinearKVPool.get_key_scales_zeros = get_key_scales_zeros
    HybridLinearKVPool.get_value_scales_zeros = get_value_scales_zeros
    HybridLinearKVPool.get_raw_kv_buffer = get_raw_kv_buffer
    HybridLinearKVPool._R_k = _R_k
    HybridLinearKVPool._R_v = _R_v
    HybridLinearKVPool._k_clip_ratio = _k_clip_ratio
    HybridLinearKVPool._v_clip_ratio = _v_clip_ratio
    # NOTE: ``head_num`` is deliberately NOT installed. Upstream's __init__
    # assigns ``self.head_num = head_num`` as a plain instance attribute, and a
    # data-descriptor property on the class would shadow that assignment and
    # raise AttributeError (no setter). The instance attribute already has the
    # right value, so nothing is needed. Same reasoning for ``v_head_dim``:
    # upstream provides ``get_v_head_dim()``.
    HybridLinearKVPool._oscar_int2_patched = True
    logger.info("[oscar] patched HybridLinearKVPool for INT2")
