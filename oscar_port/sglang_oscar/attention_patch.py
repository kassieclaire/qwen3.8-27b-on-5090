"""Monkeypatch: teach the NEW upstream TritonAttnBackend to READ int2 KV.

*** THIS IS THE PART THE TASK BRIEF DID NOT ANTICIPATE. ***
Newer upstream SGLang has NO int2 attention path at all -- ``grep -c int2`` on
``sglang/kernels/ops/attention/decode_attention.py`` returns 0. The OSCAR fork
carried ~1600 lines of int2 Triton decode kernels plus a whole
``srt/layers/attention/quantized_kv_prefill.py``. Without them, an int2 pool is
write-only: ``get_key_buffer`` hands back a packed uint8 ``[rows, H, hd//4]``
tensor that upstream's bf16 kernels would silently misinterpret.

What this module wires up
-------------------------
DECODE  -> ``sglang_oscar.kernels.int2_decode_attention.decode_attention_fwd_quantized``
           (ported verbatim from the OSCAR fork; dequantizes inside the kernel).
           Q is rotated by R_k first; the output is inverse-rotated by R_v.T.

PREFILL -> the prefix slots are dequantized to dense bf16 with
           ``dequantize_kv_int2_triton``, concatenated with the freshly-rotated
           extend K/V, and fed to torch SDPA per request. This mirrors the OSCAR
           fork's ``_forward_extend_quantized_dense``, but ALWAYS takes the SDPA
           branch rather than FlashAttention:
             * head_dim is 256 here, and FA3 in this image is not built for
               sm_120 (Blackwell consumer) -- the fork's own comment says the
               same and falls back to SDPA for exactly these reasons;
             * it avoids depending on any particular flash_attn symbol surviving
               the upstream reshuffle.
           This is CORRECT but SLOWER than a fused kernel. See README.

Rotation math (identical to the fork):
    write: K' = K @ R_k,  V' = V @ R_v  (V skipped if absorbed into qkv_proj)
    read : Q' = Q @ R_k   so that Q'.K'^T == Q.(R_k R_k^T).K^T == Q.K^T
           out = (attn @ V') @ R_v^T    undoes the V rotation
Under GQA a per-head (V2 ``[kv_heads, hd, hd]``) rotation is repeat-interleaved
across the ``kv_group_num`` query heads that share each KV head.
"""

from __future__ import annotations

import logging

import torch

from .kernels.int2_decode_attention import decode_attention_fwd_quantized
from .kv_pool_patch import INT2, dequantize_int2_slots

logger = logging.getLogger(__name__)


def _pool_is_int2(pool) -> bool:
    return getattr(pool, "dtype", None) == INT2


def _rotate_q(q3: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """q3 is [tokens, q_heads, hd]. R is [hd,hd] or [kv_heads,hd,hd].

    Single broadcast matmul per call: [t, Hkv, kvg, d] @ [Hkv, 1, d, d].
    (The previous einsum + repeat_interleave materialized a [Hq, d, d]
    copy and ran 2-3 kernels per rotation -- ~115 tiny bf16 GEMMs per
    spec cycle, ~3.7 ms.)
    """
    x = q3.to(R.dtype)
    if R.dim() == 2:
        return (x @ R).contiguous()
    hkv = R.shape[0]
    kvg = max(1, x.shape[1] // hkv)
    if kvg == 1:
        # per-KV-head bmm, one kernel
        return torch.bmm(x.transpose(0, 1), R).transpose(0, 1).contiguous()
    xb = x.view(x.shape[0], hkv, kvg, R.shape[-1])
    out = torch.matmul(xb, R.unsqueeze(1))          # broadcast over kvg
    return out.reshape(x.shape[0], hkv * kvg, R.shape[-1]).contiguous()


def _inverse_rotate_out(o3: torch.Tensor, R_v: torch.Tensor) -> torch.Tensor:
    """out @ R_v^T. o3 is [tokens, q_heads, v_hd]."""
    x = o3.to(R_v.dtype)
    if R_v.dim() == 2:
        return (x @ R_v.T).to(o3.dtype)
    hkv = R_v.shape[0]
    kvg = max(1, x.shape[1] // hkv)
    if kvg == 1:
        return torch.bmm(x.transpose(0, 1), R_v.transpose(-1, -2)).transpose(0, 1).to(o3.dtype)
    xb = x.view(x.shape[0], hkv, kvg, R_v.shape[-1])
    out = torch.matmul(xb, R_v.transpose(-1, -2).unsqueeze(1))
    return out.reshape(x.shape[0], hkv * kvg, R_v.shape[-1]).to(o3.dtype).contiguous()


def _get_pool(self, forward_batch):
    """Resolve the KV pool across upstream API drift.

    The OSCAR fork read ``forward_batch.token_to_kv_pool``. Current upstream
    removed that attribute from ForwardBatch and keeps the pool on the
    attention backend instead (``self.token_to_kv_pool``, assigned from
    ``model_runner.token_to_kv_pool`` in TritonAttnBackend.__init__). Try the
    backend first, then fall back so this works on either layout.

    For a hybrid model the backend holds the OUTER HybridLinearKVPool; the
    int2 accessors are forwarded onto it by hybrid_patch, so returning the
    outer pool is correct.
    """
    pool = getattr(self, "token_to_kv_pool", None)
    if pool is None:
        pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None:
        mr = getattr(self, "model_runner", None)
        pool = getattr(mr, "token_to_kv_pool", None) if mr is not None else None
    return pool


def install(TritonAttnBackend):
    if getattr(TritonAttnBackend, "_oscar_int2_patched", False):
        return
    orig_decode = TritonAttnBackend.forward_decode
    orig_extend = TritonAttnBackend.forward_extend

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None,
        score_mod=None, aux_tensors=None,
    ):
        pool = _get_pool(self, forward_batch)
        if not _pool_is_int2(pool):
            return orig_decode(
                self, q, k, v, layer, forward_batch, save_kv_cache, sinks,
                score_mod, aux_tensors,
            )
        if score_mod is not None:
            raise NotImplementedError("score_mod is not supported with int2 KV.")
        if self.dcp_size > 1:
            raise NotImplementedError("DCP is not supported with int2 KV.")

        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = (
            q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            if layer.qk_head_dim != layer.v_head_dim
            else torch.empty_like(q)
        )

        if save_kv_cache and k is not None and v is not None:
            pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                ),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        from sglang.srt.layers.radix_attention import (  # noqa: F401
            AttentionType,
        )

        logits_soft_cap = _soft_cap(layer)

        q3 = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        lid = layer.layer_id
        R_k = pool._R_k[lid]
        R_v = pool._R_v[lid]
        # Rotation fold: pass NATURAL q; the grouped stage1 kernel applies
        # R_k in-registers (16 quarter-dots, one kv-head per program). The
        # output still comes back in R_v space, so the inverse below remains.
        q_pass = q3
        Rk_pass = R_k if R_k.dim() in (2, 3) else None

        o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        decode_attention_fwd_quantized(
            q_pass,
            pool.get_raw_key_buffer(lid),
            pool.get_raw_value_buffer(lid),
            pool.get_key_scales_zeros(lid),
            pool.get_value_scales_zeros(lid),
            o3,
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            INT2,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            R_k=Rk_pass,
        )
        o3.copy_(_inverse_rotate_out(o3, R_v))
        return o

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None,
        score_mod=None, aux_tensors=None,
    ):
        pool = _get_pool(self, forward_batch)
        if not _pool_is_int2(pool):
            return orig_extend(
                self, q, k, v, layer, forward_batch, save_kv_cache, sinks,
                score_mod, aux_tensors,
            )
        if k is None or v is None:
            raise NotImplementedError(
                "int2 extend requires explicit k/v (the None/None re-read path "
                "would need a dequantizing gather)."
            )
        if score_mod is not None:
            raise NotImplementedError("score_mod is not supported with int2 KV.")

        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        lid = layer.layer_id
        R_k = pool._R_k[lid]
        R_v = pool._R_v[lid]
        v_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        # ROTATION-FOLDED PATH: q stays NATURAL (un-rotated); the tile kernel
        # applies R_k in-registers after the Q load and applies R_v^T before
        # the store. k/v are still rotated on write (below) because the pool
        # layout is unchanged. This removes ~32 tiny torch GEMMs per verify
        # cycle (2 per layer: q-rotate + out-inverse) plus their HBM
        # round-trips, keeping only the k/v rotations that fold into the
        # quantize kernel's epilogue... which still run as one torch GEMM
        # each here -- see _rotate_kv_one_launch below.
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k3r, v3r = _rotate_kv_one_launch(k.contiguous(), v.contiguous(), R_k, R_v)
        k3 = k3r
        v3 = v3r if not v_absorbed else v.contiguous().to(R_v.dtype)

        # Write the ALREADY-ROTATED tensors so the pool does not rotate twice.
        if save_kv_cache:
            pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                ),
                k3,
                v3,
                layer.k_scale,
                layer.v_scale,
                already_hadamard_transformed=True,
            )

        prefix_indptr = self.forward_metadata.kv_indptr

        # ---------------------------------------------------------------
        # CUDA-GRAPH-SAFE PATH (required: --disable-cuda-graph is not an
        # option, it costs far too much decode throughput).
        #
        # DSPARK's TARGET_VERIFY is captured by the DECODE cuda-graph runner
        # (ForwardMode.TARGET_VERIFY is in is_extend(), NOT is_decode(), so it
        # lands here in forward_extend regardless of --speculative-attention-
        # mode). Everything below the capture boundary must therefore be
        # 100% device-side: no .item(), no .tolist(), no int(tensor), no
        # data-dependent Python loop, no allocation whose SHAPE depends on a
        # device value.
        #
        # The dense-SDPA fallback further below violates all of those AND
        # materialises the whole dequantised prefix ([seq_len, H, 256] bf16)
        # every step -- at 256k that is ~1 GiB per layer per step. It is kept
        # only for the non-capturable, genuinely-ragged first prefill.
        #
        # Strategy for the capturable case: split attention into two halves
        # that each have a static launch geometry, then merge them with the
        # standard log-sum-exp identity:
        #
        #   prefix half : int2 KV already in the pool -> reuse the SAME
        #                 int2 decode kernel used by forward_decode, which
        #                 dequantises INSIDE the kernel (no dense prefix
        #                 tensor ever exists) and is bounded by kv_indptr.
        #   extend half : the freshly-rotated k3/v3 for this step, run
        #                 through upstream's extend kernel with
        #                 skip_prefix=True so it only does the diagonal
        #                 block. Its k_buffer/v_buffer args are unused in
        #                 that mode, so the int2 packed buffers are never
        #                 misread as bf16.
        #
        #   out = (o_pre * exp(lse_pre - m) + o_ext * exp(lse_ext - m))
        #         / (exp(lse_pre - m) + exp(lse_ext - m))
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # FUSED CAUSAL TILE PATH (int2_causal_tile_attention)
        #
        # One kernel per layer replaces: the per-query-row int2 launch loop
        # + upstream dense extend kernel + the ~12-kernel LSE merge. It reads
        # the prefix AND this step's diagonal straight from the packed pool
        # (set_kv_buffer already ran above), with causal masking done by
        # absolute position. Numerics validated by tests/test_int2_tile_attention.py
        # (rel-err ~0.005 == pure int2 noise) across verify/prefill/draft
        # geometries, empty prefixes, grouped scales and multi-split.
        #
        # Requires: uniform extend length (TARGET_VERIFY by construction;
        # ordinary chunked prefill with equal lens or bs==1), no sinks, no
        # sliding window, causal. Falls back to the older two-kernel path or
        # the dense SDPA path otherwise.
        # ---------------------------------------------------------------
        if _can_use_graph_safe_extend(
            self, forward_batch, layer, sinks, q3.shape[0]
        ):
            ext_len_tok = q3.shape[0] // max(1, self.forward_metadata.kv_indptr.shape[0] - 1)
            fused_ok = (
                ext_len_tok > 0
                and not v_absorbed_inv_needed(layer)
                and R_k is not None and R_v is not None   # 2-D and 3-D both fold now
            )
            if fused_ok:
                return _extend_int2_fused_tile(
                    self, forward_batch, layer, pool, q3, k3, v3, R_k, R_v, lid,
                    ext_len_tok,
                )
            q3 = _rotate_q(q3, R_k)   # fallback paths expect rotated q
            return _extend_int2_graph_safe(
                self, forward_batch, layer, pool, q3, k3, v3, R_v, lid,
                sinks=sinks,
            )

        q3 = _rotate_q(q3, R_k)       # dense SDPA fallback also expects rotated

        # Dequantize the cached prefix (rotated space) and run SDPA per request.
        #
        # IMPORTANT: kv_indices is a PREALLOCATED buffer sized for the worst
        # case; only its first kv_indptr[-1] entries are valid for this batch.
        # The tail holds stale/garbage slot ids. Upstream's triton kernels are
        # bounded by kv_indptr and never touch it, but our gather (buf[idx])
        # reads the whole tensor -> "vectorized gather kernel index out of
        # bounds" device-side assert (seen under DSPARK TARGET_VERIFY, where
        # the buffer is sized for gamma+1 windows). Slice to the valid region:
        # it is both correct and strictly less work.
        _n_prefix = int(prefix_indptr[-1]) if prefix_indptr.numel() else 0
        prefix_k, prefix_v = dequantize_int2_slots(
            pool, lid, self.forward_metadata.kv_indices[:_n_prefix], q3.dtype
        )
        extend_start_loc = forward_batch.extend_start_loc
        # Same caveat as extend_seq_lens_cpu below: may be absent under
        # TARGET_VERIFY. Derived lazily from the reconstructed lengths.
        _derive_start_loc = extend_start_loc is None

        num_q_heads = q3.shape[1]
        out = q3.new_empty((q3.shape[0], num_q_heads, layer.v_head_dim))
        causal = _is_causal(layer, forward_batch)
        window = (
            layer.sliding_window_size
            if layer.sliding_window_size is not None and layer.sliding_window_size > 0
            else -1
        )

        # ``extend_seq_lens_cpu`` is None in some extend-family modes -- notably
        # DSPARK's TARGET_VERIFY, where every request verifies the SAME
        # fixed-width window (gamma+1) so upstream never materializes a host-side
        # per-request length list. Reconstruct it rather than crashing with
        # "TypeError: 'NoneType' object is not iterable".
        ext_lens_cpu = forward_batch.extend_seq_lens_cpu
        if ext_lens_cpu is None:
            esl = getattr(forward_batch, "extend_seq_lens", None)
            if esl is not None:
                ext_lens_cpu = esl.tolist()
            else:
                # Uniform split of the token block across the batch.
                bs = max(int(prefix_indptr.shape[0]) - 1, 1)
                total = int(q3.shape[0])
                if total % bs != 0:
                    raise RuntimeError(
                        "int2 prefill: cannot infer per-request extend lengths "
                        f"({total} tokens, {bs} requests, no extend_seq_lens)."
                    )
                ext_lens_cpu = [total // bs] * bs

        if _derive_start_loc:
            _acc, extend_start_loc = 0, []
            for _l in ext_lens_cpu:
                extend_start_loc.append(_acc)
                _acc += int(_l)

        for i, ext_len in enumerate(ext_lens_cpu):
            ext_len = int(ext_len)
            if ext_len == 0:
                continue
            ps, pe = int(prefix_indptr[i]), int(prefix_indptr[i + 1])
            es = int(extend_start_loc[i])
            ee = es + ext_len
            ki = torch.cat([prefix_k[ps:pe], k3[es:ee]], dim=0)
            vi = torch.cat([prefix_v[ps:pe], v3[es:ee]], dim=0)
            k_len = ki.shape[0]

            qi = q3[es:ee].transpose(0, 1).unsqueeze(0)   # [1,Hq,q,hd]
            kib = ki.transpose(0, 1).unsqueeze(0)         # [1,Hkv,k,hd]
            vib = vi.transpose(0, 1).unsqueeze(0)

            # Query position p maps to absolute key index (k_len - ext_len + p).
            q_abs = torch.arange(
                k_len - ext_len, k_len, device=q3.device
            ).unsqueeze(1)
            k_abs = torch.arange(k_len, device=q3.device).unsqueeze(0)
            allowed = torch.ones(
                (ext_len, k_len), dtype=torch.bool, device=q3.device
            )
            if causal:
                allowed &= k_abs <= q_abs
            if window >= 0:
                allowed &= k_abs >= (q_abs - window)
            mask = torch.zeros((ext_len, k_len), dtype=qi.dtype, device=q3.device)
            mask.masked_fill_(~allowed, float("-inf"))

            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, kib, vib,
                attn_mask=mask,
                scale=layer.scaling,
                enable_gqa=(num_q_heads != kib.shape[1]),
            )
            out[es:ee] = oi.squeeze(0).transpose(0, 1)

        out = _inverse_rotate_out(out, R_v)
        o = getattr(forward_batch, "_attn_output", None)
        if o is None:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        o.view(-1, num_q_heads, layer.v_head_dim).copy_(out)
        return o

    TritonAttnBackend.forward_decode = forward_decode
    TritonAttnBackend.forward_extend = forward_extend
    TritonAttnBackend._oscar_int2_patched = True
    logger.info("[oscar] patched TritonAttnBackend for INT2 read path")


def _can_use_graph_safe_extend(self, forward_batch, layer, sinks, ntok) -> bool:
    """Is the fused (prefix-int2 + extend) two-kernel path usable?

    Requirements:
      * uniform per-request extend length -- true for TARGET_VERIFY (every
        request verifies the same gamma+1 window). This is what makes
        ``qo_indptr`` a pure arange and lets ``per_req`` come from SHAPES
        rather than device values (mandatory under cuda-graph capture).
      * no sliding window (this model has none; the windowed path uses a
        different indptr pair and would need its own merge).
      * no attention sinks: a sink term must be folded into the softmax
        denominator exactly once, but our two halves each run their own
        softmax. Applying it in one half and not the other would be wrong,
        so bail out rather than compute a subtly incorrect result.
      * causal, non-cross attention.
    """
    fm = self.forward_metadata
    if getattr(fm, "qo_indptr", None) is None:
        return False
    if getattr(fm, "kv_indptr", None) is None:
        return False
    if sinks is not None:
        return False
    if layer.sliding_window_size is not None and layer.sliding_window_size > 0:
        return False
    if layer.is_cross_attention:
        return False
    if not hasattr(self, "extend_attention_fwd"):
        return False
    kvp = fm.kv_indptr
    bs = kvp.shape[0] - 1
    if bs <= 0 or ntok % bs != 0:
        return False

    mode = forward_batch.forward_mode
    if getattr(mode, "is_target_verify", lambda: False)():
        # Captured by the decode cuda-graph runner: SHAPE-ONLY checks above.
        # Every request verifies the same gamma+1 window, so uniformity holds
        # by construction and needs no device read.
        return True

    # Ordinary (chunked) prefill. NOT cuda-graph captured -- we run with
    # --disable-prefill-cuda-graph because the prefill graph costs ~1.1GB for
    # little gain -- so a host-side uniformity check is allowed here.
    #
    # This matters a lot: the dense-SDPA fallback materialises the entire
    # dequantised prefix per chunk, which measured ~95 tok/s of prefill (i.e.
    # ~14 minutes for an 80k-token prompt). The two-kernel path keeps the
    # prefix packed and is orders of magnitude cheaper.
    if not getattr(mode, "is_extend", lambda: False)():
        return False
    if bs == 1:
        return True   # trivially uniform
    lens = forward_batch.extend_seq_lens_cpu
    if lens is None:
        return False
    return len(set(int(x) for x in lens)) == 1


def _extend_int2_graph_safe(
    self, forward_batch, layer, pool, q3, k3, v3, R_v, lid, sinks=None
):
    """Two static kernel launches + an LSE merge. No host syncs anywhere.

    Returns the flattened ``[tokens, q_heads * v_head_dim]`` output that
    ``RadixAttention.forward`` expects.
    """
    fm = self.forward_metadata
    nq = layer.tp_q_head_num
    hd = q3.shape[-1]
    vhd = layer.v_head_dim
    ntok = q3.shape[0]
    dev = q3.device

    qo = fm.qo_indptr
    kvp = fm.kv_indptr
    # Both derived from tensor SHAPES, not values -> static under capture.
    bs = kvp.shape[0] - 1
    per_req = ntok // max(bs, 1)

    # NOTE on the prefix/extend split (investigated 2026-08-15, do not "fix"):
    # dspark_verify.py DOES add the verify window to batch.seq_lens
    # (`batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w`, line ~252, taken
    # because the predicate is `hasattr(backend,
    # "make_forward_metadata_from_raw_verify")` and the triton backend has no
    # such attr). BUT `_forward_prepared_verify` RESTORES the backup at line
    # ~285 *before* `target_worker.forward_batch_generation`, so the inflated
    # value only sizes the allocation -- the metadata the backend actually
    # sees is prefix-only. The extend kernel confirms the convention:
    #     cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend   (line 388)
    # i.e. kv_indptr covers ONLY the prefix and the diagonal comes from
    # qo_indptr. So the two halves here are already disjoint.
    # Shortening the prefix by per_req was tried and is WRONG: it drops real
    # context and collapsed accept-len from 6.95 to ~1.35.
    kv_indices = fm.kv_indices

    # ---- half 1: prefix, straight out of the int2 pool -------------------
    # The int2 decode kernel's batch axis is indexed by kv_indptr, so it must
    # be the REQUEST axis (kv_indptr has bs+1 entries). Under TARGET_VERIFY
    # each request carries per_req (= gamma+1) query rows, which do not fit
    # that axis.
    #
    # A previous version folded the window onto the HEAD axis
    # (flat = real_head*per_req + t). The index arithmetic for that is sound
    # -- kv_head = flat//(H_flat//H_kv) does map back correctly, and
    # tests/verify_path_check.py confirms the round trip -- but the KERNEL
    # does not honour it: tests/int2_verify_kernel_check.py measures rel err
    # 0.007 at per_req=1 (normal 2-bit loss) versus 1.27 at per_req=7. The
    # grouped-GQA tiling evidently assumes each kv head owns a CONTIGUOUS run
    # of q heads, which head-major folding breaks. That silent wrongness is
    # exactly what made DSPARK emit `wfile.write.write` and dropped indents.
    #
    # So: issue one kernel call per window position, each with the plain
    # [bs, nq, hd] layout the kernel is known-correct for. per_req is derived
    # from SHAPES (ntok // bs), so the loop length is a Python constant at
    # capture time -- the graph just records per_req launches. All prefix
    # positions attend to the same prefix, so kvp/kv_indices are reused as-is.
    budget_rows = 256
    splits = max(1, min(int(self.max_kv_splits), budget_rows // max(nq, 1)))
    attn_logits = torch.empty(
        (bs, nq, splits, vhd), dtype=torch.float32, device=dev,
    )
    attn_lse = torch.empty(
        (bs, nq, splits), dtype=torch.float32, device=dev,
    )
    # forward_metadata.num_kv_splits is sized for the decode path and is not
    # guaranteed to be populated here; own it (shape [bs] is static, so this
    # stays capture-safe).
    num_kv_splits = torch.full((bs,), splits, dtype=torch.int32, device=dev)

    q_win = q3.view(bs, per_req, nq, hd)
    o_pre = q3.new_empty((bs, per_req, nq, vhd))
    lse_pre = torch.empty((bs, per_req, nq), dtype=torch.float32, device=dev)

    k_raw = pool.get_raw_key_buffer(lid)
    v_raw = pool.get_raw_value_buffer(lid)
    k_sz = pool.get_key_scales_zeros(lid)
    v_sz = pool.get_value_scales_zeros(lid)

    for t in range(per_req):
        decode_attention_fwd_quantized(
            q_win[:, t].contiguous(),
            k_raw,
            v_raw,
            k_sz,
            v_sz,
            o_pre[:, t],
            kvp,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            splits,   # must match attn_logits.shape[2] (the kernel asserts it)
            layer.scaling,
            INT2,
            logit_cap=_soft_cap(layer),
            sinks=None,   # gated off in _can_use_graph_safe_extend
            xai_temperature_len=layer.xai_temperature_len,
            output_lse=lse_pre[:, t],
        )

    o_pre = o_pre.reshape(ntok, nq, vhd)
    lse_pre = lse_pre.reshape(ntok, nq)

    # ---- half 2: the diagonal (this step's freshly-rotated k3/v3) --------
    # skip_prefix=True => the kernel never *loads* from k_buffer/v_buffer.
    # BUT it still SPECIALIZES on their dtype at Triton compile time:
    #     qk = tl.dot(q.to(k.dtype), k)
    # With the packed uint8 int2 buffers that becomes tl.dot(uint8, uint8) ->
    # "AssertionError: only int8 supported!" during capture. So hand it a
    # zero-element tensor of the MODEL dtype: same rank/strides for the
    # compiler, no storage, never read.
    # custom_mask/mask_indptr are passed through rather than forced to None so
    # a tree-shaped draft (if ever used) keeps its mask; skip_prefix_custom_mask
    # defaults True, i.e. the mask applies only to this diagonal block.
    kbuf_dummy = q3.new_empty((0, k3.shape[1], hd))
    vbuf_dummy = q3.new_empty((0, v3.shape[1], vhd))
    _zeros_indptr = torch.zeros_like(kvp)
    o_ext = q3.new_empty((ntok, nq, vhd))
    lse_ext = torch.empty((ntok, nq), dtype=torch.float32, device=dev)
    self.extend_attention_fwd(
        q3,
        k3.to(q3.dtype),
        v3.to(q3.dtype),
        o_ext,
        kbuf_dummy,
        vbuf_dummy,
        qo,
        # Zero prefix length for this half: the prefix is handled entirely by
        # the int2 kernel above. Passing the real kv_indptr here would make
        # the kernel offset its causal mask by cur_seq_len_prefix (see
        # `offs_qidx = cur_seq_len_prefix + ...`), shifting the diagonal.
        _zeros_indptr,
        fm.kv_indices,
        # custom_mask is indexed against the FULL (prefix+extend) key axis, so
        # it cannot be reused now that this half sees prefix_len == 0. DSPARK
        # runs topk=1 (a causal chain, block_size=7 -> gamma+1 window), for
        # which plain causal masking over the diagonal block is exactly right.
        # _can_use_graph_safe_extend() restricts this path to TARGET_VERIFY, so
        # a tree-shaped draft never reaches here.
        None,                       # custom_mask
        True,                       # is_causal
        None,                       # mask_indptr
        per_req,                    # max_len_extend (uniform by construction)
        # k_descale / v_descale, NOT the layer's Optional[Tensor] k_scale.
        # Upstream's own forward_extend passes 1.0 when layer.k_scale is None
        # (a None reaches Triton as a typeless arg -> "'NoneType' object has no
        # attribute 'type'"). Our K/V are already dequantized-to-model-dtype by
        # the rotation step, so no descale is wanted here regardless.
        1.0,
        1.0,
        sm_scale=layer.scaling,
        logit_cap=_soft_cap(layer),
        lse_extend=lse_ext,
        skip_prefix=True,
    )

    # ---- merge the two halves by log-sum-exp ----------------------------
    # An empty prefix (seq_len == extend_len, i.e. a brand-new request) makes
    # lse_pre == -inf and o_pre garbage/NaN. m would then be finite (lse_ext),
    # so w_pre == exp(-inf) == 0 -- correct weight -- but 0 * NaN == NaN. So
    # sanitise both halves before combining rather than relying on the weight.
    #
    # Both halves are also fully masked-out for no other reason here (causal
    # diagonal always has >=1 visible key), so lse_ext is always finite.
    neg_inf_pre = torch.isneginf(lse_pre) | torch.isnan(lse_pre)
    lse_pre = torch.where(neg_inf_pre, torch.full_like(lse_pre, -1e30), lse_pre)
    o_pre = torch.nan_to_num(o_pre, nan=0.0, posinf=0.0, neginf=0.0)

    m = torch.maximum(lse_pre, lse_ext)
    w_pre = torch.exp(lse_pre - m).unsqueeze(-1)
    w_ext = torch.exp(lse_ext - m).unsqueeze(-1)
    out = (o_pre.float() * w_pre + o_ext.float() * w_ext) / (w_pre + w_ext)
    out = out.to(q3.dtype)

    out = _inverse_rotate_out(out, R_v)
    o = q3.new_empty((ntok, nq * vhd))
    o.view(ntok, nq, vhd).copy_(out)
    return o


def _soft_cap(layer):
    try:
        from sglang.srt.layers.attention.triton_backend import logit_capping_mod

        return logit_capping_mod(layer.logit_capping_method, layer.logit_cap)
    except Exception:
        return getattr(layer, "logit_cap", 0.0) or 0.0


def _is_causal(layer, forward_batch) -> bool:
    from sglang.srt.layers.radix_attention import AttentionType

    if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
        return False
    return True


def v_absorbed_inv_needed(layer) -> bool:
    """The fused tile kernel writes output in the V-rotated space, so the
    inverse rotation must still be applied by the caller (it is, below).
    This predicate exists to keep the gate explicit if a future variant
    skips it."""
    return False


def _rotate_kv_one_launch(k, v, R_k, R_v):
    """Rotate k by R_k and v by R_v. [t, Hkv, d] inputs with [Hkv, d, d]
    rotations: ONE bmm per tensor (no repeat_interleave copy)."""
    kb = k.to(R_k.dtype)
    vb = v.to(R_v.dtype)
    if R_k.dim() == 2:
        kr = (kb @ R_k).contiguous()
    else:
        kr = torch.bmm(kb.transpose(0, 1), R_k).transpose(0, 1).contiguous()
    if R_v.dim() == 2:
        vr = (vb @ R_v).contiguous()
    else:
        vr = torch.bmm(vb.transpose(0, 1), R_v).transpose(0, 1).contiguous()
    return kr, vr


def _extend_int2_fused_tile(
    self, forward_batch, layer, pool, q3, k3, v3, R_k, R_v, lid, ext_len_tok,
):
    """Single fused causal int2 kernel: prefix + diagonal from the pool.

    The diagonal tokens were already quantized into the pool by
    ``set_kv_buffer`` above (out_cache_loc slots), so attention sees ONE
    contiguous key axis and needs no separate extend kernel, no LSE merge,
    and -- for prefill -- reads each prefix byte BLOCK_M times fewer than
    the old per-row loop.

    Rotations are folded INTO the kernel: q arrives natural, the kernel
    applies R_k[h] in-registers before the QK dots and applies R_v[h]^T
    in-registers before the store, so the caller performs no torch-side
    q-rotate / out-inverse GEMMs for this path.
    """
    from .kernels.int2_tile_attention import int2_causal_tile_attention

    fm = self.forward_metadata
    nq = layer.tp_q_head_num
    hd = q3.shape[-1]
    vhd = layer.v_head_dim
    ntok = q3.shape[0]

    kv_group_num = nq // (pool.get_raw_key_buffer(lid).shape[1])

    Rk_pass = R_k.contiguous() if R_k.dim() == 3 else (
        R_k.contiguous().unsqueeze(0) if R_k.dim() == 2 else None)
    Rv_pass = R_v.contiguous() if R_v.dim() == 3 else (
        R_v.contiguous().unsqueeze(0) if R_v.dim() == 2 else None)

    out = int2_causal_tile_attention(
        q3,
        pool.get_raw_key_buffer(lid),
        pool.get_raw_value_buffer(lid),
        pool.get_key_scales_zeros(lid),
        pool.get_value_scales_zeros(lid),
        fm.kv_indptr,
        fm.kv_indices,
        forward_batch.out_cache_loc,
        None,                      # prefix_lens: derived in-kernel from indptr
        ext_len_tok,
        kv_group_num,
        layer.scaling,
        logit_cap=_soft_cap(layer),
        splits=1,
        Rk=Rk_pass,
        Rv=Rv_pass,
    )

    if Rk_pass is None:
        raise AssertionError("fused tile path requires per-head [Hkv, d, d] rotations")
    o = q3.new_empty((ntok, nq * vhd))
    o.view(ntok, nq, vhd).copy_(out)
    return o
