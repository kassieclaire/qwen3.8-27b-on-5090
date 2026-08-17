"""Fused causal INT2-KV tile attention for extend/verify — 1-2 launches per layer.

Replaces ``attention_patch._extend_int2_graph_safe``'s per-query-row launch
loop (EXT launches/layer: 8 for DSPARK verify, 2048 for a prefill chunk) +
upstream dense extend kernel + ~12-kernel LSE merge.

Why ONE kernel suffices
-----------------------
``set_kv_buffer`` runs BEFORE attention, so this step's diagonal (extend)
tokens are ALREADY quantized into the int2 pool at ``out_cache_loc`` slots.
The metadata's ``kv_indptr``/``kv_indices`` cover ONLY the prefix (verified:
dspark's verify path restores prefix-only seq_lens before the target
forward), so the kernel walks TWO key sets with one online-softmax
accumulator:

  prefix  : slots kv_indices[kv_indptr[r] : kv_indptr[r+1]], absolute
            positions [0, pfx_r) -- visible to every query row (causal).
  diagonal: slots out_cache_loc[r*EXT + j], absolute positions
            [pfx_r, pfx_r+EXT); query row for token t sees diagonal key j
            iff j <= t.

For prefill this also amortizes every prefix K/V byte over BLOCK_M query
rows instead of 1 (the old loop re-read the whole prefix PER query row).

Row order / GQA (zero-copy)
---------------------------
Q keeps its natural [ntok, Hq, D] layout. The M axis enumerates
(req, kv_head, ext_token, g) with g = q-head-within-kv-head:
row m of tile (req, kv_head) is (tok=m//KVG, g=m%KVG); its Q/O address is
(req*EXT + tok)*stride_tok + (kv_head*KVG + g)*stride_h. KVG constexpr →
div/mod free. One tile serves exactly one (req, kv-head) pair, which is
what lets a single K/V tile feed all BLOCK_M rows.

INT2 unpack (identical to the vendored decode kernel + pool writer):
  element d -> byte d % (L/4), crumb d // (L/4); dequant (crumb - zero)*scale.
  K is loaded TRANSPOSED [D/4, N] (decode-kernel pattern) and the crumb
  planes join+permute+reshape into a natural-order [D, N] matrix, so
  tl.dot(q_nat, k_nat) is a plain GEMM and Q needs no reordering.
  V stays in crumb order; the PV products accumulate into 4 quarter slices
  stored at natural dim offsets k*L/4.
  Scales: per (token, head) interleaved [s0, z0, s1, z1, ...];
  GROUP_SIZE = L // num_groups; the pool guarantees num_groups == 1 or
  (power-of-two and %4 == 0), so a group never straddles a crumb plane:
  group(crumb q, byte i) = q*(num_groups/4) + i // GROUP_SIZE.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MIN_BLOCK_KV = 32


@triton.jit
def _tanh_(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _int2_tile_causal_stage1(
    Q,                    # [ntok, Hq, D] bf16 (natural, rotated space)
    K_Buffer,             # [cache, Hkv, L/4] uint8
    V_Buffer,             # [cache, Hkv, L/4] uint8
    K_SZ,                 # [cache, Hkv, 2*groups] f32
    V_SZ,                 # [cache, Hkv, 2*groups] f32
    sm_scale,
    kv_indptr,            # [bs+1] int32 -- PREFIX span per request
    kv_indices,           # int32 prefix slot ids
    out_cache_loc,        # int32 [>=bs*EXT] diagonal slot ids
    Out,                  # SPLITS==1: [ntok, Hq, D] model-dtype output
    Mid,                  # SPLITS>1 : [ntok, Hq, SPLITS, D] f32 partials
    Lse,                  # SPLITS>1 : [ntok, Hq, SPLITS] f32
    stride_q_tok,
    stride_q_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_o_tok,
    stride_o_h,
    stride_o_s,
    Rk_ptr,                # [Hkv, D, D] bf16 or null (HAS_ROT=0)
    Rv_ptr,                # [Hkv, D, D] bf16 or null
    EXT,                  # runtime int: extend tokens per request (uniform)
    TILES_PER_RH,         # runtime int: cdiv(EXT*KVG, BLOCK_M)
    KVG: tl.constexpr,    # q heads per kv head
    HKV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    L: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_ROT: tl.constexpr,  # 1: rotate q by R_k[h % NRK] and out by R_v[h % NRV]^T in-kernel
    NRK: tl.constexpr,     # number of distinct R_k matrices (1 for shared 2-D)
    NRV: tl.constexpr,
):
    pid0 = tl.program_id(0)
    split_kv_id = tl.program_id(1)

    GROUPED: tl.constexpr = GROUP_SIZE < L
    NUM_GROUPS: tl.constexpr = L // GROUP_SIZE
    NGQ: tl.constexpr = NUM_GROUPS // 4  # groups per crumb plane

    rh = pid0 // TILES_PER_RH
    tile = pid0 % TILES_PER_RH
    req = rh // HKV
    cur_kv_head = rh % HKV

    kv_start = tl.load(kv_indptr + req)
    pfx = tl.load(kv_indptr + req + 1) - kv_start
    kv_len = pfx + EXT

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(kv_len, SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, kv_len)

    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < (EXT * KVG)
    tok = offs_m // KVG
    g = offs_m % KVG
    q_head = cur_kv_head * KVG + g

    offs_qp = tl.arange(0, BLOCK_D // 4)
    qmask = row_ok[:, None] & (offs_qp[None, :] < (L // 4))

    # Q quarters stay in crumb-aligned order: no join/permute (SM120 has only
    # 99KB shared; layout-converting a [M,256] tile through tl.permute
    # exceeded it). The QK product is 4 quarter-dots summed -- mathematically
    # identical to one full dot because both operands share the crumb order.
    q_row = (Q + (req * EXT + tok)[:, None] * stride_q_tok
             + q_head[:, None] * stride_q_h)
    q_q0 = tl.load(q_row + offs_qp[None, :], mask=qmask, other=0.0)
    q_q1 = tl.load(q_row + (offs_qp + L // 4)[None, :], mask=qmask, other=0.0)
    q_q2 = tl.load(q_row + (offs_qp + 2 * (L // 4))[None, :], mask=qmask, other=0.0)
    q_q3 = tl.load(q_row + (offs_qp + 3 * (L // 4))[None, :], mask=qmask, other=0.0)

    if HAS_ROT:
        # Fold the q-side OSCAR rotation into the kernel. The pool stores
        # k already rotated (k @ R_k[h]); scores need (q @ R_k[h]) . k_stored.
        # R_k[h] is identical for every Q head of kv head h and this tile
        # serves exactly one (req, h) pair, so 16 [M,QTR]x[QTR,QTR] block-dots
        # in registers replace a separate [ntok, Hq, D] GEMM launch + HBM
        # round-trip. Quarter e of q covers natural dims [e*QTR, (e+1)*QTR);
        # out quarter c = sum_e q_e @ R[e*QTR.., c*QTR..].
        QTR: tl.constexpr = L // 4
        rk_base = Rk_ptr + (cur_kv_head % NRK).to(tl.int64) * L * L
        r_q0 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
        r_q1 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
        r_q2 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
        r_q3 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
        offs_col = tl.arange(0, BLOCK_D // 4)
        # e = 0
        rb = rk_base + 0 * L
        r_q0 += tl.dot(q_q0, tl.load(rb + offs_qp[:, None] * L + offs_col[None, :]).to(q_q0.dtype))
        r_q1 += tl.dot(q_q0, tl.load(rb + offs_qp[:, None] * L + QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q2 += tl.dot(q_q0, tl.load(rb + offs_qp[:, None] * L + 2 * QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q3 += tl.dot(q_q0, tl.load(rb + offs_qp[:, None] * L + 3 * QTR + offs_col[None, :]).to(q_q0.dtype))
        # e = 1
        rb = rk_base + QTR * L
        r_q0 += tl.dot(q_q1, tl.load(rb + offs_qp[:, None] * L + offs_col[None, :]).to(q_q0.dtype))
        r_q1 += tl.dot(q_q1, tl.load(rb + offs_qp[:, None] * L + QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q2 += tl.dot(q_q1, tl.load(rb + offs_qp[:, None] * L + 2 * QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q3 += tl.dot(q_q1, tl.load(rb + offs_qp[:, None] * L + 3 * QTR + offs_col[None, :]).to(q_q0.dtype))
        # e = 2
        rb = rk_base + 2 * QTR * L
        r_q0 += tl.dot(q_q2, tl.load(rb + offs_qp[:, None] * L + offs_col[None, :]).to(q_q0.dtype))
        r_q1 += tl.dot(q_q2, tl.load(rb + offs_qp[:, None] * L + QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q2 += tl.dot(q_q2, tl.load(rb + offs_qp[:, None] * L + 2 * QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q3 += tl.dot(q_q2, tl.load(rb + offs_qp[:, None] * L + 3 * QTR + offs_col[None, :]).to(q_q0.dtype))
        # e = 3
        rb = rk_base + 3 * QTR * L
        r_q0 += tl.dot(q_q3, tl.load(rb + offs_qp[:, None] * L + offs_col[None, :]).to(q_q0.dtype))
        r_q1 += tl.dot(q_q3, tl.load(rb + offs_qp[:, None] * L + QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q2 += tl.dot(q_q3, tl.load(rb + offs_qp[:, None] * L + 2 * QTR + offs_col[None, :]).to(q_q0.dtype))
        r_q3 += tl.dot(q_q3, tl.load(rb + offs_qp[:, None] * L + 3 * QTR + offs_col[None, :]).to(q_q0.dtype))
        q_q0 = r_q0.to(q_q0.dtype)
        q_q1 = r_q1.to(q_q0.dtype)
        q_q2 = r_q2.to(q_q0.dtype)
        q_q3 = r_q3.to(q_q0.dtype)

    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_q0 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)

    if split_end > split_start:
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < split_end
            is_pfx = offs_n < pfx
            j = tl.where(is_pfx, 0, offs_n - pfx)   # safe: never negative in addr
            slot_p = tl.load(kv_indices + kv_start + offs_n,
                             mask=n_ok & is_pfx, other=0)
            slot_d = tl.load(out_cache_loc + req * EXT + j,
                             mask=n_ok & (~is_pfx), other=0)
            slot = tl.where(is_pfx, slot_p, slot_d)

            # ---- K packed, TRANSPOSED [D/4, N] (decode-kernel pattern) ----
            k_pack = tl.load(
                K_Buffer + slot[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh + offs_qp[:, None],
                mask=n_ok[None, :] & (offs_qp[:, None] < (L // 4)),
                other=0,
            )
            if GROUPED:
                gq = offs_qp // GROUP_SIZE
                szk = slot[None, :] * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_s0 = tl.load(K_SZ + szk + 2 * gq[:, None], mask=n_ok[None, :], other=1.0).to(q_q0.dtype)
                k_z0 = tl.load(K_SZ + szk + 2 * gq[:, None] + 1, mask=n_ok[None, :], other=0.0).to(q_q0.dtype)
                k_s1 = tl.load(K_SZ + szk + 2 * (gq + NGQ)[:, None], mask=n_ok[None, :], other=1.0).to(q_q0.dtype)
                k_z1 = tl.load(K_SZ + szk + 2 * (gq + NGQ)[:, None] + 1, mask=n_ok[None, :], other=0.0).to(q_q0.dtype)
                k_s2 = tl.load(K_SZ + szk + 2 * (gq + 2 * NGQ)[:, None], mask=n_ok[None, :], other=1.0).to(q_q0.dtype)
                k_z2 = tl.load(K_SZ + szk + 2 * (gq + 2 * NGQ)[:, None] + 1, mask=n_ok[None, :], other=0.0).to(q_q0.dtype)
                k_s3 = tl.load(K_SZ + szk + 2 * (gq + 3 * NGQ)[:, None], mask=n_ok[None, :], other=1.0).to(q_q0.dtype)
                k_z3 = tl.load(K_SZ + szk + 2 * (gq + 3 * NGQ)[:, None] + 1, mask=n_ok[None, :], other=0.0).to(q_q0.dtype)
            else:
                szk = slot * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_s0 = tl.load(K_SZ + szk + 0, mask=n_ok, other=1.0).to(q_q0.dtype)
                k_z0 = tl.load(K_SZ + szk + 1, mask=n_ok, other=0.0).to(q_q0.dtype)
                k_s0 = k_s0[None, :]
                k_z0 = k_z0[None, :]
                k_s1, k_z1 = k_s0, k_z0
                k_s2, k_z2 = k_s0, k_z0
                k_s3, k_z3 = k_s0, k_z0

            k_q0 = ((k_pack & 0x03).to(q_q0.dtype) - k_z0) * k_s0
            k_q1 = (((k_pack >> 2) & 0x03).to(q_q0.dtype) - k_z1) * k_s1
            k_q2 = (((k_pack >> 4) & 0x03).to(q_q0.dtype) - k_z2) * k_s2
            k_q3 = (((k_pack >> 6) & 0x03).to(q_q0.dtype) - k_z3) * k_s3
            # 4 quarter-dots (both operands crumb-ordered; no permute needed)
            qk = (tl.dot(q_q0, k_q0) + tl.dot(q_q1, k_q1)
                  + tl.dot(q_q2, k_q2) + tl.dot(q_q3, k_q3)) * sm_scale
            if logit_cap > 0:
                qk = logit_cap * _tanh_(qk / logit_cap)

            # causal: prefix always visible; diagonal j <= tok; tail masked
            visible = n_ok[None, :] & (is_pfx[None, :] | (j[None, :] <= tok[:, None]))
            qk = tl.where(visible & row_ok[:, None], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]

            # ---- V packed [N, D/4], crumb order ---------------------------
            v_pack = tl.load(
                V_Buffer + slot[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh + offs_qp[None, :],
                mask=n_ok[:, None] & (offs_qp[None, :] < (L // 4)),
                other=0,
            )
            if GROUPED:
                gqv = offs_qp // GROUP_SIZE
                szv = slot[:, None] * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_s0 = tl.load(V_SZ + szv + 2 * gqv[None, :], mask=n_ok[:, None], other=1.0).to(q_q0.dtype)
                v_z0 = tl.load(V_SZ + szv + 2 * gqv[None, :] + 1, mask=n_ok[:, None], other=0.0).to(q_q0.dtype)
                v_s1 = tl.load(V_SZ + szv + 2 * (gqv + NGQ)[None, :], mask=n_ok[:, None], other=1.0).to(q_q0.dtype)
                v_z1 = tl.load(V_SZ + szv + 2 * (gqv + NGQ)[None, :] + 1, mask=n_ok[:, None], other=0.0).to(q_q0.dtype)
                v_s2 = tl.load(V_SZ + szv + 2 * (gqv + 2 * NGQ)[None, :], mask=n_ok[:, None], other=1.0).to(q_q0.dtype)
                v_z2 = tl.load(V_SZ + szv + 2 * (gqv + 2 * NGQ)[None, :] + 1, mask=n_ok[:, None], other=0.0).to(q_q0.dtype)
                v_s3 = tl.load(V_SZ + szv + 2 * (gqv + 3 * NGQ)[None, :], mask=n_ok[:, None], other=1.0).to(q_q0.dtype)
                v_z3 = tl.load(V_SZ + szv + 2 * (gqv + 3 * NGQ)[None, :] + 1, mask=n_ok[:, None], other=0.0).to(q_q0.dtype)
            else:
                szv = slot * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_s0 = tl.load(V_SZ + szv + 0, mask=n_ok, other=1.0).to(q_q0.dtype)
                v_z0 = tl.load(V_SZ + szv + 1, mask=n_ok, other=0.0).to(q_q0.dtype)
                v_s1, v_z1 = v_s0, v_z0
                v_s2, v_z2 = v_s0, v_z0
                v_s3, v_z3 = v_s0, v_z0
                v_z0 = v_z0[:, None]
                v_s0 = v_s0[:, None]
                v_z1, v_s1 = v_z0, v_s0
                v_z2, v_s2 = v_z0, v_s0
                v_z3, v_s3 = v_z0, v_s0

            v_q0 = ((v_pack & 0x03).to(q_q0.dtype) - v_z0) * v_s0
            v_q1 = (((v_pack >> 2) & 0x03).to(q_q0.dtype) - v_z1) * v_s1
            v_q2 = (((v_pack >> 4) & 0x03).to(q_q0.dtype) - v_z2) * v_s2
            v_q3 = (((v_pack >> 6) & 0x03).to(q_q0.dtype) - v_z3) * v_s3

            p_bf = p.to(v_q0.dtype)
            acc_q0 += tl.dot(p_bf, v_q0)
            acc_q1 += tl.dot(p_bf, v_q1)
            acc_q2 += tl.dot(p_bf, v_q2)
            acc_q3 += tl.dot(p_bf, v_q3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # ---- writeback ---------------------------------------------------------
    safe_inv = tl.where(e_sum > 0, 1.0 / e_sum, 0.0)
    o_row = (req * EXT + tok)[:, None] * stride_o_tok + q_head[:, None] * stride_o_h
    wmask = row_ok[:, None] & (offs_qp[None, :] < (L // 4))

    if SPLITS == 1:
        out_ty = Out.dtype.element_ty
        if HAS_ROT:
            # Output fold: the pool stores v rotated (v @ R_v[h]), so this
            # kernel's output is attn_out @ R_v[h]; the model needs attn_out,
            # i.e. multiply by R_v[h]^T: rot_e = sum_c acc_c @ Rv[e.., c..]^T
            # with Wc[m, i] = Rv[(e*QTR+i)*L + (c*QTR+m)] -> [QTR_m, QTR_i].
            # (QTR was defined by the q-fold above; Python/if scopes share it.)
            n0 = (acc_q0 * safe_inv[:, None]).to(out_ty)
            n1 = (acc_q1 * safe_inv[:, None]).to(out_ty)
            n2 = (acc_q2 * safe_inv[:, None]).to(out_ty)
            n3 = (acc_q3 * safe_inv[:, None]).to(out_ty)
            rv_base = Rv_ptr + (cur_kv_head % NRV).to(tl.int64) * L * L
            offs_i = tl.arange(0, BLOCK_D // 4)
            ro0 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
            ro1 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
            ro2 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
            ro3 = tl.zeros([BLOCK_M, BLOCK_D // 4], dtype=tl.float32)
            # c = 0
            w = tl.load(rv_base + 0 + offs_i[None, :] * L + 0 + offs_qp[:, None]).to(out_ty)
            ro0 += tl.dot(n0, w)
            w = tl.load(rv_base + QTR * L + offs_i[None, :] * L + 0 + offs_qp[:, None]).to(out_ty)
            ro1 += tl.dot(n0, w)
            w = tl.load(rv_base + 2 * QTR * L + offs_i[None, :] * L + 0 + offs_qp[:, None]).to(out_ty)
            ro2 += tl.dot(n0, w)
            w = tl.load(rv_base + 3 * QTR * L + offs_i[None, :] * L + 0 + offs_qp[:, None]).to(out_ty)
            ro3 += tl.dot(n0, w)
            # c = 1
            w = tl.load(rv_base + 0 + offs_i[None, :] * L + QTR + offs_qp[:, None]).to(out_ty)
            ro0 += tl.dot(n1, w)
            w = tl.load(rv_base + QTR * L + offs_i[None, :] * L + QTR + offs_qp[:, None]).to(out_ty)
            ro1 += tl.dot(n1, w)
            w = tl.load(rv_base + 2 * QTR * L + offs_i[None, :] * L + QTR + offs_qp[:, None]).to(out_ty)
            ro2 += tl.dot(n1, w)
            w = tl.load(rv_base + 3 * QTR * L + offs_i[None, :] * L + QTR + offs_qp[:, None]).to(out_ty)
            ro3 += tl.dot(n1, w)
            # c = 2
            w = tl.load(rv_base + 0 + offs_i[None, :] * L + 2 * QTR + offs_qp[:, None]).to(out_ty)
            ro0 += tl.dot(n2, w)
            w = tl.load(rv_base + QTR * L + offs_i[None, :] * L + 2 * QTR + offs_qp[:, None]).to(out_ty)
            ro1 += tl.dot(n2, w)
            w = tl.load(rv_base + 2 * QTR * L + offs_i[None, :] * L + 2 * QTR + offs_qp[:, None]).to(out_ty)
            ro2 += tl.dot(n2, w)
            w = tl.load(rv_base + 3 * QTR * L + offs_i[None, :] * L + 2 * QTR + offs_qp[:, None]).to(out_ty)
            ro3 += tl.dot(n2, w)
            # c = 3
            w = tl.load(rv_base + 0 + offs_i[None, :] * L + 3 * QTR + offs_qp[:, None]).to(out_ty)
            ro0 += tl.dot(n3, w)
            w = tl.load(rv_base + QTR * L + offs_i[None, :] * L + 3 * QTR + offs_qp[:, None]).to(out_ty)
            ro1 += tl.dot(n3, w)
            w = tl.load(rv_base + 2 * QTR * L + offs_i[None, :] * L + 3 * QTR + offs_qp[:, None]).to(out_ty)
            ro2 += tl.dot(n3, w)
            w = tl.load(rv_base + 3 * QTR * L + offs_i[None, :] * L + 3 * QTR + offs_qp[:, None]).to(out_ty)
            ro3 += tl.dot(n3, w)
            tl.store(Out + o_row + offs_qp[None, :], ro0.to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + L // 4)[None, :], ro1.to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + 2 * (L // 4))[None, :], ro2.to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + 3 * (L // 4))[None, :], ro3.to(out_ty), mask=wmask)
        else:
            tl.store(Out + o_row + offs_qp[None, :],
                     (acc_q0 * safe_inv[:, None]).to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + L // 4)[None, :],
                     (acc_q1 * safe_inv[:, None]).to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + 2 * (L // 4))[None, :],
                     (acc_q2 * safe_inv[:, None]).to(out_ty), mask=wmask)
            tl.store(Out + o_row + (offs_qp + 3 * (L // 4))[None, :],
                     (acc_q3 * safe_inv[:, None]).to(out_ty), mask=wmask)
    else:
        m_row = o_row + split_kv_id * stride_o_s
        tl.store(Mid + m_row + offs_qp[None, :], acc_q0 * safe_inv[:, None], mask=wmask)
        tl.store(Mid + m_row + (offs_qp + L // 4)[None, :], acc_q1 * safe_inv[:, None], mask=wmask)
        tl.store(Mid + m_row + (offs_qp + 2 * (L // 4))[None, :], acc_q2 * safe_inv[:, None], mask=wmask)
        tl.store(Mid + m_row + (offs_qp + 3 * (L // 4))[None, :], acc_q3 * safe_inv[:, None], mask=wmask)
        # Lse layout [ntok, Hq, S] contiguous: idx = ((req*EXT+tok)*Hq + q_head)*S + s
        lse_idx = ((req * EXT + tok) * HKV * KVG + q_head) * SPLITS + split_kv_id
        lse_val = tl.where(e_sum > 0, e_max + tl.log(e_sum), -float("inf"))
        tl.store(Lse + lse_idx, lse_val, mask=row_ok)


@triton.jit
def _int2_tile_reduce(
    Mid,                  # [ntok, Hq, S, D] f32
    Lse,                  # [ntok, Hq, S] f32
    Out,                  # [ntok, Hq, D] model dtype
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # == (tok*Hq + h)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    base = pid.to(tl.int64) * SPLITS
    m = -float("inf")
    for s in range(SPLITS):
        m = tl.maximum(m, tl.load(Lse + base + s))

    denom = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(SPLITS):
        lse_s = tl.load(Lse + base + s)
        w = tl.exp(lse_s - m)
        denom += w
        part = tl.load(Mid + base * D + s * D + offs_d, mask=mask_d, other=0.0)
        acc += part * w

    inv = tl.where(denom > 0, 1.0 / denom, 0.0)
    tl.store(Out + pid.to(tl.int64) * D + offs_d,
             (acc * inv).to(Out.dtype.element_ty), mask=mask_d)


# ---------------------------------------------------------------------------
# Launcher: graph-safe (no host reads of device tensors, static shapes)
# ---------------------------------------------------------------------------

def int2_causal_tile_attention(
    q,                    # [ntok, Hq, D] model dtype (NATURAL order; kernel rotates)
    k_raw, v_raw,         # [cache, Hkv, L/4] uint8
    k_sz, v_sz,           # [cache, Hkv, 2*groups] f32
    kv_indptr,            # [bs+1] int32 (prefix spans)
    kv_indices,           # int32 prefix slot ids
    out_cache_loc,        # int32 [bs*EXT] diagonal slots
    prefix_lens,          # [bs] int32 (NOT read by the kernel; kept for API parity)
    ext_len,              # python int (static per graph)
    kv_group_num,         # Hq // Hkv
    sm_scale,
    *,
    logit_cap=0.0,
    splits=1,
    out=None,
    workspace=None,       # optional dict of preallocated Mid/Lse (graph buffers)
    Rk=None,              # [Hkv, D, D] bf16 fused: kernel computes (q@Rk) and out@Rv^T
    Rv=None,
):
    """Compute causal attention for a uniform extend batch straight from the
    int2 pool. Returns [ntok, Hq, D] in q.dtype, in the NATURAL (un-rotated)
    space when Rk/Rv are passed; q must be given un-rotated and k/v are read
    pre-rotated from the pool.

    ``workspace``: for SPLITS>1 mode pass a dict with 'mid' and 'lse' tensors
    sized [ntok, Hq, splits, D] f32 / [ntok, Hq, splits] f32 (cuda-graph
    static buffers); they are allocated here otherwise.
    """
    ntok, hq, d = q.shape
    hkv = k_raw.shape[1]
    assert hq == hkv * kv_group_num
    L = k_raw.shape[-1] * 4
    assert d == L, f"q head_dim {d} != pool head_dim {L}"
    bs = kv_indptr.shape[0] - 1
    assert ntok == bs * ext_len, f"ntok {ntok} != bs*EXT {bs}*{ext_len}"
    dev = q.device

    group_size = L // (k_sz.shape[-1] // 2)
    BLOCK_D = triton.next_power_of_2(L)
    q = q.contiguous()
    has_rot = Rk is not None and Rv is not None
    if has_rot:
        assert Rk.dim() in (2, 3) and Rv.dim() in (2, 3), "rotations must be [D,D] or [Hkv,D,D]"
        assert Rk.shape[-1] == L and Rv.shape[-1] == L, "R geometry mismatch"
        # 2-D shared rotation -> view as [1, D, D]; the kernel broadcasts by
        # taking head index modulo the (compile-time constant) rotation count.
        if Rk.dim() == 2:
            Rk = Rk.contiguous().unsqueeze(0)
        else:
            Rk = Rk.contiguous()
        if Rv.dim() == 2:
            Rv = Rv.contiguous().unsqueeze(0)
        else:
            Rv = Rv.contiguous()
    else:
        Rk = q  # dummy pointers (never dereferenced when HAS_ROT=0)
        Rv = q

    if splits == 1:
        rows_per_rh = ext_len * kv_group_num
        BLOCK_M = min(16, max(16, triton.next_power_of_2(rows_per_rh)))
        if rows_per_rh < BLOCK_M:
            BLOCK_M = triton.next_power_of_2(rows_per_rh)
        tiles_per_rh = triton.cdiv(rows_per_rh, BLOCK_M)
        grid = (tiles_per_rh * bs * hkv, 1)
        if out is None:
            out = torch.empty((ntok, hq, d), dtype=q.dtype, device=dev)
        _int2_tile_causal_stage1[grid](
            q, k_raw, v_raw, k_sz, v_sz, sm_scale,
            kv_indptr, kv_indices, out_cache_loc,
            out, out, out,          # Out / Mid / Lse (Mid, Lse unused in SPLITS=1)
            q.stride(0), q.stride(1),
            k_raw.stride(0), k_raw.stride(1),
            v_raw.stride(0), v_raw.stride(1),
            k_sz.stride(0), k_sz.stride(1),
            v_sz.stride(0), v_sz.stride(1),
            out.stride(0), out.stride(1), 0,
            Rk, Rv,
            ext_len, tiles_per_rh,
            KVG=kv_group_num, HKV=hkv,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_N=int(os.environ.get("SGL_INT2_TILE_BLOCK_N", 128)),
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            logit_cap=logit_cap,
            L=L,
            GROUP_SIZE=group_size,
            SPLITS=1,
            HAS_ROT=has_rot,
            NRK=Rk.shape[0],
            NRV=Rv.shape[0],
            num_warps=int(os.environ.get("SGL_INT2_TILE_WARPS", 4)),
            num_stages=int(os.environ.get("SGL_INT2_TILE_STAGES", 1)),
        )
        return out

    # SPLITS>1 path does not implement HAS_ROT; do torch-side rotations.
    if has_rot and splits > 1:
        q = torch.bmm(
            q.view(-1, hkv, kv_group_num, L).permute(1, 0, 2, 3).reshape(hkv, -1, L),
            Rk,
        ).view(hkv, -1, kv_group_num, L).permute(1, 0, 2, 3).reshape(-1, hq, L).contiguous()

    # SPLITS>1: fixed grid for graph capture (bs small, EXT*KVG <= BLOCK_M)
    rows_per_rh = ext_len * kv_group_num
    BLOCK_M = triton.next_power_of_2(max(rows_per_rh, 16))
    assert rows_per_rh <= BLOCK_M
    if workspace is None or "mid" not in workspace:
        mid = torch.zeros((ntok, hq, splits, d), dtype=torch.float32, device=dev)
        lse = torch.full((ntok, hq, splits), -float("inf"), dtype=torch.float32, device=dev)
    else:
        mid, lse = workspace["mid"], workspace["lse"]
        mid.zero_()
        lse.fill_(-float("inf"))
    if out is None:
        out = torch.empty((ntok, hq, d), dtype=q.dtype, device=dev)
    grid = (bs * hkv, splits)
    _int2_tile_causal_stage1[grid](
        q, k_raw, v_raw, k_sz, v_sz, sm_scale,
        kv_indptr, kv_indices, out_cache_loc,
        out, mid, lse,
        q.stride(0), q.stride(1),
        k_raw.stride(0), k_raw.stride(1),
        v_raw.stride(0), v_raw.stride(1),
        k_sz.stride(0), k_sz.stride(1),
        v_sz.stride(0), v_sz.stride(1),
        mid.stride(0), mid.stride(1), mid.stride(2),
        Rk, Rv,
        ext_len, 1,
        KVG=kv_group_num, HKV=hkv,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_N=int(os.environ.get("SGL_INT2_TILE_BLOCK_N", 128)),
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        L=L,
        GROUP_SIZE=group_size,
        SPLITS=splits,
        HAS_ROT=False,          # SPLITS path stores rotated partials; the
        NRK=1, NRV=1,           # caller inverts via torch bmm
        num_warps=int(os.environ.get("SGL_INT2_TILE_WARPS", 4)),
        num_stages=int(os.environ.get("SGL_INT2_TILE_STAGES", 1)),
    )
    _int2_tile_reduce[(ntok * hq,)](
        mid, lse, out,
        D=L, SPLITS=splits, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    if has_rot and splits > 1:
        out = torch.bmm(
            out.view(-1, hkv, kv_group_num, L).permute(1, 0, 2, 3).reshape(hkv, -1, L),
            Rv.transpose(-1, -2),
        ).view(hkv, -1, kv_group_num, L).permute(1, 0, 2, 3).reshape(-1, hq, L).contiguous()
    return out
