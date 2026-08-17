"""INT2 quantized decode-attention Triton kernels (OSCAR).

Extracted VERBATIM from the OSCAR fork's
``sglang/srt/layers/attention/triton_ops/decode_attention.py``:

  * ``_get_scale_group_size`` / ``_get_shared_kv_scale_group_size``  (helpers)
  * ``tanh``                                                        (triton jit)
  * ``_fwd_kernel_stage2`` / ``_decode_softmax_reducev_fwd``         (stage-2 reduce)
  * ``decode_attention_fwd_quantized``                              (dispatcher)
  * ``_fwd_kernel_stage1_quant_int2`` + grouped variant + launchers  (stage-1)

WHY THIS FILE EXISTS
--------------------
Newer upstream SGLang (``sglang.kernels.ops.attention.decode_attention``) has
NO int2 support at all -- zero occurrences of "int2" in that module. Porting
only the pool write-side would produce a KV cache that is written but can never
be read back. These are the read-side kernels.

Stage-2 (``_fwd_kernel_stage2`` / ``_decode_softmax_reducev_fwd``) is vendored
here rather than imported from upstream because the OSCAR variant carries an
extra ``output_lse`` / ``WRITE_LSE`` parameter that upstream's copy replaced
with ``use_pdl``. Vendoring keeps this module self-contained and immune to
further upstream churn in that signature.

DELIBERATELY NOT PORTED: ``decode_attention_fwd_int2_unified`` and
``_fwd_kernel_stage2_unified`` -- those belong to the mixed-KV (HP window)
path, which is spec-decoding-incompatible and is not part of this port.

Pure Python + Triton. No sglang imports beyond ``is_hip``, which is inlined
below so the module imports standalone.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl


def _detect_hip() -> bool:
    try:
        return torch.version.hip is not None
    except Exception:
        return False


_is_hip = _detect_hip()

logger = logging.getLogger(__name__)

_MIN_BLOCK_KV = 32


def _get_scale_group_size(head_dim: int, scales_zeros) -> int:
    """Return the per-group head-dim span for quantized KV scales.

    ``scales_zeros`` has last-dim layout ``[scale_0, zero_0, scale_1, zero_1, ...]``,
    i.e. ``2 * num_groups`` entries. Returns ``head_dim // num_groups``; when the
    cache uses a single scale/zero pair per head this equals ``head_dim``.
    """
    num_groups = scales_zeros.shape[-1] // 2
    if head_dim % num_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by quant scale groups ({num_groups})"
        )
    return head_dim // num_groups


def _get_shared_kv_scale_group_size(
    Lk: int, Lv: int, k_scales_zeros, v_scales_zeros
) -> int:
    """Return the shared configured INT2 KV group size.

    K and V may have different head dims in MLA/DPE-style layouts, so the
    scalar one-group case can report different per-tensor group sizes. Once
    either side is actually grouped, the configured group size must match.
    """
    k_group_size = _get_scale_group_size(Lk, k_scales_zeros)
    v_group_size = _get_scale_group_size(Lv, v_scales_zeros)
    k_grouped = k_group_size < Lk
    v_grouped = v_group_size < Lv

    if (k_grouped or v_grouped) and k_group_size != v_group_size:
        raise ValueError(
            "INT2 KV cache requires K and V to use the same quant group size "
            f"when grouped, got K={k_group_size}, V={v_group_size}"
        )
    return k_group_size if (k_grouped or v_grouped) else max(k_group_size, v_group_size)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    O_lse,
    v_scale,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )
    if WRITE_LSE:
        # Per-seq log-sum-exp = e_max + log(e_sum). O_lse has shape
        # [bs, num_heads]; batch stride = stride_obs // Lv = num_heads.
        tl.store(O_lse + cur_batch * (stride_obs // Lv) + cur_head, e_max + tl.log(e_sum))


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_scale,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
    output_lse=None,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None
    WRITE_LSE = output_lse is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        output_lse,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        WRITE_LSE=WRITE_LSE,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )



def decode_attention_fwd_quantized(
    q,
    k_buffer,  # Quantized INT2 packed uint8
    v_buffer,  # Quantized INT2 packed uint8
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_dtype,  # must be "int2"
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
    R_k=None,
):
    """
    Attention forward with INT2 quantized KV cache.
    Dispatches between MHA and GQA/MQA paths based on ``kv_group_num``.
    """
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]
    assert kv_dtype == "int2", f"Only int2 quant KV is supported, got {kv_dtype}"

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        decode_attention_fwd_normal_quant_int2(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )
    else:
        decode_attention_fwd_grouped_quant_int2(
            q,
            k_buffer,
            v_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
            R_k=R_k,
        )


# ---------------------------------------------------------------------------
# INT2 quantized decode attention kernels
# ---------------------------------------------------------------------------
# INT2 packs 4 values per byte (2-bit crumbs).  Storage is head_dim // 4
# packed uint8 bytes.  Unpacking uses masks 0x03, shifts >> 2, >> 4, >> 6.
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_stage1_quant_int2(
    Q,
    K_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    V_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2*k_groups] float32, interleaved scale/zero pairs
    V_Scales_Zeros,  # [cache_size, num_heads, 2*v_groups] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    K_GROUPED: tl.constexpr = GROUP_SIZE < Lk
    V_GROUPED: tl.constexpr = GROUP_SIZE < Lv

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    # For INT2, work with 4 quarters separately
    acc_q0 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_DV // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_d_quarter = tl.arange(0, BLOCK_DMODEL // 4)
        mask_d_quarter = offs_d_quarter < (Lk // 4)

        q_main = tl.load(Q + off_q, mask=mask_d, other=0.0)
        # Split Q into 4 quarters
        q_q0 = tl.where(mask_d_quarter, tl.gather(q_main, offs_d_quarter, 0), 0.0)
        idx_q1 = (Lk // 4) + offs_d_quarter
        q_q1 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q1, 0), 0.0)
        idx_q2 = 2 * (Lk // 4) + offs_d_quarter
        q_q2 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q2, 0), 0.0)
        idx_q3 = 3 * (Lk // 4) + offs_d_quarter
        q_q3 = tl.where(mask_d_quarter, tl.gather(q_main, idx_q3, 0), 0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT2 K (uint8, 4 values per byte)
            offs_d_packed = tl.arange(0, BLOCK_DMODEL // 4)
            mask_d_packed = offs_d_packed < (Lk // 4)

            offs_buf_k_packed = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[None, :]
            )
            k_quant_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for K
            if K_GROUPED:
                offs_group_k_q0 = offs_d_packed // GROUP_SIZE
                offs_group_k_q1 = (offs_d_packed + (Lk // 4)) // GROUP_SIZE
                offs_group_k_q2 = (offs_d_packed + 2 * (Lk // 4)) // GROUP_SIZE
                offs_group_k_q3 = (offs_d_packed + 3 * (Lk // 4)) // GROUP_SIZE
                safe_group_k_q0 = tl.where(mask_d_packed, offs_group_k_q0, 0)
                safe_group_k_q1 = tl.where(mask_d_packed, offs_group_k_q1, 0)
                safe_group_k_q2 = tl.where(mask_d_packed, offs_group_k_q2, 0)
                safe_group_k_q3 = tl.where(mask_d_packed, offs_group_k_q3, 0)
                offs_sz_k = (
                    kv_loc[:, None] * stride_sz_kbs + cur_kv_head * stride_sz_kh
                )
                k_scale_q0 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q0[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q0 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q0[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q1 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q1[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q1 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q1[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q2 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q2[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q2 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q2[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                k_scale_q3 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q3[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=1.0,
                )
                k_zero_q3 = tl.load(
                    K_Scales_Zeros + offs_sz_k + 2 * safe_group_k_q3[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d_packed[None, :]),
                    other=0.0,
                )
                # Dequantize INT2 K inline: unpack 4 crumbs and dequantize per-group.
                k_q0 = (
                    ((k_quant_packed & 0x03).to(tl.float32) - k_zero_q0)
                    * k_scale_q0
                ).to(q_q0.dtype)
                k_q1 = (
                    (((k_quant_packed >> 2) & 0x03).to(tl.float32) - k_zero_q1)
                    * k_scale_q1
                ).to(q_q0.dtype)
                k_q2 = (
                    (((k_quant_packed >> 4) & 0x03).to(tl.float32) - k_zero_q2)
                    * k_scale_q2
                ).to(q_q0.dtype)
                k_q3 = (
                    (((k_quant_packed >> 6) & 0x03).to(tl.float32) - k_zero_q3)
                    * k_scale_q3
                ).to(q_q0.dtype)
            else:
                offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_scale_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                )
                k_zero_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                )
                k_q0 = (
                    ((k_quant_packed & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q1 = (
                    (((k_quant_packed >> 2) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q2 = (
                    (((k_quant_packed >> 4) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)
                k_q3 = (
                    (((k_quant_packed >> 6) & 0x03).to(tl.float32) - k_zero_1d[:, None])
                    * k_scale_1d[:, None]
                ).to(q_q0.dtype)

            # Compute QK from 4 partial dot products
            qk = (
                tl.sum(q_q0[None, :] * k_q0, 1)
                + tl.sum(q_q1[None, :] * k_q1, 1)
                + tl.sum(q_q2[None, :] * k_q2, 1)
                + tl.sum(q_q3[None, :] * k_q3, 1)
            )
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # Load packed INT2 V
            offs_dv_packed = tl.arange(0, BLOCK_DV // 4)
            mask_dv_packed = offs_dv_packed < (Lv // 4)

            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv_packed[None, :]
            )
            v_quant_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                other=0,
            )

            # Load scales and zeros for V
            if V_GROUPED:
                offs_group_v_q0 = offs_dv_packed // GROUP_SIZE
                offs_group_v_q1 = (offs_dv_packed + (Lv // 4)) // GROUP_SIZE
                offs_group_v_q2 = (offs_dv_packed + 2 * (Lv // 4)) // GROUP_SIZE
                offs_group_v_q3 = (offs_dv_packed + 3 * (Lv // 4)) // GROUP_SIZE
                safe_group_v_q0 = tl.where(mask_dv_packed, offs_group_v_q0, 0)
                safe_group_v_q1 = tl.where(mask_dv_packed, offs_group_v_q1, 0)
                safe_group_v_q2 = tl.where(mask_dv_packed, offs_group_v_q2, 0)
                safe_group_v_q3 = tl.where(mask_dv_packed, offs_group_v_q3, 0)
                offs_sz_v = (
                    kv_loc[:, None] * stride_sz_vbs + cur_kv_head * stride_sz_vh
                )
                v_scale_q0 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q0[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q0 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q0[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q1 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q1[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q1 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q1[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q2 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q2[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q2 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q2[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                v_scale_q3 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q3[None, :],
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=1.0,
                )
                v_zero_q3 = tl.load(
                    V_Scales_Zeros + offs_sz_v + 2 * safe_group_v_q3[None, :] + 1,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv_packed[None, :]),
                    other=0.0,
                )
                # Dequantize INT2 V inline: unpack 4 crumbs per-group.
                v_q0 = (
                    ((v_quant_packed & 0x03).to(tl.float32) - v_zero_q0)
                    * v_scale_q0
                ).to(q_q0.dtype)
                v_q1 = (
                    (((v_quant_packed >> 2) & 0x03).to(tl.float32) - v_zero_q1)
                    * v_scale_q1
                ).to(q_q0.dtype)
                v_q2 = (
                    (((v_quant_packed >> 4) & 0x03).to(tl.float32) - v_zero_q2)
                    * v_scale_q2
                ).to(q_q0.dtype)
                v_q3 = (
                    (((v_quant_packed >> 6) & 0x03).to(tl.float32) - v_zero_q3)
                    * v_scale_q3
                ).to(q_q0.dtype)
            else:
                offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_scale_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                )
                v_zero_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                )
                v_q0 = (
                    ((v_quant_packed & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q1 = (
                    (((v_quant_packed >> 2) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q2 = (
                    (((v_quant_packed >> 4) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)
                v_q3 = (
                    (((v_quant_packed >> 6) & 0x03).to(tl.float32) - v_zero_1d[:, None])
                    * v_scale_1d[:, None]
                ).to(q_q0.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)

            # Accumulate separately for 4 quarters
            acc_q0 *= re_scale
            acc_q1 *= re_scale
            acc_q2 *= re_scale
            acc_q3 *= re_scale
            acc_q0 += tl.sum(p[:, None] * v_q0, 0)
            acc_q1 += tl.sum(p[:, None] * v_q1, 0)
            acc_q2 += tl.sum(p[:, None] * v_q2, 0)
            acc_q3 += tl.sum(p[:, None] * v_q3, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # Store 4 quarters separately
        # Quarter 0: indices [0, Lv//4)
        offs_dv_q0 = tl.arange(0, BLOCK_DV // 4)
        mask_dv_quarter = offs_dv_q0 < (Lv // 4)
        offs_mid_o_q0 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q0
        )
        tl.store(
            Att_Out + offs_mid_o_q0,
            acc_q0 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 1: indices [Lv//4, Lv//2)
        offs_dv_q1 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q1
            + Lv // 4
        )
        tl.store(
            Att_Out + offs_mid_o_q1,
            acc_q1 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 2: indices [Lv//2, 3*Lv//4)
        offs_dv_q2 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q2 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q2
            + 2 * (Lv // 4)
        )
        tl.store(
            Att_Out + offs_mid_o_q2,
            acc_q2 / e_sum,
            mask=mask_dv_quarter,
        )

        # Quarter 3: indices [3*Lv//4, Lv)
        offs_dv_q3 = tl.arange(0, BLOCK_DV // 4)
        offs_mid_o_q3 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_q3
            + 3 * (Lv // 4)
        )
        tl.store(
            Att_Out + offs_mid_o_q3,
            acc_q3 / e_sum,
            mask=mask_dv_quarter,
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_grouped_kernel_stage1_quant_int2(
    Q,
    K_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    V_Buffer,  # Quantized INT2 [cache_size, num_heads, head_dim//4] uint8 (packed)
    K_Scales_Zeros,  # [cache_size, num_heads, 2*groups] float32, interleaved scale/zero pairs
    V_Scales_Zeros,  # [cache_size, num_heads, 2*groups] float32
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_kbs,  # K scales_zeros stride for cache
    stride_sz_kh,  # K scales_zeros stride for head
    stride_sz_vbs,  # V scales_zeros stride for cache
    stride_sz_vh,  # V scales_zeros stride for head
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_ROT: tl.constexpr,
    NRK: tl.constexpr,
    R_k_ptr,                # [NRK, L, L] bf16 (dummy when HAS_ROT=0)
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    GROUPED: tl.constexpr = GROUP_SIZE < L
    FAST: tl.constexpr = (BLOCK_D // 4) >= GROUP_SIZE
    # Hoisted out of the per-block KV loop: all loop-invariant (depend only on
    # the BLOCK_D/GROUP_SIZE constexprs). Defining them inside the loop made
    # newer Triton raise "constexpr cannot be reassigned" on the 2nd iteration.
    NUM_GROUPS_QUARTER: tl.constexpr = (BLOCK_D // 4) // GROUP_SIZE
    grp_q0: tl.constexpr = (0 * (BLOCK_D // 4)) // GROUP_SIZE
    grp_q1: tl.constexpr = (1 * (BLOCK_D // 4)) // GROUP_SIZE
    grp_q2: tl.constexpr = (2 * (BLOCK_D // 4)) // GROUP_SIZE
    grp_q3: tl.constexpr = (3 * (BLOCK_D // 4)) // GROUP_SIZE
    v_grp_q0: tl.constexpr = (0 * (BLOCK_D // 4)) // GROUP_SIZE
    v_grp_q1: tl.constexpr = (1 * (BLOCK_D // 4)) // GROUP_SIZE
    v_grp_q2: tl.constexpr = (2 * (BLOCK_D // 4)) // GROUP_SIZE
    v_grp_q3: tl.constexpr = (3 * (BLOCK_D // 4)) // GROUP_SIZE

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < L

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # Use 4 separate accumulators for INT2 quarters
    acc_q0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_d_q0 = tl.arange(0, BLOCK_D // 4)
        offs_d_q1 = tl.arange(BLOCK_D // 4, 2 * (BLOCK_D // 4))
        offs_d_q2 = tl.arange(2 * (BLOCK_D // 4), 3 * (BLOCK_D // 4))
        offs_d_q3 = tl.arange(3 * (BLOCK_D // 4), BLOCK_D)
        mask_d_quarter = offs_d_q0 < (L // 4)

        q_main = tl.load(
            Q + offs_q,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        if HAS_ROT:
            # Fold R_k into the kernel: pool stores k already rotated
            # (k @ R_k[h]); scores need (q @ R_k[h]) . k_stored. All BLOCK_H
            # rows of this program belong to ONE kv head (cur_kv_head), so
            # 16 [BLOCK_H, D/4] x [D/4, D/4] quarter-dots suffice, in
            # registers, replacing a separate [ntok, Hq, D] GEMM launch.
            QTR: tl.constexpr = L // 4
            rk_base = R_k_ptr + (cur_kv_head % NRK) * L * L
            q0 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(q_main, tl.broadcast_to(offs_d_q0[None, :], [BLOCK_H, BLOCK_D // 4]), 1), 0.0,
            )
            q1 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(q_main, tl.broadcast_to(offs_d_q1[None, :], [BLOCK_H, BLOCK_D // 4]), 1), 0.0,
            )
            q2 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(q_main, tl.broadcast_to(offs_d_q2[None, :], [BLOCK_H, BLOCK_D // 4]), 1), 0.0,
            )
            q3 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(q_main, tl.broadcast_to(offs_d_q3[None, :], [BLOCK_H, BLOCK_D // 4]), 1), 0.0,
            )
            r0 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
            r1 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
            r2 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
            r3 = tl.zeros([BLOCK_H, BLOCK_D // 4], dtype=tl.float32)
            offs_col = tl.arange(0, BLOCK_D // 4)
            # e = 0
            rb = rk_base + 0
            r0 += tl.dot(q0, tl.load(rb + offs_col[:, None] * L + offs_col[None, :]).to(q0.dtype))
            r1 += tl.dot(q0, tl.load(rb + offs_col[:, None] * L + QTR + offs_col[None, :]).to(q0.dtype))
            r2 += tl.dot(q0, tl.load(rb + offs_col[:, None] * L + 2 * QTR + offs_col[None, :]).to(q0.dtype))
            r3 += tl.dot(q0, tl.load(rb + offs_col[:, None] * L + 3 * QTR + offs_col[None, :]).to(q0.dtype))
            # e = 1
            rb = rk_base + QTR * L
            r0 += tl.dot(q1, tl.load(rb + offs_col[:, None] * L + offs_col[None, :]).to(q0.dtype))
            r1 += tl.dot(q1, tl.load(rb + offs_col[:, None] * L + QTR + offs_col[None, :]).to(q0.dtype))
            r2 += tl.dot(q1, tl.load(rb + offs_col[:, None] * L + 2 * QTR + offs_col[None, :]).to(q0.dtype))
            r3 += tl.dot(q1, tl.load(rb + offs_col[:, None] * L + 3 * QTR + offs_col[None, :]).to(q0.dtype))
            # e = 2
            rb = rk_base + 2 * QTR * L
            r0 += tl.dot(q2, tl.load(rb + offs_col[:, None] * L + offs_col[None, :]).to(q0.dtype))
            r1 += tl.dot(q2, tl.load(rb + offs_col[:, None] * L + QTR + offs_col[None, :]).to(q0.dtype))
            r2 += tl.dot(q2, tl.load(rb + offs_col[:, None] * L + 2 * QTR + offs_col[None, :]).to(q0.dtype))
            r3 += tl.dot(q2, tl.load(rb + offs_col[:, None] * L + 3 * QTR + offs_col[None, :]).to(q0.dtype))
            # e = 3
            rb = rk_base + 3 * QTR * L
            r0 += tl.dot(q3, tl.load(rb + offs_col[:, None] * L + offs_col[None, :]).to(q0.dtype))
            r1 += tl.dot(q3, tl.load(rb + offs_col[:, None] * L + QTR + offs_col[None, :]).to(q0.dtype))
            r2 += tl.dot(q3, tl.load(rb + offs_col[:, None] * L + 2 * QTR + offs_col[None, :]).to(q0.dtype))
            r3 += tl.dot(q3, tl.load(rb + offs_col[:, None] * L + 3 * QTR + offs_col[None, :]).to(q0.dtype))
            q_q0 = r0.to(q0.dtype)
            q_q1 = r1.to(q0.dtype)
            q_q2 = r2.to(q0.dtype)
            q_q3 = r3.to(q0.dtype)
        else:
            q_q0 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(
                    q_main,
                    tl.broadcast_to(offs_d_q0[None, :], [BLOCK_H, BLOCK_D // 4]),
                    1,
                ),
                0.0,
            )
            q_q1 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(
                    q_main,
                    tl.broadcast_to(offs_d_q1[None, :], [BLOCK_H, BLOCK_D // 4]),
                    1,
                ),
                0.0,
            )
            q_q2 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(
                    q_main,
                    tl.broadcast_to(offs_d_q2[None, :], [BLOCK_H, BLOCK_D // 4]),
                    1,
                ),
                0.0,
            )
            q_q3 = tl.where(
                (mask_h[:, None]) & (mask_d_quarter[None, :]),
                tl.gather(
                    q_main,
                    tl.broadcast_to(offs_d_q3[None, :], [BLOCK_H, BLOCK_D // 4]),
                    1,
                ),
                0.0,
            )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # Load packed INT2 K in transposed format for efficient dot product
            offs_d_packed = tl.arange(0, BLOCK_D // 4)
            mask_d_packed = offs_d_packed < (L // 4)

            offs_buf_k_packed = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d_packed[:, None]
            )
            k_packed = tl.load(
                K_Buffer + offs_buf_k_packed,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d_packed[:, None]),
                other=0,
            )

            # Load K scales and zeros for dequantization
            if GROUPED:
                # When GROUP_SIZE divides into the per-quarter dim
                # (BLOCK_D // 4), use the fast per-group-load + broadcast
                # path. Otherwise (group spans multiple quarters), fall back
                # to the per-element load.
                if FAST:
                    offs_grp_k = tl.arange(0, NUM_GROUPS_QUARTER)
                    offs_grp_k_q1 = (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_grp_k_q2 = 2 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_grp_k_q3 = 3 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_k
                    offs_sz_k = (
                        kv_loc[None, :] * stride_sz_kbs + cur_kv_head * stride_sz_kh
                    )
                    k_scale_q0_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q0_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q1_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q1[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q1_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q1[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q2_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q2[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q2_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q2[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    k_scale_q3_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q3[:, None],
                        mask=offs_n[None, :] < split_kv_end, other=1.0,
                    )
                    k_zero_q3_grp = tl.load(
                        K_Scales_Zeros + offs_sz_k + 2 * offs_grp_k_q3[:, None] + 1,
                        mask=offs_n[None, :] < split_kv_end, other=0.0,
                    )
                    # Broadcast per-group across GROUP_SIZE dims via reshape.
                    k_scale_q0 = tl.reshape(
                        tl.broadcast_to(k_scale_q0_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q0 = tl.reshape(
                        tl.broadcast_to(k_zero_q0_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q1 = tl.reshape(
                        tl.broadcast_to(k_scale_q1_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q1 = tl.reshape(
                        tl.broadcast_to(k_zero_q1_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q2 = tl.reshape(
                        tl.broadcast_to(k_scale_q2_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q2 = tl.reshape(
                        tl.broadcast_to(k_zero_q2_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_scale_q3 = tl.reshape(
                        tl.broadcast_to(k_scale_q3_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                    k_zero_q3 = tl.reshape(
                        tl.broadcast_to(k_zero_q3_grp[:, None, :],
                                        (NUM_GROUPS_QUARTER, GROUP_SIZE, BLOCK_N)),
                        (BLOCK_D // 4, BLOCK_N),
                    )
                else:
                    # Fallback: group spans multiple quarters. Each quarter is
                    # entirely within a single group, so just load 1 (scale,
                    # zero) per (quarter, token) and broadcast across all dims.
                    offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                    k_scale_q0_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q0,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q0_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q0 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q1_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q1,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q1_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q1 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q2_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q2,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q2_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q2 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q3_t = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q3,
                                           mask=offs_n < split_kv_end, other=1.0)
                    k_zero_q3_t  = tl.load(K_Scales_Zeros + offs_sz_k_1d + 2 * grp_q3 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    k_scale_q0 = tl.broadcast_to(k_scale_q0_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q0  = tl.broadcast_to(k_zero_q0_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q1 = tl.broadcast_to(k_scale_q1_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q1  = tl.broadcast_to(k_zero_q1_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q2 = tl.broadcast_to(k_scale_q2_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q2  = tl.broadcast_to(k_zero_q2_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                    k_scale_q3 = tl.broadcast_to(k_scale_q3_t[None, :], (BLOCK_D // 4, BLOCK_N))
                    k_zero_q3  = tl.broadcast_to(k_zero_q3_t[None, :],  (BLOCK_D // 4, BLOCK_N))
                # Cast scales/zeros to q's dtype ONCE so the per-element dequant
                # below stays entirely in bf16 (saves 2 fp32↔bf16 casts per crumb).
                k_scale_q0 = k_scale_q0.to(q_q0.dtype)
                k_zero_q0  = k_zero_q0.to(q_q0.dtype)
                k_scale_q1 = k_scale_q1.to(q_q0.dtype)
                k_zero_q1  = k_zero_q1.to(q_q0.dtype)
                k_scale_q2 = k_scale_q2.to(q_q0.dtype)
                k_zero_q2  = k_zero_q2.to(q_q0.dtype)
                k_scale_q3 = k_scale_q3.to(q_q0.dtype)
                k_zero_q3  = k_zero_q3.to(q_q0.dtype)
                # Dequantize INT2 K inline: unpack 4 crumbs per-group.
                # k_packed shape: [BLOCK_D//4, BLOCK_N] (transposed)
                k_q0 = ((k_packed & 0x03).to(q_q0.dtype) - k_zero_q0) * k_scale_q0
                k_q1 = (((k_packed >> 2) & 0x03).to(q_q0.dtype) - k_zero_q1) * k_scale_q1
                k_q2 = (((k_packed >> 4) & 0x03).to(q_q0.dtype) - k_zero_q2) * k_scale_q2
                k_q3 = (((k_packed >> 6) & 0x03).to(q_q0.dtype) - k_zero_q3) * k_scale_q3
            else:
                offs_sz_k_1d = kv_loc * stride_sz_kbs + cur_kv_head * stride_sz_kh
                k_scale_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                ).to(q_q0.dtype)
                k_zero_1d = tl.load(
                    K_Scales_Zeros + offs_sz_k_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                ).to(q_q0.dtype)
                k_q0 = (
                    (k_packed & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q1 = (
                    ((k_packed >> 2) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q2 = (
                    ((k_packed >> 4) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]
                k_q3 = (
                    ((k_packed >> 6) & 0x03).to(q_q0.dtype) - k_zero_1d[None, :]
                ) * k_scale_1d[None, :]

            # Compute QK as ONE fused MMA instead of 4 small ones by stacking
            # the 4 dequantized quarters into a contiguous D axis.
            # The int2 unpack assigns crumb i to original dim positions
            # [i*L//4, (i+1)*L//4), so concatenating q0|q1|q2|q3 along
            # D reconstructs the natural K layout.
            #
            # We use tl.join (which adds a new last axis) + tl.reshape to
            # interleave: [BLOCK_D//4, BLOCK_N] -> [4, BLOCK_D//4, BLOCK_N]
            # via two binary joins -> permute -> reshape to [BLOCK_D, BLOCK_N].
            k_01 = tl.join(k_q0, k_q1)        # [BLOCK_D//4, BLOCK_N, 2]
            k_23 = tl.join(k_q2, k_q3)        # [BLOCK_D//4, BLOCK_N, 2]
            k_full = tl.join(k_01, k_23)      # [BLOCK_D//4, BLOCK_N, 2, 2]
            k_full = tl.reshape(k_full, (BLOCK_D // 4, BLOCK_N, 4))
            k_full = tl.permute(k_full, (2, 0, 1))      # [4, BLOCK_D//4, BLOCK_N]
            k_full = tl.reshape(k_full, (BLOCK_D, BLOCK_N))

            q_01 = tl.join(q_q0, q_q1)        # [BLOCK_H, BLOCK_D//4, 2]
            q_23 = tl.join(q_q2, q_q3)
            q_full = tl.join(q_01, q_23)      # [BLOCK_H, BLOCK_D//4, 2, 2]
            q_full = tl.reshape(q_full, (BLOCK_H, BLOCK_D // 4, 4))
            q_full = tl.permute(q_full, (0, 2, 1))      # [BLOCK_H, 4, BLOCK_D//4]
            q_full = tl.reshape(q_full, (BLOCK_H, BLOCK_D))

            qk = tl.dot(q_full, k_full)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # Load packed INT2 V and dequantize. V layout: [BLOCK_N, BLOCK_D//4]
            offs_d_packed_v = tl.arange(0, BLOCK_D // 4)
            offs_buf_v_packed = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_d_packed_v[None, :]
            )
            v_packed = tl.load(
                V_Buffer + offs_buf_v_packed,
                mask=(offs_n[:, None] < split_kv_end) & (offs_d_packed_v[None, :] < (L // 4)),
                other=0,
            )

            # Load V scales and zeros for dequantization
            if GROUPED:
                if FAST:
                    offs_grp_v = tl.arange(0, NUM_GROUPS_QUARTER)
                    offs_grp_v_q1 = (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_grp_v_q2 = 2 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_grp_v_q3 = 3 * (BLOCK_D // 4) // GROUP_SIZE + offs_grp_v
                    offs_sz_v = (
                        kv_loc[:, None] * stride_sz_vbs + cur_kv_head * stride_sz_vh
                    )
                    v_scale_q0_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q0_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q1_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q1[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q1_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q1[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q2_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q2[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q2_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q2[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q3_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q3[None, :],
                        mask=offs_n[:, None] < split_kv_end, other=1.0,
                    )
                    v_zero_q3_grp = tl.load(
                        V_Scales_Zeros + offs_sz_v + 2 * offs_grp_v_q3[None, :] + 1,
                        mask=offs_n[:, None] < split_kv_end, other=0.0,
                    )
                    v_scale_q0 = tl.reshape(
                        tl.broadcast_to(v_scale_q0_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q0 = tl.reshape(
                        tl.broadcast_to(v_zero_q0_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q1 = tl.reshape(
                        tl.broadcast_to(v_scale_q1_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q1 = tl.reshape(
                        tl.broadcast_to(v_zero_q1_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q2 = tl.reshape(
                        tl.broadcast_to(v_scale_q2_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q2 = tl.reshape(
                        tl.broadcast_to(v_zero_q2_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_scale_q3 = tl.reshape(
                        tl.broadcast_to(v_scale_q3_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                    v_zero_q3 = tl.reshape(
                        tl.broadcast_to(v_zero_q3_grp[:, :, None],
                                        (BLOCK_N, NUM_GROUPS_QUARTER, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_D // 4),
                    )
                else:
                    # Fallback: group spans multiple quarters.
                    offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                    v_scale_q0_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q0,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q0_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q0 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q1_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q1,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q1_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q1 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q2_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q2,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q2_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q2 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q3_t = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q3,
                                           mask=offs_n < split_kv_end, other=1.0)
                    v_zero_q3_t  = tl.load(V_Scales_Zeros + offs_sz_v_1d + 2 * v_grp_q3 + 1,
                                           mask=offs_n < split_kv_end, other=0.0)
                    v_scale_q0 = tl.broadcast_to(v_scale_q0_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q0  = tl.broadcast_to(v_zero_q0_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q1 = tl.broadcast_to(v_scale_q1_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q1  = tl.broadcast_to(v_zero_q1_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q2 = tl.broadcast_to(v_scale_q2_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q2  = tl.broadcast_to(v_zero_q2_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                    v_scale_q3 = tl.broadcast_to(v_scale_q3_t[:, None], (BLOCK_N, BLOCK_D // 4))
                    v_zero_q3  = tl.broadcast_to(v_zero_q3_t[:, None],  (BLOCK_N, BLOCK_D // 4))
                # Cast V scales/zeros to q's dtype ONCE so per-element dequant
                # below stays in bf16 (saves 2 fp32↔bf16 casts per crumb).
                v_scale_q0 = v_scale_q0.to(q_q0.dtype)
                v_zero_q0  = v_zero_q0.to(q_q0.dtype)
                v_scale_q1 = v_scale_q1.to(q_q0.dtype)
                v_zero_q1  = v_zero_q1.to(q_q0.dtype)
                v_scale_q2 = v_scale_q2.to(q_q0.dtype)
                v_zero_q2  = v_zero_q2.to(q_q0.dtype)
                v_scale_q3 = v_scale_q3.to(q_q0.dtype)
                v_zero_q3  = v_zero_q3.to(q_q0.dtype)
                # Dequantize INT2 V inline: unpack 4 crumbs per-group.
                v_q0 = ((v_packed & 0x03).to(q_q0.dtype) - v_zero_q0) * v_scale_q0
                v_q1 = (((v_packed >> 2) & 0x03).to(q_q0.dtype) - v_zero_q1) * v_scale_q1
                v_q2 = (((v_packed >> 4) & 0x03).to(q_q0.dtype) - v_zero_q2) * v_scale_q2
                v_q3 = (((v_packed >> 6) & 0x03).to(q_q0.dtype) - v_zero_q3) * v_scale_q3
            else:
                offs_sz_v_1d = kv_loc * stride_sz_vbs + cur_kv_head * stride_sz_vh
                v_scale_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 0,
                    mask=offs_n < split_kv_end,
                    other=1.0,
                ).to(q_q0.dtype)
                v_zero_1d = tl.load(
                    V_Scales_Zeros + offs_sz_v_1d + 1,
                    mask=offs_n < split_kv_end,
                    other=0.0,
                ).to(q_q0.dtype)
                v_q0 = (
                    (v_packed & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q1 = (
                    ((v_packed >> 2) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q2 = (
                    ((v_packed >> 4) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]
                v_q3 = (
                    ((v_packed >> 6) & 0x03).to(q_q0.dtype) - v_zero_1d[:, None]
                ) * v_scale_1d[:, None]

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Scale existing accumulators
            acc_q0 *= re_scale[:, None]
            acc_q1 *= re_scale[:, None]
            acc_q2 *= re_scale[:, None]
            acc_q3 *= re_scale[:, None]

            # Accumulate attention-weighted V for 4 quarters
            acc_q0 += tl.dot(p.to(v_q0.dtype), v_q0)
            acc_q1 += tl.dot(p.to(v_q1.dtype), v_q1)
            acc_q2 += tl.dot(p.to(v_q2.dtype), v_q2)
            acc_q3 += tl.dot(p.to(v_q3.dtype), v_q3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store 4 quarters separately to indices [k*L//4, (k+1)*L//4)
        offs_dv = tl.arange(0, BLOCK_D // 4)
        mask_dv_quarter = offs_dv < (L // 4)
        base_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        tl.store(
            Att_Out + base_mid_o + offs_dv[None, :],
            acc_q0 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + L // 4)[None, :],
            acc_q1 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 2 * (L // 4))[None, :],
            acc_q2 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )
        tl.store(
            Att_Out + base_mid_o + (offs_dv + 3 * (L // 4))[None, :],
            acc_q3 / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv_quarter[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // L

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_att_m_fwd_quant_int2(
    q,
    k_buffer,  # Quantized INT2 (packed)
    v_buffer,  # Quantized INT2 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    """
    INT2 quantized KV cache attention wrapper (MHA).
    Dequantizes KV cache on-the-fly inside the kernel.
    """
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    # For INT2, the buffer stores packed values (head_dim//4)
    # But we need to work with the actual head_dim
    Lk = k_buffer.shape[-1] * 4  # Unpack to get real dimension
    Lv = v_buffer.shape[-1] * 4

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    group_size = _get_shared_kv_scale_group_size(
        Lk, Lv, k_scales_zeros, v_scales_zeros
    )

    _fwd_kernel_stage1_quant_int2[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        GROUP_SIZE=group_size,
    )


def _decode_grouped_att_m_fwd_quant_int2(
    q,
    k_buffer,  # Quantized INT2 (packed)
    v_buffer,  # Quantized INT2 (packed)
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    R_k=None,             # [1 or Hkv, L, L] bf16: fold q-rotation into the kernel
):
    """
    INT2 quantized KV cache attention wrapper (GQA/MQA).
    Dequantizes KV cache on-the-fly inside the kernel.

    Tuning history (Qwen3-8B 32Q/8KV, head_dim=128, bs=1, seq=80k, H100):
      knobs                                   | mean ms (seq=80k bs=1)
      ----------------------------------------+----------------------
      BLOCK_N=32  BLOCK_H=16 W=4 S=2 (legacy) | 0.650
      BLOCK_N=32  BLOCK_H=16 W=4 S=3          | 0.165  (+splits=32 default)
      BLOCK_N=128 BLOCK_H=8  W=4 S=3 (current)| 0.096  ← 1.74x over previous tune
    Bigger BLOCK_N amortizes the per-iteration dependency chain (load packed
    crumb → mask/shift → cast → sub zero → mul scale → tl.dot) over more KV
    tokens; smaller BLOCK_H lowers register pressure so more blocks fit per SM.
    """
    # For INT2, k_buffer is packed, so actual head dim is 4x the last dimension.
    # K and V share the same head dim in this path (no MLA/DPE split).
    L = k_buffer.shape[-1] * 4
    assert v_buffer.shape[-1] * 4 == L, "INT2 KV cache requires Lk == Lv"
    BLOCK_D = triton.next_power_of_2(L)
    group_size = _get_shared_kv_scale_group_size(
        L, L, k_scales_zeros, v_scales_zeros
    )

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    MAX_KV_SPLITS = max_kv_splits

    # Tile heuristic 
    if kv_group_num <= 8:
        if batch >= 16:
            _bn_default, _bh_default, _nw_default = 32, 4, 1
        elif batch >= 4:
            _bn_default, _bh_default, _nw_default = 64, 8, 2
        else:
            _bn_default, _bh_default, _nw_default = 128, 8, 4
    else:
        _bn_default = 128
        _bh_default = 16 if batch >= 16 else 8
        _nw_default = 4
    BLOCK = int(os.environ.get("SGL_INT2_BLOCK_N", _bn_default))
    BLOCK_H = int(os.environ.get("SGL_INT2_BLOCK_H", _bh_default))
    num_warps = int(os.environ.get("SGL_INT2_NUM_WARPS", _nw_default))
    num_stages = int(os.environ.get("SGL_INT2_NUM_STAGES", 3))

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    if R_k is not None:
        Rk_use = R_k if R_k.dim() == 3 else R_k.unsqueeze(0)
        Rk_use = Rk_use.contiguous()
        has_rot = True
        nrk = Rk_use.shape[0]
    else:
        Rk_use = q  # dummy pointer, never dereferenced (HAS_ROT=0)
        has_rot = False
        nrk = 1

    _fwd_grouped_kernel_stage1_quant_int2[grid](
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        v_scales_zeros.stride(0),
        v_scales_zeros.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=num_stages,
        L=L,
        GROUP_SIZE=group_size,
        HAS_ROT=has_rot,
        NRK=nrk,
        R_k_ptr=Rk_use,
        **extra_kargs,
    )


def decode_attention_fwd_normal_quant_int2(
    q,
    k_buffer,  # Quantized INT2
    v_buffer,  # Quantized INT2
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    """
    Normal (MHA) attention forward with INT2 quantized KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.
    """
    # Stage 1: Compute attention scores and accumulate values
    _decode_att_m_fwd_quant_int2(
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
    )
    # For INT2, v_buffer is packed (quarter size), but stage2 needs full dimension
    # o has the correct output dimension
    v_buf_for_stage2 = o

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale=1.0,
        v_buffer=v_buf_for_stage2,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd_grouped_quant_int2(
    q,
    k_buffer,  # Quantized INT2
    v_buffer,  # Quantized INT2
    k_scales_zeros,  # Scales and zeros for K
    v_scales_zeros,  # Scales and zeros for V
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
    R_k=None,
):
    """
    Grouped (GQA/MQA) attention forward with INT2 quantized KV cache.
    Dequantizes on-the-fly inside the kernel, avoiding global memory writes.
    """
    # Stage 1: Compute attention scores and accumulate values
    _decode_grouped_att_m_fwd_quant_int2(
        q,
        k_buffer,
        v_buffer,
        k_scales_zeros,
        v_scales_zeros,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        xai_temperature_len,
        R_k=R_k,
    )
    # For INT2, v_buffer is packed (quarter size), but stage2 needs full dimension
    # o has the correct output dimension
    v_buf_for_stage2 = o

    # Stage 2: Reduce across KV splits and compute final output
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale=1.0,
        v_buffer=v_buf_for_stage2,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
        output_lse=output_lse,
    )
