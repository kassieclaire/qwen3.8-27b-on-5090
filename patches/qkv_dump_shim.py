"""
Standalone post-RoPE Q/K/V dump hook for OSCAR rotation calibration.

Ported out of OSCAR's vendored triton_backend.py (branch zhongzhu/hybrid-model,
forward_extend) so that calibration can run on UPSTREAM sglang (CUDA 13) without
having to port the whole attention backend.

It wraps TritonAttnBackend.forward_extend and saves, per full-attention layer:
    <DUMP_KVCACHE_DIR>/layer_<gid>/q/<chunk>.pt      [T, Hq, qk_head_dim]
    <DUMP_KVCACHE_DIR>/layer_<gid>/k/<chunk>.pt      [T, Hkv, head_dim]
    <DUMP_KVCACHE_DIR>/layer_<gid>/v/<chunk>.pt      [T, Hkv, head_dim]
    <DUMP_KVCACHE_DIR>/layer_<gid>/seq_lens/<chunk>.pt   int32 per-request lens

Only fires on layers that actually go through the triton attention backend, so on
a hybrid model (Qwen3.5/Qwen3.8 GatedDeltaNet) the linear-attention layers are
skipped automatically -- no filtering needed.

This assumes TP == 1 (single RTX 5090), so the all_gather branch of the original
is intentionally dropped.

Activate with:
    DUMP_KVCACHE=true
    DUMP_KVCACHE_DIR=/path/out
    DUMP_KVCACHE_TOKENS=30000
    SGLANG_QKV_DUMP_SHIM=1     (so sitecustomize installs it)
"""

import os
import logging

logger = logging.getLogger(__name__)


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def install():
    import torch
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    dump_dir = os.environ.get("DUMP_KVCACHE_DIR", "./qkv_dump")
    max_tokens = int(os.environ.get("DUMP_KVCACHE_TOKENS", "30000"))
    os.makedirs(dump_dir, exist_ok=True)

    # When speculative decoding is on, BOTH the target and the draft push
    # tensors through this backend, and their layer_ids overlap (target uses
    # global ids 3,7,...,63; the DSPARK draft uses 0..4) -- so an unfiltered
    # dump silently interleaves two different geometries into the same
    # layer_<id> directory. Filter by head_dim, which differs (target 256 vs
    # draft 128), to capture exactly one model per run.
    only_head_dim = os.environ.get("DUMP_KVCACHE_ONLY_HEAD_DIM", "").strip()
    only_head_dim = int(only_head_dim) if only_head_dim else None
    capture_decode = _truthy(os.environ.get("DUMP_KVCACHE_DECODE", ""))
    if only_head_dim is not None:
        logger.warning(
            "[qkv_dump] restricting dump to layers with head_dim=%d",
            only_head_dim,
        )

    state = {
        "saved": {},   # layer_id -> tokens saved
        "chunk": {},   # layer_id -> next chunk index
        "done": set(),
    }

    orig_forward_extend = TritonAttnBackend.forward_extend
    orig_forward_decode = TritonAttnBackend.forward_decode

    def _capture(q, k, v, layer, forward_batch):
        """Shared capture body for both the extend and decode paths."""
        try:
            lid = layer.layer_id
            if only_head_dim is not None and k is not None:
                # layer.head_dim is the logical per-head width and is correct on
                # both the 3D extend layout and the 2D decode layout.
                if int(getattr(layer, "head_dim", k.shape[-1])) != only_head_dim:
                    return
            if k is not None and v is not None and lid not in state["done"]:
                saved = state["saved"].get(lid, 0)
                remaining = max_tokens - saved
                if remaining > 0:
                    n = min(q.shape[0], remaining)
                    torch.cuda.synchronize()
                    q_d = (
                        q[:n]
                        .reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        .contiguous()
                        .detach()
                    )
                    # k/v arrive [T, Hkv, hd] on the extend path but may be
                    # flattened to [T, Hkv*hd] on the decode path.
                    if k.dim() >= 3:
                        kv_heads = int(k.shape[1])
                    else:
                        kv_heads = max(int(k.shape[-1]) // int(layer.head_dim), 1)
                    k_d = k[:n].reshape(n, kv_heads, -1).contiguous().detach()
                    v_d = v[:n].reshape(n, kv_heads, -1).contiguous().detach()

                    lens = []
                    esl = getattr(forward_batch, "extend_seq_lens", None)
                    if esl is not None:
                        remain = n
                        for slen in esl.tolist():
                            if remain <= 0:
                                break
                            take = min(slen, remain)
                            lens.append(take)
                            remain -= take
                    else:
                        lens = [n]

                    ci = state["chunk"].get(lid, 0)
                    for name, t in (("q", q_d), ("k", k_d), ("v", v_d)):
                        d = os.path.join(dump_dir, f"layer_{lid}", name)
                        os.makedirs(d, exist_ok=True)
                        torch.save(t.cpu(), os.path.join(d, f"{ci}.pt"))
                    sd = os.path.join(dump_dir, f"layer_{lid}", "seq_lens")
                    os.makedirs(sd, exist_ok=True)
                    torch.save(
                        torch.tensor(lens, dtype=torch.int32),
                        os.path.join(sd, f"{ci}.pt"),
                    )

                    state["saved"][lid] = saved + n
                    state["chunk"][lid] = ci + 1
                    if saved + n >= max_tokens:
                        state["done"].add(lid)
                        logger.warning(
                            f"[qkv_dump] layer {lid} complete "
                            f"({saved + n} tokens, {ci + 1} chunks)"
                        )
        except Exception as e:  # never break serving because of the dump
            logger.error(f"[qkv_dump] layer dump failed: {e}", exc_info=True)

    def forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
        _capture(q, k, v, layer, forward_batch)
        return orig_forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs):
        # The DSPARK draft model writes almost all of its KV during DECODE
        # (its prefill only ever sees the gamma+1 verify window), so an
        # extend-only hook captures ~21 tokens and the rotation fit is
        # meaningless. Capture the decode path too.
        if capture_decode:
            _capture(q, k, v, layer, forward_batch)
        return orig_forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs)

    TritonAttnBackend.forward_extend = forward_extend
    TritonAttnBackend.forward_decode = forward_decode
    logger.warning(
        f"[qkv_dump] hook installed dir={dump_dir} max_tokens={max_tokens} "
        f"capture_decode={capture_decode}"
    )


if _truthy(os.environ.get("SGLANG_QKV_DUMP_SHIM", "")) and _truthy(
    os.environ.get("DUMP_KVCACHE", "")
):
    install()
