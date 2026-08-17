"""Post-load hook: fold R_v into Qwen3.5 ``qkv_proj``'s V slice.

Reimplementation of the OSCAR fork's ``_maybe_absorb_oscar_v_rotation_qwen35``
against the NEW upstream ``qwen3_5.py``. Enabled by
``SGLANG_OSCAR_ABSORB_V_ROTATION=1``.

WHY IT EXISTS
-------------
The int2 write path stores ``V @ R_v``. If R_v is instead folded into the
weights that PRODUCE V, the rotation is free at runtime: the qkv_proj output is
already in R_v space, so ``set_kv_buffer`` skips the V GEMM. The pool learns
this from ``layer.oscar_v_rotation_absorbed``.

THE OFFSET MATH -- VERIFIED AGAINST NEW UPSTREAM (checked, unchanged)
---------------------------------------------------------------------
Qwen3.5 sets ``attn_output_gate=True`` and builds::

    QKVParallelLinear(hidden, head_dim,
                      total_num_heads * (1 + attn_output_gate),   # Q + gate
                      total_num_kv_heads, bias=False)

so the qkv_proj output rows are laid out ``[q | gate | k | v]`` with
``len(gate) == len(q) == q_size``. NEW upstream confirms this: the forward does
``qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)``.
Therefore::

    v_offset = 2 * q_size + kv_size          (NOT the generic q_size + kv_size)

Using the generic offset would land inside the K block and silently corrupt
weights. The layer class is still named ``Qwen3_5AttentionDecoderLayer`` and
still exposes ``q_size`` / ``kv_size`` / ``head_dim`` / ``num_kv_heads`` /
``attn_output_gate``, so the fork's math transfers unchanged.

*** IMPORTANT LIMITATION (unchanged from the fork, and it BITES here) ***
Absorption requires a DENSE float ``qkv_proj.weight``. For
unsloth/Qwen3.8-27B-NVFP4 the attention q/k/v projections are fp8
(``float-quantized`` group_0), NOT bf16 -- so this hook will REFUSE to fold and
raise. That is intentional: silently folding into fp8 codes would corrupt the
weights. Run with ``SGLANG_OSCAR_ABSORB_V_ROTATION=0`` (the default) on that
checkpoint and let the pool rotate V at runtime instead. Note the OSCAR project
itself found ABSORB_V=1 to be WRONG for per-head/format_version-2 rotations
anyway.
"""

from __future__ import annotations

import logging

import torch

from .rotations import env_bool, load_oscar_rotation_config

logger = logging.getLogger(__name__)

_DENSE_FLOAT = (torch.float32, torch.float16, torch.bfloat16)


def absorb_v_rotation(model, model_label: str = "Qwen3.5") -> bool:
    """Fold R_v into every full-attention layer's qkv_proj V slice.

    Returns True if anything was folded. Idempotent: layers already marked
    ``oscar_v_rotation_absorbed`` are skipped.
    """
    if not env_bool("SGLANG_OSCAR_ABSORB_V_ROTATION", False):
        return False

    try:
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
    except ImportError:  # pragma: no cover
        logger.warning("[oscar] qwen3_5 not importable; skipping V absorption")
        return False

    cfg = load_oscar_rotation_config()
    layers = getattr(model, "layers", None)
    if layers is None:
        logger.warning("[oscar] %s has no .layers; skipping V absorption", model_label)
        return False

    start = int(getattr(model, "start_layer", 0))
    end = int(getattr(model, "end_layer", len(layers)))
    full_ids = [
        lid
        for lid in range(start, end)
        if isinstance(layers[lid], Qwen3_5AttentionDecoderLayer)
    ]
    if not full_ids:
        raise RuntimeError(
            f"SGLANG_OSCAR_ABSORB_V_ROTATION=1 for {model_label} but no "
            f"Qwen3_5AttentionDecoderLayer in layer range [{start}, {end})."
        )

    first = layers[full_ids[0]]
    w = getattr(first.qkv_proj, "weight", None)
    if w is None or w.ndim != 2:
        raise RuntimeError(
            f"SGLANG_OSCAR_ABSORB_V_ROTATION=1: {model_label} layer "
            f"{full_ids[0]} qkv_proj.weight is not 2D; only dense qkv_proj "
            f"layouts are supported."
        )
    if w.dtype not in _DENSE_FLOAT:
        raise RuntimeError(
            f"SGLANG_OSCAR_ABSORB_V_ROTATION=1: {model_label} qkv_proj has "
            f"dtype={w.dtype}; only dense float types are supported. "
            f"NVFP4/FP8 checkpoints (e.g. unsloth/Qwen3.8-27B-NVFP4, whose "
            f"attention projections are fp8) CANNOT use absorption -- unset "
            f"SGLANG_OSCAR_ABSORB_V_ROTATION and let the KV pool rotate V at "
            f"runtime."
        )

    state = torch.load(cfg.v_rotation_path, map_location="cpu", weights_only=False)
    if "layers" not in state:
        raise ValueError(
            f"OSCAR V-rotation checkpoint at {cfg.v_rotation_path} missing "
            f"'layers' key"
        )
    rot_layers = state["layers"]

    folded = skipped = 0
    for lid in full_ids:
        attn = layers[lid]
        if getattr(attn.attn, "oscar_v_rotation_absorbed", False):
            skipped += 1
            continue

        key = lid if lid in rot_layers else str(lid)
        if key not in rot_layers:
            raise ValueError(
                f"OSCAR V-rotation checkpoint missing entry for {model_label} "
                f"layer {lid} (have: {sorted(map(str, rot_layers))[:8]}...)"
            )
        R = rot_layers[key]["rotation"]
        head_dim = attn.head_dim
        if R.dim() != 2 or R.shape != (head_dim, head_dim):
            raise ValueError(
                f"OSCAR V-rotation layer {lid} has shape {tuple(R.shape)}, "
                f"expected ({head_dim}, {head_dim}). Per-head (3-D) rotations "
                f"cannot be absorbed into qkv_proj by this helper."
            )

        weight = attn.qkv_proj.weight
        R_v = R.to(dtype=torch.float32, device=weight.device)

        # [q | gate | k | v]; gate_size == q_size when attn_output_gate.
        gate = bool(getattr(attn, "attn_output_gate", True))
        v_offset = attn.q_size * (2 if gate else 1) + attn.kv_size

        v_w = weight.data.narrow(0, v_offset, attn.kv_size)
        v_w_3d = v_w.reshape(attn.num_kv_heads, attn.head_dim, -1)
        folded_w = torch.matmul(R_v.T, v_w_3d.to(torch.float32)).to(weight.dtype)
        v_w.copy_(folded_w.reshape_as(v_w))

        attn.attn.oscar_v_rotation_absorbed = True
        folded += 1

    if folded == 0 and skipped == 0:
        raise RuntimeError(
            f"SGLANG_OSCAR_ABSORB_V_ROTATION=1: {model_label} no layers folded."
        )
    if folded == 0:
        return False

    logger.info(
        "[oscar] absorbed V rotation into %s qkv_proj for %d/%d full-attention "
        "layers (skipped %d already-absorbed; attn_output_gate=%s, "
        "v_offset=2*q_size+kv_size)",
        model_label,
        folded,
        len(full_ids),
        skipped,
        bool(getattr(first, "attn_output_gate", True)),
    )
    return True


def install(qwen3_5_mod):
    """Hook ``absorb_v_rotation`` onto every Qwen3.5 ``load_weights``."""
    if getattr(qwen3_5_mod, "_oscar_absorb_patched", False):
        return
    if not env_bool("SGLANG_OSCAR_ABSORB_V_ROTATION", False):
        # Nothing to do; leave load_weights untouched.
        qwen3_5_mod._oscar_absorb_patched = True
        return

    import inspect

    patched = []
    for name, obj in vars(qwen3_5_mod).items():
        if not inspect.isclass(obj) or not hasattr(obj, "load_weights"):
            continue
        if not name.startswith("Qwen3_5"):
            continue
        orig = obj.load_weights

        def make(orig_fn, label):
            def load_weights(self, *a, **kw):
                out = orig_fn(self, *a, **kw)
                target = getattr(self, "model", self)
                # VL wrappers nest the text model one more level.
                if not hasattr(target, "layers") and hasattr(target, "language_model"):
                    target = target.language_model
                try:
                    absorb_v_rotation(target, model_label=label)
                except Exception:
                    logger.exception("[oscar] V-rotation absorption failed")
                    raise
                return out

            return load_weights

        obj.load_weights = make(orig, name)
        patched.append(name)

    qwen3_5_mod._oscar_absorb_patched = True
    logger.info("[oscar] hooked V-rotation absorption into: %s", patched)
