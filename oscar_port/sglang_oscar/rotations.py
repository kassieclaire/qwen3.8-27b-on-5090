"""OSCAR rotation config + checkpoint loading.

Ported from the OSCAR fork's ``srt/mem_cache/memory_pool.py``
(``OscarRotationConfig``, ``load_oscar_rotation_config``, ``load_oscar_rotations``).

PORT NOTE -- env access
-----------------------
The OSCAR fork read these through ``sglang.srt.environ.envs.SGLANG_OSCAR_*``,
which only works because that fork *registered* them in its ``environ.py``.
Newer upstream's ``environ.py`` has no such entries, and its ``EnvField``
registry rejects unknown attributes. So we read ``os.environ`` directly here.
That keeps the shim independent of upstream's env registry and makes the
defaults explicit and auditable.

Env vars (name / type / default) -- values match the OSCAR fork exactly:

    SGLANG_OSCAR_K_ROTATION_PATH   str    ""      K rotation checkpoint
    SGLANG_OSCAR_V_ROTATION_PATH   str    ""      V rotation checkpoint
    SGLANG_OSCAR_K_CLIP_RATIO      float  0.0     per-row |.| quantile clip for K
    SGLANG_OSCAR_V_CLIP_RATIO      float  0.0     per-row |.| quantile clip for V
    SGLANG_LLOYD_MAX               bool   False   Lloyd-Max buckets (num_groups==1 only)
    SGLANG_OSCAR_ABSORB_V_ROTATION bool   False   fold R_v into qkv_proj's V slice
    SGLANG_MIXED_KV_SCALE_DTYPE    str    float32 dtype of the int2 scale/zero buffers

NOT ported: SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT (default False in the fork).
It selects ``quantized_set_kv_int2_oscar_rotate_k_clip_triton``, a fused
rotate+clip+pack kernel that additionally requires V-rotation absorption and a
single-scale layout. We always take the pretransformed-clip kernel instead --
see README "Deliberately omitted".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "y", "on", "t"}
_FALSE = {"0", "false", "no", "n", "off", "f", ""}


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a float") from exc


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean")


_SCALE_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def scale_dtype() -> torch.dtype:
    """dtype of the interleaved [scale, zero] buffers (SGLANG_MIXED_KV_SCALE_DTYPE)."""
    name = env_str("SGLANG_MIXED_KV_SCALE_DTYPE", "float32").strip().lower()
    if name not in _SCALE_DTYPES:
        raise ValueError(
            f"SGLANG_MIXED_KV_SCALE_DTYPE={name!r} unsupported; "
            f"choose from {sorted(_SCALE_DTYPES)}"
        )
    return _SCALE_DTYPES[name]


@dataclass(frozen=True)
class OscarRotationConfig:
    """Config for the OSCAR learned rotation + per-row clip applied to int2 KV.

    Verbatim semantics from the OSCAR fork, including the validation rules.
    """

    k_rotation_path: str
    v_rotation_path: str
    k_clip_ratio: float
    v_clip_ratio: float

    def __post_init__(self):
        for name, r in (("k", self.k_clip_ratio), ("v", self.v_clip_ratio)):
            if not (0.0 <= r <= 1.0):
                raise ValueError(
                    f"SGLANG_OSCAR_{name.upper()}_CLIP_RATIO must be in [0, 1], got {r}"
                )
        if not (self.k_rotation_path and self.v_rotation_path):
            raise ValueError(
                "OSCAR int2 KV cache requires both SGLANG_OSCAR_K_ROTATION_PATH "
                "and SGLANG_OSCAR_V_ROTATION_PATH to point at rotation checkpoints"
            )


def load_oscar_rotation_config(draft: bool = False) -> OscarRotationConfig:
    """Build an :class:`OscarRotationConfig` from the ``SGLANG_OSCAR_*`` env vars.

    Read on every call (not at import time) so tests can flip env between pool
    constructions -- same contract as the OSCAR fork.

    When ``draft`` is True, prefer the ``SGLANG_OSCAR_DRAFT_*`` variants. The
    DSPARK draft model has a DIFFERENT KV geometry from the target (5 full
    layers / 8 kv heads / head_dim 128, versus the target's 16 layers /
    4 kv heads / head_dim 256), so it needs rotations fitted against its own
    activations. Reusing the target's rotations is not merely suboptimal -- the
    matrices are the wrong SHAPE (256x256 vs 128x128) and keyed by the wrong
    layer ids.

    The draft pool must still be int2: it is allocated one slot per target
    token, so at 256k it rivals the target pool (5 GiB bf16 / 2.5 GiB fp8),
    and any non-int2 dtype would be a second KV quantization method.
    """
    if draft:
        return OscarRotationConfig(
            k_rotation_path=env_str("SGLANG_OSCAR_DRAFT_K_ROTATION_PATH"),
            v_rotation_path=env_str("SGLANG_OSCAR_DRAFT_V_ROTATION_PATH"),
            k_clip_ratio=env_float(
                "SGLANG_OSCAR_DRAFT_K_CLIP_RATIO",
                env_float("SGLANG_OSCAR_K_CLIP_RATIO", 0.0),
            ),
            v_clip_ratio=env_float(
                "SGLANG_OSCAR_DRAFT_V_CLIP_RATIO",
                env_float("SGLANG_OSCAR_V_CLIP_RATIO", 0.0),
            ),
        )
    return OscarRotationConfig(
        k_rotation_path=env_str("SGLANG_OSCAR_K_ROTATION_PATH"),
        v_rotation_path=env_str("SGLANG_OSCAR_V_ROTATION_PATH"),
        k_clip_ratio=env_float("SGLANG_OSCAR_K_CLIP_RATIO", 0.0),
        v_clip_ratio=env_float("SGLANG_OSCAR_V_CLIP_RATIO", 0.0),
    )


def draft_oscar_enabled() -> bool:
    """True when draft-specific rotation checkpoints are configured."""
    return bool(
        env_str("SGLANG_OSCAR_DRAFT_K_ROTATION_PATH")
        or env_str("SGLANG_OSCAR_DRAFT_V_ROTATION_PATH")
    )


def oscar_enabled() -> bool:
    """True when at least one rotation path is set (mirrors the OSCAR fork's
    ``dtype == "int2" and (K_ROTATION_PATH or V_ROTATION_PATH)`` gate)."""
    return bool(
        env_str("SGLANG_OSCAR_K_ROTATION_PATH")
        or env_str("SGLANG_OSCAR_V_ROTATION_PATH")
    )


def load_oscar_rotations(
    path: str,
    layer_num: int,
    start_layer: int,
    head_dim,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    layer_ids: Optional[List[int]] = None,
):
    """Load per-layer OSCAR rotation matrices from ``path``.

    Checkpoint schema (produced by the offline OSCAR pipeline)::

        {"layers": {layer_id: {"rotation": Tensor}, ...}}

    ``layer_ids`` -- **this is the load-bearing argument for hybrid models.**
    The checkpoint is keyed by GLOBAL layer id (Qwen3.8-27B full-attention
    layers are 3, 7, 11, ..., 63) but the inner KV pool only holds the 16
    full-attention layers and indexes them by LOCAL id 0..15. Passing
    ``layer_ids=[3, 7, ..., 63]`` maps global -> local in that order. Without
    it the loader would use the contiguous range
    ``[start_layer, start_layer + layer_num)`` and silently load the wrong
    matrices (or raise "missing layer").

    ``head_dim`` may be a scalar (uniform geometry -> returns a stacked
    ``[layer_num, hd, hd]`` tensor) or a per-local-layer sequence
    (heterogeneous geometry -> returns a list of ``layer_num`` tensors).

    Both rotation formats are accepted and preserved:
      * V1 per-layer  ``[hd, hd]``
      * V2 per-head   ``[num_kv_heads, hd, hd]``
    Downstream consumers branch on ``R.dim()``.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "layers" not in state:
        raise ValueError(f"OSCAR rotation checkpoint at {path} missing 'layers' key")
    layers = state["layers"]

    if layer_ids is not None:
        if len(layer_ids) != layer_num:
            raise ValueError(
                f"load_oscar_rotations: layer_ids has {len(layer_ids)} entries "
                f"but layer_num={layer_num}"
            )
        global_layer_ids = list(layer_ids)
    else:
        global_layer_ids = [start_layer + local for local in range(layer_num)]

    per_layer = not isinstance(head_dim, int)
    if per_layer:
        head_dims = list(head_dim)
        if len(head_dims) != layer_num:
            raise ValueError(
                f"per-layer head_dim list len {len(head_dims)} != layer_num {layer_num}"
            )
    else:
        head_dims = [head_dim] * layer_num

    mats = []
    for local, global_lid in enumerate(global_layer_ids):
        hd = head_dims[local]
        if global_lid not in layers and str(global_lid) not in layers:
            raise ValueError(
                f"OSCAR rotation checkpoint at {path} missing layer {global_lid}. "
                f"Available (first 8): {sorted(map(str, layers))[:8]}"
            )
        ldata = layers.get(global_lid, layers.get(str(global_lid)))
        R = ldata["rotation"]
        if R.dim() == 3:
            if R.shape[1:] != (hd, hd):
                raise ValueError(
                    f"OSCAR per-head rotation layer {global_lid} has shape "
                    f"{tuple(R.shape)}, expected (num_kv_heads, {hd}, {hd})"
                )
        elif R.shape != (hd, hd):
            raise ValueError(
                f"OSCAR rotation layer {global_lid} has shape {tuple(R.shape)}, "
                f"expected ({hd}, {hd}) or (num_kv_heads, {hd}, {hd})"
            )
        mats.append(R.to(dtype))

    logger.info(
        "[oscar] loaded rotation from %s for layers %s head_dim=%s dtype=%s%s",
        path,
        (
            global_layer_ids
            if layer_ids is not None
            else f"[{start_layer}, {start_layer + layer_num})"
        ),
        ("per-layer" if per_layer else head_dim),
        dtype,
        (" [per-head: %d kv heads]" % mats[0].shape[0]) if mats[0].dim() == 3 else "",
    )

    if per_layer:
        return [m.to(device) for m in mats]
    return torch.stack(mats, dim=0).to(device)
