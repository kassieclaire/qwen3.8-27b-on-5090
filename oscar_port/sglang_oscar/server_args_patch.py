"""Monkeypatch: make NEW upstream accept ``--kv-cache-dtype int2``.

Upstream does not know the value ``int2``. Two places reject it:

1. ``ServerArgs.add_cli_args`` registers ``--kv-cache-dtype`` with an explicit
   ``choices=[...]`` list that has no ``int2`` -> argparse errors out.
2. ``sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype`` maps the
   string to a torch dtype and ends with
   ``raise ValueError(f"Unsupported kv_cache_dtype: ...")``.

We patch both. ``configure_kv_cache_dtype`` returns the STRING ``"int2"`` as the
pool dtype (that is exactly what the OSCAR fork did -- ``self.dtype == "int2"``
is the sentinel the whole int2 path keys off), and ``resolved_kv_cache_dtype``
is also set to ``"int2"`` so backends see a quantized tag.

Also registered here: ``--kv-cache-quant-group-size``, matching the OSCAR
fork's flag. NOTE the validation is STRICTER than the fork's: for head_dim 256
a group size of 128 yields num_groups == 2, which the packed-byte Triton
kernels cannot express (they need num_groups == 1, or a power of two divisible
by 4). This was verified on hardware -- the kernel raises NotImplementedError.
The fork accepted the flag and only failed later at the kernel; we fail at
startup instead.

ACTIVATION WITHOUT THE FLAG
---------------------------
If you would rather not patch argparse, set ``SGLANG_OSCAR_FORCE_INT2=1``.
``apply_patch`` then rewrites the parsed ``ServerArgs.kv_cache_dtype`` to
``int2`` after parsing. Both routes end in the same place.
"""

from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)

INT2 = "int2"


def install_cli(ServerArgs):
    """Add ``int2`` to the --kv-cache-dtype choices and register the group-size flag."""
    if getattr(ServerArgs, "_oscar_int2_cli_patched", False):
        return
    # ``add_cli_args`` is declared ``@staticmethod``; plain attribute access
    # already yields the underlying function on py3, but go through
    # ``inspect.getattr_static`` + ``__func__`` when present so this also works
    # if upstream ever switches it to a classmethod.
    raw = inspect.getattr_static(ServerArgs, "add_cli_args")
    orig_add = getattr(raw, "__func__", raw)

    def add_cli_args(parser):
        orig_add(parser)
        for action in parser._actions:
            if "--kv-cache-dtype" in getattr(action, "option_strings", ()):
                if action.choices is not None and INT2 not in action.choices:
                    action.choices = list(action.choices) + [INT2]
                    action.help = (action.help or "") + (
                        ' "int2" enables the OSCAR rotated 2-bit KV cache '
                        "(requires SGLANG_OSCAR_K_ROTATION_PATH / "
                        "SGLANG_OSCAR_V_ROTATION_PATH)."
                    )
        existing = {
            o for a in parser._actions for o in getattr(a, "option_strings", ())
        }
        if "--kv-cache-quant-group-size" not in existing:
            parser.add_argument(
                "--kv-cache-quant-group-size",
                type=int,
                default=None,
                help=(
                    "Head-dimension group size for int2 KV quantization. Only "
                    "valid with --kv-cache-dtype int2. Default (unset) = one "
                    "scale/zero pair per head. For head_dim=256 the only other "
                    "supported value is 64; 128 is NOT supported (it yields 2 "
                    "groups and the packed-byte kernels need num_groups %% 4 == 0)."
                ),
            )
        return parser

    ServerArgs.add_cli_args = staticmethod(add_cli_args)
    if not hasattr(ServerArgs, "kv_cache_quant_group_size"):
        ServerArgs.kv_cache_quant_group_size = None
    ServerArgs._oscar_int2_cli_patched = True
    logger.info("[oscar] patched ServerArgs CLI: --kv-cache-dtype int2")


def install_dtype_resolver(kv_cache_dtype_mod):
    """Let ``configure_kv_cache_dtype`` return the ``"int2"`` sentinel."""
    if getattr(kv_cache_dtype_mod, "_oscar_int2_patched", False):
        return
    orig = kv_cache_dtype_mod.configure_kv_cache_dtype

    def configure_kv_cache_dtype(*args, **kwargs):
        requested = kwargs.get("server_args_kv_cache_dtype")
        if requested is None and args:
            requested = args[0]
        draft = kwargs.get("speculative_draft_kv_cache_dtype")
        is_draft = bool(kwargs.get("is_draft_worker"))
        # The DSPARK draft worker gets its own pool. Only force int2 on it when
        # the user explicitly asked for it via --speculative-draft-kv-cache-dtype;
        # otherwise leave the draft on its default dtype.
        from . import kv_pool_patch as _kvp
        from .rotations import draft_oscar_enabled

        # Tell the pool constructor which worker it is building for, so it can
        # pick the matching rotation checkpoints.
        _kvp.set_draft_context(is_draft)

        if is_draft and draft is not None and draft == INT2:
            return INT2, INT2
        if requested == INT2:
            if is_draft and draft_oscar_enabled():
                # Draft rotations are configured -> the draft pool is ALSO
                # OSCAR int2. This is the required configuration when the task
                # allows no KV method other than OSCAR int2: the draft pool is
                # sized one slot per target token, so leaving it bf16/fp8 would
                # mean a second quantization scheme holding the larger share of
                # the KV budget (5.0 / 2.5 GiB vs the target's 2.0 GiB @256k).
                logger.info(
                    "[oscar] draft worker: using OSCAR int2 with draft-specific "
                    "rotations."
                )
                return INT2, INT2
            if is_draft:
                # The draft worker has NO OSCAR rotations of its own (they are
                # fitted against the target's 16 full-attention layers, global
                # ids 3,7,...,63; the draft has 5 layers with different
                # geometry). So it must never take the int2 path, or rotation
                # loading dies with "missing layer 0".
                #
                # If the user asked for a specific draft dtype (e.g.
                # --speculative-draft-kv-cache-dtype fp8_e4m3, which halves the
                # draft KV -- 5.0 -> 2.5 GiB at 256k), honour it by delegating
                # to upstream with the target's "int2" swapped out for "auto".
                # Otherwise fall back to the model dtype.
                model_dtype = kwargs.get("model_dtype")
                if model_dtype is None and len(args) >= 3:
                    model_dtype = args[2]
                if draft is not None:
                    logger.info(
                        "[oscar] draft worker: target is int2 but the draft has "
                        "no rotations; using requested draft dtype %s.",
                        draft,
                    )
                    if kwargs.get("server_args_kv_cache_dtype") is not None:
                        kwargs = dict(kwargs)
                        kwargs["server_args_kv_cache_dtype"] = "auto"
                    else:
                        args = list(args)
                        args[0] = "auto"
                        args = tuple(args)
                    return orig(*args, **kwargs)
                logger.warning(
                    "[oscar] draft worker requested int2 KV but has no OSCAR "
                    "rotations; using %s for the draft pool.",
                    model_dtype,
                )
                return "auto", model_dtype
            return INT2, INT2
        return orig(*args, **kwargs)

    kv_cache_dtype_mod.configure_kv_cache_dtype = configure_kv_cache_dtype
    kv_cache_dtype_mod._oscar_int2_patched = True
    logger.info("[oscar] patched configure_kv_cache_dtype for int2")


def install_element_size() -> None:
    """Teach ``torch._utils._element_size`` about the ``"int2"`` sentinel.

    Upstream's ``pool_configurator._compute_cell_size`` does::

        kv_size = torch._utils._element_size(kv_cache_dtype)

    to size the KV pool. Our pool dtype is the STRING ``"int2"``, so the stock
    implementation raises::

        RuntimeError: expected torch.dtype, but got <class 'str'>

    The OSCAR fork never hit this because its (older) pool_configurator had an
    explicit int2 branch; upstream restructured that code.

    int2 stores 4 values per byte, so the honest per-element size is 0.25 bytes.
    ``_element_size`` must return an int, and returning 0 would make the
    scheduler believe the KV cache is free (it would then size the pool
    unboundedly). We therefore return **1** and let the caller be conservative:
    the pool is sized as if KV were int8, i.e. we UNDER-claim the token budget
    by ~4x rather than over-claiming it. Over-claiming would OOM mid-run;
    under-claiming merely leaves capacity on the table and can be recovered
    explicitly with --max-total-tokens.

    Note the scale/zero side-buffers are genuine extra bytes per group, so the
    true cost is 0.25 + (2*4/group_size) bytes/elem; at group_size == 256 that
    is ~0.28. Returning 1 covers those too.
    """
    import torch

    orig = torch._utils._element_size

    if getattr(orig, "_oscar_int2_patched", False):
        return

    def _element_size(dtype):
        if isinstance(dtype, str) and dtype == INT2:
            return 1  # conservative; see docstring
        return orig(dtype)

    _element_size._oscar_int2_patched = True
    torch._utils._element_size = _element_size
    logger.info("[oscar] patched torch._utils._element_size for int2")


def install_cell_size() -> None:
    """Correct the KV pool sizing for int2 (recover the ~4x under-claim).

    ``install_element_size`` returns 1 byte/elem for the ``"int2"`` sentinel so
    that upstream's integer arithmetic keeps working. That is SAFE but sizes the
    pool as if KV were int8, so we only get ~1/4 of the tokens we could hold --
    at 256k that is the difference between a 30k-token pool and a full one.

    Rather than re-deriving the whole cell-size expression (upstream branches
    over MLA / DSA / MiniMax-sparse / FP4 variants), wrap the single public
    entry point and rescale its RESULT by the true bytes-per-element ratio:

        int2 packed          = 0.25 bytes/elem
        scale+zero side data = 2 * 4 / group_size bytes/elem
                             = 0.03125 at group_size=256

    so ~0.28 bytes/elem vs the 1.0 assumed => multiply the cell size by ~0.28.
    We deliberately keep a small safety margin (round UP) because the pool also
    stores page padding and the scale buffers are allocated per page.
    """
    import math

    try:
        from sglang.srt.model_executor import pool_configurator as pc
    except Exception as exc:  # pragma: no cover
        logger.warning("[oscar] cannot patch pool_configurator: %s", exc)
        return

    if getattr(pc, "_oscar_cell_size_patched", False):
        return

    orig = pc.DefaultPoolConfigurator._compute_cell_size

    def _compute_cell_size(self, kvc, num_layers: int) -> int:
        cell = orig(self, kvc, num_layers)
        if getattr(kvc, "kv_cache_dtype", None) != INT2:
            return cell
        group_size = int(os.environ.get("SGLANG_OSCAR_GROUP_SIZE", "256") or 256)
        # bytes/elem actually stored, relative to the 1.0 assumed upstream.
        ratio = 0.25 + (2 * 4.0 / max(group_size, 1))
        scaled = int(math.ceil(cell * ratio))
        logger.info(
            "[oscar] int2 cell size %d -> %d bytes/token "
            "(ratio %.4f, group_size %d)",
            cell,
            scaled,
            ratio,
            group_size,
        )
        return scaled

    pc.DefaultPoolConfigurator._compute_cell_size = _compute_cell_size
    pc._oscar_cell_size_patched = True
    logger.info("[oscar] patched DefaultPoolConfigurator._compute_cell_size")

    # ---- DSPARK/DFLASH draft term -------------------------------------
    # _compute_cell_size only covers the TARGET pool. For the dflash family
    # (which DSPARK belongs to) upstream then ADDS a flat per-token term for
    # the draft's own KV pool:
    #
    #   self._cell_size = scale_kv_cell_size_per_token_for_dflash(
    #       target_cell_size_per_token=<our scaled value>, ...,
    #       draft_cell_size_per_token=_dflash_draft_cell_size(kvc))
    #
    # That draft term comes from spec_aux_config.dflash_draft_cell_size_per_token,
    # which is computed from the draft's config assuming a DENSE dtype. Since we
    # now run the draft pool on int2 as well, it over-prices the draft by ~7x
    # (20480 B/token instead of 2880), and the profiler then hands back a
    # max_total_num_tokens roughly 2.5x smaller than what actually fits --
    # silently capping the servable context (e.g. 133k instead of >256k).
    #
    # Scale that term by the same int2 ratio.
    try:
        orig_draft = pc._dflash_draft_cell_size
    except Exception:  # pragma: no cover
        logger.warning("[oscar] no _dflash_draft_cell_size to patch")
        return

    def _dflash_draft_cell_size(kvc) -> int:
        cell = orig_draft(kvc)
        if not cell or getattr(kvc, "kv_cache_dtype", None) != INT2:
            return cell
        # The draft's head_dim (128) differs from the target's (256), so its
        # scale/zero overhead per element is larger for the same group size.
        # num_groups = ceil(head_dim / group_size), min 1.
        group_size = int(os.environ.get("SGLANG_OSCAR_GROUP_SIZE", "256") or 256)
        draft_head_dim = int(
            os.environ.get("SGLANG_OSCAR_DRAFT_HEAD_DIM", "128") or 128
        )
        n_groups = max(1, -(-draft_head_dim // max(group_size, 1)))
        ratio = 0.25 + (2 * 4.0 * n_groups / max(draft_head_dim, 1))
        scaled = int(math.ceil(cell * ratio))
        logger.info(
            "[oscar] int2 DRAFT cell size %d -> %d bytes/token (ratio %.4f)",
            cell, scaled, ratio,
        )
        return scaled

    pc._dflash_draft_cell_size = _dflash_draft_cell_size
    logger.info("[oscar] patched _dflash_draft_cell_size for int2")


def maybe_force_int2(server_args) -> None:
    """Env-var activation path (``SGLANG_OSCAR_FORCE_INT2=1``)."""
    if os.environ.get("SGLANG_OSCAR_FORCE_INT2", "0").lower() in (
        "1", "true", "yes", "on",
    ):
        if server_args.kv_cache_dtype != INT2:
            logger.info(
                "[oscar] SGLANG_OSCAR_FORCE_INT2=1: rewriting kv_cache_dtype "
                "%s -> int2",
                server_args.kv_cache_dtype,
            )
            server_args.kv_cache_dtype = INT2


def validate(server_args) -> None:
    """Fail fast on combinations known to be broken, at startup rather than
    mid-forward."""
    if getattr(server_args, "kv_cache_dtype", None) != INT2:
        return

    from .rotations import oscar_enabled

    if not oscar_enabled():
        raise ValueError(
            "--kv-cache-dtype int2 requires SGLANG_OSCAR_K_ROTATION_PATH and "
            "SGLANG_OSCAR_V_ROTATION_PATH to point at OSCAR rotation "
            "checkpoints. This port implements only the rotated OSCAR int2 "
            "path."
        )

    # int2 has no FA3/flashinfer reader; the read kernels ported here are the
    # Triton ones. Mirror the OSCAR fork's startup check.
    bad = None
    ab = getattr(server_args, "attention_backend", None)
    dab = getattr(server_args, "decode_attention_backend", None)
    if dab not in (None, "triton"):
        bad = f"--decode-attention-backend {dab}"
    elif dab is None and ab not in (None, "triton"):
        bad = f"--attention-backend {ab}"
    if bad:
        raise ValueError(
            f"--kv-cache-dtype int2 requires the Triton decode path, got {bad}. "
            "Use `--attention-backend triton` (or `--decode-attention-backend "
            "triton`)."
        )

    gsz = getattr(server_args, "kv_cache_quant_group_size", None)
    if gsz is not None and gsz <= 0:
        raise ValueError("--kv-cache-quant-group-size must be positive")
