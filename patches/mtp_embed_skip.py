"""Skip loading embed_tokens/lm_head into the Qwen3.5 MTP draft model.

The EAGLE worker replaces them with the target's tensors via
``set_embed_and_head`` right after load, so the draft's own copies are pure
waste. Two things must be prevented:

1. *Allocation*: ``nn.Embedding(248k, 5120)`` + ``ParallelLMHead`` empty
   buffers burn ~5.1 GB of VRAM at construction. We re-create those two
   parameters as empty meta tensors immediately after ``__init__``.
2. *Loading*: the checkpoint tensors are dropped in ``load_weights``.

``set_embed_and_head`` later does ``del ...weight`` then reassigns the
target's live tensors, so the meta placeholders are never read.
"""
from __future__ import annotations

import os

import torch
from torch import nn

_PATCH_FLAG = os.environ.get("SGLANG_SKIP_MTP_EMBED_LOAD", "1") == "1"


def _apply_init_patch(m) -> None:
    orig_init = m.Qwen3_5ForCausalLMMTP.__init__

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            self.model.embed_tokens.weight = nn.Parameter(
                torch.empty(0, device="meta"), requires_grad=False
            )
            if not getattr(self.config, "tie_word_embeddings", False):
                self.lm_head.weight = nn.Parameter(
                    torch.empty(0, device="meta"), requires_grad=False
                )
            print(
                "[mtp_embed_skip] draft embed/lm_head kept on meta device",
                flush=True,
            )
        except Exception as e:
            print(f"[mtp_embed_skip] init patch failed: {e}", flush=True)

    m.Qwen3_5ForCausalLMMTP.__init__ = new_init


def _apply_load_patch(m) -> None:
    orig = m.Qwen3_5ForCausalLMMTP.load_weights

    def load_weights(self, weights, is_mtp: bool = False):
        def filtered():
            skipped = 0
            for name, tensor in weights:
                short = name
                for prefix in ("model.language_model.", "model."):
                    if short.startswith(prefix):
                        short = short[len(prefix):]
                if short in ("embed_tokens.weight", "lm_head.weight"):
                    skipped += 1
                    continue
                yield name, tensor
            if skipped:
                print(
                    f"[mtp_embed_skip] skipped {skipped} embed/lm_head tensors "
                    f"on {type(self).__name__}",
                    flush=True,
                )

        return orig(self, filtered(), is_mtp=is_mtp)

    m.Qwen3_5ForCausalLMMTP.load_weights = load_weights


def apply() -> None:
    if not _PATCH_FLAG:
        return
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        print(f"[mtp_embed_skip] import failed: {e}", flush=True)
        return
    _apply_init_patch(m)
    _apply_load_patch(m)
    print("[mtp_embed_skip] MTP embed/lm_head load skipped", flush=True)


apply()
