"""Diagnostic: log a stack trace at every MHATokenToKVPool construction.

Used to find which component allocates a *duplicate* bf16 target-geometry KV
pool alongside the OSCAR int2 pool. Activate with SGLANG_TRACE_KVPOOL=1.
Purely diagnostic -- has no effect on behaviour.
"""

import os
import traceback


def _install():
    import sglang.srt.mem_cache.memory_pool as mp

    _orig = mp.MHATokenToKVPool.__init__

    def traced(self, *a, **k):
        dt = k.get("dtype")
        ln = k.get("layer_num")
        hn = k.get("head_num")
        hd = k.get("head_dim")
        print(
            f"[kvtrace] MHATokenToKVPool(dtype={dt}, layer_num={ln}, "
            f"head_num={hn}, head_dim={hd})",
            flush=True,
        )
        for line in traceback.format_stack()[-9:-1]:
            s = line.strip().splitlines()[0]
            if "sglang" in s or "oscar" in s:
                print("[kvtrace]     " + s, flush=True)
        return _orig(self, *a, **k)

    mp.MHATokenToKVPool.__init__ = traced
    print("[kvtrace] installed", flush=True)


if str(os.environ.get("SGLANG_TRACE_KVPOOL", "")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
):
    _install()
