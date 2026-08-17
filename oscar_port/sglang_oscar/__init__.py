"""OSCAR INT2 KV-cache port for upstream SGLang (path B: plain int2, no mixed-KV).

Import ``apply_patch`` (one directory up) to install; nothing here patches on
import, so this package is safe to introspect.
"""

__all__ = [
    "attention_patch",
    "hybrid_patch",
    "kernels",
    "kv_pool_patch",
    "qwen35_absorb",
    "rotations",
    "server_args_patch",
]
