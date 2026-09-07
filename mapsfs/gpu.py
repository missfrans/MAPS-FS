from __future__ import annotations

_CONFIGURED = False


def configure_tensorflow(cfg):
    global _CONFIGURED
    import tensorflow as tf
    if _CONFIGURED:
        return tf
    runtime = cfg.get("runtime", {})
    if bool(runtime.get("gpu_memory_growth", True)):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    if bool(runtime.get("mixed_precision", False)):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    _CONFIGURED = True
    return tf
