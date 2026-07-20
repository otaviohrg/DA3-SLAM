"""
DA3-SLAM package.

Public entry points are re-exported lazily so that lightweight modules
(da3_slam.config, da3_slam.frontend.keyframe_selector) can be imported
without pulling in the GPU stack (torch, gtsam, Depth Anything 3).
"""

import importlib
import sys as _sys

# Python 3.11.0rc1 is missing sys.get_int_max_str_digits (added in 3.11.0 final).
# PyTorch ≥2.5 references it in torch._dynamo.polyfills.sys at import time.
if not hasattr(_sys, "get_int_max_str_digits"):
    _sys.get_int_max_str_digits = lambda: 4300
    _sys.set_int_max_str_digits = lambda n: None

# Public name → defining module.  Resolved on first attribute access (PEP 562)
# so that `from da3_slam import load_slam_config` does not import torch.
_LAZY_EXPORTS = {
    "DA3SLAM": "da3_slam.slam",
    "SLAMResult": "da3_slam.slam",
    "SLAMConfig": "da3_slam.config",
    "load_slam_config": "da3_slam.config",
    "DEFAULT_YAML": "da3_slam.config",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache so __getattr__ runs only once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
