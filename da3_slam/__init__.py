import sys as _sys

# Python 3.11.0rc1 is missing sys.get_int_max_str_digits (added in 3.11.0 final).
# PyTorch ≥2.5 references it in torch._dynamo.polyfills.sys at import time.
if not hasattr(_sys, "get_int_max_str_digits"):
    _sys.get_int_max_str_digits = lambda: 4300
    _sys.set_int_max_str_digits = lambda n: None

from da3_slam.slam import DA3SLAM, SLAMConfig, SLAMResult
from da3_slam.config import load_slam_config, DEFAULT_YAML
