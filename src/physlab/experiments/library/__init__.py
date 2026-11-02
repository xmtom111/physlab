"""Built-in pendulum, incline, and thin-lens experiments."""

from physlab.experiments.library.incline import SIM_FRICTION, incline_protocol
from physlab.experiments.library.lens import TRUE_FOCAL_M, lens_protocol
from physlab.experiments.library.pendulum import SIM_GRAVITY, pendulum_protocol

__all__ = [
    "SIM_FRICTION",
    "SIM_GRAVITY",
    "TRUE_FOCAL_M",
    "incline_protocol",
    "lens_protocol",
    "pendulum_protocol",
]
