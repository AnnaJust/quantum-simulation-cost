import cudaq
import numpy as np
from .layers import create_layers
from .bell import create_bell
from .ghz import create_ghz

CIRCUITS = {
    "bell": create_bell,
    "ghz": create_ghz,
    "layers": create_layers,
}

TARGETS = {
    ("cpu", "complex128"): ("qpp-cpu", {}),
    ("gpu", "complex64"): ("nvidia", {"option": "fp32"}),
    ("gpu", "complex128"): ("nvidia", {"option": "fp64"}),
}


def simulate(circuit: tuple, dtype: str, device: str) -> np.ndarray:

    kernel, *arguments = circuit

    target, options = TARGETS[(device, dtype)]
    cudaq.set_target(target, **options)
    state = cudaq.get_state(kernel, *arguments)

    return np.asarray(state)
