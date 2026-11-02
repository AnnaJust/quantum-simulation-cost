import cudaq
import numpy as np

from .bell import create_bell
from .ghz import create_ghz


def simulate(circuit_name: str, nr_of_qubits: int) -> np.ndarray:

    if circuit_name == "bell":
        kernel, size = create_bell(nr_of_qubits)
    elif circuit_name == "ghz":
        kernel, size = create_ghz(nr_of_qubits)
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    cudaq.set_target("qpp-cpu") # ustawia silnik symulacji na CPU (inne np. nvidia dla gpu)
    state = cudaq.get_state(kernel, size) #zwraca amplitudy stanu kwantowego

    return np.asarray(state)
