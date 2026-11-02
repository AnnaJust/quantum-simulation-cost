import numpy as np

from .bell import create_bell
from .ghz import create_ghz


def simulate(circuit_name: str, nr_of_qubits: int) -> np.ndarray:

    if circuit_name == "bell":
        qv = create_bell(nr_of_qubits)
    elif circuit_name == "ghz":
        qv = create_ghz(nr_of_qubits)
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    return np.asarray(qv.qs.statevector(return_type="array")).flatten() # wbudowana ścieżka symulacji, nie jest identycznie skonfigurowanym silnikiem
