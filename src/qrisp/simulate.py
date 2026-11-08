import numpy as np
from qrisp import QuantumVariable
from .bell import create_bell
from .ghz import create_ghz

CIRCUITS = {
    "bell": create_bell,
    "ghz": create_ghz,
}

# The Qrisp simulator is always CPU and complex64, so dtype and device are accepted only to keep the interface identical to the other frameworks
def simulate(qv: QuantumVariable, dtype: str, device: str) -> np.ndarray:
    return np.asarray(qv.qs.statevector(return_type="array")).flatten()