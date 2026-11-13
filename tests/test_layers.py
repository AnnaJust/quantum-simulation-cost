import importlib
import os

import numpy as np
import pytest

from src.config import CONFIGS, FIDELITY_TOLERANCE, MODULES
from src.layers import REVERSED_QUBIT_ORDER, reference_state, reverse_qubits

QUBITS = [2, 4, 6, 8, 10]
DEPTHS = [0, 1, 2, 5]
SEED = 1234
IMPORT_NAMES = {"qiskit_aer": "qiskit_aer", "qrisp": "qrisp", "cuda_q": "cudaq"}
CASES = [(framework, device, dtype) for framework, configs in CONFIGS.items() for device, dtype in configs]


# Return |<a|b>|^2 for two statevectors.
def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


# Check that every package computes the same layered-circuit state as the NumPy reference.
@pytest.mark.parametrize("framework, device, dtype", CASES)
@pytest.mark.parametrize("nr_of_qubits", QUBITS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_layers_match_reference(framework: str, device: str, dtype: str, nr_of_qubits: int, depth: int) -> None:

    pytest.importorskip(IMPORT_NAMES[framework])
    if device == "gpu" and os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("GPU tests run only with RUN_GPU_TESTS=1")

    module = importlib.import_module(MODULES[framework])
    circuit = module.CIRCUITS["layers"](nr_of_qubits, depth, SEED)
    state = np.asarray(module.simulate(circuit, dtype, device)).reshape(-1)
    if framework in REVERSED_QUBIT_ORDER:
        state = reverse_qubits(state, nr_of_qubits)
    expected = reference_state(nr_of_qubits, depth, SEED)

    state_fidelity = fidelity(expected, state)
    reversed_fidelity = fidelity(expected, reverse_qubits(state, nr_of_qubits))

    assert state.size == 2**nr_of_qubits
    assert state_fidelity > 1 - FIDELITY_TOLERANCE[dtype], (
        f"fidelity {state_fidelity:.8f}; with reversed qubit order {reversed_fidelity:.8f}"
    )