import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .bell import create_bell
from .ghz import create_ghz
from .layers import create_layers

CIRCUITS = {
    "bell": create_bell,
    "ghz": create_ghz,
    "layers": create_layers,
}

DEVICES = {
    "cpu": "CPU",
    "gpu": "GPU",
}

PRECISIONS = {
    "complex64": "single",
    "complex128": "double",
}


def simulate(circuit: QuantumCircuit, dtype: str, device: str) -> np.ndarray:
    
    circuit.save_statevector()

    simulator = AerSimulator(
        method="statevector",
        device=DEVICES[device],
        precision=PRECISIONS[dtype],
        max_parallel_threads=1,
    )

    result = simulator.run(circuit, shots=1).result()
    return np.asarray(result.get_statevector().data)
