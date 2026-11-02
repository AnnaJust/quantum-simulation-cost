import numpy as np
from qiskit_aer import AerSimulator

from .bell import create_bell
from .ghz import create_ghz


def simulate(circuit_name: str, nr_of_qubits: int) -> np.ndarray:

    if circuit_name == "bell":
        circuit = create_bell(nr_of_qubits)
    elif circuit_name == "ghz":
        circuit = create_ghz(nr_of_qubits)
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    circuit.save_statevector() # symulator musi wiedzieć, że chcemy zapisać wektor stanu po wykonaniu obwodu

    simulator = AerSimulator(
        method="statevector",
        device="CPU", # można też użyć GPU
        precision="double",
        max_parallel_threads=1
    )

    result = simulator.run(circuit, shots=1).result() # shots oznacza liczbę powtórzeń symulacji, w tym przypadku 1, bo chcemy tylko wektor stanu

    return np.asarray(result.get_statevector().data)
