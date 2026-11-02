from qiskit import QuantumCircuit

def create_bell(nr_of_qubits: int) -> QuantumCircuit:

    if nr_of_qubits < 2 or nr_of_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even and >= 2")

    circuit = QuantumCircuit(nr_of_qubits)

    for i in range(0, nr_of_qubits, 2):
        circuit.h(i)
        circuit.cx(i, i + 1)

    return circuit