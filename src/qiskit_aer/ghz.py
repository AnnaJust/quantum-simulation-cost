from qiskit import QuantumCircuit

def create_ghz(nr_of_qubits: int) -> QuantumCircuit:
    
    if nr_of_qubits < 2:
        raise ValueError("Number of qubits must be >= 2")

    circuit = QuantumCircuit(nr_of_qubits)
    circuit.h(0)

    for i in range(nr_of_qubits - 1):
        circuit.cx(i, i + 1)

    return circuit