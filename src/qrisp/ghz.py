from qrisp import QuantumVariable, h, cx

def create_ghz(nr_of_qubits: int) -> QuantumVariable:
    
    if nr_of_qubits < 2:
        raise ValueError("Number of qubits must be >= 2")

    qv = QuantumVariable(nr_of_qubits)
    h(qv[0])

    for i in range(nr_of_qubits - 1):
        cx(qv[i], qv[i + 1])

    return qv