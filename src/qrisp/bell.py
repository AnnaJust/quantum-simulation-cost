from qrisp import QuantumVariable, h, cx

def create_bell(nr_of_qubits: int) -> QuantumVariable:

    if nr_of_qubits < 2 or nr_of_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even and >= 2")

    qv = QuantumVariable(nr_of_qubits)

    for i in range(0, nr_of_qubits, 2):
        h(qv[i])
        cx(qv[i], qv[i + 1])

    return qv
