import cudaq

@cudaq.kernel
def ghz_kernel(nr_of_qubits: int):
    
    qubits = cudaq.qvector(nr_of_qubits)
    h(qubits[0])

    for i in range(nr_of_qubits - 1):
        x.ctrl(qubits[i], qubits[i + 1])


def create_ghz(nr_of_qubits: int) -> tuple:

    if nr_of_qubits < 2:
        raise ValueError("Number of qubits must be >= 2")

    return ghz_kernel, nr_of_qubits