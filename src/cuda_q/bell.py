import cudaq

@cudaq.kernel
def bell_kernel(nr_of_qubits: int):

    qubits = cudaq.qvector(nr_of_qubits)

    for i in range(0, nr_of_qubits, 2):
        h(qubits[i])
        x.ctrl(qubits[i], qubits[i + 1])


def create_bell(nr_of_qubits: int) -> tuple:

    if nr_of_qubits < 2 or nr_of_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even and >= 2")

    return bell_kernel, nr_of_qubits