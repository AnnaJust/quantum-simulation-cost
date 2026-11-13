import cudaq

from src.layers import layer_angles


# Kernel of the random layered circuit; the angles are passed as a flat list (layer, qubit, [ry, rz]).
# The CNOT pairs repeat cnot_pairs() from src/layers.py, because a kernel cannot call ordinary Python functions.
@cudaq.kernel
def layers_kernel(nr_of_qubits: int, depth: int, angles: list[float]):

    qubits = cudaq.qvector(nr_of_qubits)

    for layer in range(depth):
        for q in range(nr_of_qubits):
            ry(angles[2 * (layer * nr_of_qubits + q)], qubits[q])
            rz(angles[2 * (layer * nr_of_qubits + q) + 1], qubits[q])
        for q in range(layer % 2, nr_of_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])


# Return the kernel together with its arguments, in the same form as create_bell and create_ghz.
def create_layers(nr_of_qubits: int, depth: int, seed: int) -> tuple:

    angles = layer_angles(nr_of_qubits, depth, seed).reshape(-1).tolist()
    return layers_kernel, nr_of_qubits, depth, angles