from qrisp import QuantumVariable, cx, ry, rz

from src.layers import cnot_pairs, layer_angles


# Build the random layered circuit: ry and rz on every qubit, then CNOTs on alternating neighbouring pairs.
def create_layers(nr_of_qubits: int, depth: int, seed: int) -> QuantumVariable:

    angles = layer_angles(nr_of_qubits, depth, seed)
    qv = QuantumVariable(nr_of_qubits)

    for layer in range(depth):
        for q in range(nr_of_qubits):
            ry(float(angles[layer, q, 0]), qv[q])
            rz(float(angles[layer, q, 1]), qv[q])
        for control, target in cnot_pairs(nr_of_qubits, layer):
            cx(qv[control], qv[target])

    return qv