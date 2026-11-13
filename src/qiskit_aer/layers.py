from qiskit import QuantumCircuit

from src.layers import cnot_pairs, layer_angles


# Build the random layered circuit: ry and rz on every qubit, then CNOTs on alternating neighbouring pairs.
def create_layers(nr_of_qubits: int, depth: int, seed: int) -> QuantumCircuit:

    angles = layer_angles(nr_of_qubits, depth, seed)
    circuit = QuantumCircuit(nr_of_qubits)

    for layer in range(depth):
        for q in range(nr_of_qubits):
            circuit.ry(float(angles[layer, q, 0]), q)
            circuit.rz(float(angles[layer, q, 1]), q)
        for control, target in cnot_pairs(nr_of_qubits, layer):
            circuit.cx(control, target)

    return circuit