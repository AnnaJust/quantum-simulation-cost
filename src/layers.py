import numpy as np

# Frameworks that return the statevector with the reverse qubit order (qubit 0 as the most significant bit).
REVERSED_QUBIT_ORDER = {"qrisp"}


# Random ry and rz angles of every qubit in every layer, shape (depth, nr_of_qubits, 2).
def layer_angles(nr_of_qubits: int, depth: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=(depth, nr_of_qubits, 2))


# CNOT pairs of one layer: (0,1), (2,3), ... in even layers and (1,2), (3,4), ... in odd layers.
def cnot_pairs(nr_of_qubits: int, layer: int) -> list[tuple[int, int]]:
    return [(q, q + 1) for q in range(layer % 2, nr_of_qubits - 1, 2)]


# ry rotation matrix.
def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


# rz rotation matrix.
def rz_matrix(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


# Reverse the qubit order of a statevector (qubit 0 <-> qubit n-1).
def reverse_qubits(state: np.ndarray, nr_of_qubits: int) -> np.ndarray:
    return state.reshape([2] * nr_of_qubits).transpose(range(nr_of_qubits - 1, -1, -1)).reshape(-1)


# Apply a 2x2 gate to one qubit; the state is a tensor with one axis per qubit.
def apply_gate(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    return np.moveaxis(state, 0, qubit)


# Apply CNOT: swap the amplitudes with control = 1 and target = 0 or 1.
def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    target_0 = [slice(None)] * state.ndim
    target_0[control], target_0[target] = 1, 0
    target_1 = list(target_0)
    target_1[target] = 1

    state = state.copy()
    state[tuple(target_0)], state[tuple(target_1)] = state[tuple(target_1)].copy(), state[tuple(target_0)].copy()
    return state


# Exact output state of the layered circuit computed with plain NumPy (reference for small circuits only),
# returned in Qiskit's qubit order.
def reference_state(nr_of_qubits: int, depth: int, seed: int) -> np.ndarray:
    state = np.zeros([2] * nr_of_qubits, dtype=complex)
    state[(0,) * nr_of_qubits] = 1
    angles = layer_angles(nr_of_qubits, depth, seed)

    for layer in range(depth):
        for q in range(nr_of_qubits):
            ry_angle, rz_angle = angles[layer, q]
            state = apply_gate(state, rz_matrix(rz_angle) @ ry_matrix(ry_angle), q)
        for control, target in cnot_pairs(nr_of_qubits, layer):
            state = apply_cnot(state, control, target)

    return reverse_qubits(state.reshape(-1), nr_of_qubits)