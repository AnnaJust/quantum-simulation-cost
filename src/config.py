from pathlib import Path

CIRCUITS = ("bell", "ghz")
QUBITS = [2, 4, 8, 12, 16, 20, 22, 24, 26]
DEVICES = ("cpu", "gpu")
DTYPES = ("complex64", "complex128")

CONFIGS = {
    "qiskit_aer": [("cpu", "complex64"), ("cpu", "complex128"), ("gpu", "complex64"), ("gpu", "complex128")],
    "qrisp": [("cpu", "complex64")],
    "cuda_q": [("cpu", "complex128"), ("gpu", "complex64"), ("gpu", "complex128")],
}

REPETITIONS = 5
TIMEOUT_S = 1800
SHUFFLE_SEED = 42
WARMUP_QUBITS = 2

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FIDELITY_TOLERANCE = {
    "complex64": 1e-4,
    "complex128": 1e-8,
}
BYTES_PER_AMPLITUDE = {
    "complex64": 8,
    "complex128": 16,
}

LAYER_TIMEOUT_S = 3600
LAYER_QUBITS = [18, 22]
LAYER_DEPTHS = [0, 1, 4, 16, 64]
LAYER_SEED = 1234

MIN_MEMORY_QUBITS = 12
GROWTH_FIT_FROM_QUBITS = 20

ROOT = Path(__file__).resolve().parents[1]

PYTHONS = {
    "qiskit_aer": ROOT / ".venv-qiskit/bin/python",
    "qrisp": ROOT / ".venv-qrisp/bin/python",
    "cuda_q": ROOT / ".venv-cudaq/bin/python",
}
MODULES = {
    "qiskit_aer": "src.qiskit_aer.simulate",
    "qrisp": "src.qrisp.simulate",
    "cuda_q": "src.cuda_q.simulate",
}
PACKAGES = {
    "qiskit_aer": "qiskit-aer",
    "qrisp": "qrisp",
    "cuda_q": "cudaq",
}