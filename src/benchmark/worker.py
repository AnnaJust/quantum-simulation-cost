import argparse
import importlib
import json
import resource
import time
import numpy as np
from importlib.metadata import PackageNotFoundError, version

from src.config import (
    BYTES_PER_AMPLITUDE, CIRCUITS, DEVICES, DTYPES, FIDELITY_TOLERANCE, MODULES, PACKAGES, WARMUP_QUBITS,
)


def framework_version(framework: str) -> str:
    try:
        return version(PACKAGES[framework])
    except PackageNotFoundError:
        return version("qiskit-aer-gpu")


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def fidelity(circuit: str, state: np.ndarray) -> float:

    if circuit == "ghz":
        overlap = (state[0] + state[-1]) / np.sqrt(2)
    else:
        bell_pair = (np.array([1, 0, 0, 1]) / np.sqrt(2)).astype(state.dtype)
        overlap = state
        while overlap.size > 1:
            overlap = overlap.reshape(-1, 4) @ bell_pair
        overlap = overlap.item()

    return float(abs(overlap) ** 2)


def validate_state(circuit: str, state: np.ndarray, nr_of_qubits: int, dtype: str) -> float:

    if state.size != 2**nr_of_qubits:
        raise ValueError(f"Invalid statevector size: {state.size}")
    state_fidelity = fidelity(circuit, state)

    if state_fidelity < 1 - FIDELITY_TOLERANCE[dtype]:
        raise ValueError(f"Wrong statevector: fidelity with the expected state = {state_fidelity}")
    
    return state_fidelity


def run_worker(framework: str, circuit: str, nr_of_qubits: int, dtype: str, device: str) -> dict:

    module = importlib.import_module(MODULES[framework])
    create_circuit = module.CIRCUITS[circuit]
    module.simulate(create_circuit(WARMUP_QUBITS), dtype, device)

    start = time.perf_counter()
    quantum_circuit = create_circuit(nr_of_qubits)
    build_s = time.perf_counter() - start # build time includes circuit creation and compilation
    rss_before_mib = peak_rss_mib()

    start = time.perf_counter()
    state = module.simulate(quantum_circuit, dtype, device)
    simulate_s = time.perf_counter() - start # simulation time includes the time to return the statevector
    rss_after_mib = peak_rss_mib()

    state = np.asarray(state).reshape(-1)
    state_fidelity = validate_state(circuit, state, nr_of_qubits, dtype)

    return {
        "framework": framework,
        "framework_version": framework_version(framework),
        "circuit": circuit,
        "qubits": nr_of_qubits,
        "device": device,
        "precision": dtype,
        "build_s": build_s,
        "simulate_s": simulate_s,
        "peak_rss_mib": rss_after_mib,
        "rss_delta_mib": rss_after_mib - rss_before_mib,
        "statevector_bytes": state.nbytes,
        "theoretical_bytes": BYTES_PER_AMPLITUDE[dtype] * 2**nr_of_qubits,
        "nonzero_amplitudes": int(np.count_nonzero(np.abs(state) ** 2 > 0.5 / state.size)),
        "dtype": str(state.dtype),
        "fidelity": state_fidelity,
        "status": "success",
    }


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=MODULES)
    parser.add_argument("--circuit", required=True, choices=CIRCUITS)
    parser.add_argument("--qubits", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=DTYPES)
    parser.add_argument("--device", required=True, choices=DEVICES)
    args = parser.parse_args()

    print(json.dumps(run_worker(args.framework, args.circuit, args.qubits, args.dtype, args.device)))


if __name__ == "__main__":
    main()