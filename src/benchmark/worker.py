import argparse
import importlib
import json
import resource
import time
from importlib.metadata import version

import numpy as np

# frameworki i ich moduły symulacyjne
MODULES = {
    "qiskit_aer": "src.qiskit_aer.simulate",
    "qrisp": "src.qrisp.simulate",
    "cuda_q": "src.cuda_q.simulate",
}

# pakiety odpowiadające frameworkom, potrzebne do pobrania wersji
PACKAGES = {
    "qiskit_aer": "qiskit-aer",
    "qrisp": "qrisp",
    "cuda_q": "cudaq",
}


def run_worker(framework: str, circuit: str, nr_of_qubits: int):

    module = importlib.import_module(MODULES[framework])

    start = time.perf_counter() # początek pomiaru
    state = module.simulate(circuit, nr_of_qubits)
    elapsed = time.perf_counter() - start # czas wykonania symulacji (budowanie obwodu + symulacja + pobranie statevectora)

    state = np.asarray(state).reshape(-1)

    if state.size != 2**nr_of_qubits: # sprawdzenie liczby amplitud w wektorze stanu (powinna być równa 2^n)
        raise ValueError("Invalid statevector size") 

    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-5): # sprawdzenie normalizacji wektora stanu (powinna być równa 1)
        raise ValueError("Statevector is not normalized")

    # statystyki pwykorzystania pamięci RAM (peak_rss) w KiB
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "framework": framework,
        "framework_version": version(PACKAGES[framework]),
        "circuit": circuit,
        "qubits": nr_of_qubits,
        "time_s": elapsed,
        "peak_rss_mib": peak_rss_kib / 1024, # konwersja z KiB na MiB
        "statevector_bytes": int(state.nbytes),
        "amplitudes": int(state.size),
        "dtype": str(state.dtype),
        "theoretical_complex128_bytes": 16 * 2**nr_of_qubits,
        "status": "success",
    }


def main():

    # parser argumentów terminala
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=MODULES)
    parser.add_argument("--circuit", required=True,choices=["bell", "ghz"])
    parser.add_argument("--qubits", required=True, type=int)

    args = parser.parse_args()

    result = run_worker(
        args.framework,
        args.circuit,
        args.qubits
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
