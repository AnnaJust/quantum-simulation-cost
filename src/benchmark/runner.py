import csv
import json
import os
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# frameworki i ich ścieżki do środowisk wirtualnych
PYTHONS = {
    "qiskit_aer": ROOT / ".venv-qiskit/bin/python",
    "qrisp": ROOT / ".venv-qrisp/bin/python",
    "cuda_q": ROOT / ".venv-cudaq/bin/python",
}

# nazwy kolumn w pliku wynikowym CSV
FIELDS = [
    "framework",
    "framework_version",
    "circuit",
    "qubits",
    "repetition",
    "time_s",
    "peak_rss_mib",
    "statevector_bytes",
    "amplitudes",
    "dtype",
    "theoretical_complex128_bytes",
    "status",
    "error",
]


def run_once(framework, circuit, nr_of_qubits): # wykonanie pojedynczego eksperymentu dla danego frameworka, obwodu i liczby kubitów
    
    command = [
        str(PYTHONS[framework]),
        "-m", "src.benchmark.worker",
        "--framework", framework,
        "--circuit", circuit,
        "--qubits", str(nr_of_qubits),
    ]

    env = os.environ.copy() # zmienne środowiskowe procesu pythona

    # ograniczenie liczby wątków, żeby uniknąć różnych poziomów równoległości w różnych frameworkach
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # uruchomienie procesu benchmarkowego i pobranie jego wyniku
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    # sprawdzenie, czy proces zakończył się sukcesem
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1500:] or f"Exit code: {completed.returncode}")

    return json.loads(completed.stdout.strip().splitlines()[-1])



def main():
    nr_of_qubits = [2, 4, 8, 12, 16, 20]
    repetitions = 5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "results" / f"benchmark_{timestamp}.csv"

    output.parent.mkdir(exist_ok=True)

    # przygotowanie listy wszystkich zadań do wykonania (framework, rodzaj obwodu, liczba kubitów, powtórzenie)
    jobs = []

    for n in nr_of_qubits:
        for circuit in ("bell", "ghz"):
            for framework in PYTHONS:
                for rep in range(1, repetitions + 1):
                    jobs.append((framework, circuit, n, rep))

    random.Random(42).shuffle(jobs) # losowe przemieszanie zadań, żeby uniknąć systematycznych różnic w czasie wykonania

    with output.open("w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for index, (framework, circuit, n, rep) in enumerate(jobs, 1):
            print(
                f"[{index}/{len(jobs)}] "
                f"{framework} {circuit} "
                f"n={n}, rep={rep}",
                flush=True,
            )

            try:
                result = run_once(framework, circuit, n)
                result["repetition"] = rep
                result["error"] = ""

            except Exception as exc:
                result = {
                    "framework": framework,
                    "circuit": circuit,
                    "qubits": n,
                    "repetition": rep,
                    "status": "failed",
                    "error": str(exc),
                }

            writer.writerow(result)
            file.flush()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
