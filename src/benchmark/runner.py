import csv
import json
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

from src.benchmark.plots import main as make_plots
from src.config import (
    CIRCUITS, CONFIGS, PYTHONS, QUBITS, REPETITIONS, ROOT, SHUFFLE_SEED, SINGLE_THREAD_ENV, TIMEOUT_S,
)

FIELDS = [
    "framework", "framework_version", "circuit", "qubits", "device", "precision", "repetition",
    "build_s", "simulate_s", "peak_rss_mib", "rss_delta_mib",
    "statevector_bytes", "theoretical_bytes", "nonzero_amplitudes", "dtype", "fidelity",
    "status", "error",
]


def run_once(framework: str, circuit: str, nr_of_qubits: int, dtype: str, device: str) -> dict:

    command = [
        str(PYTHONS[framework]), "-m", "src.benchmark.worker",
        "--framework", framework,
        "--circuit", circuit,
        "--qubits", str(nr_of_qubits),
        "--dtype", dtype,
        "--device", device,
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env={**os.environ, **SINGLE_THREAD_ENV},
    )

    return json.loads(completed.stdout.strip().splitlines()[-1])


def build_jobs() -> list[tuple]:

    jobs = [
        (framework, circuit, n, device, dtype, rep)
        for n in QUBITS
        for circuit in CIRCUITS
        for framework in PYTHONS
        for device, dtype in CONFIGS[framework]
        for rep in range(1, REPETITIONS + 1)
    ]

    random.Random(SHUFFLE_SEED).shuffle(jobs)
    return jobs


def read_output(command: list[str]) -> str | None:

    try:
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_proc_value(path: str, key: str) -> str | None:
    
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    return next((line.split(":", 1)[1].strip() for line in lines if line.startswith(key)), None)


def write_metadata(path: Path) -> None:

    metadata = {
        "started": datetime.now().isoformat(),
        "git_commit": read_output(["git", "rev-parse", "HEAD"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": read_proc_value("/proc/cpuinfo", "model name"),
        "ram_total": read_proc_value("/proc/meminfo", "MemTotal"),
        "gpu": read_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "circuits": CIRCUITS,
        "qubits": QUBITS,
        "repetitions": REPETITIONS,
        "timeout_s": TIMEOUT_S,
        "cpu_threads": 1,
        "configs": CONFIGS,
    }

    path.write_text(json.dumps(metadata, indent=2))


def main() -> None:

    output = ROOT / "results" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output.parent.mkdir(exist_ok=True)
    write_metadata(output.with_name(output.stem + "_meta.json"))

    jobs = build_jobs()
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for index, (framework, circuit, n, device, dtype, rep) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {framework} {circuit} {device} {dtype} n={n}, rep={rep}", flush=True)
            try:
                result = run_once(framework, circuit, n, dtype, device)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                result = {
                    "framework": framework, "circuit": circuit, "qubits": n,
                    "device": device, "precision": dtype, "status": "failed", "error": str(exc),
                }
            writer.writerow({**result, "repetition": rep})
            file.flush()

    print(f"Saved: {output}")
    make_plots(output)


if __name__ == "__main__":
    main()
