# Classical Simulation Cost of Basic Quantum Information Primitives

This project investigates how the computational cost of classical quantum-state simulation changes with the number of qubits.

The experiments compare three Python-based simulation frameworks:

- **Qiskit Aer** — statevector simulator on CPU and GPU (`qiskit-aer-gpu`)
- **Qrisp** — built-in statevector simulation (CPU only)
- **NVIDIA CUDA-Q** — `qpp-cpu` backend on CPU and `nvidia` (cuStateVec) backend on GPU

## Experiments

Two types of quantum circuits are simulated:

- **Bell pairs:** $n/2$ independent Bell pairs
- **GHZ state:** one $n$-qubit GHZ state

The experiments use 2, 4, 8, 12, 16, 20, 22, 24 and 26 qubits (`QUBITS` in `src/config.py`).

Each framework is run in every device and precision configuration it supports (`src/config.py`):

| Framework  | CPU complex64 | CPU complex128 | GPU complex64 | GPU complex128 |
|------------|:-------------:|:--------------:|:-------------:|:--------------:|
| Qiskit Aer | ✓             | ✓              | ✓             | ✓              |
| Qrisp      | ✓             | –              | –             | –              |
| CUDA-Q     | –             | ✓              | ✓             | ✓              |

Each configuration is executed five times (720 runs in total), each time in a separate single-threaded Python process. Jobs run in a fixed random order.

## Measurements

Each worker process first runs an untimed 2-qubit warm-up (CUDA context, JIT compilation), then records:

- **`build_s`** — circuit construction time
- **`simulate_s`** — simulation and statevector retrieval time (main metric)
- **`peak_rss_mib`** — peak RAM of the whole process (including the package import)
- **`rss_delta_mib`** — increase of the process peak RAM during the simulation
- **`statevector_bytes`** — memory occupied by the returned statevector array
- **`theoretical_bytes`** — theoretical statevector size, $2^n$ × 8 B (complex64) or 16 B (complex128)
- **`fidelity`** — fidelity with the expected Bell/GHZ state; a run fails if it is below $1 - 10^{-4}$ (complex64) or $1 - 10^{-8}$ (complex128)
- **`nonzero_amplitudes`** — number of non-zero amplitudes, showing how sparse the state is

Hardware, software and experiment settings are saved to `results/benchmark_*_meta.json`.

## Running the benchmark

The project uses three separate Python environments: `.venv-qiskit` (Qiskit Aer, pandas, matplotlib), `.venv-qrisp` and `.venv-cudaq`. Exact package versions are recorded in `results/*-lock.txt`; after changing an environment, refresh its lock file, for example:

```bash
uv pip freeze --python .venv-qiskit/bin/python > results/qiskit-lock.txt
```

Run the benchmark from the project root:

```bash
.venv-qiskit/bin/python -m src.benchmark.runner
```

Results are saved to `results/benchmark_*.csv`, together with plots (`_time` as a circuit × device grid, `_speedup`, `_memory`, `_rss_bell`, `_rss_ghz`, `_statevector`), medians with the spread of the times (`_summary.csv`) and growth rates (`_growth.csv`). Plots for an existing CSV file can be generated with:

```bash
.venv-qiskit/bin/python -m src.benchmark.plots results/benchmark_XXXX.csv
```