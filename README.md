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

## Limitations

- **CPU vs. GPU is not an equal comparison:** the CPU used a single core, the GPU was used in full.
- **Qrisp has it easier:** Bell and GHZ states are almost all zeros, and Qrisp stores only the non-zero amplitudes, while the other packages store all $2^n$.
- **These circuits could be simulated much faster:** they use only H and CNOT gates, which special (stabilizer) simulators handle quickly; the exponential cost comes from storing the full statevector.
- **RAM below ~20 qubits shows mostly the package itself:** importing a package takes 300–450 MiB, while the statevector is still tiny.
- **GPU memory was not measured:** the readings available under WSL2 were not reliable.

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

## Results

The full statevector has $2^n$ numbers, so every extra qubit doubles the memory needed. `_growth.csv` shows how many times the simulation time grows per added qubit (fit for $n \ge 20$); a value close to 2 means the time grows together with the statevector.

Main findings (run `results/benchmark_20261108_175941.csv`, AMD Ryzen 9 9950X3D, NVIDIA RTX 5090; all 720 runs passed the fidelity check):

- **Each qubit roughly doubles the time, but only for larger circuits:** from about 16 qubits on, the time on the CPU grows 2.1–2.2× per added qubit and memory grows with the statevector; below that, fixed overheads dominate.
- **The package matters a lot:** with 26 qubits on one CPU core, the same circuit takes 0.3–0.5 s in Qrisp, 4.5–6.2 s in Qiskit Aer and about 300 s in CUDA-Q (`qpp-cpu`), i.e. two orders of magnitude between dense simulators and almost three with Qrisp.
- **The GPU pays off only for larger circuits:** it is slower for small ones and faster from about 12 qubits (CUDA-Q) or 16 qubits (Qiskit Aer); at 26 qubits CUDA-Q on the GPU is 7–12× faster than Qiskit Aer on the CPU in double precision and 17–35× in single precision.
- **Lower precision does not always help:** Qiskit Aer computes in complex64 but returns complex128, so it gains no speed on the CPU.
