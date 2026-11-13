# Classical Simulation Cost of Basic Quantum Information Primitives

This project investigates how the computational cost of classical quantum-state simulation changes with the number of qubits and with the number of gates.

The experiments compare three Python-based simulation frameworks:

- **Qiskit Aer** — statevector simulator on CPU and GPU (`qiskit-aer-gpu`)
- **Qrisp** — built-in statevector simulation (CPU only)
- **NVIDIA CUDA-Q** — `qpp-cpu` backend on CPU and `nvidia` (cuStateVec) backend on GPU

All three are used as ideal (noiseless) statevector emulators with all-to-all qubit connectivity, so circuits are simulated as written, without transpilation to a hardware topology.

## Experiments

Each framework is run in every device and precision configuration it supports (`CONFIGS` in `src/config.py`):

| Framework  | CPU complex64 | CPU complex128 | GPU complex64 | GPU complex128 |
|------------|:-------------:|:--------------:|:-------------:|:--------------:|
| Qiskit Aer | ✓             | ✓              | ✓             | ✓              |
| Qrisp      | ✓             | –              | –             | –              |
| CUDA-Q     | –             | ✓              | ✓             | ✓              |

Every run is executed in a separate single-threaded Python process, five times per configuration, and jobs run in a fixed random order.

### Experiment 1: Bell pairs and GHZ state

- **Bell pairs:** $n/2$ independent Bell pairs
- **GHZ state:** one $n$-qubit GHZ state

The experiment uses 2, 4, 8, 12, 16, 20, 22, 24 and 26 qubits (`QUBITS` in `src/config.py`), 720 runs in total. These circuits contain very few gates, so the measured time is dominated by allocating, preparing and returning the statevector.

### Experiment 2: random layered circuit

Each layer applies `ry` and `rz` rotations with random angles to every qubit, followed by CNOTs on neighbouring pairs: (0,1), (2,3), ... in even layers and (1,2), (3,4), ... in odd layers (`src/layers.py`). The angles are drawn from a fixed seed, so every framework simulates exactly the same circuit, and the output is a dense state with no simple structure.

The experiment uses 18 and 22 qubits and 0, 1, 4, 16 and 64 layers (`LAYER_QUBITS`, `LAYER_DEPTHS`, `LAYER_SEED` in `src/config.py`), 400 runs in total. For every configuration the median time is fitted with

$$t(d) = a + b \cdot d,$$

where $d$ is the number of layers, $a$ the fixed cost (setup, allocation, returning the statevector), $b$ the cost of one layer, and $d^* = a/b$ the depth at which both costs are equal.

In the stored run (`results/layers/benchmark_layers_20261112_232429.csv`), 20 runs (Qrisp and CUDA-Q GPU, complex64, 22 qubits, 1 and 4 layers) were first rejected by a too strict norm check computed in single precision. After the check was fixed, only these runs were repeated on the same machine; the original file is kept as `results/layers/benchmark_layers_20261112_232429_before_rerun.csv`.

## Measurements

Each worker process first runs an untimed 2-qubit warm-up of the same circuit (CUDA context, JIT compilation), then records:

- **`build_s`** — circuit construction time
- **`simulate_s`** — simulation and statevector retrieval time (main metric)
- **`peak_rss_mib`** — peak RAM of the whole process (including the package import)
- **`rss_delta_mib`** — increase of the process peak RAM during the simulation
- **`statevector_bytes`** — memory occupied by the returned statevector array
- **`theoretical_bytes`** — theoretical statevector size, $2^n$ × 8 B (complex64) or 16 B (complex128)
- **`fidelity`** — fidelity with the expected Bell/GHZ state; a run fails if it is below $1 - 10^{-4}$ (complex64) or $1 - 10^{-8}$ (complex128). For the layered circuit this column is empty: the run fails if the squared norm of the state differs from 1 by more than the same tolerance, and the correctness of the circuit itself is checked by the tests below
- **`nonzero_amplitudes`** — number of non-zero amplitudes, showing how sparse the state is
- **`depth`**, **`seed`** — number of layers and random seed (layered circuit only)

Hardware, software and experiment settings are saved next to every CSV file as `*_meta.json`.

## Results folder

```
results/
├── bell_ghz/      Experiment 1: CSV, metadata, plots, summary and growth rates
├── layers/        Experiment 2: CSV (and the file before the rerun), metadata, plot, summary, fit, run log
├── diagnostics/   single-thread check (check_threads.log)
└── environment/   exact package versions of the three environments (*-lock.txt)
```

## Running the benchmark

The project uses three separate Python environments: `.venv-qiskit` (Qiskit Aer, pandas, matplotlib), `.venv-qrisp` and `.venv-cudaq`. Exact package versions are recorded in `results/environment/*-lock.txt`; after changing an environment, refresh its lock file, for example:

```bash
uv pip freeze --python .venv-qiskit/bin/python > results/environment/qiskit-lock.txt
```

Run the benchmark from the project root. Experiment 1 (Bell and GHZ):

```bash
.venv-qiskit/bin/python -m src.benchmark.runner
```

Results are saved to `results/bell_ghz/benchmark_*.csv`, together with plots (`_time` as a circuit × device grid, `_speedup`, `_memory`, `_rss_bell`, `_rss_ghz`, `_statevector`), medians with the spread of the times (`_summary.csv`) and growth rates (`_growth.csv`). Plots for an existing CSV file can be generated with:

```bash
.venv-qiskit/bin/python -m src.benchmark.plots results/bell_ghz/benchmark_XXXX.csv
```

Experiment 2 (random layered circuit):

```bash
.venv-qiskit/bin/python -m src.benchmark.runner --experiment layers
.venv-qiskit/bin/python -m src.benchmark.plots_layers results/layers/benchmark_layers_XXXX.csv
```

Results are saved to `results/layers/benchmark_layers_*.csv`; the second command creates the time-vs-layers plot (`_time.png`), the medians (`_summary.csv`) and the fitted $a$, $b$ and $d^*$ (`_fit.csv`). A single layered run at 22 qubits and 64 layers takes about 36 minutes in CUDA-Q on one CPU thread, so the whole experiment takes about 5 hours.

## Tests

`tests/test_layers.py` compares the layered circuit of every configuration with an exact NumPy reference (`reference_state` in `src/layers.py`) for 2–10 qubits and 0–5 layers. Qrisp returns the statevector with the reverse qubit order, which is taken into account when comparing (`REVERSED_QUBIT_ORDER`). Run the tests in each environment (`pytest` is needed in it); GPU tests run only with `RUN_GPU_TESTS=1`:

```bash
.venv-qiskit/bin/python -m pytest tests/test_layers.py -q
.venv-qrisp/bin/python -m pytest tests/test_layers.py -q
.venv-cudaq/bin/python -m pytest tests/test_layers.py -q
RUN_GPU_TESTS=1 .venv-cudaq/bin/python -m pytest tests/test_layers.py -q
```

## Single-thread check

`scripts/check_threads.sh` verifies that CPU runs really use one thread by comparing CPU time with wall-clock time on long layered runs (about 100% means one core). It also repeats Qrisp and CUDA-Q without the limit as a control:

```bash
bash scripts/check_threads.sh 2>&1 | tee results/diagnostics/check_threads.log
```

In the stored check, every CPU configuration used 99–100% of one core. Without the limit Qrisp and CUDA-Q used about 27–28 cores; CUDA-Q was then about 12× faster, Qrisp was not faster.