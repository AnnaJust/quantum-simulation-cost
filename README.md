# Classical Simulation Cost of Basic Quantum Information Primitives

This project investigates how the computational cost of classical quantum-state simulation changes with the number of qubits.

The experiments compare three Python-based simulation frameworks:

- **Qiskit Aer** — CPU statevector simulator
- **Qrisp** — built-in statevector simulation
- **NVIDIA CUDA-Q** — `qpp-cpu` backend

## Experiments

Two types of quantum circuits are simulated:

- **Bell pairs:** \(n/2\) independent Bell pairs
- **GHZ state:** one \(n\)-qubit GHZ state

The experiments use 2, 4, 8, 12, 16, and 20 qubits.

Each configuration is executed five times in separate Python processes.

## Measurements

The benchmark records:

- **Execution time:** circuit construction, simulation, and statevector retrieval
- **Peak RAM:** maximum resident memory usage of the worker process
- **Statevector size:** memory occupied by the returned statevector array

All simulations use CPU backends. The frameworks use their respective statevector implementations and may differ in numerical precision.

## Running the benchmark

The project uses three separate Python environments with Qiskit Aer, Qrisp, and CUDA-Q installed.

From the project root, run:

```bash
.venv-qiskit/bin/python -m src.benchmark.runner
```

Results are saved to `results/benchmark_*.csv`.

## Results

The full statevector requires \(2^n\) complex amplitudes, resulting in exponential memory growth. Execution time and peak process memory also depend on the framework implementation, numerical precision, and runtime overhead.