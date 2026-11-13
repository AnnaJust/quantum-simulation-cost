#!/usr/bin/env bash
# Check how many CPU cores each CPU configuration really uses: CPU usage = (user + sys) / real time.
# About 100% means one thread; N x 100% means N threads working at the same time.
# Run from the project root: bash scripts/check_threads.sh

TIMEFORMAT="   real %R s, user %U s, sys %S s, CPU usage %P%%"
SINGLE_THREAD_VARS=(OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS)

# Run one long layered circuit on the CPU and print its wall-clock time, CPU time and CPU usage.
run_case() {
    local venv=$1 framework=$2 dtype=$3 qubits=$4 depth=$5
    echo "== $framework cpu $dtype, n=$qubits, depth=$depth"
    time (".venv-$venv/bin/python" -m src.benchmark.worker --framework "$framework" --device cpu --dtype "$dtype" \
        --circuit layers --qubits "$qubits" --depth "$depth" > /dev/null 2>&1)
}

echo "##### WITH the single-thread limit (as in the benchmark)"
for var in "${SINGLE_THREAD_VARS[@]}"; do
    export "$var=1"
done
run_case qiskit qiskit_aer complex64 22 256
run_case qiskit qiskit_aer complex128 22 256
run_case qrisp qrisp complex64 22 256
run_case cudaq cuda_q complex128 18 64

echo "##### WITHOUT the limit (control; Qiskit Aer is skipped because max_parallel_threads=1 is set in its code)"
unset "${SINGLE_THREAD_VARS[@]}"
run_case qrisp qrisp complex64 22 256
run_case cudaq cuda_q complex128 18 16