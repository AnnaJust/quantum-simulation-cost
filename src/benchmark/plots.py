import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BYTES_PER_AMPLITUDE, DEVICES, DTYPES, GROWTH_FIT_FROM_QUBITS, MIN_MEMORY_QUBITS, QUBITS

COLORS = {
    ("qiskit_aer", "complex128"): "#1F77B4", ("qiskit_aer", "complex64"): "#9467BD",
    ("cuda_q", "complex128"): "#2CA02C", ("cuda_q", "complex64"): "#D62728",
    ("qrisp", "complex64"): "#FF7F0E",
}
MARKERS = {"cuda_q": "s", "qrisp": "^", "qiskit_aer": "o"}
NAMES = {"cuda_q": "CUDA-Q", "qrisp": "Qrisp", "qiskit_aer": "Qiskit Aer"}
LINE_STYLES = {"complex64": "--", "complex128": "-"}
CONFIG_KEYS = ["framework", "device", "precision", "circuit"]
QUBITS_LABEL = "number of qubits"
FULL_WIDTH_IN = 7.2
HALF_WIDTH_IN = 3.8

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
})


# Combine the repetitions of each configuration: time median, min, max, std, IQR and median memory.
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.status == "success"]
    return ok.groupby(CONFIG_KEYS + ["qubits"]).agg(
        t_med=("simulate_s", "median"), t_min=("simulate_s", "min"), t_max=("simulate_s", "max"),
        t_std=("simulate_s", "std"), t_iqr=("simulate_s", lambda t: t.quantile(0.75) - t.quantile(0.25)),
        rss=("peak_rss_mib", "median"), rss_delta=("rss_delta_mib", "median"),
        sv_mib=("statevector_bytes", lambda b: b.median() / 2**20),
    ).reset_index()


# Draw one configuration: its own colour, a marker per framework, dashed line with hollow markers for complex64.
def plot_line(ax, g: pd.DataFrame, column: str, framework: str, precision: str, label: str) -> None:
    g = g.sort_values("qubits")
    color = COLORS[framework, precision]
    ax.plot(g.qubits, g[column], color=color, ls=LINE_STYLES[precision], marker=MARKERS[framework],
            mfc="white" if precision == "complex64" else color, label=label)


# Put a tick at every measured qubit count and use the same x range on every plot.
def qubit_axis(ax) -> None:
    ax.set(xticks=QUBITS, xlim=(min(QUBITS) - 1, max(QUBITS) + 1))


# Save the figure as PNG and free its memory.
def save(fig, out: Path) -> None:
    fig.savefig(out)
    plt.close(fig)


# Plot the median simulation time with a min-max band in a 2 x 2 grid (rows: circuits, columns: CPU/GPU).
def plot_time(s: pd.DataFrame, out: Path) -> None:
    circuits = sorted(s.circuit.unique())
    fig, axes = plt.subplots(len(circuits), len(DEVICES), figsize=(FULL_WIDTH_IN, 2.0 * len(circuits) + 0.4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for row, circuit in zip(axes, circuits):
        for ax, device in zip(row, DEVICES):
            panel = s[(s.circuit == circuit) & (s.device == device)]
            for (fw, prec), g in panel.groupby(["framework", "precision"]):
                plot_line(ax, g, "t_med", fw, prec, f"{NAMES[fw]} ({prec})")
                g = g.sort_values("qubits")
                ax.fill_between(g.qubits, g.t_min, g.t_max, color=COLORS[fw, prec], alpha=0.15, lw=0)
            ax.set(yscale="log", title=f"{circuit.upper()}, {device.upper()}")
            ax.yaxis.set_minor_locator(plt.NullLocator())
            qubit_axis(ax)
            ax.legend(loc="upper left")
        row[0].set_ylabel("simulation time [s]")
    for ax in axes[-1]:
        ax.set_xlabel(QUBITS_LABEL)
    save(fig, out)


# Plot the GPU speedup (CPU time / GPU time) for configurations run on both devices.
def plot_speedup(s: pd.DataFrame, out: Path) -> None:
    times = s.pivot_table(index=["framework", "precision", "circuit", "qubits"], columns="device", values="t_med")
    if not {"cpu", "gpu"} <= set(times.columns):
        return
    speedup = (times["cpu"] / times["gpu"]).dropna().rename("speedup").reset_index()

    circuits = sorted(speedup.circuit.unique())
    fig, axes = plt.subplots(1, len(circuits), figsize=(FULL_WIDTH_IN, 2.6), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, c in zip(axes[0], circuits):
        for (fw, prec), g in speedup[speedup.circuit == c].groupby(["framework", "precision"]):
            plot_line(ax, g, "speedup", fw, prec, f"{NAMES[fw]} ({prec})")
        ax.axhline(1, color="black", lw=0.6)
        ax.set(yscale="log", title=c.upper(), xlabel=QUBITS_LABEL)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        qubit_axis(ax)
    axes[0][0].set_ylabel("CPU time / GPU time")
    axes[0][-1].legend(loc="upper left")
    save(fig, out)


# Plot the RAM increase during the simulation against the theoretical statevector size (CPU only).
def plot_memory(cpu: pd.DataFrame, out: Path) -> None:
    cpu = cpu[cpu.qubits >= MIN_MEMORY_QUBITS]
    circuits = sorted(cpu.circuit.unique())
    ns = np.array(sorted(cpu.qubits.unique()))
    fig, axes = plt.subplots(1, len(circuits), figsize=(FULL_WIDTH_IN, 2.6), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, c in zip(axes[0], circuits):
        for (fw, prec), g in cpu[cpu.circuit == c].groupby(["framework", "precision"]):
            g = g[g.rss_delta > 0]
            plot_line(ax, g, "rss_delta", fw, prec, f"{NAMES[fw]} ({prec})")
        for prec, ls in (("complex64", ":"), ("complex128", "-.")):
            ax.plot(ns, BYTES_PER_AMPLITUDE[prec] * 2.0**ns / 2**20, color="black", ls=ls, lw=0.8,
                    label=f"statevector {prec}")
        ax.set_yscale("log", base=2)
        ax.set(title=c.upper(), xlabel=QUBITS_LABEL, xticks=ns)
    axes[0][0].set_ylabel("RAM increase [MiB]")
    axes[0][-1].legend(loc="upper left")
    save(fig, out)


# Plot the peak RAM of the whole process for one circuit, including the package import (CPU only).
def plot_rss(cpu: pd.DataFrame, circuit: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(HALF_WIDTH_IN, 2.5), layout="constrained")
    for (fw, prec), g in cpu[cpu.circuit == circuit].groupby(["framework", "precision"]):
        plot_line(ax, g, "rss", fw, prec, f"{NAMES[fw]} ({prec})")
    ax.set(ylim=(0, None), xlabel=QUBITS_LABEL, ylabel="peak RAM of the process [MiB]")
    qubit_axis(ax)
    ax.legend(loc="upper left")
    save(fig, out)


# Plot the size of the returned statevector against the theoretical 2^n size, one panel per circuit.
def plot_statevector(s: pd.DataFrame, out: Path) -> None:
    circuits = sorted(s.circuit.unique())
    ns = np.linspace(s.qubits.min(), s.qubits.max(), 200)
    fig, axes = plt.subplots(1, len(circuits), figsize=(FULL_WIDTH_IN, 2.6), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, c in zip(axes[0], circuits):
        sizes = s[s.circuit == c].groupby(["framework", "precision", "qubits"]).sv_mib.median().reset_index()
        for (fw, prec), g in sizes.groupby(["framework", "precision"]):
            plot_line(ax, g, "sv_mib", fw, prec, f"{NAMES[fw]} ({prec})")
        for prec, ls in (("complex64", ":"), ("complex128", "-.")):
            ax.plot(ns, BYTES_PER_AMPLITUDE[prec] * 2.0**ns / 2**20, color="black", ls=ls, lw=0.8,
                    label=f"theoretical {prec}")
        ax.set(title=c.upper(), xlabel=QUBITS_LABEL, ylim=(0, None))
        qubit_axis(ax)
    axes[0][0].set_ylabel("returned statevector [MiB]")
    axes[0][-1].legend(loc="upper left")
    save(fig, out)


# Fit log2(median time) against qubits for large n; factor_per_qubit close to 2 means time grows as 2^n.
def growth_rates(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in s[s.qubits >= GROWTH_FIT_FROM_QUBITS].groupby(CONFIG_KEYS):
        if len(g) >= 3:
            slope = np.polyfit(g.qubits, np.log2(g.t_med), 1)[0]
            rows.append({**dict(zip(CONFIG_KEYS, key)), "factor_per_qubit": 2**slope})
    return pd.DataFrame(rows)


# Report failed runs, then save the plots, the summary table and the growth-rate table.
def main(path: str | Path) -> None:
    path = Path(path)
    df = pd.read_csv(path)

    failed = df[df.status != "success"]
    if len(failed):
        print("Failed runs (see the 'error' column in the CSV):")
        print(failed.groupby(["framework", "device", "precision"]).size().to_string())

    s = summarize(df)
    if s.empty:
        print("No successful runs - nothing to plot")
        return

    # Build an output path next to the CSV file, e.g. benchmark_XXXX_time.png.
    def output(suffix: str) -> Path:
        return path.with_name(f"{path.stem}_{suffix}")

    plot_time(s, output("time.png"))
    plot_speedup(s, output("speedup.png"))
    plot_statevector(s, output("statevector.png"))
    cpu = s[s.device == "cpu"]
    plot_memory(cpu, output("memory.png"))
    for circuit in sorted(cpu.circuit.unique()):
        plot_rss(cpu, circuit, output(f"rss_{circuit}.png"))

    s.to_csv(output("summary.csv"), index=False)
    growth = growth_rates(s)
    growth.to_csv(output("growth.csv"), index=False)
    print(growth.to_string(index=False))
    print(f"Plots saved next to {path.name}")


if __name__ == "__main__":
    main(sys.argv[1])