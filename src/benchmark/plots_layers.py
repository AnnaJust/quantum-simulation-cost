import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.benchmark.plots import COLORS, FULL_WIDTH_IN, LINE_STYLES, MARKERS, NAMES, save
from src.config import DEVICES, LAYER_DEPTHS

CONFIG_KEYS = ["framework", "device", "precision", "qubits"]


# Median, min and max simulation time of every configuration and depth.
def summarize_layers(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.status == "success"]
    return ok.groupby(CONFIG_KEYS + ["depth"]).agg(
        t_med=("simulate_s", "median"), t_min=("simulate_s", "min"), t_max=("simulate_s", "max"),
    ).reset_index()


# Fixed cost a = t(0), cost of one layer b = (t(max) - t(0)) / max and d_star = a / b.
def fit_layers(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in s.groupby(CONFIG_KEYS):
        t = g.set_index("depth").t_med
        max_depth = t.index.max()
        a = t[0]
        b = (t[max_depth] - a) / max_depth
        rows.append({**dict(zip(CONFIG_KEYS, key)), "a_s": a, "b_s_per_layer": b, "d_star": a / b})
    return pd.DataFrame(rows)


# Draw one panel: medians with min-max bars and the line a + b * depth for each configuration.
def plot_panel(ax, panel: pd.DataFrame, fits: pd.DataFrame) -> None:
    depths = np.linspace(0, max(LAYER_DEPTHS), 400)
    for (fw, device, prec, n), g in panel.groupby(CONFIG_KEYS):
        a, b = fits.loc[(fw, device, prec, n), ["a_s", "b_s_per_layer"]]
        color, ls, marker = COLORS[fw, prec], LINE_STYLES[prec], MARKERS[fw]
        mfc = "white" if prec == "complex64" else color

        ax.plot(depths, a + b * depths, color=color, ls=ls, lw=0.9)
        ax.errorbar(g.depth, g.t_med, yerr=[g.t_med - g.t_min, g.t_max - g.t_med], color=color, ls="none",
                    marker=marker, mfc=mfc, elinewidth=0.6, capsize=1.5)
        ax.plot([], [], color=color, ls=ls, marker=marker, mfc=mfc, label=f"{NAMES[fw]} ({prec})")

    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set(yscale="log", xticks=LAYER_DEPTHS, xticklabels=[str(d) for d in LAYER_DEPTHS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper left")


# Time vs number of layers: rows are qubit counts, columns CPU and GPU.
def plot_layers_time(s: pd.DataFrame, fits: pd.DataFrame, out: Path) -> None:
    qubits = sorted(s.qubits.unique())
    fig, axes = plt.subplots(len(qubits), len(DEVICES), figsize=(FULL_WIDTH_IN, 2.2 * len(qubits) + 0.4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    fits = fits.set_index(CONFIG_KEYS)

    for row, n in zip(axes, qubits):
        for ax, device in zip(row, DEVICES):
            plot_panel(ax, s[(s.qubits == n) & (s.device == device)], fits)
            ax.set_title(f"{n} qubits, {device.upper()}")
        row[0].set_ylabel("simulation time [s]")
    for ax in axes[-1]:
        ax.set_xlabel("number of layers")

    # Extra space at the top so the legends do not cover the data.
    low, high = axes[0][0].get_ylim()
    axes[0][0].set_ylim(low, high * (high / low) ** 0.3)
    save(fig, out)


# Report failed runs and save the plot, the summary and the fit next to the CSV file.
def main(path: str | Path) -> None:
    path = Path(path)
    df = pd.read_csv(path)

    failed = df[df.status != "success"]
    if len(failed):
        print("Failed runs (see the 'error' column in the CSV):")
        print(failed.groupby(CONFIG_KEYS + ["depth"]).size().to_string())

    s = summarize_layers(df)
    if s.empty:
        print("No successful runs - nothing to plot")
        return

    fits = fit_layers(s)
    plot_layers_time(s, fits, path.with_name(f"{path.stem}_time.png"))
    s.to_csv(path.with_name(f"{path.stem}_summary.csv"), index=False)
    fits.to_csv(path.with_name(f"{path.stem}_fit.csv"), index=False)
    print(fits.to_string(index=False))
    print(f"Plots saved next to {path.name}")


if __name__ == "__main__":
    main(sys.argv[1])