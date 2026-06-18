"""Plot correctness scores across runs as a grouped point-range.

Reads the per-question correctness arrays that
compute_stats_basics saved ({metric}_{config_name}.txt), computes
mean +/- SEM, and draws one matplotlib axes: each model on the
x-axis, the four metrics (pass / naive / weighted / maj) dodged
within each model as color-coded points with SEM error bars.

Local alternative to a W&B custom panel -- no auth, no Vega, full
control over the figure. Edit RUNS / METRICS below.

    python plot_scores.py
"""

import numpy as np
import matplotlib.pyplot as plt

# (short label, result_dir, config_name). The .txt arrays live at
# {result_dir}/{metric}_{config_name}.txt.
RUNS = [
    (
        "Llama-1B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--level-4--Llama3.2-1B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--level-4--Llama3.2-1B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
    ),
    (
        "Llama-3B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--level-4--Llama3.2-3B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--level-4--Llama3.2-3B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
    ),
    (
        "Qwen-Math-1.5B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--level-4--Qwen2.5-Math-1.5B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--level-4--Qwen2.5-Math-1.5B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
    ),
]

# Metric key -> display label, in plot order.
METRICS = {
    "pass_gb": "pass@gb",
    "naive_gb": "naive@gb",
    "weighted_gb": "weighted@gb",
    "maj_gb": "maj@gb",
}

OUT_PATH = "results/prm800k/mcts_cnt--level-4/scores_pointrange.pdf"


def mean_sem(arr):
    """Mean and standard error of the mean (nan-aware)."""
    mean = np.nanmean(arr)
    sem = np.nanstd(arr, ddof=1) / np.sqrt(len(arr))
    return float(mean), float(sem)


def main():
    metric_keys = list(METRICS)
    n_metric = len(metric_keys)

    # means[metric][run], sems[metric][run]
    means = {m: [] for m in metric_keys}
    sems = {m: [] for m in metric_keys}
    for _, result_dir, config_name in RUNS:
        for m in metric_keys:
            arr = np.loadtxt(f"{result_dir}/{m}_{config_name}.txt")
            mean, sem = mean_sem(arr)
            means[m].append(mean)
            sems[m].append(sem)

    labels = [label for label, _, _ in RUNS]
    x = np.arange(len(RUNS))
    # Dodge the metrics within each model's x slot.
    width = 0.8
    offsets = np.linspace(
        -width / 2, width / 2, n_metric, endpoint=True,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, m in enumerate(metric_keys):
        ax.errorbar(
            x + offsets[j], means[m], yerr=sems[m],
            fmt="o", capsize=3, markersize=6,
            label=METRICS[m],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("correctness")
    ax.set_title("Correctness by run (mean ± SEM)")
    ax.legend(title="metric", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUT_PATH, dpi=300)
    print(f"saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
