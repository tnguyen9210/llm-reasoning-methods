"""Log a cross-run comparison table for a Vega point-range panel.

Each generation run logs its own scalar eval metrics, but a custom
Vega panel reads a single wandb.Table. So this collects mean +/- sem
for every (run, metric) into ONE table and logs it to a dedicated
"summary" run, which the panel then plots as a point-range
(point = mean, whisker = mean +/- sem).

Reads the per-question correctness arrays that compute_stats_basics
saved ({metric}_{config_name}.txt) and recomputes mean +/- sem, so no
PRM / GPU and no re-scoring needed. Edit RUNS below to point at the
result dirs + config_names you want to compare.

    python plot_comparison_wandb.py
"""

import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)

import numpy as np
import wandb

# (display label, result_dir, config_name). The .txt arrays live at
# {result_dir}/{metric}_{config_name}.txt.
RUNS = [
    (
        "Llama3.2-1B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--Llama3.2-1B--tmpl-native--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--Llama3.2-1B--tmpl-native--bs-4--d-20--b-080--cpuct-2.0",
    ),
    (
        "Llama3.2-3B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--Llama3.2-3B--tmpl-native--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--Llama3.2-3B--tmpl-native--bs-4--d-20--b-080--cpuct-2.0",
    ),
    (
        "Qwen2.5-Math-1.5B",
        "results/prm800k/mcts_cnt--level-4"
        "/mcts_cnt--Qwen2.5-Math-1.5B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
        "mcts_cnt--Qwen2.5-Math-1.5B--tmpl-native"
        "--bs-4--d-20--b-080--cpuct-2.0",
    ),
]

# Correctness metrics to put on the comparison plot.
METRICS = ["pass_gb", "naive_gb", "weighted_gb", "maj_gb"]


def mean_sem(arr):
    """Mean and standard error of the mean (nan-aware)."""
    mean = np.nanmean(arr)
    sem = np.nanstd(arr, ddof=1) / np.sqrt(len(arr))
    return float(mean), float(sem)


def main():
    rows = []
    for label, result_dir, config_name in RUNS:
        for metric in METRICS:
            path = f"{result_dir}/{metric}_{config_name}.txt"
            arr = np.loadtxt(path)
            mean, sem = mean_sem(arr)
            rows.append([label, metric, mean, sem,
                         mean - sem, mean + sem])

    table = wandb.Table(
        columns=["run", "metric", "mean", "sem", "lower", "upper"],
        data=rows,
    )

    wandb.init(project="llm-reasoning", name="comparison-summary")
    wandb.log({"comparison": table})
    wandb.finish()
    print(f"logged {len(rows)} rows to run 'comparison-summary'")


if __name__ == "__main__":
    main()
