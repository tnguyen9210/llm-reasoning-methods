"""Compute performance stats for a scored search run.

CPU-only post-processing: grades the scored per-question dataset
(pass@gb / naive / weighted / maj correctness + search-cost stats)
and prints a mean ± SEM summary, via utils.metrics.compute_stats_basics.

Takes the SAME Hydra config that produced the run, so config_name and
result_dir resolve to the same paths the scored files live in — no
hardcoded run name. The search method is selected by --config-name.
Examples:

    # mcts_cnt, Llama-3B native, 1 trial
    python compute_stats.py --config-name mcts_cnt_prm800k \\
        llm=llama_3b gen.use_custom_template=false run.num_trials=1

    # bon
    python compute_stats.py --config-name bon_prm800k run.num_trials=2

No PRM / GPU needed (scoring already happened in
prepare_scored_dataset).
"""

import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)

import hydra
from hydra.core.config_store import ConfigStore
import wandb

from utils import metrics
from utils.configs import (
    ExpConfig, BoNConfig, MCTSCntConfig, config_name, level_dir,
    load_wandb_run_id,
)

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(group="search", name="bon_schema", node=BoNConfig)
cs.store(group="search", name="mcts_cnt_schema", node=MCTSCntConfig)


@hydra.main(
    config_path="conf",
    config_name="mcts_cnt_prm800k",
    version_base=None,
)
def main(cfg: ExpConfig):
    root_dir = hydra.utils.get_original_cwd()

    run_name = config_name(cfg)
    result_dir = (
        f"{root_dir}/results/{cfg.data.name}"
        f"/{level_dir(cfg)}/{run_name}"
    )
    print(f"config_name = {run_name}")

    # Summary columns:
    #   pass@gb, naive@gb, weighted@gb, maj@gb,
    #   ncomps, depth, nphases, ndepths   (each: mean ± SEM)
    summary = metrics.compute_stats_basics(
        result_dir, run_name, cfg.run.num_trials,
    )

    # Reattach to the generation run (id saved at generation time) and
    # log the metrics onto it, so scores + stats live in one place. If
    # no id sidecar exists (run predates this), skip W&B silently.
    run_id = load_wandb_run_id(result_dir)
    if run_id is None:
        print("no wandb_run_id.txt found; skipping W&B logging")
        return

    wandb.init(
        project="llm-reasoning", id=run_id, resume="must",
    )
    log_data = {}
    for metric, (mean, sem) in summary.items():
        log_data[f"eval/{metric}"] = mean
        log_data[f"eval/{metric}_sem"] = sem
    wandb.log(log_data)
    # Also write into summary explicitly. log() lands in history and
    # only propagates to summary on a clean live finish; on a reattach
    # (resume="must") that propagation is unreliable, and custom-panel
    # queries read summary -- so force it here.
    wandb.run.summary.update(log_data)
    wandb.finish()
    print(f"logged eval metrics to W&B run {run_id}")


if __name__ == "__main__":
    main()
