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
        llm=llama_3b llm.use_custom_template=false run.num_trials=1

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
    ExpConfig, BoNConfig, MCTSCntConfig, BLMCTSCntConfig,
    BLMCTSCntV02Config, BLMCTSKubeV01Config, BLMCTSKubeV02Config,
    BLMCTSKdepthV01Config, BLMCTSKdepthV02Config, BLMCTSSemConfig,
    BLMCTSSemV02Config, MCTSSemV01Config, MCTSSemV02Config,
    resolve_result_dir, load_wandb_run_id,
)

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(group="search", name="bon_schema", node=BoNConfig)
cs.store(group="search", name="mcts_cnt_schema", node=MCTSCntConfig)
cs.store(
    group="search", name="mcts_bl_cnt_v01_schema", node=BLMCTSCntConfig,
)
cs.store(
    group="search", name="mcts_bl_cnt_v02_schema", node=BLMCTSCntV02Config,
)
cs.store(
    group="search", name="mcts_bl_kube_v01_schema", node=BLMCTSKubeV01Config,
)
cs.store(
    group="search", name="mcts_bl_kube_v02_schema", node=BLMCTSKubeV02Config,
)
cs.store(
    group="search", name="mcts_bl_kdepth_v01_schema",
    node=BLMCTSKdepthV01Config,
)
cs.store(
    group="search", name="mcts_bl_kdepth_v02_schema",
    node=BLMCTSKdepthV02Config,
)
cs.store(group="search", name="mcts_sem_v01_schema", node=MCTSSemV01Config)
cs.store(group="search", name="mcts_sem_v02_schema", node=MCTSSemV02Config)
cs.store(
    group="search", name="mcts_bl_sem_v01_schema", node=BLMCTSSemConfig,
)
cs.store(
    group="search", name="mcts_bl_sem_v02_schema", node=BLMCTSSemV02Config,
)


@hydra.main(
    config_path="conf",
    config_name="mcts_cnt_prm800k",
    version_base=None,
)
def main(cfg: ExpConfig):
    root_dir = hydra.utils.get_original_cwd()

    # Locate the run by its recorded identity (manifest hash), or by
    # an explicit +result_dir=... override for old/un-backfilled dirs.
    result_dir, run_name = resolve_result_dir(
        root_dir, cfg, override=cfg.get("result_dir", None),
    )
    print(f"result_dir = {result_dir}")
    print(f"config_name = {run_name}")

    # Summary columns:
    #   pass@gb, naive@gb, weighted@gb, maj@gb,
    #   ncomps, depth, nphases, ndepths,
    #   total_gens, capped                (each: mean ± SEM)
    #   peak_gb / peak_b — best naive@b over stopping budgets
    #   b <= gen_budget, and the budget where it happens
    #   (naive_gb <= peak_gb <= pass_gb; nan for runs without a
    #   comp_gen axis, e.g. bon).
    # +num_proc=N parallelizes grading over questions (default 48;
    # use 1 when many compute_stats processes run concurrently —
    # all of them share one SLURM job cgroup's CPU/memory).
    # num_phases is the phase ceiling `capped` compares against; bon
    # has no phase loop, so it resolves to None and capped stays nan.
    summary = metrics.compute_stats_basics(
        result_dir, run_name, cfg.run.num_trials, cfg.data.grader_name,
        num_proc=cfg.get("num_proc", 48),
        num_phases=cfg.search.get("num_phases", None),
        step_budget=cfg.search.get("gen_budget", None),
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
        # Separate prefix so _sem doesn't clutter the "eval" panel
        # section in the W&B UI (sections are auto-grouped by prefix).
        log_data[f"eval_sem/{metric}_sem"] = sem
    # Write directly to summary, skip wandb.log/history entirely.
    # compute_stats.py is re-runnable (e.g. once num_trials is
    # updated), and any wandb.log call -- even with an explicit step
    # -- collides with this run's existing step cursor (the generator
    # already logged per-trial timing on steps 0..num_trials-1) and
    # gets silently dropped. eval/* is a single cross-trial average,
    # not a series, so it belongs in summary only: always overwrites,
    # never grows into a multi-point line.
    wandb.run.summary.update(log_data)
    wandb.finish()
    print(f"logged eval metrics to W&B run {run_id}")


if __name__ == "__main__":
    main()
