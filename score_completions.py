"""Score existing raw search completions into a HF Dataset.

Standalone post-processing for any search run (bon, mcts_cnt, ...).
Reads each trial's raw generate_*.jsonl and writes the scored
per-question dataset via the same core.scoring.build_scored_dataset
that generate_mcts_cnt runs inline. build_scored_dataset is
method-agnostic, so it attaches whatever per-question stats the
method emitted (mcts_cnt tree stats; bon completion_ntokens).

This is the primary scoring path for **bon** (generate_bon does not
auto-score: BoN's large n + the 8B PRM beside the vLLM generator
risks OOM, so scoring is deliberately decoupled here). For mcts_cnt
it doubles as re-scoring / recovery after an inline scoring failure.

Run it with the SAME config that produced the run, so run_name and
result_dir resolve to the same paths. The search method is selected
by --config-name. Examples:

    # mcts_cnt (its config already includes the prm group)
    python score_completions.py --config-name mcts_cnt_prm800k \\
        run.num_trials=1

    # bon (its config has no prm group, so add one)
    python score_completions.py --config-name bon_prm800k \\
        +prm=llama_prm run.num_trials=1
"""

import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)
logging.disable(logging.CRITICAL)

import json

import torch
import hydra
from hydra.core.config_store import ConfigStore

from core.reward_models import RLHFlowPRM
from core.scoring import build_scored_dataset
from utils.configs import (
    ExpConfig, BoNConfig, MCTSCntConfig, config_name, level_dir,
)
from utils.load_data import load_data_hf

# Register every search schema so any --config-name resolves its
# search group. The actual method is chosen by the selected config.
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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    prm = RLHFlowPRM(
        model_path=cfg.prm.prm_dir, device=cfg.prm.device_map,
    )

    load_kwargs = {"ds_split": cfg.data.ds_split}
    if cfg.data.level is not None:
        load_kwargs["level"] = cfg.data.level
    dataset = load_data_hf(cfg.data.ds_dir, **load_kwargs)

    run_name = config_name(cfg)
    result_dir = (
        f"{root_dir}/results/{cfg.data.name}"
        f"/{level_dir(cfg)}/{run_name}"
    )
    print(run_name)

    for trial_idx in range(cfg.run.num_trials):
        raw_path = (
            f"{result_dir}/generate_{run_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        if not os.path.exists(raw_path):
            print(f"skip trial {trial_idx}: {raw_path} not found")
            continue

        with open(raw_path, 'r', encoding='utf-8') as fin:
            results = json.loads(fin.readline())

        build_scored_dataset(
            results, dataset, prm, result_dir, run_name,
            trial_idx, agg_strategy=cfg.gen.agg_strategy,
            n=0, batch_size=cfg.prm.score_batch_size,
            num_proc=cfg.run.num_proc,
        )


if __name__ == "__main__":
    main()
