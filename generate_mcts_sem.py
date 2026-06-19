import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)
logging.disable(logging.CRITICAL)

import time
import json
import socket

import numpy as np
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from vllm import LLM
import wandb

from core import (
    mcts_sem_search_v01_00_00,
    mcts_sem_search_v02_00_00,
)
from core.reward_models import RLHFlowPRM
from core.scoring import build_scored_dataset
from utils.configs import (
    ExpConfig, MCTSSemV01Config, MCTSSemV02Config, config_name,
    level_dir, save_wandb_run_id, load_wandb_run_id,
)
from utils.load_data import load_data_hf

# One launcher, two semantic-MCTS variants. cfg.algo picks the core
# search module: v01 sources diversity embeds from a second vLLM
# pooling engine on the policy; v02 sources them from the PRM and
# skips that engine. The differing wiring (whether to build the
# pooling engine) is driven by cfg.search.embeds_source below, not
# by this dict — the dict only selects the algorithm.
algo_dict = {
    "mcts_sem_v01": mcts_sem_search_v01_00_00,
    "mcts_sem_v02": mcts_sem_search_v02_00_00,
}

# Register the structured schemas so the YAML binds onto typed,
# validated dataclasses instead of a plain DictConfig. Both search
# subclasses are registered under the "search" group; conf/search/
# mcts_sem_v01|v02 selects one (ExpConfig.search is the base type, so
# the concrete schema must come from the group).
cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(group="search", name="mcts_sem_v01_schema", node=MCTSSemV01Config)
cs.store(group="search", name="mcts_sem_v02_schema", node=MCTSSemV02Config)


def _make_result_dir(path: str) -> None:
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except OSError as e:
        raise OSError(f"Error creating directory: {e}") from e


@hydra.main(
    config_path="conf",
    config_name="mcts_sem_v01_prm800k",
    version_base=None,
)
def main(cfg: ExpConfig):
    root_dir = hydra.utils.get_original_cwd()
    algo = algo_dict[cfg.algo]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Generative engine.
    llm_vllm = LLM(
        model=cfg.llm.llm_dir,
        tensor_parallel_size=cfg.llm.tensor_parallel_size,
        max_model_len=cfg.llm.max_model_len,
        gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
        enforce_eager=cfg.llm.enforce_eager,
        distributed_executor_backend=None,
        dtype=cfg.llm.dtype,
        seed=cfg.gen.seed,
    )

    # Second engine in pooling mode: returns per-token hidden states
    # for the embedding-diversity term. Same checkpoint, its own
    # GPU-memory share (search.embeds_gpu_memory_utilization). Built
    # ONLY when embeds come from the policy (v01). For embeds_source ==
    # "prm" (v02) the embeds are pulled from the PRM forward pass, so
    # this engine isn't loaded and its GPU share goes back to the
    # generative engine — the search core receives None for it.
    llm_vllm_embeds = None
    if cfg.search.embeds_source == "policy":
        llm_vllm_embeds = LLM(
            model=cfg.llm.llm_dir,
            runner="pooling",
            tensor_parallel_size=cfg.llm.tensor_parallel_size,
            max_model_len=cfg.llm.max_model_len,
            gpu_memory_utilization=cfg.search.embeds_gpu_memory_utilization,
            enforce_eager=cfg.llm.enforce_eager,
            distributed_executor_backend=None,
            dtype=cfg.llm.dtype,
            seed=cfg.gen.seed,
        )

    prm = RLHFlowPRM(model_path=cfg.prm.prm_dir, device=cfg.prm.device_map)

    load_kwargs = {"ds_split": cfg.data.ds_split}
    if cfg.data.level is not None:
        load_kwargs["level"] = cfg.data.level
    dataset = load_data_hf(cfg.data.ds_dir, **load_kwargs)

    batch_of_questions = [q[cfg.data.question_field] for q in dataset]
    if cfg.run.num_questions > 0:
        batch_of_questions = batch_of_questions[:cfg.run.num_questions]
    num_questions = len(batch_of_questions)
    num_trials = cfg.run.num_trials
    print(f"num_questions = {num_questions}, num_trials = {num_trials}")

    # Load the held-out embedding mean when centering is on. Stored on
    # the search config so the core's _extract_embeds can subtract it.
    # With embeds_proj="sparse" the mean must be in the POST-projection
    # space (built with the same fixed projection); _extract_embeds
    # guards on the shape and raises if it's the raw-source dim instead.
    if cfg.search.embeds_center:
        mean_path = f"{root_dir}/results/{cfg.search.embeds_mean_dir}.npy"
        cfg.search.embeds_mean = np.load(mean_path).flatten()
        print(f"embeds_mean.shape = {cfg.search.embeds_mean.shape}")

    run_name = config_name(cfg)
    print(run_name)
    result_dir = (
        f"{root_dir}/results/{cfg.data.name}"
        f"/{level_dir(cfg)}/{run_name}"
    )
    _make_result_dir(result_dir)

    # Resume onto the same run if this is a restart. load_ returns None
    # on a fresh launch (no sidecar) -> W&B mints a new id. resume="allow"
    # (not "must") so both first launch and restart share this one path.
    run_id = load_wandb_run_id(result_dir)
    wandb_run = wandb.init(
        project="llm-reasoning",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=run_id,
        resume="allow",
    )
    # Persist the run id so a restart -- and compute_stats -- can
    # reattach and log onto this same run. Idempotent on a resume.
    save_wandb_run_id(result_dir, wandb_run.id)

    print(f"node = {socket.gethostname()}")

    # Skip trials already completed in a prior (interrupted) launch. A
    # trial's .done marker is written only after its raw results are
    # dumped, so resume restarts at the first trial without one.
    start_trial = 0
    while start_trial < num_trials and os.path.exists(
        f"{result_dir}/generate_{run_name}"
        f"--trial-{start_trial:03d}.done"
    ):
        start_trial += 1
    if start_trial > 0:
        print(f"resuming: {start_trial}/{num_trials} trials already done")

    total_start = time.time()
    for trial_idx in range(start_trial, num_trials):
        trial_start = time.time()
        print(f"trial {trial_idx}")

        results = algo._search(
            batch_of_questions, cfg, trial_idx,
            llm_vllm, llm_vllm_embeds, prm,
        )

        # 1. Dump the raw results. Write to a temp path and rename so a
        # crash mid-write never leaves a half-written .jsonl visible
        # under the real name (os.replace is atomic on one filesystem).
        out_path = (
            f"{result_dir}/generate_{run_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        tmp_path = out_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout)
            fout.write('\n')
        os.replace(tmp_path, out_path)

        # 2. Log timing to W&B.
        elapsed = time.time() - trial_start
        print(f"it takes {elapsed / num_questions:0.4f}s per question")
        print(f"it takes {elapsed / 3600:0.2f}h per trial")
        wandb.log({
            "time_per_question_s": elapsed / num_questions,
            "time_per_trial_hr": elapsed / 3600,
        }, step=trial_idx)

        # 3. Mark the trial done. The marker's presence means
        # "generation finished + raw results dumped" -- NOT "scored"
        # (scoring runs below and is re-runnable separately). A future
        # resume can skip any trial whose marker already exists.
        done_path = (
            f"{result_dir}/generate_{run_name}"
            f"--trial-{trial_idx:03d}.done"
        )
        with open(done_path, 'w', encoding='utf-8') as fout:
            fout.write('')

        # 4. Post-process: score completions and write the per-question
        # HF dataset. Wrapped so a scoring failure never discards the
        # raw results already written above; re-runnable separately
        # via prepare_scored_dataset.py.
        try:
            build_scored_dataset(
                results, dataset, prm, result_dir, run_name,
                trial_idx, agg_strategy=cfg.gen.agg_strategy,
                n="gb", batch_size=cfg.prm.score_batch_size,
                num_proc=cfg.run.num_proc,
            )
        except Exception as e:
            print(f"scoring failed for trial {trial_idx}: {e!r}")
            print("raw results saved; re-run prepare_scored_dataset.py")

    print(f"it takes {time.time() - total_start:0.4f}s in total")
    wandb.finish()


if __name__ == "__main__":
    main()
