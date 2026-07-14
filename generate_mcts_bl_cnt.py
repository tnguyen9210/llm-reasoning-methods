import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)
logging.disable(logging.CRITICAL)

import time
import json
import socket

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from vllm import LLM
import wandb

from core import (
    mcts_bl_cnt_search_v01_00_00,
    mcts_bl_cnt_search_v02_00_00,
    mcts_bl_cnt_search_v03_00_00,
)
from core.reward_models import build_prm
from core.scoring import build_scored_dataset
from utils.configs import (
    ExpConfig, BLMCTSCntConfig, BLMCTSCntV02Config, BLMCTSCntV03Config,
    config_name, level_dir, results_root,
    write_manifest, load_wandb_run_id,
    save_timing_state, load_timing_state,
)
from utils.load_data import load_data_hf

# One launcher, three budget-limited-cnt-MCTS variants. All three share
# an identical _search(batch_of_questions, cfg, trial_idx, llm_vllm,
# prm) signature and the same best-first frontier skeleton -- they
# differ only in the leaf-selection index (PUCT / fractional-KUBE /
# depth-shaping knapsack), which lives entirely in each core module, so
# there is no per-variant runtime wiring here (contrast
# generate_mcts_sem.py, which branches on whether to build a second
# pooling engine). cfg.algo picks the core module; --config-name picks
# the root config (and therefore the search schema, via its
# search: mcts_bl_cnt_v0N group file).
algo_dict = {
    "mcts_bl_cnt_v01": mcts_bl_cnt_search_v01_00_00,
    "mcts_bl_cnt_v02": mcts_bl_cnt_search_v02_00_00,
    "mcts_bl_cnt_v03": mcts_bl_cnt_search_v03_00_00,
}

# Register the structured schemas so the YAML binds onto typed,
# validated dataclasses instead of a plain DictConfig. The search
# subclasses are registered under the "search" group; conf/search/
# mcts_bl_cnt_v01|v02|v03 selects one (ExpConfig.search is the base
# type, so the concrete schema must come from the group).
cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(
    group="search", name="mcts_bl_cnt_v01_schema", node=BLMCTSCntConfig,
)
cs.store(
    group="search", name="mcts_bl_cnt_v02_schema", node=BLMCTSCntV02Config,
)
cs.store(
    group="search", name="mcts_bl_cnt_v03_schema", node=BLMCTSCntV03Config,
)


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
    config_name="mcts_bl_cnt_v01_prm800k",
    version_base=None,
)
def main(cfg: ExpConfig):
    root_dir = hydra.utils.get_original_cwd()
    algo = algo_dict[cfg.algo]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    llm_vllm = LLM(
        model=cfg.llm.llm_dir,
        tensor_parallel_size=cfg.llm.tensor_parallel_size,
        max_model_len=cfg.llm.max_model_len,
        gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
        enforce_eager=cfg.llm.enforce_eager,
        distributed_executor_backend=None,
        dtype=cfg.llm.dtype,
        quantization=cfg.llm.quantization,
        load_format=cfg.llm.load_format,
        seed=cfg.gen.seed,
    )

    prm = build_prm(cfg.prm.kind, cfg.prm.prm_dir, device=cfg.prm.device_map)

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

    run_name = config_name(cfg)
    print(run_name)
    result_dir = (
        f"{root_dir}/results/{results_root(cfg)}"
        f"/{level_dir(cfg)}/{run_name}"
    )
    _make_result_dir(result_dir)
    # Resume onto the same run if this is a restart. load_ returns None
    # on a fresh launch (no manifest run_id) -> W&B mints a new id.
    # resume="allow" (not "must") so both first launch and restart
    # share this one path. MUST read this BEFORE write_manifest below:
    # write_manifest with run_id=None would overwrite a saved id with
    # null, so a restart would lose it and mint a fresh run every time.
    run_id = load_wandb_run_id(result_dir)
    # Record the full config identity so post-processing can locate
    # this run by recorded hash (find_run_dir), not by re-deriving the
    # name. Pass the loaded run_id through so the pre-init write
    # preserves it on a resume. Idempotent (atomic overwrite).
    write_manifest(result_dir, cfg, run_id=run_id)
    wandb_run = wandb.init(
        project="llm-reasoning",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=run_id,
        resume="allow",
    )
    # Persist the run id into the manifest so a restart -- and
    # compute_stats -- can reattach and log onto this same run.
    # Idempotent on a resume.
    write_manifest(result_dir, cfg, run_id=wandb_run.id)

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

    # Running average of per-trial timing, logged to W&B as
    # time_per_question_s / time_per_trial_hr. Seeded from the sidecar
    # so a resume continues the SAME average rather than restarting it
    # from this process's first trial (n_done/avg are (0, 0.0, 0.0) on
    # a fresh launch, since load_timing_state has nothing to load).
    n_done, avg_q_s, avg_trial_hr = load_timing_state(result_dir)

    total_start = time.time()
    for trial_idx in range(start_trial, num_trials):
        trial_start = time.time()
        print(f"trial {trial_idx}")

        results = algo._search(
            batch_of_questions, cfg, trial_idx, llm_vllm, prm,
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

        # 2. Fold this trial into the running average and log it to
        # W&B (cumulative over all trials so far, not just this one).
        elapsed = time.time() - trial_start
        avg_q_s = (avg_q_s * n_done + elapsed / num_questions) / (n_done + 1)
        avg_trial_hr = (avg_trial_hr * n_done + elapsed / 3600) / (n_done + 1)
        n_done += 1
        save_timing_state(result_dir, n_done, avg_q_s, avg_trial_hr)
        print(f"running avg: {avg_q_s:0.4f}s per question")
        print(f"running avg: {avg_trial_hr:0.2f}h per trial")
        wandb.log({
            "time_per_question_s": avg_q_s,
            "time_per_trial_hr": avg_trial_hr,
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
                question_field=cfg.data.question_field,
            )
        except Exception as e:
            print(f"scoring failed for trial {trial_idx}: {e!r}")
            print("raw results saved; re-run prepare_scored_dataset.py")

    print(f"it takes {time.time() - total_start:0.4f}s in total")
    wandb.finish()


if __name__ == "__main__":
    main()
