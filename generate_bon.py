import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)
logging.disable(logging.CRITICAL)

import time
import json

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from vllm import LLM
import wandb

from core import bon_search_v01_0_0
from utils.configs import (
    ExpConfig, BoNConfig, config_name, level_dir, save_wandb_run_id,
)
from utils.load_data import load_data_hf

algo_dict = {
    "bon": bon_search_v01_0_0,
}

# Register the structured schemas so the YAML binds onto typed,
# validated dataclasses instead of a plain DictConfig. The search
# subclass is registered under the "search" group; conf/search/bon
# selects it (ExpConfig.search is the base type, so the concrete
# schema must come from the group).
cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(group="search", name="bon_schema", node=BoNConfig)


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
    config_name="bon_prm800k",
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
        seed=cfg.gen.seed,
    )

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
        f"{root_dir}/results/{cfg.data.name}"
        f"/{level_dir(cfg)}/{run_name}"
    )
    _make_result_dir(result_dir)

    # Tag by difficulty level so runs are filterable in W&B
    # (level=None means the whole split -> "level-all").
    level_tag = (
        f"level-{cfg.data.level}"
        if cfg.data.level is not None else "level-all"
    )
    wandb_run = wandb.init(
        project="llm-reasoning",
        name=run_name,
        tags=[level_tag],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    # Persist the run id so compute_stats can reattach and log the
    # post-processing metrics onto this same run.
    save_wandb_run_id(result_dir, wandb_run.id)

    total_start = time.time()
    for trial_idx in range(num_trials):
        trial_start = time.time()
        print(f"trial {trial_idx}")

        results = algo._search(
            batch_of_questions, cfg, trial_idx, llm_vllm,
        )
        out_path = (
            f"{result_dir}/generate_{run_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout)
            fout.write('\n')

        elapsed = time.time() - trial_start
        print(f"it takes {elapsed / num_questions:0.4f}s per question")
        print(f"it takes {elapsed / 3600:0.2f}h per trial")
        wandb.log({
            "time_per_question_s": elapsed / num_questions,
            "time_per_trial_hr": elapsed / 3600,
        }, step=trial_idx)

    print(f"it takes {time.time() - total_start:0.4f}s in total")
    wandb.finish()


if __name__ == "__main__":
    main()
