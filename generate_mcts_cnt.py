import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)

import time
import json

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM
import wandb

from sal.config import Config
from core import mcts_cnt_search_v05_00_00
from core.reward_models import RLHFFlow
from utils.load_data import load_data_hf

algo_dict = {
    "mcts_cnt": mcts_cnt_search_v05_00_00,
}


def _make_result_dir(path: str) -> None:
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except OSError as e:
        raise OSError(f"Error creating directory: {e}") from e


def _build_config_name(cfg: DictConfig) -> str:
    level_str = f"--level-{cfg.level}" if cfg.level is not None else ""
    gen_budget = cfg.num_batches * cfg.max_depths
    return (
        f"mcts_cnt{level_str}"
        f"--n-{cfg.n}--d-{cfg.max_depths}"
        f"--b-{gen_budget:03d}"
        f"--cpuct-{cfg.cpuct}"
    )


@hydra.main(
    config_path="conf",
    config_name="mcts_cnt_prm800k",
    version_base=None,
)
def main(cfg: DictConfig):
    root_dir = hydra.utils.get_original_cwd()
    algo = algo_dict[cfg.algo]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = Config()
    config.agg_strategy      = cfg.agg_strategy
    config.temperature       = cfg.temperature
    config.n                 = cfg.n
    config.lookahead         = cfg.lookahead
    config.max_depths        = cfg.max_depths
    config.filter_duplicates = cfg.filter_duplicates
    config.date_string       = cfg.date_string
    config.seed              = cfg.seed

    config.num_phases        = cfg.num_phases
    config.gen_budget       = cfg.num_batches * cfg.max_depths
    config.cpuct             = cfg.cpuct
    config.negative_reward   = cfg.negative_reward

    llm_vllm = LLM(
        model=cfg.llm_dir,
        tensor_parallel_size=cfg.tensor_parallel_size,
        swap_space=cfg.swap_space,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype=cfg.dtype,
        seed=cfg.seed,
    )

    prm = RLHFFlow(model_path=cfg.prm_dir, device_map="cuda:0")

    load_kwargs = {"ds_split": cfg.ds_split}
    if cfg.level is not None:
        load_kwargs["level"] = cfg.level
    dataset = load_data_hf(cfg.ds_dir, **load_kwargs)

    batch_of_questions = [q[cfg.question_field] for q in dataset]
    if cfg.num_questions > 0:
        batch_of_questions = batch_of_questions[:cfg.num_questions]
    num_questions = len(batch_of_questions)
    num_trials = cfg.num_trials
    print(f"num_questions = {num_questions}, num_trials = {num_trials}")

    config_name = _build_config_name(cfg)
    print(config_name)
    result_dir = f"{root_dir}/results/{cfg.ds_name}/{config_name}"
    _make_result_dir(result_dir)

    wandb.init(
        project="llm-reasoning",
        name=config_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    total_start = time.time()
    for trial_idx in range(num_trials):
        trial_start = time.time()
        print(f"trial {trial_idx}")

        results = algo._search(
            batch_of_questions, config, trial_idx, llm_vllm, prm,
        )
        out_path = (
            f"{result_dir}/generate_{config_name}"
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
