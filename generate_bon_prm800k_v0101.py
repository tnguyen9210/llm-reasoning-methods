import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL+1)

import time
import json

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM
import wandb

from sal.config import Config
from core import bon_search_v01_0_0
from utils.load_data import load_data_hf

algo_dict = {
    "bon": bon_search_v01_0_0,
}


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
def main(cfg: DictConfig):
    root_dir = hydra.utils.get_original_cwd()
    algo = algo_dict[cfg.algo]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = Config()
    config.agg_strategy      = cfg.agg_strategy
    config.temperature       = cfg.temperature
    config.max_tokens        = cfg.max_tokens
    config.bs                = cfg.bs
    config.filter_duplicates = cfg.filter_duplicates
    config.date_string       = cfg.date_string
    config.seed              = cfg.seed
    config.version           = cfg.algo_version

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

    dataset = load_data_hf(
        cfg.ds_dir, ds_split=cfg.ds_split, level=cfg.level
    )
    batch_of_questions = [q['problem'] for q in dataset]
    if cfg.num_questions > 0:
        batch_of_questions = batch_of_questions[:cfg.num_questions]
    num_questions = len(batch_of_questions)
    num_trials = cfg.num_trials
    print(f"num_questions = {num_questions}, num_trials = {num_trials}")

    config_name = (
        f"bon--level-{cfg.level}--{cfg.algo_version}"
        f"--bs-{cfg.bs}--temp-{cfg.temperature}"
    )
    print(config_name)
    result_dir = (
        f"{root_dir}/results/{cfg.ds_name}/bon--level-{cfg.level}"
        f"/{config_name}"
    )
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
            batch_of_questions, config, trial_idx, llm_vllm
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
