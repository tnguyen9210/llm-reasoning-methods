'''
Generate best-of-n (BoN) completions for PRM800K/MATH problems
with vLLM.

This script loads the PRM800K MATH test split, filters by problem
level, runs multiple BoN generation trials with a configured vLLM
model, logs timing metrics to Weights & Biases, and writes each
trial's completions to JSONL files for downstream analysis.

v0101: base BoN generation script with wandb experiment tracking.
'''


import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL+1)

import time
import json

import torch
from vllm import LLM

from sal.config import Config

from core import bon_search_v01_0_0
from utils.load_data import load_data_hf

import wandb

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


def main():
    algo_version = "v01_0_0"
    algo_name = "bon"
    algo = algo_dict[algo_name]

    base_dir = '/groups/chichengz/tnn/datasets/'
    ds_name  = "prm800k"
    ds_split = "test"
    ds_dir   = os.path.join(base_dir, "prm800k/math_splits")
    llm_dir  = os.path.join(base_dir, "Llama-3.2-1B-Instruct")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = Config()
    config.agg_strategy     = 'last'
    config.temperature      = 0.8
    config.max_tokens       = 2048
    config.bs               = 256
    config.filter_duplicates = True
    config.date_string      = "Aug 1 2025"
    config.seed             = 0
    config.version          = algo_version

    llm_vllm = LLM(
        model=llm_dir,
        tensor_parallel_size=1,
        swap_space=16,
        max_model_len=5000,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=config.seed,
    )

    level      = 4
    num_trials = 4
    dataset    = load_data_hf(ds_dir, ds_split=ds_split, level=level)
    batch_of_questions = [q['problem'] for q in dataset]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}, num_trials = {num_trials}")

    config_name = (
        f"bon--level-{level}--{config.version}"
        f"--bs-{config.bs}--temp-{config.temperature}"
    )
    print(config_name)
    result_dir = f"results/{ds_name}/bon--level-{level}/{config_name}"
    _make_result_dir(result_dir)

    wandb.init(
        project="MCTSDiv",
        name=config_name,
        config=vars(config),
    )

    total_start = time.time()
    for trial_idx in range(num_trials):
        trial_start = time.time()
        print(f"trial {trial_idx}")

        results = algo._search(batch_of_questions, config, trial_idx, llm_vllm)
        out_path = f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout)
            fout.write('\n')

        elapsed = time.time() - trial_start
        print(f"it takes {elapsed / num_questions:0.4f}s per question")
        print(f"it takes {elapsed / 3600:0.2f}h per trial")
        wandb.log({
            "trial": trial_idx,
            "time_per_question": elapsed / num_questions,
            "time_per_trial_hr": elapsed / 3600,
        })

    print(f"it takes {time.time() - total_start:0.4f}s in total")
    wandb.finish()


if __name__ == "__main__":
    main()
