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


@hydra.main(
    config_path="conf",
    config_name="bon_gsm8k",
    version_base=None,
)
def main(cfg: DictConfig):
    root_dir = hydra.utils.get_original_cwd()
    algo = algo_dict[cfg.algo]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    print(GPUS)

    config = Config()
    config.agg_strategy = cfg.agg_strategy
    config.bs = cfg.bs
    config.filter_duplicates = cfg.filter_duplicates
    config.date_string = cfg.date_string
    config.seed = cfg.seed
    config.version = cfg.algo_version

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

    num_trials = cfg.num_trials
    print(f"num_trials = {num_trials}")

    dataset = load_data_hf(cfg.data_dir, ds_split="test_subset")
    num_questions = len(dataset)
    print(f"num_questions = {num_questions}")

    batch_of_questions = [
        dataset[q_idx]['question'] for q_idx in range(num_questions)
    ]

    config_name = f"bon--{cfg.algo_version}--bs-{cfg.bs}"
    print(config_name)
    result_dir = f"{root_dir}/results/gsm8k/bon/{config_name}"
    try:
        os.makedirs(result_dir)
        print(f"Directory '{result_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{result_dir}' already exists.")
    except OSError as e:
        raise OSError(f"Error creating directory: {e}") from e

    wandb.init(
        project="llm-reasoning",
        name=config_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    start_time1 = time.time()
    for trial_idx in range(num_trials):
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

        if trial_idx % 1 == 0:
            total_time = time.time() - start_time1
            time_per_trial = total_time / (trial_idx + 1)
            time_per_question = time_per_trial / num_questions
            print(f"it takes {time_per_question:0.4f}s per question")
            print(f"it takes {time_per_trial:0.4f}s per trial")
            wandb.log({
                "time_per_question_s": time_per_question,
                "time_per_trial_hr": time_per_trial / 3600,
            }, step=trial_idx)

    total_time = time.time() - start_time1
    print(f"it takes {total_time:0.4f}s in total")
    wandb.finish()


if __name__ == "__main__":
    main()
