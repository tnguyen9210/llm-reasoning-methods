import os
import time
import json

import torch
from vllm import LLM

from sal.config import Config

from core import mcts_search_extra_v71
from core.reward_models import RLHFFlow

from utils.load_data import load_data_prm800k


def _make_result_dir(path: str) -> None:
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except OSError as e:
        raise OSError(f"Error creating directory: {e}") from e


def main():
    base_dir = '/groups/kjun/tnn/datasets/'
    data_dir = os.path.join(base_dir, "prm800k/math_splits")
    llm_dir  = os.path.join(base_dir, "Llama-3.2-1B-Instruct")
    prm_dir  = os.path.join(base_dir, "Llama3.1-8B-PRM-Deepseek-Data")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = Config()
    config.agg_strategy    = 'last'
    config.n               = 4      # candidates generated per depth
    config.beam_width      = 4      # nodes kept after selection
    config.lookahead       = 0
    config.max_depths      = 20
    config.sort_completed  = False
    config.filter_duplicates = True
    config.date_string     = "Aug 1 2025"
    config.seed            = 0

    config.num_batches  = 4
    config.step_budget  = config.num_batches * config.max_depths
    config.num_phases   = 1000
    config.lam          = 0.1
    config.normalize_embeds = True
    config.use_ppl      = True
    config.ds_beta      = 1.0
    config.ds_alpha     = 100.0
    config.negative_reward = 0
    config.version      = "h71"

    level = 3

    llm_gen_gpu_util   = 0.2
    llm_embed_gpu_util = 0.2   # total budget 0.4, split evenly

    _llm_kwargs = dict(
        tensor_parallel_size=1,
        swap_space=16,
        max_model_len=5000,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=config.seed,
    )
    llm_vllm       = LLM(model=llm_dir, gpu_memory_utilization=llm_gen_gpu_util,   **_llm_kwargs)
    llm_vllm_embeds = LLM(model=llm_dir, gpu_memory_utilization=llm_embed_gpu_util, task="embed", **_llm_kwargs)
    prm = RLHFFlow(model_path=prm_dir, device_map='cuda:0')

    data_by_levels = load_data_prm800k(data_dir)
    batch_of_questions = [q['problem'] for q in data_by_levels[level]]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}")

    for alpha in [100.0]:
        config.ds_alpha = alpha

        config_name = (
            f"mcts--level-{level}--{config.version}"
            f"--n-{config.n}--d-{config.max_depths}--b-{config.step_budget}"
            f"--lam-{config.lam}--dalpha-{config.ds_alpha}--dbeta-{config.ds_beta}"
            f"--ppl-{config.use_ppl}--normalize-{config.normalize_embeds}"
        )
        print(config_name)
        result_dir = f"results/mcts--level-{level}/{config_name}"
        _make_result_dir(result_dir)

        total_start = time.time()
        for trial_idx in [1]:
            trial_start = time.time()
            print(f"trial {trial_idx}")

            results = mcts_search_extra_v71._search(
                batch_of_questions, config, trial_idx, llm_vllm, llm_vllm_embeds, prm
            )
            out_path = f"{result_dir}/generate_{config_name}--trial-{trial_idx}.jsonl"
            with open(out_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout)
                fout.write('\n')

            elapsed = time.time() - trial_start
            print(f"it takes {elapsed / num_questions:0.4f}s per question")
            print(f"it takes {elapsed / 3600:0.2f}h per trial")

        print(f"it takes {time.time() - total_start:0.4f}s in total")


if __name__ == "__main__":
    main()