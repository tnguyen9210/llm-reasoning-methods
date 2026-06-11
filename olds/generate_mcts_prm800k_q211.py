'''
MCTS implementation from the rStar-Math's repo
'''

import os, psutil, gc
import time 
import json
import pprint

from collections import defaultdict
import random
import numpy as np

import torch 
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

from sal.config import Config

from core import mcts_search_extra_v21
from core.reward_models import RLHFFlow

from datasets import load_dataset

# from utils.load_data import load_data_prm800k_hf
from utils.utils import add_completions_to_dataset

algo_dict = {
    "cnt_mcts": mcts_search_extra_v21
}


def main():

    algo_name = "cnt_mcts"
    algo = algo_dict[algo_name]
    
    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits"
    
    # llm and prm path
    # llm_dir = base_dir + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    # prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
    
    llm_dir = base_dir + "/Llama-3.2-1B-Instruct"
    prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(GPUS)
    else:
        print("CUDA is not available.")

    # general params
    config = Config()
    config.agg_strategy = 'last'
    config.n = 4                      # number of budgets to be generated per depth
    config.beam_width = 4             # number of nodes left after selection
    config.lookahead = 0              # don't use it for now
    config.max_depths = 20            # max depths, after reaching max_depth then terminate search 
    config.sort_completed = False      
    config.filter_duplicates = True   # remove any duplicates in the last list of trajs
    config.date_string = "Aug 1 2025"
    config.seed = 0

    # mcts parameter
    config.num_batches = 20
    config.step_budget = config.num_batches*config.max_depths 
    config.num_phases = 1000

    config.cpuct = 2
    config.negative_reward = 0

    config.version = "q21"

    level = 4                                   # level of difficulty of questions
    
    llm_total_gpu = 0.4
    llm_gpu_memory_utilization = 0.2
    
    llm_vllm = LLM(
        model=llm_dir, 
        tensor_parallel_size=1, 
        # trust_remote_code=True,
        swap_space=16,
        max_model_len=5000,
        gpu_memory_utilization=llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=config.seed,
    )

    prm = RLHFFlow(model_path=prm_dir, device_map='cuda:0')

    #  load data 
    level = 4                                   # level of difficulty of questions
    num_trials = 20
    print(f"num_trials = {num_trials}")

    q_idxes = [79, 19, 0]

    for q_idx in q_idxes:
        # data_by_levels = load_data_prm800k(data_dir)
        print(f"\n-> q_idx = {q_idx}")
        dataset = load_dataset("json", data_files = f"{data_dir}/test.jsonl", split='train')
        dataset = dataset.filter(lambda example: example['level'] == level)
        if q_idx is not None:
            dataset = dataset.select([q_idx])
        
        num_questions = len(dataset)  # level 4 has 128 questions
        # num_questions = 2
        print(f"num_questions = {num_questions}")
    
        # get batch of questions ['q1', 'q2', ...]
        # batch_of_questions = [data_by_levels[level][q_idx]['problem'] for q_idx in range(num_questions)]
        batch_of_questions = [dataset[idx]['problem'] for idx in range(num_questions)]
    
        # run search_algo and save results
        config_name = f"mcts--level-{level}--q-{q_idx:03d}--{config.version}--n-{config.n}--d-{config.max_depths}--b-{config.step_budget:03d}--cpuct-{config.cpuct}"
        print(config_name)
        result_dir = f"results/mcts--level-{level}/prompts/q-{q_idx:03d}/{config_name}"
        try:
            os.makedirs(result_dir, exist_ok=True)
            print(f"Directory '{result_dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{result_dir}' already exists.")
        except OSError as e:
            print(f"Error creating directory: {e}")
            stop
                
        start_time1 = time.time()
        for trial_idx in range(num_trials):
            print(f"trial {trial_idx:03d}")
            start_time = time.time()
            
            # best_of_n(batch_of_questions, config, llm_vllm, random_seeds[trial_idx])
            results = algo._search(batch_of_questions, config, trial_idx, llm_vllm, prm)
            with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'w', encoding = 'utf-8') as fout:
                json.dump(results, fout)
                fout.write('\n')
        
            # compute the time
            if trial_idx % 1 == 0:
                total_time = time.time() - start_time
                # time_per_trial = total_time/(trial_idx+1)
                time_per_trial = total_time
                time_per_question = time_per_trial/num_questions
                # print(f"trial {trial_idx}")
                print(f"it takes {time_per_question:0.4f}s per question")
                print(f"it takes {time_per_trial/3600:0.2f}h per trial")
    
            add_completions_to_dataset(result_dir, config_name, dataset, prm, trial_idx, config)
            
        total_time = time.time() - start_time1
        print(f"it takes {total_time/3600:0.2f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()