'''
Modified the code to use a single GPU
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

from core import mcts_embeds_search_v03_02_00
from core.reward_models import RLHFFlow

from utils.load_data import load_data_prm800k_hf
from utils.utils import add_completions_to_dataset_tree

algo_dict = {
    "mcts_embeds": mcts_embeds_search_v03_02_00
}

def main():

    algo_version = "v03_02_00"
    algo_name = "mcts_embeds"
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
    config.max_depths = 10            # max depths, after reaching max_depth then terminate search 
    config.sort_completed = False      
    config.filter_duplicates = True   # remove any duplicates in the last list of trajs
    config.date_string = "Aug 1 2025"
    config.seed = 0

    # mcts parameter
    config.num_batches = 16
    config.step_budget = config.num_batches*config.max_depths 
    config.num_phases = 1000
    
    config.lam = 0.01 
    config.use_ppl = True
    config.embeds_normalizing = True
    config.embeds_centering = True
    config.embeds_mean_dir = "embeds_mean_bon--level-4--v01_0_0--bs-256"

    config.ds_beta = 1.0
    config.ds_alpha = 100.0
    config.negative_reward = 0
    
    config.version = algo_version  
    
    # baseline: gpu_memory_utilization=0.2
    # use the standard model 
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

    llm_vllm_embeds = LLM(
        model=llm_dir, 
        tensor_parallel_size=1, 
        # trust_remote_code=True,
        task="embed",
        swap_space=16,
        max_model_len=5000,
        gpu_memory_utilization=llm_total_gpu-llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=config.seed,
    )
    
    prm = RLHFFlow(model_path=prm_dir, device_map='cuda:0')

    #  load data 
    level = 4                                   # level of difficulty of questions
    num_trials = 5
    # q_idx = 0
    print(f"num_trials = {num_trials}")

    dataset = load_data_prm800k_hf(data_dir, level)
    
    num_questions = len(dataset)  # level 4 has 128 questions
    # num_questions = 2
    print(f"num_questions = {num_questions}")

    # get batch of questions ['q1', 'q2', ...]
    # batch_of_questions = [data_by_levels[level][q_idx]['problem'] for q_idx in range(num_questions)]
    batch_of_questions = [dataset[idx]['problem'] for idx in range(num_questions)]

    if config.embeds_centering:
        config.embeds_mean = np.load(f"results/{config.embeds_mean_dir}.npy")
        config.embeds_mean = config.embeds_mean.flatten()
        print(config.embeds_mean.shape)

    # run search_algo and save results
    config_name = f"mcts_embeds--level-{level}--{config.version}--n-{config.n}--d-{config.max_depths}--b-{config.step_budget:03d}--lam-{config.lam}--dalpha-{config.ds_alpha}--dbeta-{config.ds_beta}--ppl-{config.use_ppl}--normalize-{config.embeds_normalizing}--mcenter-{config.embeds_centering}"
    print(config_name)
    result_dir = f"results/mcts_embeds--level-{level}/{config_name}"
    try:
        os.makedirs(result_dir)
        print(f"Directory '{result_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{result_dir}' already exists.")
    except OSError as e:
        print(f"Error creating directory: {e}")
        stop
            
    start_time1 = time.time()
    for trial_idx in [0,2,4]:
        print(f"trial {trial_idx}")
        start_time = time.time()
        
        # best_of_n(batch_of_questions, config, llm_vllm, random_seeds[trial_idx])
        results = algo._search(batch_of_questions, config, trial_idx, llm_vllm, llm_vllm_embeds, prm)
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

        add_completions_to_dataset_tree(result_dir, config_name, dataset, prm, trial_idx, config)
            
    
    total_time = time.time() - start_time1
    print(f"it takes {total_time/3600:0.2f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()