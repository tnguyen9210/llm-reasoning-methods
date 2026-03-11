import os, psutil, gc
import json, pickle
import time
import pprint
import copy 

from collections import defaultdict
import random
import numpy as np

import torch
import torch.distributed as dist

import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL+1)

from vllm import LLM, SamplingParams, PoolingParams
from sal.config import Config

from utils.load_data import load_data_prm800k_hf
from utils import utils


def main():
    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits1"
    
    # llm and prm path
    # llm_dir = base_dir + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    # prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
    
    llm_dir = base_dir + "/Llama-3.2-1B-Instruct"
    prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(GPUS)
    else:
        print("CUDA is not available.")

    llm_gpu_memory_utilization = 0.5
    llm_vllm_embeds = LLM(
        model=llm_dir, 
        tensor_parallel_size=1, 
        # trust_remote_code=True,
        task="embed",
        swap_space=16,
        max_model_len=6000,
        gpu_memory_utilization=llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=0,
    )

    config = Config()
    config.agg_strategy = 'last'
    config.date_string = "Aug 1 2025"
    config.normalize_embeds = True

    # load data
    level = 2
    num_trials = 5 
    print(f"num_trials = {num_trials}")
     
    dataset_orig = load_data_prm800k_hf(data_dir, level)

    num_questions = len(dataset_orig)
    # num_questions = 2
    print(f"num_questions = {num_questions}")
    
    config_name = f"bon--level-2--v01_0_0--bs-256"
    # config_name = f"mcts--level-4--q-000--q31--n-4--d-20--b-80--cpuct-2"
    result_dir = f"results/bon--level-{level}/{config_name}"
    print(f"config_name = {config_name}")

    start_time1 = time.time()
    all_embeds = []
    for trial_idx in range(num_trials):
        print(f"\n-> trial_idx = {trial_idx}")
        trial_embeds = utils.extract_completions_embeds(
            result_dir, config_name, trial_idx, dataset_orig, llm_vllm_embeds, config=config)
        all_embeds += trial_embeds

        trial_embeds = np.vstack(trial_embeds)
        print(len(trial_embeds))
        
        np.save(f"{result_dir}/embeds_{config_name}--trial-{trial_idx:03d}.npy", trial_embeds)
        
        # compute the time
        if trial_idx % 1 == 0:
            total_time = time.time() - start_time1
            time_per_trial = total_time/(trial_idx+1)
            time_per_question = time_per_trial/num_questions
            # print(f"trial {trial_idx}")
            print(f"it takes {time_per_question:0.4f}s per question")
            print(f"it takes {time_per_trial:0.4f}s per trial")

    # all_embeds = np.vstack(all_embeds) 
    # np.savetxt(f"{result_dir}/embeds_{config_name}.npy", all_embeds)
    total_time = time.time() - start_time1
    print(f"it takes {total_time:0.4f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()