import os, psutil, gc
import time 
import json
import pprint

from collections import defaultdict
import random
import numpy as np

import torch 
import torch.distributed as dist
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams, PoolingParams

# from sal.config import Config
# from sal.utils.score import score, aggregate_scores
# from core.reward_models import RLHFFlow

# from utils.load_data import load_data_prm800k
from utils.load_data import load_data_prm800k_hf


def extract_completions_info(config_name, dataset, trial_idx):

    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}")

    start_time = time.time()    
    with open(f"results/{config_name}/generate_{config_name}--trial-{trial_idx}.jsonl", 'r', encoding = 'utf-8') as fin:
        for line in fin:

            trial_data = json.loads(line)
            # print(len(trial_data["completions"][0]))
            # stop
            
            # batch_completions = [trial_data["completions"][q_idx] for q_idx in range(num_questions)]
            # print(f"{len(batch_completions)}")
            # print(f"len = {len(batch_completions[0])}")

            nphases_arr = []
            ndepths_arr = []
            for q_idx in range(num_questions):
                nphases_arr.append(trial_data["last_phases"][q_idx])
                ndepths_arr.append(np.mean(trial_data["ndepths_arr"][q_idx]))
                
            nphases_mean = np.mean(nphases_arr)
            nphases_std = np.std(nphases_arr, ddof=1)/np.sqrt(len(nphases_arr)) 
            print(f"nphases_mean: {nphases_mean:0.1f} (\u00B1{nphases_std:0.1f})")
            
            ndepths_mean = np.mean(ndepths_arr)
            ndepths_std = np.std(ndepths_arr, ddof=1)/np.sqrt(len(ndepths_arr)) 
            print(f"ndepths_mean: {ndepths_mean:0.1f} (\u00B1{ndepths_std:0.1f})")
            
    
    total_time = time.time() - start_time
    print(f"it takes {total_time:0.4f}s in total")

def main():
    
    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits"
    
    # llm and prm path
    llm_dir = base_dir + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
    
    llm_tokenizer_dir = base_dir + "/Llama-3.2-1B-Instruct"
    prm_tokenizer_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data"
    
    level = 5
    trial_idx = 0
    
    #  load data 
    dataset = load_data_prm800k_hf(data_dir, split='test')
    dataset = dataset.filter(lambda example: example['level'] == level)
    
    # run search_algo and save results
    config_name = f"mcts--v51--n-8--d-40--lam-10--dalpha-10.0--dbeta-1.0--cpuct-0-2--ppl-True--normalize-True--level-4"
    config_name = f"mcts--e15--n-4--d-40--nb-2--cpuct-2--level-5"
    print(f"config_name = {config_name}--trial-{trial_idx}")
    
    extract_completions_info(config_name, dataset, trial_idx)
    
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()