
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
from sal.utils.score import score, aggregate_scores
from core.reward_models import RLHFFlow

from datasets import load_dataset

from utils.load_data import load_data_prm800k_hf
from utils.utils import add_completions_to_dataset

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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(GPUS)
    else:
        print("CUDA is not available.")

    prm = RLHFFlow(model_path=prm_dir, device_map='cuda:3')

    # general params
    config = Config()
    config.agg_strategy = 'last'

    level = 4
    num_trials = 20
    q_idx = 0
    
    #  load data 
    dataset_orig = load_data_prm800k_hf(data_dir, level, q_idx)
    
    # run search_algo and save results
    config_name = f"mcts--level-4--q-000--q71--n-4--d-20--b-400--lam-0.01--dalpha-100.0--dbeta-1.0--ppl-True--normalize-True"
    # config_name = f"mcts--level-4--q-000--q31--n-4--d-20--b-80--cpuct-2"
    result_dir = f"results/mcts--level-{level}/prompts/q-{q_idx:03d}/{config_name}"
    # config_name = f"mcts--level-4--e21--n-4--d-10--nb-8--cpuct-2"
    # result_dir = f"results/mcts--level-{level}/{config_name}"
    print(f"config_name = {config_name}")

    for trial_idx in range(num_trials):
        add_completions_to_dataset(result_dir, config_name, dataset_orig, prm, trial_idx, config=config)
    
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()