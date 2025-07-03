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

from core import mcts_search_v21
from core.reward_models import RLHFFlow

from utils.load_data import load_data_prm800k


# APPROACHES = {
#     # "bon": best_of_n.best_of_n_v11,
#     "diverse_reward_search": diverse_reward_search.diverse_search
# }

def main():
    
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
    
    config.n = 8                      # number of budgets to be generated per depth
    config.beam_width = 4             # number of nodes left after selection
    config.lookahead = 0              # don't use it for now
    config.max_depths = 40            # max depths, after reaching max_depth then terminate search 
    config.sort_completed = False      
    config.filter_duplicates = True   # remove any duplicates in the last list of trajs
    config.seed = 0

    # mcts parameter
    config.num_phases = 100
    config.batch_budget = config.max_depths 
    
    config.lam = 10 
    config.normalize_embeds = True

    config.cpuct = 2
    config.ds_beta = 1.0
    config.ds_alpha = 100.0
    config.use_ppl = True

    config.version = "v21"
    
    # baseline: gpu_memory_utilization=0.2
    # use the standard model 
    llm_gpu_memory_utilization = 0.5
    # llm_vllm = LLM(
    #     model = llm_dir,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization = 0.7,  # Utilize 50% of GPU memory
    #     # enable_prefix_caching=True,  # V100 doesn't support enable_prefix_caching 
    #     # enable_chunked_prefill=False, # and enable_chunked_prefill
    #     max_model_len = 5000,
    #     dtype = "float16",
    #     seed = config.seed)
    
    llm_vllm = LLM(
        model=llm_dir, 
        tensor_parallel_size=1, 
        # trust_remote_code=True,
        swap_space=16,
        max_model_len=5000,
        gpu_memory_utilization=llm_gpu_memory_utilization,
        # enforce_eager=True,
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
        gpu_memory_utilization=0.98-llm_gpu_memory_utilization,
        # enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=0,
    )
    
    prm = RLHFFlow(model_path=prm_dir, device_map='cuda:1')

    #  load data 
    data_by_levels = load_data_prm800k(data_dir)
    
    level = 4                                   # level of difficulty of questions
    num_questions = len(data_by_levels[level])  # level 4 has 128 questions
    # num_questions = 2
    num_trials = 5
    print(f"num_questions = {num_questions}")
    print(f"num_trials = {num_trials}")
    
    # get batch of questions ['q1', 'q2', ...]
    batch_of_questions = [data_by_levels[level][q_idx]['problem'] for q_idx in range(num_questions)]

    # # select search algo
    # search_name = 'diverse_reward_search'
    # search_algo = APPROACHES[search_name]
    # print(search_algo)

    # run search_algo and save results
    config_name = f"mcts--n-{config.n}--d-{config.max_depths}--lam-{config.lam}--dalpha-{config.ds_alpha}--dbeta-{config.ds_beta}--cpuct-{config.cpuct}--ppl-{config.use_ppl}--normalize-{config.normalize_embeds}--level-{level}--{config.version}"
    print(config_name)
            
    start_time = time.time()
    for trial_idx in range(3,5):
        np.random.seed(100000+trial_idx)
        random.seed(100000+trial_idx)
        torch.manual_seed(100000+trial_idx)
        torch.cuda.manual_seed(100000+trial_idx)
        
        # best_of_n(batch_of_questions, config, llm_vllm, random_seeds[trial_idx])
        results = mcts_search_v21._search(batch_of_questions, config, llm_vllm, llm_vllm_embeds, prm)
        with open(f"results/generate_{config_name}--trial-{trial_idx}.jsonl", 'w', encoding = 'utf-8') as fout:
            json.dump(results, fout)
            fout.write('\n')
    
        # compute the time
        if trial_idx % 1 == 0:
            total_time = time.time() - start_time
            time_per_trial = total_time/(trial_idx+1)
            time_per_question = time_per_trial/num_questions
            print(f"trial {trial_idx}")
            print(f"it takes {time_per_question:0.4f}s per question")
            print(f"it takes {time_per_trial:0.4f}s per trial")
    
    total_time = time.time() - start_time
    print(f"it takes {total_time:0.4f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()