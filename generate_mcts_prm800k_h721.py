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

from core import mcts_search_extra_v72
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
    
    config.n = 4                      # number of budgets to be generated per depth
    config.beam_width = 4             # number of nodes left after selection
    config.lookahead = 0              # don't use it for now
    config.max_depths = 20            # max depths, after reaching max_depth then terminate search 
    config.sort_completed = False      
    config.filter_duplicates = True   # remove any duplicates in the last list of trajs
    config.date_string = "Aug 1 2025"
    config.seed = 0

    # mcts parameter
    config.num_batches = 4
    # config.step_budget = 10
    config.step_budget = config.num_batches*config.max_depths 
    config.num_phases = 1000
    
    config.lam = 0.01 
    config.normalize_embeds = True
    config.use_ppl = True

    config.ds_beta = 1.0
    config.ds_alpha = 100.0
    config.negative_reward = 0

    config.version = "h72"

    level = 3                                   # level of difficulty of questions
    
    # baseline: gpu_memory_utilization=0.2
    # use the standard model 
    llm_total_gpu = 0.4
    llm_gpu_memory_utilization = 0.2
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
    data_by_levels = load_data_prm800k(data_dir)
    
    
    num_questions = len(data_by_levels[level])  # level 4 has 128 questions
    # num_questions = 1
    num_trials = 5
    print(f"num_questions = {num_questions}")
    print(f"num_trials = {num_trials}")
    
    # get batch of questions ['q1', 'q2', ...]
    batch_of_questions = [data_by_levels[level][q_idx]['problem'] for q_idx in range(num_questions)]

    # # select search algo
    # search_name = 'diverse_reward_search'
    # search_algo = APPROACHES[search_name]
    # print(search_algo)

    # # run search_algo and save results
    # config_name = f"mcts--level-{level}--{config.version}--n-{config.n}--d-{config.max_depths}--nb-{config.num_batches}--lam-{config.lam}--dalpha-{config.ds_alpha}--dbeta-{config.ds_beta}--ppl-{config.use_ppl}--normalize-{config.normalize_embeds}"
    # print(config_name)
    # result_dir = f"results/mcts--level-{level}/{config_name}"
    # try:
    #     os.mkdir(result_dir)
    #     print(f"Directory '{result_dir}' created successfully.")
    # except FileExistsError:
    #     print(f"Directory '{result_dir}' already exists.")
    # except OSError as e:
    #     print(f"Error creating directory: {e}")
    #     stop
            
    
    alpha_list = [1000.0]   
    for alpha in alpha_list:
        config.ds_alpha = alpha
        
        # run search_algo and save results
        config_name = f"mcts--level-{level}--{config.version}--n-{config.n}--d-{config.max_depths}--b-{config.step_budget}--lam-{config.lam}--dalpha-{config.ds_alpha}--dbeta-{config.ds_beta}--ppl-{config.use_ppl}--normalize-{config.normalize_embeds}"
        print(config_name)
        result_dir = f"results/mcts--level-{level}/{config_name}"
        try:
            os.mkdir(result_dir)
            print(f"Directory '{result_dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{result_dir}' already exists.")
        except OSError as e:
            print(f"Error creating directory: {e}")
            stop

        start_time1 = time.time()
        for trial_idx in [1,0]:
            print(f"trial {trial_idx}")
            start_time = time.time()
            # np.random.seed(100000+trial_idx)
            # random.seed(100000+trial_idx)
            # torch.manual_seed(100000+trial_idx)
            # torch.cuda.manual_seed(100000+trial_idx)
            
            # best_of_n(batch_of_questions, config, llm_vllm, random_seeds[trial_idx])
            results = mcts_search_extra_v72._search(batch_of_questions, config, trial_idx, llm_vllm, llm_vllm_embeds, prm)
            with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx}.jsonl", 'w', encoding = 'utf-8') as fout:
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
    
        total_time = time.time() - start_time1
        print(f"it takes {total_time:0.4f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()