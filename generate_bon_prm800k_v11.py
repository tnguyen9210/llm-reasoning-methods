import os, psutil, gc
import time 
import json
import pprint

from collections import defaultdict
import random

import torch 
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

from sal.config import Config

from core import bon_search_v01_0_0
from core.reward_models import RLHFFlow

from utils.load_data import load_data_prm800k_hf
# from utils.utils import add_completions_to_dataset_simple

algo_dict = {
    "bon": bon_search_v01_0_0,
}


def main():

    algo_version = "v01_0_0"
    algo_name = "bon"
    algo = algo_dict[algo_name]
    
    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits1"
    
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
    
    config.bs = 256
    config.filter_duplicates = True 
    config.date_string = "Aug 1 2025"
    config.seed = 0

    config.version = algo_version
    
    # baseline: gpu_memory_utilization=0.2
    # use the standard model 
    llm_total_gpu = 0.4
    llm_gpu_memory_utilization = 0.7
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

    # prm = RLHFFlow(model_path=prm_dir, device_map='cuda:0')

    # load data
    level = 1
    num_trials = 5
    print(f"num_trials = {num_trials}")
    
    dataset = load_data_prm800k_hf(data_dir, level)

    num_questions = len(dataset)
    # num_questions = 2
    print(f"num_questions = {num_questions}")
    
    # get batch of questions
    batch_of_questions = [dataset[q_idx]['problem'] for q_idx in range(num_questions)]

    # run search_algo and save results
    config_name = f"bon--level-{level}--{config.version}--bs-{config.bs}"
    print(config_name)
    result_dir = f"results/bon--level-{level}/{config_name}"
    try:
        os.makedirs(result_dir)
        print(f"Directory '{result_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{result_dir}' already exists.")
    except OSError as e:
        print(f"Error creating directory: {e}")
        stop

    start_time1 = time.time()
    for trial_idx in range(num_trials):
        print(f"trial {trial_idx}")
        # start_time = time.time()
        # np.random.seed(100000+trial_idx)
        # random.seed(100000+trial_idx)
        # torch.manual_seed(100000+trial_idx)
        # torch.cuda.manual_seed(100000+trial_idx)
        
        results = algo._search(batch_of_questions, config, trial_idx, llm_vllm)
        with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'w', encoding = 'utf-8') as fout:
            json.dump(results, fout)
            fout.write('\n')
    
        # compute the time
        if trial_idx % 1 == 0:
            total_time = time.time() - start_time1
            time_per_trial = total_time/(trial_idx+1)
            time_per_question = time_per_trial/num_questions
            # print(f"trial {trial_idx}")
            print(f"it takes {time_per_question:0.4f}s per question")
            print(f"it takes {time_per_trial:0.4f}s per trial")

        # add_completions_to_dataset(result_dir, config_name, dataset, prm, trial_idx, config)
    
    total_time = time.time() - start_time1
    print(f"it takes {total_time:0.4f}s in total")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()