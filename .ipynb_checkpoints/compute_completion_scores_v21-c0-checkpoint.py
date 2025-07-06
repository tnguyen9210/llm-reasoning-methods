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

# from utils.load_data import load_data_prm800k
from utils.load_data import load_data_prm800k_hf


def compute_completion_scores(config_name, dataset, prm, num_trials, config):

    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}")

    start_time = time.time()    
    with open(f"results/generate_{config_name}.jsonl", 'r', encoding = 'utf-8') as fin:
        trial_idx = 0
        for line in fin:
            
            if trial_idx >= num_trials:
                break
                
            trial_data = json.loads(line)
            # print(len(trial_data["completions"][0]))
            # stop
            
            batch_completions = [trial_data["completions"][q_idx] for q_idx in range(num_questions)]
            print(f"{len(batch_completions)}")
            print(f"len = {len(batch_completions[0])}")
            
            batch_scores = prm.score(batch_of_questions, batch_completions, batch_size=4)
            # # print(batch_of_questions)
            # batch_scores = []
            # for q_idx in range(len(batch_of_questions)):
            #     # print(f"q_idx: {q_idx} - {len(completions[q_idx])}")
            #     # print([batch_of_questions[q_idx]])
            #     # print([completions[q_idx]])
            #     scores = prm.score([batch_of_questions[q_idx]], [completions[q_idx]])
            #     batch_scores += scores
            #     # print(batch_scores)
            #     # gc.collect();torch.cuda.empty_cache();
            #     # print('#--- memory:', torch.cuda.memory_allocated(2)/(1024**3))
            print(len(batch_scores))
            print(len(batch_scores[0]))
            
            _dataset = dataset.add_column("completions", batch_completions)
            _dataset = _dataset.add_column("scores", batch_scores)
            print(_dataset)
    
            _dataset = score(_dataset, config)
    
            # _dataset.push_to_hub(dataset_id, config_name=f"{config_name}--trial-{trial_idx}", split='test')
            # _dataset.to_json(f"results/{config_name}--trial-{trial_idx}.jsonl")
            _dataset.to_json(f"results/{config_name}.jsonl")
            
            # compute the time
            if trial_idx % 1 == 0:
                total_time = time.time() - start_time
                time_per_trial = total_time/(trial_idx+1)
                time_per_question = time_per_trial/num_questions
                print(f"trial {trial_idx}")
                print(f"it takes {time_per_question:0.4f}s per question")
                print(f"it takes {time_per_trial:0.4f}s per trial")
    
            trial_idx += 1
        
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(GPUS)
    else:
        print("CUDA is not available.")

    prm = RLHFFlow(model_path=prm_tokenizer_dir, device_map='cuda:2')

    # general params
    config = Config()
    config.agg_strategy = 'last'
    config.n = 8
    config.dataset_start = None
    config.dataset_end = None
    
    level = 4
    num_trials = 1
    
    #  load data 
    dataset = load_data_prm800k_hf(data_dir, split=config.dataset_split)
    dataset = dataset.filter(lambda example: example['level'] == level)
    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    
    # run search_algo and save results
    config_name = f"mcts--n-8--d-40--nb-5--lam-10--dalpha-10.0--dbeta-1.0--cpuct-0-2--ppl-True--normalize-True--level-4--v51--trial-1"
    print(f"config_name = {config_name}")
    
    compute_completion_scores(config_name, dataset, prm, num_trials=1, config=config)
    
    # dist.destroy_process_group()

if __name__ == "__main__":
    main()