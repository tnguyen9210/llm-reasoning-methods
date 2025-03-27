import os, psutil, gc
import time 
import json
import pprint

from collections import defaultdict
import random


import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

# from core.reward_models import RLHFFlow, SkyworkO1
from core import reward_models

from core import best_of_n
from utils.load_data import load_data_prm800k


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

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    if torch.cuda.is_available():
        GPUs = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(f"GPUs = {GPUs}")
    else:
        print("CUDA is not available.") 

    prm_dir = base_dir + "/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    prm = reward_models.SkyworkO1(model_path=prm_dir, device_map="auto")
    
    gc.collect();torch.cuda.empty_cache();
    print('#--- memory:', torch.cuda.memory_allocated(0)/(1024**3))
    print('#--- memory:', torch.cuda.memory_allocated(1)/(1024**3))
    print('#--- memory:', torch.cuda.memory_allocated(2)/(1024**3))
    print('#--- memory:', torch.cuda.memory_allocated(3)/(1024**3))

    data_by_levels = load_data_prm800k(data_dir)
    

    # general params
    config = Config()
    config.n = 16
    config.agg_strategy = 'last'
    
    level = '1'
    num_questions = len(data_by_levels[level])
    num_questions = 20
    num_trials = 1
    print(f"num_questions = {num_questions}")
    
    # get batch of questions
    batch_of_questions = [data_by_levels[level][q_idx]['problem'] for q_idx in range(num_questions)]
    
    # load completions 
    completions_dir = f"results/generate_bon_prm800k_level{level}_n16_v11.jsonl"
    scores_dir = f"../results/scores_bon_prm800k_level{level}_n16_v11.jsonl"
    
    # compute results
    fout = open(scores_dir, 'w', encoding = 'utf-8')
    start_time = time.time()
    with open(scores_dir, 'w', encoding = 'utf-8') as fout:
        with open(completions_dir, 'r', encoding = 'utf-8') as fin:
            trial_idx = 0
            for line in fin:
                if trial_idx >= num_trials:
                    break
                    
                trial_data = json.loads(line)
        
                # Compute the scores of completions
                # print(len(trial_data["completions"]))
                # print(len(batch_of_questions))
                # print(trial_data["completions"][0][0])
                scores = prm.score(batch_of_questions, trial_data["completions"][:num_questions])
                agg_scores = [
                    [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
                ]
                # print(agg_scores)
                
                trial_data["agg_scores"] = agg_scores
                # print(trial_data.keys())
                json.dump(trial_data, fout)
                fout.write('\n')
        
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

    dist.destroy_process_group()



if __name__ == "__main__":
    main()
