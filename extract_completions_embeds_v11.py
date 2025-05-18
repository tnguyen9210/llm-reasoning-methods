import os, psutil, gc
import signal
import json
import pickle
import pprint
import time 
from tqdm import tqdm

import re

from collections import defaultdict
import random

import numpy as np 

import torch 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.search.utils import build_conv, generate_k_steps, last

# from core.reward_models import RLHFFlow

from datasets import Dataset, load_dataset

from utils import grader2
from utils import parser


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()
    
def run_with_timeout(fn_extract_answer, fn_grade, completion, gt_answer, timeout=2):
    # Set the signal handler for SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Schedule an alarm after `timeout` seconds
    try:
        c_answer = fn_extract_answer(completion, 'math')
        result = fn_grade(c_answer, gt_answer)
    except TimeoutException:
        print(f"Timeout: {completion}")
        c_answer = None
        result = None
    finally:
        signal.alarm(0)  # Cancel alarm if function returns early
    return c_answer, result



def extract_completion_embeds(config_name, level, tokenizer, llm_tf, config, tqdm_disable=False):
    all_data = []

    # go through each problem/prompt 
    dataset_by_level = load_dataset("json", data_files = f"results/{config_name}.jsonl", split='train')
    
    for q_idx, data in enumerate(tqdm(dataset_by_level, desc="Processing questions", disable=tqdm_disable)):
        # if q_idx > 2:
        #     continue
        # pprint.pprint(data)
        # print(len(data["scores"]))
    
        # extract the problem and grounth truth answer (gt_answer)
        problem = data["problem"]
        gt_cot, gt_answer  = parser.parse_ground_truth(data, 'math')
    
        # go through each completion
        cnt = 0
        for c_idx, completion in enumerate(tqdm(data['completions'], desc="Processing completion", disable=True)):
            # if depth >= len(scores):
            #     continue
            # if c_idx > 2:
            #     continue
    
            # check whether the completion provides the 
            c_answer, is_correct = run_with_timeout(parser.extract_answer, grader2.math_equal, completion, gt_answer)
            if is_correct is None: # skip the completion that can not be evaluated
                continue 
    
            scores = data["scores"][c_idx]
            
            # split the completion into steps by double newlines
            steps = completion.split("\n\n")
            if len(scores) != len(steps):
                continue
                
            conversation =  []
            current_text = ""
            
            for s_idx, step in enumerate(steps):
                # add step to current_text
                # print(f"\n-> s_idx = {s_idx}")
                if s_idx == 0:
                    current_text += step
                else:
                    current_text += "\n\n" + step 
                # print(current_text)
                convs = [
                    build_conv(problem, current_text, config.system_prompt)
                ]            
    
                templated_convs = tokenizer.apply_chat_template(
                    convs,
                    add_generation_prompt=False,
                    continue_final_message=True, # if False, add <|eot_id|> into the message
                    tokenize=False,
                )
                # print(templated_convs[0])
                
                inputs = tokenizer(templated_convs[0], return_tensors="pt").to(llm_tf.device)
    
                with torch.no_grad():
                    outputs = llm_tf(**inputs, output_hidden_states=True)
        
                    # Get last_token_embeds
                    last_hidden_state = outputs.hidden_states[-1]
                    last_token_embeds = last_hidden_state[:, -1, :].squeeze(0).detach().cpu().numpy()
                    
                    # Compute otuput_log_prob
                    # Prepare labels: shift input_ids to the right by one
                    labels = inputs['input_ids'][:, 1:]   
                    shifted_logits = outputs.logits[:, :-1, :]
                    loss_fct = CrossEntropyLoss(reduction='sum')
                    completion_log_prob = -loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1)).detach().cpu().numpy()

                x = defaultdict()
                x["problem"] = problem                                   # 
                x["level"] = level                                       # the difficulty level
                x["step_num"] = s_idx                                    # the current step #
                x["current_text"] = current_text                         # the current text includes all steps up to this point
                x["is_completed"] = 1 if s_idx == len(steps) - 1 else 0  # whether the current step is the last step 
                x["gt"] = gt_answer                                      # ground-truth answer
                x["pred"] = c_answer                                     # prediction extracted from the trajectory/completion
                x["is_correct"] = is_correct                             # whether the trajectory/completion leads to the correct answer 
                x["embeds"] = last_token_embeds                          # the hidden embeds of the last token
                x["log_prob"] = completion_log_prob                      # the log probability of the completion
                x["prm_scores"] = scores[s_idx]
    
                all_data.append(x)

    with open(f"results/{config_name}.pkl", 'wb') as fout:
        pickle.dump(all_data, fout)
        

def main():
    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits"
    # data_dir = base_dir + "/math500"
    
    # llm and prm path
    llm_dir = base_dir + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
    
    llm_tokenizer_dir = base_dir + "/Llama-3.2-1B-Instruct"
    prm_tokenizer_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
    llm_tf = AutoModelForCausalLM.from_pretrained(llm_tokenizer_dir).to("cuda:0")
    # model_regular.generation_config.pad_token_id = tokenizer.eos_token_id
    gc.collect();torch.cuda.empty_cache();
    print('#--- memory:', torch.cuda.memory_allocated(3)/(1024**3))

    config = Config()
    level = 1
    config_name = f"bon--n-256--level-{level}--train--v01--chunk-0_200--trial-0"
    print(f"config_name  = {config_name}")
    extract_completion_embeds(config_name, level, tokenizer, llm_tf, config)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()