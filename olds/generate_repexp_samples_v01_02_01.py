'''
This file implements Algorithm 1 (RepExp) from the Tuyls 2025's paper 
This reads a batch of generated completions, extracts their hidden state representations using vLLM, and applies 
the RepExp algorithm to greedily select a maximally diverse subset.

v01_01_01: base
v01_02_01: alternative from v01_01_01. Changed from response-only encoding to contextual encoding 
    continue_final_message=True: this does not include the <|eot_id|> at the end of response 
'''

import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL+1)

import time 
import json
import pprint
import importlib

import random
import numpy as np
np.set_printoptions(precision=4)
 
from utils import load_data

import torch 
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

from sal.config import Config
from sal.search.utils import build_conv

from tqdm import tqdm

def _repexp_select(all_embeds, k, embeds_dim=2048, lam=1.0):
    """
    Algorithm 1: Representation-Based Exploration (RepExp)
    
    This function iteratively selects a diverse set of generations based on 
    their hidden state representations.
    
    Args:
        all_embeds (np.ndarray): An array of shape (N, d), representing the 
                                 hidden states for N generated sequences. 
                                 (e.g., projected down to d=512 and mean-centered).
        k (int): The budget (number of responses to select).
        lam (float): The regularization parameter (lambda) for the inverse 
                       covariance matrix initialization.
        
    Returns:
        A_idxes (List[int]): A list of arm indices corresponding to the selected diverse generations.
    """
    N = len(all_embeds)
    
    A_idxes = []
    A_idxes_arr = np.zeros(N, dtype=bool)

    # Initialize inverse covariance
    Lam_t = (1/lam)*np.eye(embeds_dim)

    # Initialize A_idxes <- {a_1}, a_1 ~ Unif(N)
    a_t = np.random.randint(0, N)
    A_idxes.append(a_t)
    A_idxes_arr[a_t] = True

    
    for t in range(1, k):
        # Update the inverse covariance matrix using the Sherman-Morrison formula
        # Λ_t <- Λ_{t-1} - Λ_{t-1} h h^T Λ_{t-1} / (1 + h^T Λ_{t-1} h)
        h_t = all_embeds[a_t]
        Lamh_t = Lam_t @ h_t
        
        numer = np.outer(Lamh_t, Lamh_t)
        denom = 1.0 + (h_t @ Lamh_t)
        Lam_t = Lam_t - numer/denom 
        
        # a_{t+1} = argmax_a h(a)^T Λ_t h(a)
        scores = np.sum(all_embeds * (all_embeds @ Lam_t), axis=1)

        # Mask already selected items
        scores[A_idxes_arr] = -np.inf

        a_t = int(np.argmax(scores))
        A_idxes.append(a_t)
        A_idxes_arr[a_t] = True 

    return A_idxes


def generate_repexp_samples(dataset_orig, result_dir, config_name, trial_idx, llm_vllm_embeds, config):

    input_file = f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl"
    output_file = f"{result_dir}/repexp_{config_name}--addprompt-{config.embeds_addprompt}--embstrat-{config.embeds_strategy}--trial-{trial_idx:03d}.jsonl"
    print(f"input_file = {input_file}")
    print(f"output_file = {output_file}")
    with open(input_file, 'r', encoding='utf-8') as fin:
        gen_results = json.load(fin)

    num_questions = len(gen_results["completions"])
    # num_questions = 2
    repexp_results = {"completions": []}

    tokenizer = llm_vllm_embeds.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    
    for q_idx in tqdm(range(num_questions), desc="Processing questions"):
        question = dataset_orig[q_idx]['problem']
        q_completions = gen_results["completions"][q_idx]
        num_completions = len(q_completions)

        cand_convs = [build_conv(question, completion, config.system_prompt) for completion in q_completions]
        cand_templated_conv = tokenizer.apply_chat_template(
            cand_convs,
            add_generation_prompt=False,
            continue_final_message=True if config.embeds_addprompt == 'v01' else False,
            date_string=config.date_string,
            tokenize=False,
        )
        # print(cand_templated_conv[0])

        # Get embeddings via vLLM
        outputs = llm_vllm_embeds.encode(cand_templated_conv, pooling_task="token_embed", use_tqdm=False)

        # Extract and format embeddings into an (N,d) array
        outputs_embeds = []
        for o in outputs:
            embeds_array = o.outputs.data.detach().cpu().numpy()

            if config.embeds_strategy == 'avg':
                embed_array = np.mean(embeds_array, axis=0)
            elif config.embeds_strategy == 'last':
                embed_array = embeds_array[-1]

            outputs_embeds.append(embed_array)

        all_embeds = np.vstack(outputs_embeds) # shape: (N, embed_dim)

        # Mean-centering the representations 
        all_embeds = all_embeds - np.mean(all_embeds, axis=0)

        # Apply _repexp_select to get the repexp ordering
        repexp_idxes = _repexp_select(all_embeds, k=num_completions, embeds_dim=config.embeds_dim, lam=config.lam)

        # Reorder the actual text completions based on the RepExp indices
        repexp_completions = [q_completions[i] for i in repexp_idxes]
            
        # Append to new dataset 
        repexp_results["completions"].append(repexp_completions)

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(repexp_results, fout)


def main():
    
    # base_dir
    base_dir = '/groups/chichengz/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/prm800k/math_splits"
    
    # llm and prm path
    # llm_dir = base_dir + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    # prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
    
    llm_dir = base_dir + "/Llama-3.2-1B-Instruct"
    prm_dir = base_dir + "/Llama3.1-8B-PRM-Deepseek-Data"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # os.environ["VLLM_LOGGING_LEVEL"] = "101"
    
    if torch.cuda.is_available():
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print(GPUS)
    else:
        print("CUDA is not available.")

    # General params
    config = Config()
    config.date_string = "Aug 1 2025"
    config.seed = 0
    
    # Repexp parameters 
    config.lam = 1.0 
    config.embeds_centering = True
    config.embeds_normalizing = True
    config.embeds_strategy = 'last'
    config.embeds_addprompt = 'v01'
    
    config.embeds_dim = 2048

    llm_total_gpu = 0.7
    llm_gpu_memory_utilization = 0.2
    llm_vllm_embeds = LLM(
        model=llm_dir, 
        tensor_parallel_size=1, 
        # trust_remote_code=True,
        # task="embed",
        runner="pooling",
        swap_space=16,
        max_model_len=5000,
        gpu_memory_utilization=llm_total_gpu-llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        disable_log_stats=True,
        dtype="float16",
        seed=config.seed,
    )
    
    level = 3
    num_trials = 4

    dataset_orig = load_data.load_data_prm800k_hf(data_dir, split='test')
    dataset_orig = dataset_orig.filter(lambda example: example['level'] == level)
    num_questions = len(dataset_orig)
    
    config_name = f"bon--level-{level}--v01_0_0--bs-256"
    result_dir = f"results/bon--level-{level}/{config_name}"
    print(f"config_name = {config_name}")
    
    for trial_idx in range(num_trials):
        generate_repexp_samples(dataset_orig, result_dir, config_name, trial_idx, llm_vllm_embeds, config)
        
if __name__ == "__main__":
    main()