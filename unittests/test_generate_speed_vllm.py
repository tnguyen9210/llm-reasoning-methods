import os, psutil
import time 
import gc

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

# base_path
base_path = '/groups/kjun/tnn/datasets/'

# dataset path
dataset_path = base_path + "/prm800k/math_splits"

# llm and prm path
llm_path = base_path + "/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
prm_path = base_path + "/Llama3.1-8B-PRM-Deepseek-Data-GGUF/Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"

llm_tokenizer_path = base_path + "/Llama-3.2-1B-Instruct"
prm_tokenizer_path = base_path + "/Llama3.1-8B-PRM-Deepseek-Data"

if torch.cuda.is_available():
    GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    print(GPUS)
    for gpu_index in  GPUS:
        print(f"\n-> gpu {gpu_index}")
        gpu_index = int(gpu_index)
        # gpu_index = 0  # Change this if you have multiple GPUs
        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu_index)
        allocated_memory = torch.cuda.memory_allocated(gpu_index)
        free_memory = reserved_memory - allocated_memory
    
        print(f"Total GPU Memory: {total_memory / 1024 ** 3:.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory / 1024 ** 3:.2f} GB")
        print(f"Available GPU Memory: {free_memory / 1024 ** 3:.2f} GB")
else:
    print("CUDA is not available.")

# baseline: gpu_memory_utilization=0.2
llm_regular = LLM(
    model = llm_tokenizer_path,
    # tensor_parallel_size=1,
    gpu_memory_utilization = 0.2,  # Utilize 50% of GPU memory
    max_model_len = 5000,
    dtype = "float16",
    seed = 123)

gc.collect();torch.cuda.empty_cache();
print('#--- memory:', torch.cuda.memory_allocated(0)/(1024**3))
print('#--- memory:', torch.cuda.memory_allocated(1)/(1024**3))