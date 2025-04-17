
import os
import copy

from datasets import Dataset, load_dataset

import torch
import torch.distributed as dist

from vllm import LLM

from sal.config import Config
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

# from sal.models.reward_models import load_prm
from core.reward_models import RLHFFlow

from utils.load_data import load_data_prm800k_hf

from datasets import Dataset, load_dataset

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


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

    # config
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    # config.approach = "best_of_n"
    config.n = 16
    config.search_batch_size = 25 
    # config.sort_completed = True
    # config.filter_duplicates = True 
    config.dataset_split = 'test'
    config.seed = 0
    config.version = "v01"
    # config.dataset_start = 0
    # config.dataset_end  = 200

    approach_fn = APPROACHES[config.approach]

    level = 4
    
    dataset_id = "tnguyen9210/LLM-Reasoning-Math-500"
    config_name = f"bon--n-{config.n}--level-{level}--{config.dataset_split}--{config.version}"
    if config.dataset_start is not None and config.dataset_end is not None:
        config_name = f"{config_name}--chunk-{config.dataset_start}_{config.dataset_end}"

    #  load data 
    # data_by_levels = load_data_prm800k(data_dir)
    # dataset = load_dataset(config.dataset_name, split=config.dataset_split, cache_dir=data_dir)
    dataset = load_data_prm800k_hf(data_dir, split=config.dataset_split)
    if level != "all":
        dataset = dataset.filter(lambda example: example['level'] == level)
        
    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    
    print(f"{level}: {len(dataset)}")
    print(f"approach = {config.approach}")
    print(f"config_name = {config_name}")
    
    # load llm and prm
    num_gpus = torch.cuda.device_count()
    # baseline: gpu_memory_utilization=0.2
    # use the standard model 
    llm_vllm = LLM(
        model = llm_tokenizer_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization = 0.7,  # Utilize 50% of GPU memory
        # enable_prefix_caching=True,  # V100 doesn't support enable_prefix_caching 
        # enable_chunked_prefill=False, # and enable_chunked_prefill
        max_model_len = 5000,
        dtype = "float16",
        seed = config.seed)
    # prm = load_prm(config)

    prm = RLHFFlow(model_path=prm_tokenizer_dir, device_map='cuda:1')    

    num_trials = 5

    for trial_idx in range(num_trials):
        torch.manual_seed(100000+trial_idx)
        torch.cuda.manual_seed(100000+trial_idx)

        _dataset = copy.deepcopy(dataset)
        _dataset = _dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm_vllm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )

        _dataset = score(_dataset, config)

        # _dataset.push_to_hub(dataset_id, config_name=f"{config_name}--trial-{trial_idx}")
        _dataset.to_json(f"results/{config_name}--trial-{trial_idx}.jsonl")

        del(_dataset)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()