import time
import json
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

from sal.utils.score import score, aggregate_scores
from sal.search.utils import build_conv, generate_k_steps, last

import logging


def extract_completions_embeds(result_dir, config_name, trial_idx, dataset, llm_vllm_embeds, config):
    logging.error(f"config_name = {config_name}")
    tokenizer = llm_vllm_embeds.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    logging.error(f"num_questions = {num_questions}")

    trial_embeds = []
    with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'r', encoding = 'utf-8') as fin:
        for line in fin:
            trial_data = json.loads(line)

            for q_idx, question in enumerate(batch_of_questions):
                q_completions = trial_data["completions"][q_idx]
                q_convs = [build_conv(question, completion, config.system_prompt) for completion in q_completions]
                q_templated_convs = tokenizer.apply_chat_template(
                    q_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    date_string=config.date_string,
                    tokenize=False,
                )
                # logging.error(q_templated_convs[0])
                # logging.error(q_templated_convs[1])
                # stop

                outputs = llm_vllm_embeds.encode(q_templated_convs, use_tqdm=False)
                outputs_embeds = []
                for o in outputs:
                    trial_embeds.append(o.outputs.data.detach().cpu().numpy())
                # logging.error(outputs_embeds[:2])
                # outputs_embeds = torch.stack(outputs_embeds, dim=0)
                # logging.error(outputs_embeds.shape)
                # logging.error(len(outputs))
                # logging.error(outputs[0])
                # outputs_embeds = outputs[0].outputs.data
                # logging.error(outputs_embeds.shape)
                # if config.normalize_embeds:
                #     outputs_embeds = F.normalize(outputs_embeds, p=2,dim=-1)
                    

    return trial_embeds 


def extract_steps_embeds(result_dir, config_name, trial_idx, dataset, llm_vllm_embeds, config):
    logging.error(f"config_name = {config_name}")
    tokenizer = llm_vllm_embeds.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    logging.error(f"num_questions = {num_questions}")

    trial_embeds = []
    with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'r', encoding = 'utf-8') as fin:
        for line in fin:
            trial_data = json.loads(line)

            for q_idx, question in enumerate(batch_of_questions):
                q_completions = trial_data["completions"][q_idx]
                q_convs = [build_conv(question, completion, config.system_prompt) for completion in q_completions]
                q_templated_convs = tokenizer.apply_chat_template(
                    q_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    date_string=config.date_string,
                    tokenize=False,
                )
                # logging.error(q_templated_convs[0])
                # logging.error(q_templated_convs[1])
                # stop

                outputs = llm_vllm_embeds.encode(q_templated_convs, use_tqdm=False)
                outputs_embeds = []
                for o in outputs:
                    trial_embeds.append(o.outputs.data.detach().cpu().numpy())
                # logging.error(outputs_embeds[:2])
                # outputs_embeds = torch.stack(outputs_embeds, dim=0)
                # logging.error(outputs_embeds.shape)
                # logging.error(len(outputs))
                # logging.error(outputs[0])
                # outputs_embeds = outputs[0].outputs.data
                # logging.error(outputs_embeds.shape)
                # if config.normalize_embeds:
                #     outputs_embeds = F.normalize(outputs_embeds, p=2,dim=-1)
                    
    return trial_embeds 


def add_completions_to_dataset_simple(result_dir, config_name, dataset, prm, trial_idx, config):

    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}")

    # start_time = time.time()    
    with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'r', encoding = 'utf-8') as fin:
        for line in fin:
            trial_data = json.loads(line)

            batch_cphases = []
            batch_cdepths = []
            batch_last_step = []
            batch_last_phase = [] 
            batch_tdepths = []
            batch_nnodes_max_depth = [] 
                
            batch_completions = [trial_data["completions"][q_idx] for q_idx in range(num_questions)]
            # batch_completions = trial_data["completions"]
            # print(f"len = {len(batch_completions[0])}")
            
            batch_scores = prm.score(batch_of_questions, batch_completions, batch_size=4)
            # print(len(batch_scores))
            # print(len(batch_scores[0]))
            
            _dataset = dataset.add_column("completions", batch_completions)
            _dataset = _dataset.add_column("scores", batch_scores)

            _dataset = score(_dataset, config)
    
            # _dataset.push_to_hub(dataset_id, config_name=f"{config_name}--trial-{trial_idx}", split='test')
            # _dataset.to_json(f"results/{config_name}--trial-{trial_idx}.jsonl")
            _dataset.to_json(f"{result_dir}/{config_name}--trial-{trial_idx:03d}.jsonl")
            
            # # compute the time
            # total_time = time.time() - start_time
            # time_per_trial = total_time/(trial_idx+1)
            # time_per_question = time_per_trial/num_questions
            # print(f"trial {trial_idx}")
            # print(f"it takes {time_per_question:0.4f}s per question")
            # print(f"it takes {time_per_trial:0.4f}s per trial")
            
    # total_time = time.time() - start_time
    # print(f"it takes {total_time:0.4f}s in total")


def add_completions_to_dataset_tree(result_dir, config_name, dataset, prm, trial_idx, config):

    # get batch of questions
    batch_of_questions = [data['problem'] for data in dataset]
    num_questions = len(batch_of_questions)
    print(f"num_questions = {num_questions}")

    # start_time = time.time()    
    with open(f"{result_dir}/generate_{config_name}--trial-{trial_idx:03d}.jsonl", 'r', encoding = 'utf-8') as fin:
        for line in fin:
            trial_data = json.loads(line)

            batch_cphases = []
            batch_cdepths = []
            batch_last_step = []
            batch_last_phase = [] 
            batch_tdepths = []
            batch_nnodes_max_depth = [] 
                
            batch_completions = [trial_data["completions"][q_idx] for q_idx in range(num_questions)]
            # batch_completions = trial_data["completions"]
            # print(f"len = {len(batch_completions[0])}")
            
            batch_scores = prm.score(batch_of_questions, batch_completions, batch_size=4)
            # print(len(batch_scores))
            # print(len(batch_scores[0]))
            
            _dataset = dataset.add_column("completions", batch_completions)
            _dataset = _dataset.add_column("scores", batch_scores)
            _dataset = _dataset.add_column("csteps", trial_data["c_step_cnts"])
            _dataset = _dataset.add_column("cdepths", trial_data["c_depths"])
            _dataset = _dataset.add_column("cphases", trial_data["c_phases"])
            _dataset = _dataset.add_column("tdepths", trial_data["ndepths_arr"])
            _dataset = _dataset.add_column("last_step", trial_data["step_cnts"])
            _dataset = _dataset.add_column("last_phase", trial_data["last_phases"])
            _dataset = _dataset.add_column("nnodes_max_depth", trial_data["cnt_max_depth"])
            # print(_dataset)

            _dataset = score(_dataset, config)
    
            # _dataset.push_to_hub(dataset_id, config_name=f"{config_name}--trial-{trial_idx}", split='test')
            # _dataset.to_json(f"results/{config_name}--trial-{trial_idx}.jsonl")
            _dataset.to_json(f"{result_dir}/{config_name}--trial-{trial_idx:03d}.jsonl")
            
            # # compute the time
            # total_time = time.time() - start_time
            # time_per_trial = total_time/(trial_idx+1)
            # time_per_question = time_per_trial/num_questions
            # print(f"trial {trial_idx}")
            # print(f"it takes {time_per_question:0.4f}s per question")
            # print(f"it takes {time_per_trial:0.4f}s per trial")
            
    # total_time = time.time() - start_time
    # print(f"it takes {total_time:0.4f}s in total")