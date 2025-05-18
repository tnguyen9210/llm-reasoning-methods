
from collections import defaultdict

import copy 
import time
import numpy as np
import multiprocessing as mp

from dataclasses import dataclass

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from vllm import LLM, SamplingParams

# from sal.search.utils import Beam, build_conv, generate_k_steps, last
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv, generate_k_steps, last


@dataclass
class Beam:
    q_idx: int
    question: str
    templated_conv: str
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0


def best_of_batch(batch_of_questions, config, llm_vllm):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )
    
    completed_beams: list[Beam] = []
    beams: list[Beam] = []
    for q_idx, question in enumerate(batch_of_questions):
        for _ in range(config.n):
            beams.append(
                Beam(
                    q_idx=q_idx,
                    question=question,
                    templated_conv="",
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            ) 

    active_beams = beams
    # for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
    start_time = time.time()
    for it in range(config.num_iterations):
        # print(f"\n-> {it}")
        
        # print(len(active_beams))
        convs = [
            build_conv(b.question, b.current_text, config.system_prompt)
            for b in active_beams
        ]

        add_generation_prompt = it == 0
        continue_final_message = it > 0
    
        tokenizer = llm_vllm.get_tokenizer()
    
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
            
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )

        # Last iteration, generate to EOS
        if it == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        lookahead = 0 if it == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm_vllm, sampling_params, 1
        )
        # total_time = time.time() - start_time
        # print(f"it takes {total_time:0.4f}s")

        # Collecct gen_results into beams
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += gen_result.next_texts[0]
            # beam.history.append(beam.next_texts[0])
            beam.templated_prompt = gen_result.prompt
            # pprint.pprint(gen_result)
            # print(f"beam.next_texts = {beam.next_texts}")
            # print(f"beam.stop_reasons = {beam.stop_reasons}")
            # print(f"beam.lookahead_texts = {beam.lookahead_texts}")
            # print(f"beam.lookahead_texts = {beam.lookahead_texts}")
            # stop
            
            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
                # continue
        
        # Filter out comleted beams 
        active_beams = [b for b in active_beams if not b.completed]
        # print(len(active_beams))

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            print("break")
            break
        
        batch_of_beams = [[] for _ in range(len(batch_of_questions))]
        for b_idx, beam in enumerate(active_beams):
            batch_of_beams[beam.q_idx].append(beam)

        extended_beams = []
        for q_idx in range(len(batch_of_questions)):
            m = len(batch_of_beams[q_idx])
            # print(f"m = {m}")
            if m == 0:
                continue
                
            extended_beams += batch_of_beams[q_idx]
            if m < config.n:
                extended_idxes = np.random.randint(0, m, config.n-m)
                # print(f"m = {m}: {extended_idxes}")
                # for idx in extended_idxes:
                #     extended_beams.append(copy.deepcopy(batch_of_beams[q_idx][idx]))
                extended_beams += [copy.deepcopy(batch_of_beams[q_idx][idx]) for idx in extended_idxes]

        active_beams = extended_beams
                
    # Filter duplicate active beams
    if config.filter_duplicates:
        # Create a dictionary to filter duplicates and retain order
        unique_beam_dict = {}
        for i, b in enumerate(completed_beams):
            if b.current_text not in unique_beam_dict:
                unique_beam_dict[b.current_text] = (
                    i  # Map the unique text to its index
                )
        completed_beams = [completed_beams[i] for i in unique_beam_dict.values()]
            
    # Collect the completions from beams
    completions = [[] for _ in range(len(batch_of_questions))]
    # completion_ntokens = [[] for _ in range(len(batch_of_questions))]
    
    for beam in completed_beams:
        completions[beam.q_idx].append(beam.current_text)
        # completion_ntokens[beam.q_idx].append(beam.current_text)

    results = defaultdict(list)
    results["completions"] = completions
    # results["completion_ntokens"] = completion_ntokens
    
    return results

