'''
If candidates have the same diversity scores than choose the one with the smallest negative loglikelihood.
Update the _diverse_select function to combine PRM and diversity scores, each weighted by its respective coefficient.
'''

from collections import defaultdict

import copy 
import time
import numpy as np
import multiprocessing as mp

from dataclasses import dataclass

import torch
import torch.nn as nn
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


def _diverse_select(K, V, q_embeds, q_nll, q_ppl, q_scores, ds_alpha, ds_beta):
    num_arms = len(q_embeds)
    _V = copy.deepcopy(V)
    A_idxes = []
    A_embeds = []
    tol = 0.0001
    for it in range(K):
        _V_inv = np.linalg.inv(_V)
        q_diversity = np.einsum('ij,jk,ik->i', q_embeds, _V_inv, q_embeds)
        q_vals = ds_beta*q_scores + ds_alpha*q_diversity
        max_val = np.max([val for idx, val in enumerate(q_vals) if idx not in A_idxes])
        # candidate_idxes = np.where(np.abs(q_vals-max_val) < tol)[0]
        candidate_idxes = [
            arm_idx for arm_idx, arm_val in enumerate(q_vals)
            if (np.abs(max_val - arm_val) <= tol) and (arm_idx not in A_idxes)
        ]

        # best_idx = min(candidate_idxes, key=lambda i: q_nll[i])
        best_idx = min(candidate_idxes, key=lambda i: q_ppl[i])
        # print(q_vals)
        # print(q_nll)
        # print(candidate_idxes)
        # print(best_idx)
        
        best_embeds = q_embeds[best_idx]
        best_embeds = best_embeds.reshape(-1, 1)
        # print(best_embeds.shape)

        # update V
        _V = _V + np.matmul(best_embeds, best_embeds.T)

        # update A
        A_idxes.append(best_idx)

        # print(_V.shape)
        # print(max_val)
        # print(max_idx)
        # print(A_idxes)

    return A_idxes

def process_diverse_select(K, V, q_idx, q_active_beams, q_embeds, q_log_probs, q_ppl, q_scores, ds_alpha, ds_beta):
    # V = config.lam*np.eye(2048)
    # K = int(config.n/config.beam_width)

    if len(q_active_beams) <= K:
        return (q_idx, None)

    selected_idxes  = _diverse_select(K, V, q_embeds, q_log_probs, q_ppl, q_scores, ds_alpha, ds_beta)

    # for idx, beam in enumerate(q_active_beams):
    #     if idx not in selected_idxes:
    #         beam.pruned = True 
            
    return (q_idx, selected_idxes)
    # return (q_idx, selected_idxes) 

def diverse_search(batch_of_questions, config, llm_vllm, llm_tf, llm_tokenizer, prm):
    # print("diverse_search")
    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    V = config.lam*np.eye(2048)
    # K = int(config.n / config.beam_width)
    K = config.beam_width
    
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

    # for i in tqdm(range(config.num_depths), desc="Beam search iterations"):
    active_beams = beams
    start_time = time.time()
    for it in range(config.num_depths):
        # print(f"\n-> {it}")

        # print(f"initial: len = {len(active_beams)}")
        # for b_idx, b in enumerate(active_beams):
        #     print(f"b_idx = {b_idx}")
        #     print(b.current_text)
        
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

        # # Last iteration, generate to EOS
        # if it == config.num_depths - 1:
        #     sampling_params = SamplingParams(
        #         temperature=config.temperature,
        #         max_tokens=config.max_tokens,
        #         top_p=config.top_p,
        #         n=1,
        #     )

        lookahead = 0 if it == config.num_depths - 1 else config.lookahead
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

        # print(f"\n-> after generation: len = {len(active_beams)}")
        # for b_idx, b in enumerate(active_beams):
        #     print(f"b_idx = {b_idx}")
        #     print(b.current_text)
            
        # Filter out comleted beams 
        active_beams = [b for b in active_beams if not b.completed]
        # print(len(active_beams))

        # print(f"\n-> after filtered completions: len = {len(active_beams)}")
        # for b_idx, b in enumerate(active_beams):
        #     print(f"b_idx = {b_idx}")
        #     print(b.current_text)

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            print("break")
            break

        # Terminate reasoning once a specified depth is reached.
        if it == config.num_depths - 1:
            break

        # if config.filter_duplicates:
        #     # Create a dictionary to filter duplicates and retain order
        #     unique_beam_dict = {}
        #     for i, b in enumerate(active_beams):
        #         if b.current_text not in unique_beam_dict:
        #             unique_beam_dict[b.current_text] = (
        #                 i  # Map the unique text to its index
        #             )
        #     active_beams = [active_beams[i] for i in unique_beam_dict.values()]
        
        # print(f"\n-> after filtered duplications: len = {len(active_beams)}")
        # for b_idx, b in enumerate(active_beams):
        #     print(f"b_idx = {b_idx}")
        #     print(b.current_text)
            
        # Compute prm scores
        all_prompts = []
        all_completions = []
        for b_idx, beam in enumerate(active_beams):
            all_prompts.append(beam.question)
            all_completions.append([beam.current_text])

        all_scores = prm.score(all_prompts, all_completions, batch_size=4)
        all_agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in all_scores
        ]
        
        # Extract completion's embeddings and other info
        batch_embeds = [[] for _ in range(len(batch_of_questions))]
        batch_nll = [[] for _ in range(len(batch_of_questions))]
        batch_ppl = [[] for _ in range(len(batch_of_questions))]
        batch_beams = [[] for _ in range(len(batch_of_questions))]
        batch_scores = [[] for _ in range(len(batch_of_questions))]
        
        for b_idx, beam in enumerate(active_beams):
            # batch_scores[beam.q_idx].append(all_agg_scores[b_idx][0])
            batch_scores[beam.q_idx].append(max(-50, min(50, all_agg_scores[b_idx][0])))
            
            with torch.no_grad():
                # get beam.current_text which include previous all steps upto now
                gen_prompt = beam.templated_prompt + beam.next_texts[0]

                
                inputs = llm_tokenizer(gen_prompt, return_tensors="pt").to(llm_tf.device)
                outputs = llm_tf(**inputs, output_hidden_states=True)
    
                # Compute otuput_log_prob
                # Prepare labels: shift input_ids to the right by one
                inputs_ids_shifted = inputs['input_ids'][:, 1:]   
                outputs_logits_shifted = outputs.logits[:, :-1, :]
                completion_nll = loss_fct(outputs_logits_shifted.view(-1, outputs_logits_shifted.size(-1)), inputs_ids_shifted.view(-1))
                # outputs_log_probs = F.log_softmax(outputs_logits_shifted, dim=-1)
                # selected_log_probs = outputs_log_probs.gather(2, inputs_ids_shifted.unsqueeze(-1)).squeeze(-1)
                # completion_nll = -selected_log_probs.sum()

                # completion_nll = completion_nll[0]
                outputs_ntokens = inputs_ids_shifted.numel()
                completion_ppl = torch.exp(completion_nll/outputs_ntokens)
                # print(sent_ppl)
                # print(loss)
                
                # Get last_token_embeds
                last_hidden_state = outputs.hidden_states[-1]
                last_token_embeds = last_hidden_state[0,-1]
                # print(last_hidden_state.shape)
                # print(last_token_embeds.shape)
                # last_token_embeds = last_hidden_state[:, -1, :].squeeze(0).detach().cpu().numpy()
                # print(last_token_embeds.shape)
                
                # normalize the embeds
                if config.normalize_embeds:
                    last_token_embeds = F.normalize(last_token_embeds, p=2, dim=-1)
    
                batch_embeds[beam.q_idx].append(last_token_embeds.detach().cpu().numpy())
                batch_nll[beam.q_idx].append(completion_nll.detach().cpu().item())
                batch_ppl[beam.q_idx].append(completion_ppl.detach().cpu().item())
                batch_beams[beam.q_idx].append(beam)

        # print(batch_nll)
        # print(batch_ppl)
        # for q_idx in range(len(batch_embeds)):
        #     for b_idx in range(len(batch_embeds[q_idx])):
        #         print(batch_embeds[q_idx][b_idx][:5])
        
        tasks = [(K, V, q_idx, batch_beams[q_idx], batch_embeds[q_idx],
                  batch_nll[q_idx], batch_ppl[q_idx], np.array(batch_scores[q_idx]), config.ds_alpha, config.ds_beta) for q_idx in range(len(batch_of_questions))]
        # tasks = [(q_idx, config) for q_idx in range(len(batch_of_questions))]

        with mp.Pool() as pool:
            pool_results = pool.starmap(process_diverse_select, tasks)

        # print('construct batch_active_beams')
        # batch_active_beams = [[] for _ in range(len(batch_of_questions))]
        next_active_beams = []
        for q_idx, selected_idxes in pool_results:
            # print(f"selected_idxes = {selected_idxes}")
            q_beams = []
            if selected_idxes is not None:
                for s_idx in selected_idxes:
                    q_beams.append(batch_beams[q_idx][s_idx])
        
            else:
                q_beams = batch_beams[q_idx]

            m = len(q_beams)
            if m == 0:
                continue
            
            repeats = (config.n // m) + 1
            next_active_beams +=  [copy.deepcopy(b) for b in (q_beams*repeats)[:config.n]]

        active_beams = next_active_beams
        
        # total_time = time.time() - start_time
        # print(f"it takes {total_time:0.4f}s")
                
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

