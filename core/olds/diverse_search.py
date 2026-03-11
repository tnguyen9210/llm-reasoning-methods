
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


def _select_diverse(X_embeds, X_lprobs, X_ppl, K, V):
    num_arms = len(X_embeds)
    _V = copy.deepcopy(V)
    A_idxes = []
    A_embeds = []
    tol = 0.0001
    for it in range(K):
        _V_inv = np.linalg.inv(_V)
        arm_vals = np.einsum('ij,jk,ik->i', X_embeds, _V_inv, X_embeds)
        max_val = np.max([val for idx, val in enumerate(arm_vals) if idx not in A_idxes])
        # candidate_idxes = np.where(np.abs(arm_vals-max_val) < tol)[0]
        candidate_idxes = [
            arm_idx for arm_idx, arm_val in enumerate(arm_vals)
            if (np.abs(max_val - arm_val) <= tol) and (arm_idx not in A_idxes)
        ]

        best_idx = max(candidate_idxes, key=lambda i: X_lprobs[i])
        # print(arm_vals)
        # print(X_lprobs)
        # print(candidate_idxes)
        # print(best_idx)
        
        best_embeds = X_embeds[best_idx]
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

def process_select_diverse(q_idx, q_active_beams, q_embeds, q_log_probs, q_ppl, K, V):
    # V = config.lam*np.eye(2048)
    # K = int(config.n/config.beam_width)

    if len(q_active_beams) <= K:
        return (q_idx, None)

    selected_idxes  = _select_diverse(q_embeds, q_log_probs, q_ppl, K, V)

    # for idx, beam in enumerate(q_active_beams):
    #     if idx not in selected_idxes:
    #         beam.pruned = True 
            
    return (q_idx, selected_idxes)
    # return (q_idx, selected_idxes) 

def diverse_search(batch_of_questions, config, llm_vllm, llm_tf, llm_tokenizer):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    V = config.lam*np.eye(2048)
    K = int(config.n / config.beam_width)
    
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
    start_time = time.time()
    for it in range(config.num_depths):
        # print(f"\n-> {it}")
        if it == 0:
            active_beams = beams
        else:
            # active_beams = [b for b in active_beams if not b.pruned]
            extended_beams = []
            for beam in active_beams:
                if beam.pruned:
                    continue 
                    
                for j in range(config.beam_width):
                    extended_beams.append(copy.deepcopy(beam))

            active_beams = extended_beams

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

        if it == config.num_depths - 1:
            break
        
        # Extract completion's embeddings and other info
        batch_embeds = [[] for _ in range(len(batch_of_questions))]
        batch_log_probs = [[] for _ in range(len(batch_of_questions))]
        batch_ppl = [[] for _ in range(len(batch_of_questions))]
        batch_beams = [[] for _ in range(len(batch_of_questions))]
    
        for b_idx, beam in enumerate(active_beams):
            with torch.no_grad():
                # get beam.current_text which include previous all steps upto now
                gen_prompt = beam.templated_prompt + beam.next_texts[0]
                # print(gen_prompt)
                # stop
                inputs = llm_tokenizer(gen_prompt, return_tensors="pt").to(llm_tf.device)
                outputs = llm_tf(**inputs, output_hidden_states=True)
    
                # Get last_token_embeds
                last_hidden_state = outputs.hidden_states[-1]
                last_token_embeds = last_hidden_state[:, -1, :].squeeze(0).detach().cpu().numpy()
                # print(last_token_embeds.shape)
    
                # Compute otuput_log_prob
                # Prepare labels: shift input_ids to the right by one
                labels = inputs['input_ids'][:, 1:]   
                shifted_logits = outputs.logits[:, :-1, :]
                loss_fct = CrossEntropyLoss(reduction='sum')
                completion_log_prob = -loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1)).detach().cpu().numpy()
                outputs_ntokens = labels.numel()
                completion_ppl = np.exp(-completion_log_prob/outputs_ntokens)
                # print(sent_ppl)
                # print(loss)
    
                # normalize the embeds
                if config.normalize_embeds:
                    norm = np.linalg.norm(last_token_embeds)
                    last_token_embeds /= norm
                    # print(np.linalg.norm(last_token_embeds))
    
                batch_embeds[beam.q_idx].append(last_token_embeds)
                batch_log_probs[beam.q_idx].append(completion_log_prob)
                batch_ppl[beam.q_idx].append(completion_ppl)
                batch_beams[beam.q_idx].append(beam)

        print(batch_log_probs)
        print(batch_ppl)
        # pprint.pprint(len(batch_completions_embeds))
        # pprint.pprint(len(batch_completions_log_probs))
        # pprint.pprint(len(batch_completions_ppl))
        # print(len(batch_beams))
        # print(len(batch_beams[0]))
        # total_time = time.time() - start_time
        # print(f"it takes {total_time:0.4f}s")

        # Use _select_diverse to diversify embeddings 
        
        # for q_idx in range(len(batch_of_questions)):
            
        #     if len(batch_beams[q_idx]) <= K:
        #         continue 
    
        #     selected_idxes = _select_diverse(
        #         batch_embeds[q_idx], batch_log_probs[q_idx], batch_ppl[q_idx], K, V)
    
        #     # print(selected_idxes)
            
        #     for idx, beam in enumerate(batch_beams[q_idx]):
        #         if idx not in selected_idxes:
        #             beam.pruned = True 

        
        tasks = [(q_idx, batch_beams[q_idx], batch_embeds[q_idx],
                  batch_log_probs[q_idx], batch_ppl[q_idx], K, V) for q_idx in range(len(batch_of_questions))]
        # tasks = [(q_idx, config) for q_idx in range(len(batch_of_questions))]
    
        with mp.Pool() as pool:
            pool_results = pool.starmap(process_select_diverse, tasks)

        # pool_results = []
        # for task in tasks:
        #     pool_results.append(process_select_diverse(*task))

        for q_idx, selected_idxes in pool_results:
            if selected_idxes is not None:
                for idx, beam in enumerate(batch_beams[q_idx]):
                    if idx not in selected_idxes:
                        beam.pruned = True 
                        
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

