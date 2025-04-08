
from collections import defaultdict

import copy 
import time
import numpy as np
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
    prompt: str
    templated_prompt: str
    index: int
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

        best_idx = max(candidate_idxes, key=lambda i: X_ppl[i])
        # print(arm_vals)
        # print(X_lprobs)
        # print(candidate_idxes)
        # print(best_idx)
        
        best_embeds = X_embeds[best_idx]
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


def _select_diverse_search(batch_of_questions, config: Config, llm: LLM, llm_tf, llm_tokenizer) -> list[Beam]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_questions:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    templated_prompt=prompt,
                    index=i,
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

    completed_beams: list[Beam] = []
    # active_beams = [b for b in beams if not b.pruned]
    # print(len(active_beams))
    
    # for b_idx, beam in enumerate(active_beams):
    #     if b_idx % 2 == 0:
    #         beam.pruned = True 

    # active_beams = [b for b in beams if not b.pruned]

    # print(len(active_beams))

    # stio

    
    # for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
    for i in range(config.num_iterations):
        # print(f"iteration {i}")
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            # print(
            #     f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            # )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, 1
        )
        # print(gen_results)
        # stop

        prompts, completions = [], []
        next_active_beams = []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            beam.templated_prompt = gen_result.prompt

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        active_beams = [b for b in active_beams if not b.completed]
        # print(active_beams)

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # # get completion's embeddings
        completions_embeds = np.zeros((len(active_beams), 2048))
        completions_log_probs = np.zeros(len(active_beams))
        completions_ppl = np.zeros(len(active_beams))
    
        for b_idx, beam in enumerate(active_beams):
            with torch.no_grad():
                # get beam.current_text which include previous all steps upto now
                gen_prompt = beam.templated_prompt + beam.next_texts[0]
                inputs = llm_tokenizer(gen_prompt, return_tensors="pt").to(llm_tf.device)
                outputs = llm_tf(**inputs, output_hidden_states=True)

                # Get last_token_embeds
                last_hidden_state = outputs.hidden_states[-1]
                last_token_embeds = last_hidden_state[:, -1, :].squeeze(0).detach().cpu().numpy()
                # print(last_token_embeds.shape)

                # Compute otuput_log_prob
                # Prepare labels: shift input_ids to the right by one
                # print(inputs)
                labels = inputs['input_ids'][:, 1:]   
                shifted_logits = outputs.logits[:, :-1, :]
                loss_fct = CrossEntropyLoss(reduction='sum')
                completion_log_prob = -loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1)).detach().cpu().numpy()
                completion_ppl = np.exp(completion_log_prob/len(labels))
                # print(sent_ppl)
                # print(loss)

                # Approach 
                # log_probs = F.log_softmax(shifted_logits, dim=-1)
                # token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                # completion_log_prob = token_log_probs.sum()
                # print(completion_log_prob)

                # normalize the embeds
                if config.normalize_embeds:
                    norm = np.linalg.norm(last_token_embeds)
                    last_token_embeds /= norm
                    # print(np.linalg.norm(last_token_embeds))

                completions_embeds[b_idx] = last_token_embeds
                completions_log_probs[b_idx] = completion_log_prob
                completions_ppl[b_idx] = completion_ppl

        # print(completions_embeds)
        # print(completions_log_probs)
        # print(completions_ppl)

        # get completion's embeddings

        V = config.lam*np.eye(2048)
        K = int(config.n / config.beam_width)
        if len(active_beams) <= K:
            continue 
            
        selected_idxes = _select_diverse(
            completions_embeds, completions_log_probs, completions_ppl, K, V)
        # print(len(completions_embeds))
        # print(selected_idxes)

        for idx, beam in enumerate(active_beams):
            if idx not in selected_idxes:
                beam.pruned = True

    # # Filter completed beams for those with top config.n scores
    # if config.sort_completed:
    #     completed_beams = sorted(
    #         completed_beams,
    #         key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
    #         reverse=True,
    #     )[: config.n]
    # else:
    #     completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        # print(
        #     f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        # )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams

def select_diverse_search(batch_of_questions, config: Config, llm: LLM, llm_tf, llm_tokenizer):

    # Collect the completions from responses
    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    for q_idx, question in enumerate(batch_of_questions):
        # print(f"question {q_idx}")
        beam_results = _select_diverse_search([question], config, llm, llm_tf, llm_tokenizer)
        for b_idx, beam in enumerate(beam_results):
            # print(beam.current_text)
            completions[q_idx].append(beam.current_text)
            completion_ntokens[q_idx].append(beam.completion_tokens)

    # print(completions)
    # print(completion_ntokens)
    # stop
    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens
    
    # print(results)
    return results

# # beam_results = _select_diverse_search(batch_of_questions, config, llm, llm_tf, tokenizer)
# beam_results = select_diverse_search(batch_of_questions, config, llm, llm_transformer, tokenizer)
# print(beam_results)