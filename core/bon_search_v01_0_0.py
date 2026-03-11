
from collections import defaultdict
import random
import numpy as np

import torch

from vllm import SamplingParams
from sal.search.utils import build_conv, generate_k_steps, last


def _search(batch_of_questions, config, trial_idx, llm_vllm):
    np.random.seed(100000+trial_idx)
    random.seed(100000+trial_idx)
    torch.manual_seed(100000+trial_idx)
    torch.cuda.manual_seed(100000+trial_idx)
    
    tokenizer = llm_vllm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    # convs = [
    #     [
    #         {"role": "system", "content": config.system_prompt},
    #         {"role": "user", "content": prompt},
    #     ]
    #     for prompt in batch_of_questions
    # ]
    convs = [build_conv(question, response="", system_prompt=config.system_prompt) 
             for question in batch_of_questions]  
        
    templated_convs = tokenizer.apply_chat_template(
        convs, 
        add_generation_prompt=True,
        date_string=config.date_string,
        tokenize=False,
    )

    # Duplicate convs to generate config.bs completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.bs=2
    # templated_convs = [c for conv in templated_convs for c in [conv] * config.bs]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=config.bs,  # generate n outputs
        best_of=config.bs,
        # stop=[
        #     "\n\n"
        # ],  # we consider that a step in the problem is indicated by a double newline
        # include_stop_str_in_output=True,
        # seed=random_seed,
    )        

    # Generate responses 
    responses = llm_vllm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Re-generate responses if we get more responses than expected
    if len(responses) != len(batch_of_questions):
        responses = llm_vllm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        assert len(responses) == len(batch_of_questions), \
            f"Generated {len(responses)} responses instead of {len(batch_of_questions)}"
    
    # Collect the completions from responses
    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    for r_idx, r in enumerate(responses):
        # print(r.request_id)
        if len(r.outputs) != config.bs:
            raise ValueError(f"Generated {len(r.outputs)} completions instead of {config.bs}")
            
        for output in r.outputs:
            # print(output.text)
            # print(output.stop_reason)
            completions[r_idx].append(output.text)
            completion_ntokens[r_idx].append(len(output.token_ids))

    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens
    
    return results

