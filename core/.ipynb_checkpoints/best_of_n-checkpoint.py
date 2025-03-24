
from collections import defaultdict
from vllm import SamplingParams

def best_of_n_v11(batch_of_questions, config, llm_vllm, random_seed):
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in batch_of_questions
    ]
    
    tokenizer = llm_vllm.get_tokenizer()
    
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    templated_convs = tokenizer.apply_chat_template(
        convs, add_generation_prompt=True, tokenize=False,
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    # templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        # temperature=0,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=config.n,  # generate n outputs
        best_of=config.n,
        # stop=[
        #     "\n\n"
        # ],  # we consider that a step in the problem is indicated by a double newline
        # include_stop_str_in_output=True,
        seed=random_seed,
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
        if len(r.outputs) != config.n:
            raise ValueError(f"Generated {len(r.outputs)} completions instead of {config.n}")
            
        for output in r.outputs:
            # print(output.text)
            # print(output.stop_reason)
            completions[r_idx].append(output.text)
            completion_ntokens[r_idx].append(len(output.token_ids))

    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens
    
    return results

def best_of_n_v12(batch_of_questions, config, llm_vllm, random_seed):
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in batch_of_questions
    ]
    
    tokenizer = llm_vllm.get_tokenizer()
    
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    templated_convs = tokenizer.apply_chat_template(
        convs, add_generation_prompt=True, tokenize=False,
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        # temperature=0,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,  # generate n outputs
        # stop=[
        #     "\n\n"
        # ],  # we consider that a step in the problem is indicated by a double newline
        # include_stop_str_in_output=True,
        seed=random_seed,
    )        

    # Generate responses 
    responses = llm_vllm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Re-generate responses if we get more responses than expected
    if len(responses) != len(batch_of_questions) * config.n:
        responses = llm_vllm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        assert len(responses) == len(batch_of_questions) * config.n, \
            f"Generated {len(responses)} responses instead of {len(batch_of_questions)}"
    
    # Collect the completions from responses
    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    # for i in range(len(completions)):
    #     completions[i] = [
    #         output.text
    #         for r in responses[i * config.n : (i + 1) * config.n]
    #         for output in r.outputs
    #     ]
    #     completion_ntokens[i] = [
    #         len(output.token_ids)
    #         for r in responses[i * config.n : (i + 1) * config.n]
    #         for output in r.outputs
    #     ]
    # print(responses)
    
    for r_idx, r in enumerate(responses):
        idx = r_idx // config.n
        output = r.outputs[0]
        # print(output.text)
        completions[idx].append(output.text)
        completion_ntokens[idx].append(output.token_ids)

    # print(completions)
    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens

    return results