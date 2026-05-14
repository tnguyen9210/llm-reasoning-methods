
from collections import defaultdict
import random
import numpy as np

import torch

from vllm import SamplingParams
from sal.search.utils import build_conv


def _set_seeds(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def _prepare_inputs(batch_of_questions, config, llm_vllm):
    tokenizer = llm_vllm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    # build_conv returns [{"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": question}]
    convs = [
        build_conv(question, response="", system_prompt=config.system_prompt)
        for question in batch_of_questions
    ]

    return tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=True,
        date_string=config.date_string,
        tokenize=False,
    )


def _generate(llm_vllm, templated_convs, sampling_params, expected_count):
    responses = llm_vllm.generate(
        templated_convs, sampling_params=sampling_params, use_tqdm=False
    )
    if len(responses) != expected_count:
        responses = llm_vllm.generate(
            templated_convs, sampling_params=sampling_params, use_tqdm=False
        )
        assert len(responses) == expected_count, \
            f"Generated {len(responses)} responses instead of {expected_count}"
    return responses


# === V1: Native — single prompt, n=config.n outputs per request ===

def best_of_n_v1(batch_of_questions, config, llm_vllm, trial_idx):
    _set_seeds(100000 + trial_idx)

    templated_convs = _prepare_inputs(batch_of_questions, config, llm_vllm)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=config.n,
    )

    responses = _generate(
        llm_vllm, templated_convs, sampling_params, len(batch_of_questions)
    )

    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    for r_idx, r in enumerate(responses):
        if len(r.outputs) != config.n:
            raise ValueError(f"Generated {len(r.outputs)} completions instead of {config.n}")
        for output in r.outputs:
            completions[r_idx].append(output.text)
            completion_ntokens[r_idx].append(len(output.token_ids))

    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens
    return results


# === V2: Prompt duplication — duplicate prompts × config.n, n=1 per request ===

def best_of_n_v2(batch_of_questions, config, llm_vllm, trial_idx):
    _set_seeds(100000 + trial_idx)

    templated_convs = _prepare_inputs(batch_of_questions, config, llm_vllm)
    # Duplicate: [p1, p2] → [p1, p1, p2, p2] for config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
    )

    responses = _generate(
        llm_vllm, templated_convs, sampling_params, len(batch_of_questions) * config.n
    )

    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    for r_idx, r in enumerate(responses):
        idx = r_idx // config.n
        output = r.outputs[0]
        completions[idx].append(output.text)
        completion_ntokens[idx].append(len(output.token_ids))

    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens
    return results
