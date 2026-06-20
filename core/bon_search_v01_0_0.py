
from collections import defaultdict
import random
import numpy as np

import torch

from vllm import SamplingParams
from sal.search.utils import build_conv


def _search(batch_of_questions, config, trial_idx, llm_vllm):
    """Best-of-N: sample config.search.n completions per question in
    a single vLLM call. Reads the nested ExpConfig (config.gen.*,
    config.search.*). Returns raw completions; scoring is downstream.
    """
    np.random.seed(100000 + trial_idx)
    random.seed(100000 + trial_idx)
    torch.manual_seed(100000 + trial_idx)
    torch.cuda.manual_seed(100000 + trial_idx)

    n = config.search.n

    tokenizer = llm_vllm.get_tokenizer()
    # Override with the vendored Llama template only when asked;
    # otherwise keep the model's native template.
    if config.llm.use_custom_template:
        tokenizer.chat_template = config.gen.custom_chat_template

    convs = [
        build_conv(
            question, response="",
            system_prompt=config.gen.system_prompt,
        )
        for question in batch_of_questions
    ]

    templated_convs = tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=True,
        date_string=config.gen.date_string,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=config.gen.temperature,
        max_tokens=config.gen.max_tokens,
        top_p=config.gen.top_p,
        n=n,  # n completions per question
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
            f"Generated {len(responses)} responses instead of " \
            f"{len(batch_of_questions)}"

    # Collect the completions from responses
    completions = [[] for _ in range(len(batch_of_questions))]
    completion_ntokens = [[] for _ in range(len(batch_of_questions))]

    for r_idx, r in enumerate(responses):
        if len(r.outputs) != n:
            raise ValueError(
                f"Generated {len(r.outputs)} completions instead "
                f"of {n}"
            )
        for output in r.outputs:
            completions[r_idx].append(output.text)
            completion_ntokens[r_idx].append(len(output.token_ids))

    results = defaultdict(list)
    results["completions"] = completions
    results["completion_ntokens"] = completion_ntokens

    return results

