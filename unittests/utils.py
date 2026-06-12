"""Shared helpers for benchmark notebooks."""

import time

import torch
from vllm import SamplingParams


def gpu_mem_used_gb(device=0):
    """Driver-level used GPU memory in GB.

    Reports what the CUDA driver sees — includes both PyTorch
    allocator pool and vLLM allocs. Useful for before/after
    comparisons when loading or deleting models.
    """
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024**3)


def measure_inference(
    backend,
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    num_runs,
    temperature,
    top_p,
    base_seed=123,
    warmup=1,
):
    """Time `num_runs` stochastic generations and return stats.

    Args:
        backend:        "hf" or "vllm".
        model:          HF model or vLLM LLM instance.
        tokenizer:      HF tokenizer (unused for vllm backend).
        prompt:         Raw string prompt.
        max_new_tokens: Max tokens to generate per run.
        num_runs:       Number of timed iterations.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling threshold.
        base_seed:      Iteration i uses seed base_seed + i,
                        keeping runs reproducible but distinct.
        warmup:         Untimed passes before timing starts;
                        absorbs cudagraph capture / JIT overhead.

    Returns:
        (latency, throughput, avg_tokens, last_text)
        latency    -- mean seconds per run
        throughput -- total tokens / total time (tok/s)
        avg_tokens -- mean newly generated tokens per run
        last_text  -- decoded text from the final timed run
    """
    def _generate(seed):
        if backend == "vllm":
            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
            )
            out = model.generate(prompt, params, use_tqdm=False)
            tok_ids = out[0].outputs[0].token_ids
            return out[0].outputs[0].text, len(tok_ids)
        else:
            inputs = tokenizer(
                prompt, return_tensors="pt"
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            pad_id = (
                tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            torch.manual_seed(seed)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                )
            new_ids = out_ids[0, prompt_len:]
            return (
                tokenizer.decode(new_ids, skip_special_tokens=True),
                len(new_ids),
            )

    for w in range(warmup):
        _generate(base_seed + 10_000 + w)

    total_time = 0.0
    total_tokens = 0
    text = ""
    for i in range(num_runs):
        start = time.perf_counter()
        text, n_tokens = _generate(base_seed + i)
        total_time += time.perf_counter() - start
        total_tokens += n_tokens

    latency = total_time / num_runs
    throughput = total_tokens / total_time
    avg_tokens = total_tokens / num_runs
    return latency, throughput, avg_tokens, text
