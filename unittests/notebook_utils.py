"""Shared helpers for benchmark and examine notebooks."""

import gc
import os
import time

import torch
from vllm import LLM, SamplingParams


def print_step_scores(steps: list[str], scores: list[float]) -> None:
    """Print per-step P(correct) scores with a truncated step preview.

    Truncates each step to 60 chars so long flamingo-style steps
    don't flood the output.
    """
    for idx, (step, score) in enumerate(zip(steps, scores), start=1):
        preview = step if len(step) <= 60 else step[:60] + "..."
        print(f"Step {idx}: P(correct) = {score:.4f}")
        print(preview)


def gpu_mem_used_gb(device=0):
    """Driver-level used GPU memory in GB.

    Runs gc.collect() + empty_cache() before measuring to evict
    unreferenced tensors. Reports what the CUDA driver sees —
    includes both PyTorch allocator pool and vLLM allocs.
    """
    gc.collect()
    torch.cuda.empty_cache()
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


def benchmark_bon_speed_llm_model(
    llm_dir,
    config,
    prompts,
    num_trials,
    gpu_memory_utilization,
    warmup=1,
    max_model_len=4096,
):
    """Load llm_dir under vLLM, warm up, time num_trials BoN runs,
    then tear down. Returns (model_name, trial_times).

    Args:
        llm_dir:                 Path to the model checkpoint.
        config:                  sal.Config with generation params.
        prompts:                 List of prompt strings.
        num_trials:              Number of timed runs.
        gpu_memory_utilization:  vLLM gpu_memory_utilization setting.
        warmup:                  Untimed warmup runs before timing.
        max_model_len:           vLLM context cap. Default 4096 fits
                                 every benchmarked model (Qwen2.5-Math
                                 caps at max_position_embeddings=4096).
    """
    from core import bon_search_v1

    model_name = os.path.basename(llm_dir.rstrip("/"))
    print(f"\n=== {model_name} ===")

    llm = LLM(
        model=llm_dir,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype="float16",
        seed=config.seed,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory used: {gpu_mem_used_gb():.2f} GB")

    for w in range(warmup):
        bon_search_v1.best_of_n_v1(prompts, config, llm, 10_000 + w)

    times = []
    for trial_idx in range(num_trials):
        start = time.perf_counter()
        bon_search_v1.best_of_n_v1(prompts, config, llm, trial_idx)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"  trial {trial_idx}: {elapsed:>7.2f}s total, "
            f"{elapsed / len(prompts):.4f}s/question"
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return model_name, times


def benchmark_bon_speed_llm_quant(
    qcfg,
    config,
    prompts,
    num_trials,
    gpu_memory_utilization,
    warmup=1,
):
    """Load a quantization config under vLLM, warm up, time num_trials
    BoN runs, then tear down. Returns (name, gpu_mem_gb, trial_times).

    Args:
        qcfg:                    Dict with keys name, model_dir,
                                 quantization, load_format, dtype.
        config:                  sal.Config with generation params.
        prompts:                 List of prompt strings.
        num_trials:              Number of timed runs.
        gpu_memory_utilization:  vLLM gpu_memory_utilization setting.
        warmup:                  Untimed warmup runs before timing.
    """
    from core import bon_search_v1

    print(f"\n=== {qcfg['name']} ===")

    llm = LLM(
        model=qcfg["model_dir"],
        tensor_parallel_size=1,
        max_model_len=5000,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype=qcfg["dtype"],
        quantization=qcfg["quantization"],
        load_format=qcfg["load_format"],
        seed=config.seed,
    )
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem = gpu_mem_used_gb()
    print(f"  GPU memory used: {gpu_mem:.2f} GB")

    for w in range(warmup):
        bon_search_v1.best_of_n_v1(prompts, config, llm, 10_000 + w)

    times = []
    for trial_idx in range(num_trials):
        start = time.perf_counter()
        bon_search_v1.best_of_n_v1(prompts, config, llm, trial_idx)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"  trial {trial_idx}: {elapsed:>7.2f}s total, "
            f"{elapsed / len(prompts):.4f}s/question"
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return qcfg["name"], gpu_mem, times
