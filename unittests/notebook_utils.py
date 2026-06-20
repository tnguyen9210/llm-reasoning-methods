"""Shared helpers for benchmark and examine notebooks."""

import gc
import os
import time

import torch
from vllm import LLM, SamplingParams


def short_model_name(name_or_dir: str) -> str:
    """Compact label for a model checkpoint, for benchmark printouts.

    Maps a full HF name or path (e.g.
    '/.../Qwen2.5-Math-7B-Instruct-GPTQ-Int4') to a short label
    matching the benchmark_speed_bon_quant style: '<family>-<size>'
    plus a precision/quant suffix when one is encoded in the name.
        Llama3.2-3B-Instruct            -> 'llama-3b'
        Qwen2.5-3B-Instruct-GPTQ        -> 'qwen-3b gptq'
        Qwen2.5-Math-7B-Instruct-GPTQ-Int4 -> 'qwen-math-7b gptq-int4'
    Unrecognized names fall back to the lowercased basename so nothing
    is silently dropped.
    """
    base = os.path.basename(name_or_dir.rstrip("/"))
    low = base.lower()

    # Family.
    if "llama" in low:
        family = "llama"
    elif "qwen" in low and "math" in low:
        family = "qwen-math"
    elif "qwen" in low:
        family = "qwen"
    else:
        return low  # unknown family: don't guess, keep it visible

    # Size: first token like '3b' / '1.5b' / '70b'.
    size = ""
    for tok in base.replace("-", " ").split():
        t = tok.lower()
        if t.endswith("b") and t[:-1].replace(".", "").isdigit():
            size = t
            break

    # Quant suffix (mirrors the quant notebook's labels).
    if "gptq-int4" in low or "gptq_int4" in low:
        quant = " gptq-int4"
    elif "gptq" in low:
        quant = " gptq"
    else:
        quant = ""

    label = f"{family}-{size}" if size else family
    return f"{label}{quant}"


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


def benchmark_bon_speed(
    cfg,
    config,
    prompts,
    num_trials,
    gpu_memory_utilization,
    warmup=1,
    max_model_len=4096,
):
    """Load one model config under vLLM, warm up, time num_trials BoN
    runs, then tear down. Returns (name, trial_times).

    Unifies the former plain-model and quantized-model benchmarks: a
    plain fp16 model and a GPTQ variant are the same run, differing
    only in the vLLM load kwargs, so one helper covers both.

    No GPU-memory figure is reported: under vLLM that's the
    pre-allocated gpu_memory_utilization budget, not the model's weight
    footprint, so it's ~constant across configs and uninformative. Use
    benchmark_llm_mem_sizes_v1 (HF Transformers) for weight memory.

    Args:
        cfg:    Dict describing the model. Required: "model_dir".
                Optional:
                  "name"         label for printouts (default: derived
                                 from model_dir via short_model_name).
                  "quantization" vLLM quantization (e.g. "gptq"); None
                                 / absent = unquantized.
                  "load_format"  vLLM load_format (default "auto").
                  "dtype"        vLLM dtype (default "float16").
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

    name = cfg.get("name") or short_model_name(cfg["model_dir"])
    print(f"\n=== {name} ===")

    llm = LLM(
        model=cfg["model_dir"],
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype=cfg.get("dtype", "float16"),
        quantization=cfg.get("quantization"),
        load_format=cfg.get("load_format", "auto"),
        seed=config.seed,
    )
    gc.collect()
    torch.cuda.empty_cache()

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
    return name, times


def benchmark_prm_score_batch(
    prm,
    questions,
    answers,
    batch_sizes,
    num_trials,
    warmup=1,
):
    """Time prm.score over a FIXED candidate set at several batch sizes.

    The same (questions, answers) grid is scored at each batch_size, so
    the only thing varying is how many (question, answer) pairs go
    through the PRM per forward pass — an apples-to-apples throughput
    sweep. Also records peak GPU memory per batch size (reset before
    each timed run via torch.cuda.reset_peak_memory_stats), which is
    meaningful here: the PRM is an HF model whose activation footprint
    grows with batch size (unlike a vLLM engine's fixed pre-allocation).

    Args:
        prm:          A loaded PRM (RLHFlowPRM / QwenPRM); uses .score.
        questions:    list[str], one problem each.
        answers:      list[list[str]] candidate answers per question
                      (each "\\n\\n"-joined steps). n_pairs = total.
        batch_sizes:  Iterable of score() batch_size values to sweep.
        num_trials:   Timed score() runs per batch size.
        warmup:       Untimed runs (per batch size) before timing.

    Returns:
        list of (batch_size, n_pairs, peak_mem_gb, trial_times).
    """
    n_pairs = sum(len(a) for a in answers)
    results = []

    for bs in batch_sizes:
        print(f"\n=== batch_size = {bs} ===")

        for _ in range(warmup):
            prm.score(questions, answers, batch_size=bs)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for trial_idx in range(num_trials):
            start = time.perf_counter()
            prm.score(questions, answers, batch_size=bs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(
                f"  trial {trial_idx}: {elapsed:>7.2f}s total, "
                f"{elapsed / n_pairs:.4f}s/pair"
            )

        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  peak GPU memory: {peak_gb:.2f} GB")
        results.append((bs, n_pairs, peak_gb, times))

    return results
