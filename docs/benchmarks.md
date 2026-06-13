# Benchmarks

## Inference speed: HF Transformers vs vLLM

**Notebook:** `unittests/benchmark_speed_simple_generation_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Model:** Qwen2.5-3B-Instruct  
**Config:** 10 runs + 1 warmup, max_new_tokens=1024,
temperature=0.8, top_p=0.95, single prompt

| Backend         | Latency (s) | Tok/s  | Avg tok |
|-----------------|-------------|--------|---------|
| HF Transformers |    17.88    |  26.41 |  472.3  |
| vLLM gmu=0.3    |     4.25    | 113.39 |  482.2  |
| vLLM gmu=0.7    |     4.26    | 113.14 |  482.2  |

**Takeaway:** vLLM is ~4.3× faster than HF eager on
single-prompt throughput on V100. The two
`gpu_memory_utilization` settings are within noise for a
single prompt; the difference matters only under concurrent
batching (more sequences live in KV cache at once).

---

**Notebook:** `unittests/benchmark_speed_simple_generation_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Model:** Llama3.2-1B-Instruct  
**Config:** 10 runs + 1 warmup, max_new_tokens=1024,
temperature=0.8, top_p=0.95, single prompt

| Backend         | Latency (s) | Tok/s  | Avg tok |
|-----------------|-------------|--------|---------|
| HF Transformers |     8.52    |  58.98 |  502.3  |
| vLLM gmu=0.3    |     1.94    | 253.10 |  490.6  |
| vLLM gmu=0.7    |     1.94    | 253.05 |  490.6  |

**Takeaway:** vLLM is ~4.3× faster than HF eager on
single-prompt throughput on V100 (consistent with the
Qwen2.5-3B result). Llama3.2-1B is roughly 2× faster than
Qwen2.5-3B in both backends, as expected from the smaller
model size.

## Best-of-N search speed across models

**Notebook:** `unittests/benchmark_speed_bon_models_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Config:** BoN n=32, 10 MATH level-4 questions, 2 trials +
1 warmup, temperature=0.8, max_tokens=2048, vLLM gmu=0.7,
fp16, enforce_eager

| Model                | Mean s/trial | Std  | s/question |
|----------------------|--------------|------|------------|
| Llama3.2-1B-Instruct |     38.36    | 0.51 |    3.84    |
| Llama3.2-3B-Instruct |     95.09    | 3.50 |    9.51    |
| Qwen2.5-3B-Instruct  |    160.24    | 5.13 |   16.02    |
| Qwen2.5-7B-Instruct  |    115.84    | 4.09 |   11.58    |

**Takeaway:** BoN time does not track parameter count:
Qwen2.5-3B is the slowest of the four, ~1.4× slower than
the 7B from the same family. Within the Llama family time
does grow with size (3B is ~2.5× slower than 1B). This
suggests per-model completion length (tokens sampled
before EOS) dominates over per-token cost — worth
verifying against avg token counts per BoN run.

BoN is benchmarked under vLLM only — no HF Transformers
counterpart; see [decisions.md](decisions.md) 2026-06-12.

## GPU memory: generative models (HF Transformers)

**Notebook:** `unittests/benchmark_llm_mem_sizes_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (transformers 4.57.6, gptqmodel 5.7.0)  
**Config:** HF Transformers loads only (vLLM section disabled).
Driver-level memory via `torch.cuda.mem_get_info`, so numbers
include the CUDA context (~0.5 GB). fp16 via
`from_pretrained(dtype="float16")`; int4 via `GPTQModel.load`.
The fp32 column is from an earlier run where no `dtype` was
passed (`from_pretrained` defaults to fp32) — kept for reference.

| Model                | fp32 (GB) | fp16 (GB) | int4 (GB) |
|----------------------|-----------|-----------|-----------|
| Llama3.2-1B-Instruct |     —     |    2.80   |     —     |
| Llama3.2-3B-Instruct |   12.47   |    6.47   |    2.57   |
| Qwen2.5-3B-Instruct  |   14.30   |    8.41   |    4.63   |
| Qwen2.5-7B-Instruct  |   31.43   |   16.86   |    7.83   |

int4 = GPTQ checkpoint (`-GPTQ` / `-GPTQ-Int4`). fp32 not
re-measured for the 1B model. All values include the ~0.5 GB
CUDA context.

**Takeaway:** fp16 roughly halves fp32, as expected; GPTQ-int4
roughly halves fp16 again (~3–5× below fp32). For the
fit-7B-LLM+PRM-on-32GB question (M4), Qwen2.5-7B is 16.9 GB at
fp16 — leaving ~15 GB for a PRM — or 7.8 GB at int4, leaving
~24 GB. fp32 at 31.4 GB barely fits the LLM alone, so it is not
a viable baseline on this 32 GB card.

## GPU memory: process reward models (HF Transformers)

**Notebook:** `unittests/benchmark_prm_mem_sizes_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (transformers 4.57.6)  
**Config:** HF Transformers, `AutoModel` (reward head, not causal
LM head). Driver-level memory via `torch.cuda.mem_get_info`;
includes ~0.5 GB CUDA context. fp16 via
`from_pretrained(dtype="float16")`. V100 (sm_70) does not support
bf16; some checkpoints are published in bf16 but load correctly
in fp16.

| Model                         | fp16 (GB) |
|-------------------------------|-----------|
| Qwen2.5-Math-PRM-7B           |   13.87   |
| Llama3.1-8B-PRM-Deepseek-Data |   14.56   |

**Takeaway:** Both PRMs are ~14 GB at fp16, close to the LLM
footprint despite similar parameter counts — the Llama-based PRM
(8B) is slightly larger than the Qwen one (7B). For the
fit-LLM+PRM-on-32GB question (M4): fp16 LLM + fp16 PRM sums to
30.7 GB (Qwen2.5-7B + Qwen2.5-Math-PRM-7B), leaving ~1.3 GB
for KV cache — extremely tight. int4 LLM + fp16 PRM gives
21.7 GB, leaving ~10 GB for KV cache — a workable margin.
