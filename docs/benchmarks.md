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
