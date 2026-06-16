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
**Date:** 2026-06-12 (Math models re-run 2026-06-16)  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Config:** BoN n=32, 10 MATH level-4 questions, temperature=0.8,
max_tokens=2048, vLLM gmu=0.7, fp16, enforce_eager.
2 trials + 1 warmup for the original run; the 2026-06-16 Math run
used 1 trial + 1 warmup (so its std is 0).

| Model                      | Mean s/trial | Std  | s/question | trials |
|----------------------------|--------------|------|------------|--------|
| Llama3.2-1B-Instruct       |     38.32    | —    |    3.83    |   1    |
| Llama3.2-3B-Instruct       |    105.07    | —    |   10.51    |   1    |
| Qwen2.5-Math-1.5B-Instruct |    121.34    | —    |   12.13    |   1    |
| Qwen2.5-Math-7B-Instruct   |    238.74    | —    |   23.87    |   1    |

Earlier 2-trial run (general Qwen, now superseded for the Qwen
rows): Llama-1B=3.84, Llama-3B=9.51, Qwen2.5-3B=16.02,
Qwen2.5-7B=11.58 s/q.

**Takeaway:** BoN time still does not track parameter count, and
the Math models are markedly slower per question than the general
Qwen models at matched size — Math-7B is 23.9 s/q, ~2× the old
general Qwen-7B (11.6 s/q), and Math-1.5B (12.1) is near the old
general 3B (16.0). The most likely cause is **completion length**:
the Math-Instruct models generate longer chains-of-thought (more
tokens before EOS), and per-token cost is secondary — consistent
with the within-family Llama growth (3B ~2.7× the 1B here). Worth
confirming against avg token counts per BoN run. Caveat: the Math
rows are single-trial (no std), and max_model_len differs from the
original run (4096 for Math vs 5000), so treat the Math-vs-general
gap as indicative, not precise.

BoN is benchmarked under vLLM only — no HF Transformers
counterpart; see [decisions.md](decisions.md) 2026-06-12.

## Best-of-N search speed across quantization levels

**Notebook:** `unittests/benchmark_speed_bon_quant_v1.ipynb`  
**Date:** 2026-06-12  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Config:** BoN n=256, 5 MATH level-4 questions, 2 trials +
1 warmup, temperature=0.8, max_tokens=2048, vLLM gmu=0.5,
enforce_eager. int4 = GPTQ checkpoint.

| Config           | GPU (GB) | Mean s/trial | Std   | s/question |
|------------------|----------|--------------|-------|------------|
| llama-3b fp16    |  16.53   |    297.55    |  8.00 |   59.51    |
| llama-3b gptq    |  16.63   |    368.36    |  2.75 |   73.67    |
| qwen-3b fp16     |  16.61   |    610.16    | 14.94 |  122.03    |
| qwen-3b gptq-int4|  16.55   |    595.16    |  1.43 |  119.03    |
| qwen-7b fp16     |  16.12   |   1184.41    | 43.85 |  236.88    |
| qwen-7b gptq-int4|  16.12   |    468.76    |  2.97 |   93.75    |

GPU column is the vLLM process footprint at gmu=0.5, not the
model-weight size — it is roughly constant because the budget
is set by `gpu_memory_utilization`, not the checkpoint.

**Takeaway:** GPTQ-int4's speed effect is non-monotonic in model
size. At 3B it does not pay off — llama-3b GPTQ is ~1.24× *slower*
than fp16, and qwen-3b int4 is within noise of fp16 — because the
dequantization overhead is not repaid at that scale. At 7B it wins
decisively: qwen-7b int4 is ~2.5× *faster* than fp16, where reduced
memory-bandwidth pressure dominates. So int4 is a speed win only at
7B; at 3B it buys VRAM headroom (see the GPU-memory table) at a
small-to-zero speed cost. Caveat: small sample (5 questions, 2
trials) and a different n than the across-models sweep (n=256 vs
n=32), so s/question is not comparable across the two tables.

## GPU memory: generative models (HF Transformers)

**Notebook:** `unittests/benchmark_llm_mem_sizes_v1.ipynb`  
**Date:** 2026-06-12 (Math-Instruct rows added 2026-06-16)  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (transformers 4.57.6, gptqmodel 5.7.0)  
**Config:** HF Transformers loads only (vLLM section disabled).
Driver-level memory via `torch.cuda.mem_get_info`, so numbers
include the CUDA context (~0.5 GB). fp16 via
`from_pretrained(dtype="float16")`; int4 via `GPTQModel.load`.
The fp32 column is from an earlier run where no `dtype` was
passed (`from_pretrained` defaults to fp32) — kept for reference.

| Model                      | fp32 (GB) | fp16 (GB) | int4 (GB) |
|----------------------------|-----------|-----------|-----------|
| Llama3.2-1B-Instruct       |     —     |    2.80   |     —     |
| Llama3.2-3B-Instruct       |   12.47   |    6.47   |    2.57   |
| Qwen2.5-3B-Instruct        |   14.30   |    8.41   |    4.63   |
| Qwen2.5-7B-Instruct        |   31.43   |   16.86   |    7.83   |
| Qwen2.5-Math-1.5B-Instruct |     —     |    5.54   |     —     |
| Qwen2.5-Math-7B-Instruct   |     —     |   16.87   |     —     |

int4 = GPTQ checkpoint (`-GPTQ` / `-GPTQ-Int4`). fp32 not
re-measured for the 1B and Math models. All values include the
~0.5 GB CUDA context.

**Takeaway:** fp16 roughly halves fp32, as expected; GPTQ-int4
roughly halves fp16 again (~3–5× below fp32). The Math-Instruct
variants weigh the same as their general-Qwen counterparts —
Math-7B at 16.87 GB ≈ general 7B at 16.86 (same architecture; the
math fine-tune doesn't change weight size). Math-1.5B is a new
small point at 5.54 GB — smaller than general Qwen-3B (8.41)
despite being the stronger math model. For the fit-LLM+PRM-on-32GB
question (M4), this 1.5B math LLM + a ~14 GB fp16 PRM sums to
~19.5 GB — far more comfortable than the 7B fp16 LLM (16.9 GB,
~1.3 GB left after a fp16 PRM). fp32 at 31.4 GB barely fits the
7B LLM alone, so it is not a viable baseline on this 32 GB card.

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
