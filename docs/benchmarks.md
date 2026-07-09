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

## Best-of-N search speed across quantization levels

**Notebook:** `unittests/benchmark_speed_bon_quant_v1.ipynb`  
**Date:** 2026-06-19  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Config:** BoN n=32, 5 MATH level-4 questions, 2 trials +
1 warmup, temperature=0.8, max_tokens=2048, vLLM gmu=0.3,
enforce_eager. int4 = GPTQ checkpoint. n lowered from 256 to 32
(and gmu from 0.5 to 0.3) to match the search's real BoN width and
`benchmark_speed_bon_models_v1`, so the two notebooks are now
directly comparable; qwen-7b fp16 and qwen-math-7b fp16 are
commented out (too large to co-reside at this gmu) — their
GPTQ-int4 variants stay. Supersedes the 2026-06-12 n=256/gmu=0.5 run.

| Config              | GPU (GB) | s/trial |  s/question |
|---------------------|----------|---------|-------------|
| llama-1b fp16       |   2.80   |  21.61  |     4.32    |
| llama-3b fp16       |   6.47   |  63.20  |    12.64    |
| llama-3b gptq       |   2.57   |  67.87  |    13.57    |
| qwen-3b fp16        |   8.41   | 122.21  |    24.44    |
| qwen-3b gptq-int4   |   4.63   | 102.38  |    20.48    |
| qwen-7b fp16        |  16.86   |    —    |      —      |
| qwen-7b gptq-int4   |   7.83   |  98.09  |    19.62    |
| qwen-math-1.5b fp16 |   5.54   |  71.19  |    14.24    |
| qwen-math-7b fp16   |  16.87   |    —    |      —      |

GPU column is HF Transformers model-weight footprint (from
`unittests/benchmark_llm_mem_sizes_v1.ipynb`, see the table
below), not the vLLM process footprint during this BoN run.
qwen-7b fp16 and qwen-math-7b fp16 were not speed-tested in this
run (commented out, too large to co-reside at gmu=0.3) — memory
only, sourced from the GPU-memory table.

**Takeaway:** At 3B, GPTQ-int4 still does not pay off — llama-3b
GPTQ is ~1.07× *slower* than fp16, and qwen-3b int4 is now ~1.19×
*faster* than fp16 (reversed from the old n=256/gmu=0.5 run, where
it was within noise) — the dequantization overhead is close to a
wash at this scale and sensitive to n/gmu. qwen-7b int4 (19.62
s/q) lands close to qwen-3b fp16 (24.44) and qwen-3b int4 (20.48)
despite having ~2.3× the parameters, consistent with int4 paying
off more as model size grows. Llama-1b fp16 (4.32 s/q) and
qwen-math-1.5b fp16 (14.24) are new low-end reference points.
Caveat: small sample (5 questions, 2 trials), so treat s/question
as indicative, not precise; qwen-7b fp16 and qwen-math-7b fp16
weren't speed-tested this run (commented out for VRAM headroom at
gmu=0.3) — memory-only rows — so the int4-win-at-7B claim from the
2026-06-12 run can't be re-confirmed at this gmu.

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

| Model         | fp32 (GB) | fp16 (GB) | int4 (GB) |
|---------------|-----------|-----------|-----------|
| llama-1b      |     —     |    2.80   |     —     |
| llama-3b      |   12.47   |    6.47   |    2.57   |
| qwen-3b       |   14.30   |    8.41   |    4.63   |
| qwen-7b       |   31.43   |   16.86   |    7.83   |
| qwen-math-1.5b|     —     |    5.54   |     —     |
| qwen-math-7b  |     —     |   16.87   |     —     |

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

## Single-question trace: cnt-mcts-bl vs sem-mcts-bl

**Notebook:** `unittests/examine_search_trace_v1.ipynb`  
**Date:** 2026-07-09  
**Hardware:** Tesla V100S-PCIE-32GB  
**Env:** py311 (vllm 0.18.1, torch 2.10.0+cu126)  
**Model:** Llama3.2-1B-Instruct, `llm_prm` (embeds source for
sem)  
**Config:** MATH level-4 question 0 ("angle between two lines"),
`gen_budget=80`, `TRIAL_IDX=0` (seed 100000), single trial, one
question — a qualitative smoke comparison, not a scored
benchmark.

| Method | search time (s) | completions | last phase | phase_depths | nodes@max_depth |
|--------|------------------|-------------|------------|--------------|------------------|
| mcts_bl_cnt_v01 (cpuct=2.0) | 153.5 | 0 | 79 | [] | 0 |
| mcts_bl_sem_v01 (ds_alpha=100, lam=0.01, schedule=global) | 276.7 | 16 | 84 | [11, 7, 9, 6, 15] | 0 |

**Takeaway:** On this single question, `mcts_bl_cnt_v01`
exhausted its full 80-generation budget without producing a
single completion — consistent with the ~18% zero-completion
rate documented in `docs/findings/` for the frontier + PUCT
combination. `mcts_bl_sem_v01` (frontier selection + semantic
diversity bonus) reached 16 completions at varied depths
(6–17) on the identical question/budget/seed, at roughly 1.8×
the wall-clock cost (276.7s vs 153.5s) — the added cost comes
from the diversity term's embedding + covariance-fold
machinery on every selection. This is one question, one trial;
not a substitute for the scored pass@gb comparisons in
`docs/exp-comparison.md`, but a concrete illustration of why
bl_cnt's zero-completion issue motivated exploring bl_sem as an
alternative frontier-selection strategy.
