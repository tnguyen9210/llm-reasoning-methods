# unittests/

Notebooks and scripts for smoke tests, unit tests, benchmarks, and
result inspection. GPU column: Y = requires GPU, N = CPU-only.

## Tests

| File | What it tests | GPU |
|------|--------------|-----|
| `unittest_extract_answer_v1.ipynb` | Answer extraction from boxed notation in model predictions | N |
| `check_trajectory_completeness.py` | Classifies search-result completions as complete / ends-at-step-sep / mid-sentence; per-depth breakdown | N |
| `test_step_separator_affect_generation.ipynb` | Step separator `\n\n` survival through templating and effect on generation (EOS vs next step); env gate | Y |
| `test_transformers_batched_generation_v1.ipynb` | Batched generation with HF Transformers; left vs right padding | Y |
| `test_transformers_batched_prompt_scoring_v1.ipynb` | Log-probability computation and last-token embeddings under padding with HF Transformers | Y |
| `test_vllm_batched_generation_v1.ipynb` | Batched generation with vLLM (continuous batching) | Y |
| `test_vllm_batched_prompt_scoring_v1.ipynb` | Batched prompt log-probability scoring with vLLM | Y |
| `test_prm_llama_scoring_v1.ipynb` | Smoke test for Llama3.1-8B-PRM-Deepseek-Data: inline per-step scoring via the marker-token approach | Y |
| `test_prm_qwen_scoring_v1.ipynb` | Smoke test for Qwen2.5-Math-PRM-7B: per-step scoring via `<extra_0>` separator tokens (fp16 for V100) | Y |
| `test_prm_rlhflow_scoring_v1.ipynb` | Smoke test for the `RLHFlowPRM` wrapper: single-step and batched paths on the flamingo toy example | Y |

## Benchmarks

| File | What it benchmarks | GPU |
|------|-------------------|-----|
| `benchmark_speed_simple_generation_v1.ipynb` | Inference latency and throughput: HF Transformers vs vLLM at various memory utilization settings | Y |
| `benchmark_speed_bon_models_v1.ipynb` | BoN search speed across models (Llama 3.2 1B/3B, Qwen 2.5 3B/7B) | Y |
| `benchmark_speed_bon_quant_v1.ipynb` | BoN search speed across quantization levels (fp16 vs GPTQ-int4) | Y |
| `benchmark_speed_bon_variants_v1.ipynb` | BoN generation: native n-sampling vs prompt-duplication | Y |
| `benchmark_llm_mem_sizes_v1.ipynb` | GPU memory usage of generative models with HF Transformers and vLLM | Y |
| `benchmark_prm_mem_sizes_v1.ipynb` | GPU memory usage of PRMs with HF Transformers | Y |

## Exploratory

| File | What it examines | GPU |
|------|-----------------|-----|
| `examine_llm_chat_templates_v1.ipynb` | Chat-template rendering per LLM: native vs. former custom `custom_chat_template`; whether the `\n\n` step separator survives templating; BOS | N |
| `examine_llm_system_prompt_v1.ipynb` | System-prompt format and behavior per LLM (currently the shared `GenConfig.system_prompt`) | N |
| `examine_completions_log_probs_v1.ipynb` | Per-step log-probabilities and perplexities of generated completions | Y |
| `examine_completions_prm_scores_v1.ipynb` | PRM scores across completion steps; correctness correlation | Y |
| `qwen_prm_toy_example.ipynb` | Toy example: load and score with Qwen2.5-Math-PRM-7B | Y |

## Modules

| File | Description | GPU |
|------|-------------|-----|
| `rlhflow_prm.py` | `RLHFlowPRM` wrapper class with `score()` entry point for per-step and batched PRM scoring | Y |
| `notebook_utils.py` | Shared helpers: `gpu_mem_used_gb` (driver-level GPU memory, always flushes), `measure_inference` (timed HF/vLLM generation), `benchmark_bon_speed_llm_model` (vLLM BoN speed by model), `benchmark_bon_speed_llm_quant` (vLLM BoN speed by quantization config) | Y |
