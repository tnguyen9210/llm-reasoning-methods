# Environments

Two micromamba envs are used side by side. `py311` is canonical for
experiment results (see `decisions-log.md`, 2026-06-11); `vllm1` is kept
for comparing environment effects on generated outputs.

Activate with `micromamba activate py311` (or `vllm1`); experiments
launch from a `py311` shell via `exp-run` (`srun --overlap`).

## Key package versions

Snapshot 2026-06-12. Full lists: `docs/envs/py311.txt`,
`docs/envs/vllm1.txt` (machine-generated via `pip freeze`; regenerate
with `pip freeze > docs/envs/<env>.txt` after any env change).

| Package | py311 (canonical) | vllm1 (comparison) |
|---|---|---|
| Python | 3.11.11 | 3.11.15 |
| torch | 2.10.0+cu126 | 2.5.1 |
| transformers | 4.57.6 | 4.45.2 |
| vllm | 0.18.1 | 0.6.4 |
| tokenizers | 0.22.2 | 0.20.3 |
| numpy | 2.2.6 | 2.4.6 |
| xformers | 0.0.30 | 0.0.28.post3 |
| flashinfer-python | 0.6.6 | — |
| jinja2 | 3.1.6 | 3.1.6 |
| accelerate | 1.13.0 | 1.13.0 |
| datasets | 4.8.5 | 4.8.5 |
| wandb | 0.25.1 | 0.27.1 |

## Hardware constraints (V100 dev node)

The interactive dev node has a Tesla V100S-PCIE-32GB (Volta,
compute capability 7.0 / sm_70). Two hard limits follow:

- **vllm ≤ 0.18.x, torch cu126 builds only.** vLLM dropped Volta
  from its build targets after 0.18.x, and PyTorch ships sm_70
  kernels only in cu126 wheels (cu128/cu130 dropped Volta). Newer
  versions fail at the first kernel launch with
  `CUDA error: no kernel image is available for execution on the
  device`.
- **No bf16.** bfloat16 needs compute capability >= 8.0; use
  `dtype="float16"` / `torch.float16` on this node. Notebooks that
  default to `torch.bfloat16` (e.g. the PRM memory benchmarks) need
  the dtype switched when run here.

History: on 2026-06-12 `py311` was downgraded from
vllm 0.22.1 / torch 2.11.0+cu130 / transformers 5.12.0 to the
versions above to restore V100 support (vllm 0.18.1 pins
`transformers<5`, which pulled transformers back to 4.57.6).
Verified by a single-prompt vLLM generation on the V100.

## Known issues / version conflicts

Leftover pip resolver complaints in `py311` after the downgrade —
harmless for the core generate/score pipeline (smoke-tested), but
relevant if these packages are used:

- `gptqmodel` is pinned to 5.7.0 (with `kernels==0.12.0`): versions
  ≥5.8 require `transformers>=5.2`, incompatible with vllm's
  `transformers<5` cap. 7.0.0 crashed at import under 4.57.6;
  resolved 2026-06-12, see
  `findings/coding-findings/gptqmodel-transformers-pin.md`.
- `xformers 0.0.30` was built against `torch==2.7.0`.
- `trl 0.9.6` requires `numpy<2`.
- `datasets 4.8.5` wants `fsspec<=2026.2.0` (installed: 2026.4.0).

## Why environments are pinned at all

Generated outputs are environment-sensitive: a step-separator lost
during chat templating flips generation from "continue next step" to
"emit EOS" (see
`findings/coding-findings/library-version-trajectory-completeness.md`
and `unittests/test_step_separator_affect_generation.ipynb`, the
env-gate notebook). Re-run that notebook after any change to either
env.
