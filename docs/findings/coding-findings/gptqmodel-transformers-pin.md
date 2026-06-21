# gptqmodel is capped at 5.7.x by vllm's transformers pin

*2026-06-12*

**Observation:** `from gptqmodel import GPTQModel` under gptqmodel
7.0.0 fails at import with `AttributeError:
module 'transformers.integrations.hub_kernels' has no attribute
'lazy_load_kernel'` — its import-time compat patch assumes the
transformers-5.x API with no `hasattr` guard.

**Why:** gptqmodel ≥5.8 requires `transformers>=5.2`, while
vllm 0.18.1 caps `transformers<5`. The ranges don't overlap, so any
gptqmodel ≥5.8 can never work in `py311`; pip installs it anyway if
told to. The newest compatible release is 5.7.0
(`transformers>=4.57.1`). Two traps when installing it:

- `pip install gptqmodel==5.7.0` side-upgrades transformers to 5.x
  (no upper cap in gptqmodel's metadata) — transformers 4.57.6,
  huggingface_hub 0.36.2, and fsspec 2026.4.0 must be re-pinned
  afterwards per the freeze file.
- gptqmodel 5.7.0 pulls in `kernels`; kernels ≥0.13 requires
  huggingface_hub ≥1.x, so `kernels==0.12.0` is the matching pin.

**Implication:** `gptqmodel==5.7.0` + `kernels==0.12.0` is the
pinned pair (recorded in `docs/envs/py311.txt`). Never
`pip install -U gptqmodel` in this env; after any reinstall, verify
transformers is still 4.57.6 before trusting results.
