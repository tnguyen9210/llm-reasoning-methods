# HF model deletion does not free GPU memory

*2026-06-12*

**Observation:** after loading a HF Transformers model with
`device_map="cuda:0"` and then running:

```python
del model_hf
del tokenizer
gc.collect()
torch.cuda.empty_cache()
```

`torch.cuda.mem_get_info()` still reports ~6.12 GB used — roughly
the full footprint of the model weights.

**Why:** `torch.cuda.mem_get_info()` reports driver-level memory,
which includes the PyTorch CUDA allocator's *reserved* pool.
`empty_cache()` releases memory from the allocator back to the
driver, but only if there are no remaining live tensors pointing
into it. Model weights loaded via `device_map` are held by the
model's `state_dict` tensors; deleting the Python model object
drops the reference count but garbage collection is not guaranteed
to run immediately. Even after `gc.collect()`, the CUDA allocator
may retain the pool for reuse rather than returning it to the
driver.

In practice the residual is not released even when a vLLM engine
is subsequently created. The ~6 GB sits on top of whatever vLLM
allocates for its own budget. For example, with a 32 GB GPU and
`gpu_memory_utilization=0.7`, the expected vLLM budget is
~22.4 GB, but observed GPU memory after vLLM init is ~29.29 GB —
consistent with 22.4 GB (vLLM) + 6.12 GB (unreleased HF
residual) + ~0.75 GB (CUDA context).

**Implication:** if you intend to run HF and vLLM sequentially
in the same kernel, account for the HF residual when choosing
`gpu_memory_utilization` — the effective budget available to vLLM
is reduced by the amount the HF model left behind. The safe
pattern is to restart the kernel between HF and vLLM runs if you
need the full GPU for vLLM.
