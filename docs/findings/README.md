# Findings

One topic per file, split by kind:

- **coding-findings/** — repo behavior: environment sensitivity,
  library quirks, format gotchas — anything that affects how
  experiments are run or interpreted, independent of which
  algorithm/model is being studied.
- **exp-findings/** — empirical results about the algorithms
  themselves (accuracy/throughput/memory tradeoffs across configs).
  Headline numbers live in W&B and the
  [exp-comparison](../exp-comparison.md) tuning
  tables; notes here are write-ups that need more room than a table
  cell.

Decisions motivated by a finding go in
[decisions-log.md](../decisions-log.md) and reference it.

## coding-findings/

- [gptqmodel-transformers-pin.md](coding-findings/gptqmodel-transformers-pin.md) — 2026-06-12, gptqmodel ≥5.8 vs vllm's transformers<5 cap
- [hf-vllm-memory-residual.md](coding-findings/hf-vllm-memory-residual.md) — 2026-06-12, HF model deletion doesn't free GPU memory
- [library-version-trajectory-completeness.md](coding-findings/library-version-trajectory-completeness.md) — 2026-06-11, library version changes generated trajectory completeness
- [prm-step-split-trailing-separator.md](coding-findings/prm-step-split-trailing-separator.md) — 2026-07-06, PRM.score splits a bogus trailing empty step; RLHFlowPRM can mask a bad step, QwenPRM only tracks it

## exp-findings/

- [prm-batch-size-throughput-memory.md](exp-findings/prm-batch-size-throughput-memory.md) — prm_batch_size sweep: throughput/memory tradeoff + why the pass@gb gap isn't statistically real
- [ds-alpha-diversity-bonus-plateau.md](exp-findings/ds-alpha-diversity-bonus-plateau.md) — 2026-06-24, ds_alpha is a switch not a dial: 0→10 is a large real jump, 10→1000 is flat within noise
