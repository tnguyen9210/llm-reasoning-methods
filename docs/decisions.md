# Design decisions

Append-only log of decisions git history can't show: cross-cutting
design choices that span multiple files, and deliberate omissions —
things chosen *not* to be built, and why. Newest first. One `##`
section per decision. Titles carry one or two area prefixes
(`Area:` or `Area, Area:`) so skimming groups by eye and
`grep '^## .*Area'` gives a per-topic view.

## 2026-06-13 — Prompting: use native chat templates, not one custom template

**Context:** the search code applied a single hardcoded Llama-3.1
`custom_chat_template` to *every* model. The
`examine_chat_templates_v1` notebook
([findings](findings.md) and the vault note `llm-chat-templates`)
showed why it was added — Llama's *native* template silently trims
the trailing `\n\n` step separator — but also that it forces Llama
format onto Qwen (overriding `<|im_start|>`) and drops Llama's BOS.
**Decision:** stop overriding the template. Use each model's
**native** chat template, and keep the separator with the existing
strip-and-reappend (`removesuffix("\n\n")` before
`apply_chat_template`, re-append after). Drop the
`tokenizer.chat_template = config.custom_chat_template` override in
the search code (done first in `mcts_cnt_search_v05_00_00`; other
search files migrate one at a time). `custom_chat_template` stays
in the config as a vendored asset but is no longer applied.
**Why:** the custom template's only real job was preserving the
separator, and strip-and-reappend already does that
(`apply_chat_template` is the one place the separator is lost;
re-appending after it is correct by construction). Native templates
give each model its own in-distribution format, which removes a
**confound**: a single forced template could penalize one family
(e.g. Qwen getting Llama format) and contaminate cross-model
comparisons. Verified that strip-and-reappend on native templates
produces a valid prompt ending in `\n\n` for both Llama and Qwen,
with no `continue_final_message` crash.
**Revisit if:** a model's native template can't be made to preserve
the separator even with strip-and-reappend, or the backlogged M2
template A/B (`llm-prm-deep-dive`) shows native is *worse* than the
custom template for some model.

## 2026-06-13 — Configs: adopt structured Hydra config schema

**Context:** the upcoming sweep spans ~6 LLMs (Llama/Qwen/Phi ×
3B/7B), 2 PRMs, 4–5 datasets, and several search methods — a
combinatorial matrix where the sum of options (~17) is far below
their product (~120). Launchers currently load a Hydra
`DictConfig`, then hand-copy ~13 fields into a separate
`sal.Config` (e.g. `generate_mcts_cnt.py`).
**Decision:** define a typed, grouped config schema in
`utils/configs.py` (`GenConfig` / `RunConfig` / `LLMConfig` /
`PRMConfig` / `DataConfig` + base `SearchConfig` with one subclass
per method, composed as `ExpConfig`) and bind YAML config groups
(`conf/llm/`, `conf/data/`, `conf/search/`, …) onto it via Hydra
structured configs. Notebooks import the same dataclasses directly
(no Hydra). Migrate one launcher (`generate_mcts_cnt`) end-to-end
as a pilot before propagating; an adapter keeps the existing flat
`core/` search code working without rewriting it.
**Why:** the matrix is past the threshold where grouped config
(one file per option, combinations on the CLI) beats flat config
(one near-duplicate file per combination); the hand-copy block is
fragile (a dropped line silently keeps a wrong default). Full
rationale — schema-vs-values, nesting benefits, the three axes,
when Hydra is justified, the pilot discipline — in the vault guide
`managing-experiment-config.md`.
**Revisit if:** the experiment matrix collapses to a handful of
combinations (then flat config is simpler), or the pilot shows the
`core/` flat-config coupling is cheaper to rewrite than to adapt.

## 2026-06-12 — Benchmarks: no HF Transformers BoN speed benchmark

**Context:** considered a Transformers-based counterpart to
`unittests/benchmark_speed_bon_models_v1.ipynb` to compare Best-of-N
generation speed across backends.
**Decision:** benchmark BoN speed under vLLM only; no separate HF
Transformers BoN benchmark.
**Why:** the simple-generation benchmark
([benchmarks.md](benchmarks.md), 2026-06-12) already shows vLLM
~4.3× faster than HF eager on two models. BoN is generation-bound,
so at n=32 the gap only widens; the benchmark would cost GPU-hours
and change no decision — vLLM is the search backend either way.
**Revisit if:** an experiment requires an HF-only pipeline, or HF
Transformers gains continuous batching.

## 2026-06-11 — Env, Experiments: py311 env is canonical; old-env results are invalid

**Context:** the 2026-06-11 finding in
[findings.md](findings.md) — the old stack (vLLM 0.6.4 /
transformers 4.45.2 / torch 2.5.1) silently dropped the trailing
step separator from continuation prompts, producing ~80% abandoned
trajectories (now guarded in code by strip-and-reappend), and
returned incompatible tokenizer outputs in PRM scoring.
**Decision:** all experiments run in the py311 environment. Results
generated under the old stack (early CNT-MCTS and BL-MCTS runs) are
not comparable and must be re-run before drawing conclusions.
**Why:** outputs differ in content, not just performance; mixing
stacks would corrupt any cross-run comparison. The code guard fixes
the known separator issue, but other version-sensitive behaviors may
remain — one canonical stack removes the variable entirely.

## 2026-06-11 — Docs: lineage lives in docs, not in module docstrings

**Context:** core files carried `History` blocks recording how each
version evolved. A `.py` file should document the *current*
implementation; evolution is a separate concern.
**Decision:** module docstrings describe only the current algorithm,
plus a one-line sibling note where multiple variants coexist (e.g.
BL-MCTS v01/v02). Version lineage moves to
[algorithms.md](algorithms.md); reasons for changes go here.
**Why:** chronological logs inside source files duplicate git history
and rot; but with multiple versions coexisting as files, the
*relationship between live variants* still needs documenting — that is
current-state information and stays in the docstring.

## 2026-06-11 — Configs: Hydra run outputs disabled

**Context:** every Hydra invocation created timestamped `outputs/` /
`multirun/` directories with config snapshots and logs.
**Decision:** all configs set `hydra.output_subdir: null`,
`hydra.run.dir: .`, and disable `job_logging` / `hydra_logging`.
**Why:** W&B already records configs and metrics; experiment outputs go
to `results/`. The Hydra dirs were pure clutter and were gitignored
anyway.

## 2026-06-11 — Configs: `gen_budget` is set directly; `num_batches` dropped

**Context:** configs exposed `num_batches`, and launchers computed
`gen_budget = num_batches * max_depths`. The derived quantity, not the
factor, is the semantically meaningful budget.
**Decision:** configs expose `gen_budget` directly (e.g. `80`);
launchers pass it through unchanged. For BoB, `gen_budget` is instead
distributed evenly across depths (`gen_budget / max_depth` per depth)
to keep comparisons with MCTS fair.
**Why:** MCTS charges budget per expansion regardless of depth, so the
per-depth factorization was an artifact of the BoB framing; setting the
total directly makes sweeps and cross-algorithm comparisons explicit.

## 2026-06-11 — Naming, Configs: BoN keeps `n`; MCTS uses `batch_size`; SAL untouched

**Context:** three distinct things were called a batch size: SAL's
`Config.n`, the number of MCTS expansion candidates, and the PRM
scoring batch. MCTS code was overloading `config.n` for generation
batching.
**Decision:**
- BoN keeps `config.n = cfg.n` — `n` is semantically "number of
  candidates to generate and select from", the defining parameter of
  best-of-n.
- MCTS configs and code use `batch_size` (`config.batch_size`);
  `config.n` is no longer set by MCTS launchers.
- SAL's `Config` class is never modified — it is an upstream library.
- PRM scoring batches are `prm_batch_size` (or a hardcoded literal at
  the call site), never conflated with generation `batch_size`.
**Why:** the same name for different algorithmic quantities caused
real confusion (OOM debugging traced to the wrong "batch size");
separate names keep the terminology aligned between code, configs, and
written notes. Also standardized `max_depths` -> `max_depth`
(singular) across MCTS files at the same time.
