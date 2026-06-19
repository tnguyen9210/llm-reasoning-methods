# Design decisions

Append-only log of decisions git history can't show: cross-cutting
design choices that span multiple files, and deliberate omissions —
things chosen *not* to be built, and why. Newest first. One `##`
section per decision. Titles carry one or two area prefixes
(`Area:` or `Area, Area:`) so skimming groups by eye and
`grep '^## .*Area'` gives a per-topic view.

## 2026-06-18 — Hardware, Experiments: fit 7B generator + PRM on a V100S via int4 LLM (primary) or a small PRM (fallback)

**Context:** M3 (semantic exploration) needs to scale past the 1B
generator — semantic diversity showed no gain at Llama-3.2-1B, possibly a
capacity issue, so the method needs a 7B+ generator. The search loop
holds the generator (vLLM) and the PRM (HF) **co-resident** on one GPU
(it interleaves generation and per-step scoring), and the target card is
a **V100S (32 GB, sm_70, fp16-only)**.
**Decision:** 7B generator + **fp16 8B PRM** does NOT fit at full
precision (see arithmetic), so two feasible paths instead, both within
the V100S — this is NOT blocked on bigger GPUs:
- **Primary — int4 (GPTQ) 7B generator + fp16 8B PRM.** ~7.8 + ~14.6 ≈
  **22 GB**, leaving real KV-cache headroom, and **keeps the
  already-validated 8B PRM** (no PRM-swap confound). Already scoped as M4
  in `llm-prm-deep-dive`. Risk: int4 generation quality — verify it isn't
  visibly degraded before committing.
- **Fallback — small (~1.5B) PRM + fp16 7B generator.** Used only if int4
  generation proves unacceptable. Requires finding + validating a small
  PRM (none in the current survey — both surveyed PRMs are 7B/8B), gated
  on the `examine_prm_scores_*` notebooks confirming it still scores
  steps sanely. Open investigation in `llm-prm-deep-dive`.
**Why:** at fp16, 7B weights (~16.9 GB measured) + 8B PRM (~14.6 GB) ≈
**30.7 GB** before any KV cache — fits 32 GB only with ~1.3 GB to spare,
too tight to run (per the M4 measurements in `llm-prm-benchmarks` /
docs/benchmarks.md). Quantizing the *generator* to int4 (measured 7B-Qwen
GPTQ = 7.83 GB) is the cheapest fix and, unlike swapping the PRM, doesn't
change the reward model — so the M3 comparison (semantic vs. count-based)
stays clean against the same 8B PRM. The small-PRM route also fits but
adds a confound (the small PRM must then be used consistently across
baseline AND method, and it may score worse, washing out the signal), so
it's the fallback, not the default.
**Revisit if:** int4 generation quality is unacceptable (switch to the
small-PRM fallback), or an ≥A100-class GPU becomes available (then fp16
7B + fp16 8B PRM fits directly and neither workaround is needed).

## 2026-06-18 — Search, Configs: online-vs-fixed centering mean is a flag, not a version

**Context:** to test the fixed-mean claim above, we want to compare it
against an online-updated centering mean (μ initialized fresh and
updated with each new embedding). Question: new `vNN` file, or a flag
on the existing one?
**Decision:** a flag — `centering_mode: "fixed" | "online"` on
`MCTSSemV01Config` (inherited by v02), default `"fixed"`. `embeds_center`
stays the on/off master switch; `centering_mode` only chooses which mean
when it's on. `config_name` appends `--center-{fixed|online}` *only when*
`embeds_center` is true. Online μ is per-question mutable state living on
the `MCTS` instance (Welford `_mean`/`_count`, reset per question), so
`_extract_embeds` gains an optional running-state argument threaded
through `_embed_candidates`/`_generate_candidates`; the fixed path
ignores it. Update discipline: center with the *current* μ, then fold the
raw projected vector in (no self-leakage).
**Why:** same lineage, same algorithm, same embedding source — only how μ
is produced differs, which is exactly what the two-tier convention
(major `vNN` = lineage; behavioral variants = config flags;
[algorithms.md](algorithms.md)) reserves a flag for. The comparison is an
ablation, and ablations belong in the run name (like `enorm-True/False`),
not in duplicated files. A new `vNN` is for a changed *contract* (search
algorithm, node/tree structure, result format) — none here. Note the
online arm is deliberately the theoretically-unsound baseline: a drifting
μ_t makes the feature map non-stationary and the covariance `V`
incoherent (the very thing the fixed mean avoids); testing against it is
the point. Full rationale in the vault note
`sparse-projection-and-embedding-normalization`.
**Revisit if:** online centering turns out to need a structurally
different search loop (then it earns its own version), or the ablation
shows centering mode is irrelevant (then drop the flag).

## 2026-06-18 — Search: sparse random projection of PRM embeds; fixed matrix; pool→project→center→normalize

**Context:** mcts_sem v02 sources diversity embeddings from the PRM's
last-layer hidden states (4096-dim for Llama3.1-8B-PRM), which sizes the
covariance `V` (4096×4096). To shrink `V` we add an optional projection
to a smaller dim. Reference: verl-recipe `elliptical_reward_model_worker`
(sklearn `SparseRandomProjection`).
**Decision:** add `embeds_proj: "none" | "sparse"` to
`MCTSSemV01Config`; `embeds_dim` keeps its meaning as the size of `V` and
becomes the *post*-projection dim (the raw source dim is read off the
pooled tensor, not configured). When `"sparse"`, project the pooled
vector to `embeds_dim` via sklearn `SparseRandomProjection(density=
"auto")` (JL-optimal sparsity 1/√d). The projection matrix is **fixed for
the whole run**, built once and cached in a module-level dict keyed by
`(in_dim, out_dim, seed)`. The seed is **not** a config knob — it's a
hardcoded internal constant (`_PROJ_SEED = 0`): JL holds w.h.p. for any
seed so the choice is empirically irrelevant, and pinning it internally
still guarantees a resume rebuilds the identical matrix. v02 YAML:
`embeds_dim: 512`, `embeds_proj: sparse`, applied to both `prm` and
`policy` sources. Reordered `_extract_embeds` to
**pool → project → center → normalize** (was normalize→center); centering
moved after projection (mean lives in the projected space) and a shape
guard raises if `embeds_mean`'s dim ≠ the projected dim. Behavior-
preserving at the `embeds_center=False` default. Added scikit-learn 1.9.0
to the py311 env (sklearn chosen over a hand-rolled numpy matrix to match
the reference exactly).
**Why:** the matrix must be fixed for *correctness*, not convenience —
`V = λI + Σ uuᵀ` accumulates features across time, so a drifting map
puts past and present vectors in different bases and makes `V⁻¹`
meaningless (this is also why we reject an online projection / online
mean). Random projection is data-free (JL holds uniformly over all
vectors), so the goal is to cheaply preserve *all* pairwise geometry into
a tractable dim, not to learn a "best" subspace. center→normalize is the
correct order because the mean must be subtracted in the linear space and
normalization is non-linear (must be last); projection is ~linear so it
composes cleanly before centering. Full derivation (JL, anisotropy, the
center/normalize ordering, computing the mean) in the vault note
`sparse-projection-and-embedding-normalization`.
**Status (2026-06-18):** validated. Unit: projector fixed/seeded, JL
distance ratio ~1.00±0.03, all pool/proj/center/normalize paths + guards.
GPU end-to-end on a 2-question v02 run: projection over real 4096-dim PRM
embeds with `V` sized 512 completed cleanly (first attempt no-op'd via
resume because `config_name` didn't encode the projection knobs — fixed,
see next decision). Bonus finding: ~2.5× faster than the un-projected v02
baseline (~147 vs ~362 s/question), because `_diverse_select`'s per-pick
matrix inverse is O(d³) and d dropped 8× (4096→512); the PRM forward cost
is unchanged, so the win is entirely in the covariance math.
**Revisit if:** the projected dim proves too small to preserve the
diversity signal (raise it, or set `embeds_proj=none` + `embeds_dim=4096`
to feed raw PRM embeds), or a data-adaptive subspace (PCA on PRM embeds)
is wanted — that's a separate experiment, not a mutation of this fixed R.

## 2026-06-18 — Configs: config_name should encode every knob that changes results

**Context:** the result dir is `config_name(cfg)`, and the launcher's
resume logic skips any trial whose `.done` marker already exists in that
dir. A v02 smoke test with `embeds_proj=sparse`/512 silently resumed and
skipped the trial from an earlier non-projection v02 run, because
`config_name` didn't encode the projection knobs — both runs mapped to
the *same* dir. The projection code never ran.
**Decision (principle):** any config knob that changes the produced
results must appear in `config_name`, so distinct configs get distinct
result dirs and the resume/`.done` mechanism can't conflate them.
Implemented for the `mcts_sem` branch: a `--proj-{mode}{embeds_dim}` tag
(e.g. `--proj-sparse512`) is appended **only when projection is on**, so
no-projection runs keep their prior names and existing dirs/W&B runs
don't orphan. `embeds_dim` rides inside that tag (not as an always-on
field) because it only affects results under projection — with
`proj="none"` it must equal the raw pooled dim, so it isn't a free knob.
The projection seed is *not* encoded: it's a fixed internal constant, so
it never varies between runs.
**Why:** `config_name` is the experiment's identity key — for the result
path, for W&B run names, and (transitively) for resume safety. A knob
that affects outputs but isn't in the name is an ablation hazard: two
different experiments overwrite or resume into each other. This is the
same discipline already applied to `enorm`/`ecenter`/`cpuct` etc.
**Revisit if:** a knob is purely cosmetic (no effect on results) — those
stay out of the name to keep dirs short (e.g. `embeds_source` is omitted
because it's implied by the v01/v02 prefix).

## 2026-06-17 — Experiments, Configs: W&B run-id sidecar so scores reattach to the generation run

**Context:** generation (`generate_*`) and post-processing
(`prepare_scored_dataset` / `compute_stats`) are separate processes —
the second runs after the run is closed. We want eval metrics logged
onto the *same* W&B run as the generation, not a new one.
**Decision:** `generate_*` writes its W&B run id to a sidecar file
`{result_dir}/wandb_run_id.txt`; post-processing reads it and reattaches
via `wandb.init(id=..., resume="must")`, logging `eval/{metric}(+_sem)`
onto that run. The id lives in a **file, not in `config_name`**. Missing
sidecar (older runs) is handled gracefully (skip W&B).
**Why:** scores + stats belong on one run for a coherent W&B view, but
the reattach key can't be baked into the result-dir name — encoding a
fresh run id in `config_name` would give every re-run a new path and
break the "dir is uniquely determined by config" invariant
(see the 2026-06-18 `config_name` decision). A sidecar decouples the
mutable run id from the stable config identity.
**Revisit if:** W&B adds a first-class way to attach late metrics to a
closed run, or runs move to a store where a content-addressed id is
natural.

## 2026-06-17 — Experiments: resume interrupted multi-trial runs; trial-body write order

**Context:** multi-trial runs on rented/preemptible GPUs get killed
mid-run (OOM, preemption). Re-running from scratch wastes completed
trials and mints duplicate W&B runs.
**Decision:** reattach to the same run (`resume="allow"`, via the run-id
sidecar above) and skip any trial that already wrote a per-trial `.done`
marker. The trial body is ordered **dump → log timing → write marker →
score**, where the dump is an atomic temp-write + rename. `compute_stats`
*also* calls `wandb.run.summary.update(...)` (not just `wandb.log`)
because `log()` doesn't reliably propagate to the run summary on a
`resume` reattach.
**Why:** the ordering makes the `.done` marker mean exactly "generation
finished and raw results are safely on disk" — a crash before the marker
leaves no marker and the trial is redone cleanly; a crash after it leaves
valid results that resume skips. Atomic rename ensures a crash mid-write
never leaves a half-written `.jsonl` under the real name. Scoring runs
*after* the marker because it's separately re-runnable
(`prepare_scored_dataset`) and a scoring failure must not discard raw
generation. The `summary.update` is a W&B quirk workaround, logged so
it isn't "cleaned up" later and silently lost.
**Caveat (see 2026-06-18):** resume keys off the `.done` marker in the
`config_name` dir, so any result-affecting knob missing from
`config_name` lets an unrelated run resume-skip a trial it shouldn't.

## 2026-06-17 — Configs: self-describing run names (config_name encodes level, model, template)

**Context:** run names / result dirs didn't carry the difficulty level,
model, or chat-template mode, so W&B runs and `results/` dirs weren't
self-describing and needed redundant side tags.
**Decision:** `config_name` bakes in `--level-{level}` (prm800k only),
the model name (minus the redundant `-Instruct`), and `--tmpl-{custom|
native}`, dispatching per search method; the now-redundant level tag was
dropped from `wandb.init`. Brought `mcts_cnt` and `bon` to the same
naming convention (and `mcts_cnt` now honors `use_custom_template` so the
`tmpl-` tag is meaningful).
**Why:** the run name is the experiment's identity — making it
self-describing means a `results/` path or W&B run is interpretable on
its own, and parallel runs across levels/models/templates don't collide.
This is the positive precedent the 2026-06-18 "encode every
result-affecting knob" decision generalizes (and which the projection
knobs currently violate).
**Revisit if:** names grow unwieldy — then move low-cardinality axes
(e.g. level) back to the parent dir alone, keeping only result-affecting
knobs in the leaf name.

## 2026-06-16 — Models: Qwen2.5-Math (not Qwen2.5-Instruct) is the primary Qwen generator family

**Context:** the experiment sweep spans a Llama family and a Qwen family
of generators ([semantic-mcts] scope: "gains hold across Llama and Qwen
families"). The Qwen side was initially the general-purpose
**Qwen2.5-Instruct** line (e.g. `conf/llm/qwen_3b.yaml` =
Qwen2.5-3B-Instruct).
**Decision:** make the math-specialized **Qwen2.5-Math** (1.5B / 7B) the
primary Qwen generator family: added `conf/llm/qwen_math_1_5b.yaml` and
`qwen_math_7b.yaml`, switched the BoN-speed benchmark onto Qwen2.5-Math,
and verified the Qwen-Math preamble scores correctly under the RLHFlow
PRM before adopting it. GPU-mem utilization on `qwen_math_1_5b` (and
`llama_3b`) was lowered for PRM co-residency on the V100. Qwen2.5-Math is
now co-equal with Llama as a generator family for the benchmarks.
There is **no Qwen2.5-Math-3B**, so `conf/llm/qwen_3b.yaml`
(Qwen2.5-3B-**Instruct**) is **kept deliberately** — it's the only way to
get a *size-matched* 3B Qwen-vs-Llama comparison against `llama_3b`. So
the repo carries two distinct Qwen roles: Qwen2.5-Math (1.5B/7B) as the
in-domain family arm, and Qwen2.5-3B-Instruct as a same-size 3B control.
**Why:** the benchmarks are math reasoning (prm800k / MATH / GSM8K /
AIME), so a math-tuned generator is the in-domain choice — for the
*family* comparison, the general Instruct model would be a weaker, less
relevant arm. The 3B-Instruct is retained for a *different* axis
(family-at-matched-size), which the Math line can't cover at 3B.
**Caveat (clean-comparison):** these two roles answer different
questions and must not be blended into one "Qwen vs Llama" curve — the
Math-1.5B/7B arm varies family with *math-tuned* models, while the
3B-Instruct arm varies family at matched size with a *general* model
(a math-tuned-vs-general confound). Keep them as separate comparisons,
and don't pull `qwen_3b` into a Qwen-Math sweep.
**Revisit if:** a Qwen2.5-Math-3B is released (then `qwen_3b` Instruct
can retire, or become an explicit general-vs-math ablation at 3B), or a
newer math-specialized Qwen release supersedes 2.5.

## 2026-06-16 — Architecture: scoring vendored in-repo; MCTS auto-scores in-loop, BoN scores standalone

**Context:** scoring (PRM rewards + answer parsing + weighted/maj/naive
prediction) lived in the external `sal` library. The project wanted to
own its generate→score→dataset path, and the two search families have
different GPU-memory profiles.
**Decision:** vendor scoring into `core/scoring.py` +
`core/qwen_math_parser.py` (sal-config-free; verified byte-identical to
sal on a 128-row reference). `build_scored_dataset` turns a trial's raw
results into a per-question HF dataset method-agnostically (auto-attaches
whatever per-question stats the method emitted). **MCTS launchers
auto-score in-loop** after each trial (raw dumped first, scoring wrapped
in try/except so a scoring failure never loses a run); **BoN
deliberately stays raw-only** and is scored by the standalone
`prepare_scored_dataset` pass.
**Why:** dropping the sal dependency removes an upstream coupling and
lets scoring evolve with the project. The MCTS-vs-BoN asymmetry is a
deliberate co-residency choice: MCTS already holds the 8B PRM resident,
so in-loop scoring is free; large-n BoN (e.g. n=256) scored beside the
generative vLLM engine risks OOM, so BoN scoring is decoupled to a
separate process where the PRM can own the GPU. The method-agnostic
stat attach keeps one scoring path for tree stats (mcts) and
completion stats (bon) alike.
**Revisit if:** BoN n shrinks enough to co-reside with the PRM (then
fold its scoring in-loop too), or scoring needs to diverge from sal's
parser semantics (then the byte-identical guarantee no longer applies).

## 2026-06-16 — Naming, Configs: PRM scoring batch and CPU procs are separate from search batch_size

**Context:** `build_scored_dataset` used `cfg.search.batch_size` (the
number of MCTS expansion candidates) as the PRM scoring micro-batch —
the same name-overload the 2026-06-11 batch-size decision warned about.
On large-n BoN it forced ~4096 sequential 8B forward passes.
**Decision:** add `prm.score_batch_size` (default 8) for the PRM forward-
pass micro-batch in scoring, and `run.num_proc` (default 1) for the CPU
answer-parsing/sympy maps; launchers and `prepare_scored_dataset` pass
both. `search.batch_size` reverts to meaning only "candidates per
expansion."
**Why:** extends the 2026-06-11 decision (BoN `n` / MCTS `batch_size` /
PRM `prm_batch_size` are distinct quantities) to the *post-hoc scoring*
path, which had quietly reused the search batch. Conflating them coupled
PRM throughput to a search hyperparameter and made large-n scoring
needlessly slow.
**Revisit if:** never expected — this is a straight de-conflation.

## 2026-06-15 — Configs: ExpConfig.search is the base type; each launcher registers its method's subclass

**Context:** the structured-config migration (2026-06-13) needs one
`ExpConfig` to serve every search method, but each method has its own
typed `SearchConfig` subclass (`MCTSCntConfig`, `BoNConfig`,
`MCTSSemV0{1,2}Config`, …).
**Decision:** type `ExpConfig.search` as the **base** `SearchConfig`, and
have each launcher register its own subclass under the Hydra `"search"`
group (`cs.store(group="search", name="..._schema", node=...)`); the
concrete schema is then selected per-run via the `conf/search/` group.
`config_name` dispatches on `search.method`.
**Why:** this is the mechanism that lets one launcher + one `ExpConfig`
dispatch across methods without a union type or per-method top-level
configs — the group binding supplies the concrete subclass at compose
time. It's the structural piece the 2026-06-13 Hydra decision set up but
didn't spell out; every multi-method launcher (`generate_bon`,
`generate_mcts_sem`) now rests on it.
**Revisit if:** a method needs fields that can't express as a
`SearchConfig` subclass (then reconsider the single-base-type binding).

## 2026-06-13 — Prompting: use native chat templates, not one custom template

**Context:** the search code applied a single hardcoded Llama-3.1
`custom_chat_template` to *every* model. The
`examine_llm_chat_templates_v1` notebook
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
