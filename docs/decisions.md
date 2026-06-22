# Design decisions

Append-only log of decisions git history can't show: cross-cutting
design choices that span multiple files, and deliberate omissions —
things chosen *not* to be built, and why. Newest first. One `##`
section per decision. Titles carry one or two area prefixes
(`Area:` or `Area, Area:`) so skimming groups by eye and
`grep '^## .*Area'` gives a per-topic view.

## 2026-06-21 — Configs: don't fold timing_state.json into manifest.json

**Context:** after folding `run_id` into `manifest.json` (below),
considered going further and folding `timing_state.json` (the
per-trial running-average sidecar written by `mcts_cnt`/`mcts_sem`)
into the same file.
**Decision:** keep them separate.
**Why:** the two sidecars have incompatible write lifecycles.
`run_id` is written exactly twice per run (before and after
`wandb.init`) — set-once-then-frozen, safe to share a file with the
mostly-static identity fields. `timing_state.json` is written once
**per trial**, in the generator's hot loop
(`save_timing_state(result_dir, n_done, avg_q_s, avg_trial_hr)` in
[generate_mcts_cnt.py](../generate_mcts_cnt.py),
[generate_mcts_sem.py](../generate_mcts_sem.py)). Folding it in
would mean every trial completion does a read-modify-write of the
*entire* manifest (identity fields included) just to bump 3 timing
numbers, and raises write-contention risk if a `compute_stats.py`/
`prepare_scored_dataset.py` post-process ever runs concurrently with
a still-generating trial loop — two atomic-replace writers on the
same file instead of two different files. Today's split keeps
"identity, rarely written" and "per-trial telemetry, written every
trial" on separate files, which is doing real work, not just
incidental structure.
**Revisit if:** the per-run file count itself becomes the
bottleneck (e.g. very many small result dirs), or `timing_state`
gains fields that need cross-referencing with manifest identity at
read time.

## 2026-06-21 — Experiments, Configs: fold the W&B run-id sidecar into manifest.json

**Context:** the 2026-06-17 decision below added a standalone
`wandb_run_id.txt` sidecar so post-processing could reattach to the
same W&B run. After the result-dir naming rework (above) gave every
run dir a `manifest.json` for identity, having a second one-line
sidecar file just for the run id was redundant.
**Decision:** add a `run_id` field to `manifest.json`; drop
`wandb_run_id.txt`. `write_manifest()` now takes an optional
`run_id` and is called twice per launch: once before `wandb.init`
(`run_id=None`), once after (`run_id=wandb_run.id`).
`load_wandb_run_id()` reads `manifest.json["run_id"]` first, falling
back to the legacy `wandb_run_id.txt` for any dir not yet migrated.
**Why:** preserves the crash-safety property the sidecar design
depended on — `write_manifest` before `wandb.init` means a crash
during the (network-dependent) `wandb.init()` call still leaves a
locatable, identity-recorded dir, since `find_run_dir` matches on
`config_hash`/`config_identity` which are written in that same first
call. Field order inside the JSON has no effect on this — only
*when* a complete file lands on disk matters, not the order of keys
within it.
**Migration:** backfilled `run_id` into all 42 existing
`manifest.json` files from their `wandb_run_id.txt` sidecars (zero
mismatches), then deleted all 42 now-redundant sidecar files.
Verified `load_wandb_run_id()` still resolves correctly post-
deletion via spot-check.

## 2026-06-21 — Configs: result-dir naming = readable prefix + config hash; locate runs by recorded manifest, not recomputed name

**Context:** `config_name(cfg)` encoded *every* result-affecting knob
into the dir name (the 2026-06-18 "encode every knob" decision —
correct for collision-safety). Side effect: each new knob extended the
name format, so post-processing that *recomputed* `config_name` could
no longer find pre-existing dirs → manual rename of old dirs, hit ~3×
in one session. Root cause (vault note
`question-config-name-experiment-naming`): the name did two jobs with
opposite stability needs — *identity* (wants to change as the schema
grows) and *addressing* (needs to stay stable). Recomputing an
addressing key against a live schema is inherently fragile.
**Decision:** split the two jobs.
- **Name = readable prefix + hash.** `config_name` is now
  `{algo}{--level-N if set}--{llm}--{prm}--d-{depth}--bs-{batch}
  --b-{budget}--cfg-{hash8}`. The prefix is a *cosmetic* curated subset
  for eyeball-skimming; the `cfg-{hash8}` (sha1 over the full
  run-affecting config, cosmetic/env fields stripped) is the
  collision-safe identity. Other knobs (cpuct, lam, proj, cov, tmpl,
  prm_batch_size, …) leave the name and live only in the hash +
  manifest.
- **`level` is an optional prefix field** — shown only when
  `data.level is not None` (omitted for a full split or a level-less
  dataset like AIME), but in the hash *unconditionally* so a level-N
  and a full-split run never collide regardless of display. No
  dataset-specific logic needed; `level=None` = "absent" covers every
  case.
- **Record the identity once; locate by recorded fact.** Launchers
  `write_manifest()` a `manifest.json` (config_name, config_hash,
  config_identity, varied) into each dir at creation. Readers
  (`compute_stats`, `prepare_scored_dataset`) locate a run via
  `resolve_result_dir` → `find_run_dir` (match the *recorded* hash in
  manifests), or an explicit `+result_dir=<path>` override — NOT by
  re-deriving the name. The dir's trial-file basename comes from the
  manifest's recorded `config_name`, so files resolve even if the name
  format changes again.
- **Launcher is the one allowed recompute site.** Resume (`.done`)
  needs deterministic config→dir to decide resume-vs-fresh, so the
  launcher recomputes `config_name`; readers never do.
**Why:** the hash gives complete collision-safety (the 2026-06-18
property, preserved — adding a knob changes the hash → new dir, never a
silent collision) while the prefix keeps names short and skimmable. The
recurring "added a knob → rename old dirs" tax disappears because
readers match recorded manifests instead of recomputing. Full analysis
(full-vs-diff hash trade, why diff-from-defaults is default-change-
fragile, why "record once" is the real fix) in the vault note +
`prompt-experiment-naming-review{,-followup}`.
**Migration:** new runs get the short prefix+hash names; existing dirs
keep their long-form names and are reached via `+result_dir=`
(verified) or after `backfill_manifests.py --write` (writes a manifest
recording the old name as `config_name`; `config_hash: null` since the
full identity isn't recoverable from an old name — so old dirs are
addressable by path/name, not by recomputed hash, which matches the
agreed design). Ran the backfill over the 45 existing dirs.
**Revisit if:** `results/` grows enough that the O(N) glob in
`find_run_dir` is slow (add an index file), or run-affecting state
starts living outside `cfg` (env var / code constant) — then the
manifest is incomplete and the hash under-identifies (currently only
the hardcoded projection seed is in this category, and it's fixed).

## 2026-06-20 — Reward models: QwenPRM gains _embed_batch; PRM-source embeds drop the scoring separators

**Context:** mcts_sem v02 sources its diversity embeddings from the PRM
(`prm.embed()` → `_embed_batch`). Only `RLHFlowPRM` implemented that;
`QwenPRM` raised `NotImplementedError`, so `v02 prm=qwen_prm` failed at
the first expansion. (QwenPRM's `_score_batch` already worked — it's
usable for v02 *scoring* via the policy-embeds v01, and for mcts_cnt
scoring; the gap was specifically the embeds-source role.)
**Decision:** implement `QwenPRM._embed_batch`, mirroring
`RLHFlowPRM._embed_batch`, with two model-specific points:
- **Embed the PLAIN candidate chat, WITHOUT the `<extra_0>`
  separators** that `_build_prompt` inserts for scoring. The embedded
  text is `system / user(question) / assistant(answer)` — the same
  shape v01 embeds with the policy — so the v01-vs-v02 source ablation
  isolates *the model*, not the text. Separators are a reward-head
  scoring artifact and must not leak into the embedding text.
- **Hook `model.model.norm`** (the inner `Qwen2Model`'s final RMSNorm)
  for the `layer=-1` fast path, same as RLHFlow. The top-level module
  is `Qwen2ForProcessRewardModel` (`model: Qwen2Model` + `score: head`),
  so the backbone norm is one level deeper but the dotted path is
  identical; the `score` reward head is simply never read. Verified the
  hook output is **bit-identical** to `hidden_states[-1]` (max abs diff
  0.0) for this checkpoint, so the memory trick (capture one layer vs
  materializing all 29) is exact.
**Why:** unblocks the PRM-source ablation *across two different PRMs*
(Llama-8B-PRM vs Qwen-Math-7B-PRM embeds), not just policy-vs-PRM. The
no-separators choice is the crux: reusing `_build_prompt` would have
embedded a different text than v01, silently confounding the source
comparison with a text-format difference.
**Caveat:** the Qwen PRM's hidden dim is **3584** (vs 4096 for the
Llama PRM). With the default `embeds_proj=sparse` the raw dim is
projected to 512 regardless, so nothing to set; but `embeds_proj=none`
with the Qwen PRM requires `search.embeds_dim=3584` or the projection
shape-guard raises. Documented in the method's docstring.
**Revisit if:** a future PRM isn't a `*Model` + head over a standard
backbone (then `model.model.norm` won't be the right hook and the
embed path needs rethinking).

## 2026-06-20 — Configs: mcts_sem_v02 generator gmu is 0.3 (was an OOM-causing 0.2); gmu is a total-GPU fraction, not PRM headroom

**Context:** `mcts_sem_v02 llm=qwen_7b_gptq_int4` OOM'd at init while the
*same model* ran fine under mcts_cnt. Cause: the v02 top-level YAML
overrode `llm.gpu_memory_utilization` to **0.2**, while mcts_cnt used
the llm-group default **0.3**. The override's own comment claimed
"kept at 0.3" — comment and value had drifted apart.
**Decision:** set the v02 override to `0.3`, matching mcts_cnt, and
rewrite the comment to state what gmu actually controls.
**Why:** vLLM's `gpu_memory_utilization` is the fraction of the
**whole GPU** it may use for weights + KV cache + activations — it is
NOT a "leave room for the co-resident PRM" reservation (the HF PRM
allocates separately, outside vLLM's budget). So a *lower* gmu causes
OOM, not avoids it: `0.2 * 32 GB (V100S) = 6.4 GB < 5.3 GB` (7B-GPTQ
weights) + activations/CUDA-graph/KV → vLLM can't even init. `0.3` =
9.6 GB clears it (the value mcts_cnt already ran these models at). The
misframing ("lower gmu = more PRM headroom") was the root error.
**Revisit if:** a larger generator needs more than 0.3*total for its
own weights+KV (raise via `llm.gpu_memory_utilization=` on the CLI),
or a bigger GPU changes the arithmetic.

## 2026-06-20 — Configs: default prm_batch_size lowered 2 -> 1

**Context:** `prm_batch_size` is the PRM forward-pass micro-batch
*inside* the search loop (distinct from `prm.score_batch_size` for the
final dataset). Default was 2 across `MCTSCntConfig`,
`MCTSSemV01Config` (inherited by v02), and the two sem YAMLs.
**Decision:** default `prm_batch_size = 1` in all four places.
**Why:** throughput-only knob (does not change accuracy — same
candidates scored, only batched differently), lowered to ease PRM
memory pressure on the V100S with the larger co-resident PRMs. Result
dirs now tag `--prmbs-1`; existing `--prmbs-2/4` runs are unaffected
and stay comparable on the metric that matters (pass@gb).
**Revisit if:** PRM scoring becomes the wall-clock bottleneck and
memory allows a larger micro-batch (raise via CLI/YAML).

## 2026-06-20 — Configs: config_name always tags projection (incl. --proj-none), reversing "append only when on"

**Context:** the 2026-06-18 projection decision appended the `--proj-`
tag to `config_name` *only when* `embeds_proj != "none"`, so that
no-projection runs kept their pre-projection names and existing dirs /
W&B runs didn't orphan. But the `embeds_proj × cov_update` sweep needs
the `none` arm as a first-class cell — and with the tag suppressed, a
`proj=none` run produced a name with *no* projection marker at all,
which (a) doesn't read as self-describing next to its `--proj-sparse512`
sibling, and (b) collides in spirit with the always-on `--cov-` tag
added in the same 2026-06-18 batch (asymmetric: cov always shown, proj
sometimes hidden).
**Decision:** always append the projection tag, including
`--proj-none{embeds_dim}` (e.g. `--proj-none4096`). `config_name`'s
`proj_str` is now unconditional, mirroring `cov_str`. Both arms of a
projection sweep thus get distinct, self-describing dirs.
**Why:** this prioritizes self-describing sweep cells over the
2026-06-18 goal of not-renaming-old-dirs — a deliberate reversal of
that specific sub-choice (the *encode-every-result-affecting-knob*
principle it served is untouched and in fact strengthened: a knob
that's swept must be in the name, and `none` is a swept value). The
one pre-existing untagged `proj=none` dir was an empty dead-init (only
a `wandb_run_id.txt`, 0 trials), so it was deleted rather than renamed
— no real data orphaned. A `NOTE` in `config_name` flags the change so
a future untagged dir is understood, not silently re-run.
**Caveat / open:** this is exactly the "adding/changing a knob's
encoding orphans old dirs" friction that motivated the broader
naming-redesign discussion (vault note
`question-config-name-experiment-naming`,
[[llm-reasoning-repo-reorganize-todo]] item B): identity-by-recomputed-
name is fragile under schema evolution. This entry is a local fix; the
structural fix (manifest + explicit `--result-dir`, or readable-prefix
+ config-hash) is still pending a decision there.
**Revisit if:** the naming redesign lands (then proj/cov tagging gets
subsumed by whatever scheme it picks).

## 2026-06-20 — Configs: cov_update value renamed "sherman_morrison" -> "sm"

**Context:** the `cov_update` knob's value was the verbose
`"sherman_morrison"`, while the `config_name` dir tag already
abbreviated it to `--cov-sm` via a conditional
(`'sm' if cov == 'sherman_morrison' else cov`). So the on-disk name
and the CLI value disagreed, and the conditional existed only to bridge
that gap.
**Decision:** make the config *value* itself `"sm"` everywhere — both
`conf/search/mcts_sem_v0{1,2}.yaml`, both search cores' `==` comparisons
+ docstrings, and the dataclass default comment. `config_name`'s
`cov_str` drops the conditional and is now plain `f"--cov-{cov}"`.
**Why:** one spelling end-to-end (CLI override `search.cov_update=sm`,
config value, and dir tag all match) removes the value↔name mismatch
and the special-case bridge. The dir tag string is unchanged
(`--cov-sm` / `--cov-exact`), so existing result dirs are NOT affected
and don't need renaming — only the accepted CLI/YAML value changed.
**Revisit if:** never expected — straight rename for consistency.

## 2026-06-19 — Architecture, Configs: PRM selection is a registry on the PRM module, not a dict per launcher

**Context:** adding `QwenPRM` alongside `RLHFlowPRM` meant each launcher
that constructs a PRM (`generate_mcts_cnt`, `generate_mcts_sem`,
`prepare_scored_dataset`) carried its own local
`prm_dict = {"rlhflow": RLHFlowPRM, "qwen": QwenPRM}` plus a lookup-and-
guard block, duplicated three times.
**Decision:** move the dict and construction logic into
`core/reward_models.py` (the module that already owns `PRM`,
`RLHFlowPRM`, `QwenPRM`) as `PRM_REGISTRY: dict[str, type[PRM]]` and
`build_prm(kind, model_path, device=..., **kwargs) -> PRM`, which raises
`ValueError` (not `KeyError`) listing valid kinds on an unknown one.
Every launcher now calls `prm = build_prm(cfg.prm.kind, cfg.prm.prm_dir,
device=cfg.prm.device_map)` instead of carrying its own dict.
**Why:** the dispatch mechanism itself (a dict keyed on `cfg.prm.kind`)
was already the right shape — the problem was that it lived in three
places instead of one, so adding a future PRM kind meant remembering to
update all three call sites. Colocating the registry with the classes it
indexes is the standard fix and needed no new pattern (no decorator-based
auto-registration, no `PRMConfig.build()` method on the dataclass): a
decorator buys nothing for a 2–3-entry registry and hides its contents
from a plain `grep`/`print`; a `build()` method on `PRMConfig` would
couple `utils/configs.py` (pure config/schema, cheap to import anywhere)
to `core/reward_models.py` (model loading + GPU code), a worse seam than
the one removed. This mirrors the algo-method dispatch (`algo_dict` in
each launcher, selecting the search core module) — not consolidated here
since it isn't duplicated.
**Revisit if:** a future PRM kind needs constructor args that don't fit
`build_prm`'s `**kwargs` passthrough, or the registry grows large enough
that a flat dict becomes hard to navigate (neither expected soon).

## 2026-06-19 — Models, Configs: chat-template default lives on LLMConfig, set per model family

**Context:** `GenConfig.use_custom_template` was a single global flag
(default `True`), so every model — Llama or Qwen — got the vendored
Llama-3.1 `custom_chat_template` unless a run explicitly overrode it.
Running Qwen with this default-on custom template produced malformed,
non-terminating output (stray `<|start_header_id|>`/`<|im_start|>`-style
tokens leaking into the completion) because the template is
Llama-3.1-specific and Qwen was never trained on it — this is the
opposite confound from the one the 2026-06-13 native-template decision
already fixed (forcing one family's format onto another). A first fix
attempt added a `resolve_use_custom_template(cfg)` helper function to
pick the default per family at call time; rejected per explicit feedback
("adding a separate helper function... may make the code harder to
track and maintain in the future" — only a few Qwen configs exist, and
the value only needs setting once).
**Decision:** drop `GenConfig.use_custom_template` and the resolver
function entirely. Add `use_custom_template: bool = True` directly to
`LLMConfig` (default custom, i.e. Llama's prior behavior unchanged), and
set it to `False` (native) in each `conf/llm/qwen_*.yaml` group
(`qwen_3b`, `qwen_3b_gptq_int4`, `qwen_7b_gptq_int4`, `qwen_math_1_5b`,
`qwen_math_7b`). All template-selection read sites (`mcts_cnt`,
`mcts_sem` v01/v02, `bon`, `mcts_bl_cnt`) and `config_name`'s `--tmpl-`
tag now read `cfg.llm.use_custom_template` instead of
`cfg.gen.use_custom_template`. A CLI override
(`llm.use_custom_template=...`) still wins over the YAML default.
**Why:** the field is per-model-family state, not a computation, so it
belongs as static config data on the dataclass that already describes
the model (`LLMConfig`), set once per YAML group — no resolver needed to
"compute" a value that's actually just a default. This keeps the
single-global-flag ergonomics (one bool, one CLI override path) while
fixing the actual bug (Qwen no longer silently gets a foreign template).
**Revisit if:** the per-family default needs to depend on more than
just "which YAML group is loaded" (e.g. on dataset or task), at which
point a real resolver would earn its complexity.

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
([findings](findings/coding-findings/library-version-trajectory-completeness.md)
and the vault note `llm-chat-templates`)
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
[findings/coding-findings/library-version-trajectory-completeness.md](findings/coding-findings/library-version-trajectory-completeness.md)
— the old stack (vLLM 0.6.4 /
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
