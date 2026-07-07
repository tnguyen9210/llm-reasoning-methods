# Design decisions log

Append-only chronological record of decisions git history can't show:
cross-cutting design choices that span multiple files, and deliberate
omissions — things chosen *not* to be built, and why. Newest first.
One `##` section per decision. Titles carry one or two area prefixes
(`Area:` or `Area, Area:`) so skimming groups by eye and
`grep '^## .*Area'` gives a per-topic view.

Every decision gets an entry here, always — this file is the
chronological spine. When a decision is substantial enough to need a
table, multiple named alternatives, or an open still-unresolved
scaffold, it also gets a standalone file in [decisions/](decisions/);
the log entry then carries a one-line pointer to it rather than
repeating the full writeup.

## 2026-07-07 — Search: `ds_alpha` needs to be ~100x `ds_beta`, and `lam` sets what "matched scale" means

**Context:** `_diverse_select` (`mcts_sem_search_v02_00_00.py`) scores
each candidate arm as `q_vals = ds_beta*q_scores + ds_alpha*
q_diversity`, where `q_diversity`'s starting scale is set by the ridge
constant `lam` (`V_0 = lam*I`). Before choosing sweep values, the
terms' scales needed checking: `q_scores` is a PRM-derived running
mean — is it actually bounded, how does it compare to `q_diversity`,
and does `lam` need tuning alongside `ds_alpha`/`ds_beta` or can it be
treated as fixed?

**Decision (as embodied by the repo's default `ds_alpha=100,
ds_beta=1, lam=0.01` and its `ds_alpha`-only sweep tables):** confirmed
`q_scores ∈ [0,1]` (both PRMs emit softmax probabilities;
`aggregate_scores`'s `min`/`prod`/`last` all preserve that range).
Derived `q_diversity`'s initial scale in closed form:
`q_diversity(x) = 1/sqrt(lam)` exactly, at `lam=0.01` giving `≈10` —
two orders of magnitude above `q_scores`. This is what the existing
`ds_alpha=100` default is compensating for — scale-matching, not a
stated belief that diversity should dominate 100x. Because only the
*ratio* `ds_alpha/ds_beta` affects the argmax, fixing `ds_beta=1` and
sweeping only `ds_alpha` is lossless. `lam` and `ds_alpha` are coupled,
not independent — changing `lam` rescales `q_diversity`'s starting
point and silently changes what an already-tuned `ds_alpha` achieves,
so the informative single quantity is really `ds_alpha * sqrt(lam)`.

**Why:** without confirming the score range and deriving the
diversity term's actual scale (as a function of `lam`, not a fixed
constant), a sweep could easily test values that don't span the
informative range, or a `ds_alpha` sweep result could be silently
misapplied after a `lam` change without anyone noticing the scale had
shifted underneath it.

**Revisit if:** the plateau conclusion this reasoning feeds
(`ds_alpha ∈ {0,10,100}` sufficient at `lam=0.01`, `1000` redundant —
see
[findings/exp-findings/ds-alpha-diversity-bonus-plateau.md](findings/exp-findings/ds-alpha-diversity-bonus-plateau.md))
is challenged at higher trial counts than the current n=2/cell, or if
`lam` is ever swept — no `lam` sweep exists in the repo yet, and the
current sweep range is scoped to `lam=0.01` specifically. Full
derivation and design-discussion writeup:
[decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md).

## 2026-07-07 — Experiments: three-layer tracking — experiments.yaml (intent) → status.py (computed) → exp-comparison.md (report)

**Context:** the experiment matrix spans many comparison tables,
algorithms, and nodes, launched out of order — and its state lives
in three places that drift apart if nothing reconciles them: the
comparison tables (intent — what *should* run), the results folders
(artifacts — what *has* run), and W&B (telemetry — what's running
now). The design was worked out 2026-06-22 (vault guide
`research-coding-practices-guides/tracking-experiment-status`) and
implemented 2026-06-23 (`status.py` + `experiments.yaml`,
commit `ca5f1c6`); `exp-comparison.md` predates the system as the
cross-algorithm tuning tracker (moved into `docs/` 2026-06-21).
This entry records the standing decision retroactively — it never
got a log entry at the time.

**Decision:** don't merge the three sources of truth (different
audiences, different update cadences); add a fourth, *computed*
layer that reconciles them on demand:
- `experiments.yaml` — append-only intent ledger, a flat priority
  queue (NOT grouped by table). One entry per launchable run:
  launcher, `config_root`, `overrides`, `trials`, plus `feeds:`
  (which table cell(s) the run populates — a list, deliberately
  loose keys, two-way reference) and `recorded:` (has the number
  been transcribed into the doc — the ONLY mutable field).
- `status.py` — read-only reconciler. Composes each entry's cfg
  offline (Hydra compose, no model load), matches its
  `config_hash` against on-disk manifests, counts `.done` markers
  vs `trials`, optionally checks W&B run state → `planned` /
  `partial`(/`stalled`) / `done` / orphan. Status is COMPUTED,
  never stored — a hand-written `status:` field goes stale within
  the hour.
- `exp-comparison.md` — the report layer; a *view* over completed
  runs, never a queue. Numbers move in only from `done` rows, and
  flipping `recorded: true` happens in the same motion.

**Why:** a flat queue dissolves "finish Table 1 before Table 2"
(one run can feed several tables via `feeds`); `stalled` detection
(partial `.done` + W&B not running) replaces log-watching for
OOM/crash/disconnect, and relaunching a stalled entry is just
rerunning the same command (resume skips `.done` trials and
reattaches the manifest `run_id`); the append-only rule (never
delete or reorder; completed entries stay) is what keeps finished
runs from reading as orphans, preserves idempotency-by-inspection,
and makes the file safe for assistant edits. The `recorded` bit is
the one stored-state exception because "is it in the doc?" is not
reliably derivable — and the done-but-not-recorded gap is exactly
the worst drift (results on disk/W&B but missing from the tables).

**Verified in practice:** the backfill/hash-collision pass caught
several doc rows marked "planned" that actually hashed to
already-done dirs — the doc was stale, not the runs (the same
class of catch repeated on 2026-07-07 with the ds_alpha llama-3b
row and the model-family table).

**Revisit if:** the queue outgrows one file (split by group and
glob — not preemptively), or the not-yet-built layers land (the
assistant recorder loop; the multi-node orchestrator, explicitly
sequenced *after* the reconciler is trustworthy). Full design:
vault guide `tracking-experiment-status`; repo-side schema docs in
the `experiments.yaml` header and `status.py` docstring.

## 2026-07-07 — Search: sem-mcts v02 child selection dispatches on visit count (first-visit q-only, subsequent q+diversity)

**Context:** `MCTS.select_child` (`mcts_sem_search_v02_00_00.py`) has
carried a two-scenario dispatch since the file's current form: a first
visit (`node.visit_count() == 1`) selects by pure q-value argmax
(`_select_by_q_value`), while any subsequent visit combines q-value
with the diversity bonus (`_select_by_diversity`). No prior log entry
recorded the reasoning behind this split; documented here as the
standing design the current code embodies, following a session
discussion of the mechanism — treat this as the current implementation
decision for now, not a closed question.

**Decision (as embodied by current code):** dispatch purely on
`node.visit_count()`, with no diversity term at all on a child's first
visit and the full `ds_beta*q + ds_alpha*sqrt(log(1+visits))*diversity`
combination on every visit after.

**Why:** right after a node is expanded, every child has
`visit_count() == 1` and a q-value equal to its raw PRM candidate
score — nothing has been backpropagated through it yet. At that
instant, `V` (the diversity covariance) hasn't accumulated *any* of
these specific children's embeddings, so the diversity bonus would
reflect only unrelated earlier selections, not real signal about how
these children differ from each other. A plain q-value argmax is
cleaner than mixing in that noise. Once revisited, `V` has accumulated
at least one of the node's own children, so the diversity term becomes
genuinely informative, and the `sqrt(log(1+visits))` factor scales
exploration pressure up the longer a node has been sunk into (a
UCB-style schedule).

Regardless of which path fires, the selected child's embedding is
**unconditionally** folded into the covariance afterward — even on the
first-visit path that never reads `V_inv` — since that path still
commits to a child, and omitting the fold-in would let `V_inv` go
stale relative to what was actually selected.

**Revisit if:** this split is found not to hold up for v01's selection
shape (v01 uses a differently-structured, within-call greedy-K batch
selector, not this persistent-state dispatcher — unverified whether
the same first-visit special case applies there), or if empirical
comparison ever suggests the first-visit q-only step costs more in
missed diversity signal than it gains in reduced noise. Full writeup:
[decisions/child-selection-design.md](decisions/child-selection-design.md).

## 2026-07-07 — Search: `embeds_scope="response"` stays unimplemented for `embeds_source="prm"`

**Context:** `mcts_sem_search_v02_00_00.py::_embed_candidates`
guards the `prm` embedding source with `if sc.embeds_scope !=
"full": raise NotImplementedError(...)` — `"response"` scope
(pool only the assistant-response tokens, not the full
system/user/assistant sequence) works for `embeds_source="policy"`
(v01) but is deliberately blocked for `embeds_source="prm"` (v02).

**Why it's blocked:** `response_start_idx` is computed once per
question, in `_compute_response_start_idx`, using the
**generator's** tokenizer and chat template
(`llm_vllm.get_tokenizer()`, via `mcts_search`). Slicing the PRM's
pooled hidden-state tensor at that index is not merely
approximate — it is a different tokenizer, over a different chat
template, so the index has no defined meaning in the PRM's token
stream. It would silently produce a valid-shaped but wrong slice
(pooling over the wrong tokens), not an error, which makes this a
worse failure mode to leave unguarded than to block outright.

**Decision:** leave `embeds_source="prm"` restricted to
`embeds_scope="full"` and raise `NotImplementedError` for
`"response"`, rather than reusing the generator's
`response_start_idx` or attempting an approximate fix.

**What a correct implementation would need**, if ever prioritized:
1. A parallel `_compute_prm_response_start_idx(question, config,
   prm.tokenizer)` that renders the PRM's own prefix-only chat
   (via the PRM's `apply_chat_template`) and counts **its** tokens
   — the generator's index cannot simply be reused or adjusted.
2. Threading the PRM's tokenizer to wherever this gets computed
   (currently only the generator's tokenizer is passed around for
   this purpose).
3. Likely a **per-row** start index rather than a single scalar,
   if `PRM._embed_batch` ever batches candidates across more than
   one question in a forward pass (today it's one question at a
   time via `_embed_candidates`, so a scalar suffices only
   incidentally).
4. Verification via decoded token spans (confirm the computed
   index actually lands at the assistant turn for the PRM's
   template), since a wrong-but-plausible index wouldn't crash —
   it would just quietly pool the wrong tokens.

**Why deferred rather than fixed now:** the real config
(`conf/search/mcts_sem_v02.yaml`) already runs
`embeds_scope=full`, so no current experiment needs this path; the
guard exists to keep a future misconfiguration loud (`raise`) 
instead of silently wrong. Revisit if a future ablation specifically
wants to isolate the response-only embedding under the PRM source.
Full design across both scope values and both embedding sources:
[decisions/embeds-scope-design.md](decisions/embeds-scope-design.md).

## 2026-07-07 — Experiments: precautionary regen of two sem-mcts+qwen-PRM cells, old dirs moved aside not deleted, new W&B run ids

**Context:** the 2026-07-06 sem-mcts strip-and-reappend fix
(below) was verified to be a no-op at every existing recorded
sem-mcts config hash — every recorded run uses a
separator-preserving template (Llama+custom or Qwen+native), so
old and new code produce byte-identical prompts. That verification
was reasoned/spot-checked, not an exhaustive re-run of every
sem-mcts result. As a precaution — not because the fix is expected
to change anything — two cells feeding the new `agg_strategy
comparison (qwen-3b, qwen-math-1.5b)` sem-mcts table (method
`mcts_sem_v02`, `prm=qwen`, `agg_strategy=last`, the repo-wide
default) are being regenerated under the current code before their
numbers go in that table:

| llm | config hash | pre-fix W&B run_id |
|---|---|---|
| qwen-math-1.5b | `cfg-7a4be169` | `q0d6yk4f` |
| qwen-3b | `cfg-77cae091` | `jun56c12` |

**Why the old dirs had to move, not just relaunch in place:** both
already had 2/2 `.done` trial markers from before the fix (June
24/25) — the launcher's resume logic
(`generate_mcts_sem.py`/`generate_mcts_cnt.py`'s "skip any trial
whose `.done` marker exists") would skip straight past them and
regenerate nothing. Moved both result dirs to a `--prefix-backup`
suffix (same directory, not deleted) so a fresh launch at the same
config hash starts clean.

**Decision:** relaunch at the identical config (same hash,
`run.num_trials=2`, unchanged seed) into the now-empty original
path, rather than restoring/resuming the old run's W&B identity.

**Consequence — new W&B run ids, by design:** `load_wandb_run_id`
(`utils/configs.py`) reads `run_id` from `{result_dir}/
manifest.json` on disk; with the old dir moved aside, the fresh
launch finds no manifest, so `wandb.init(id=None, resume="allow")`
mints a **new** run. The pre-fix W&B runs (`q0d6yk4f`, `jun56c12`)
are untouched — they remain the historical record of pre-fix
generation, not resumed into or overwritten. No manual W&B edit
was made or needed; this is the same `write_manifest`/
`load_wandb_run_id` mechanism from the 2026-06-24 resume-
fragmentation-bug decision, behaving as designed (fresh manifest →
fresh run) rather than fragmenting an existing run.

**Revisit if:** the regenerated raw `.jsonl` differs from the
`--prefix-backup` copy at all — that would mean the "no-op at
existing hashes" verification from the 2026-07-06 entry was wrong,
and every other sem-mcts result would need the same scrutiny, not
just these two cells. (Not yet checked as of this writing — regen
launched manually on separate nodes, diff to follow.)

## 2026-07-06 — PRM, Scoring: shared `_split_steps` strips the trailing separator before splitting

**Context:** `QwenPRM._build_prompt` and
`RLHFlowPRM._build_conversations` (`core/reward_models.py`) each
split a candidate answer into steps with `answer.split("\n\n")`.
vLLM's `include_stop_str_in_output=True` with `stop=["\n\n"]`
means non-terminal candidates — generation cut mid-search by the
stop string, not EOS/length — keep a trailing `"\n\n"`; a plain
split on that trailing separator produces a bogus empty final
step, which gets its own scored `<extra_0>` position.

**Bug:** under `agg_strategy="last"` (`core/scoring.py::
aggregate_scores`), the bogus step's score silently replaced the
trajectory's true last-step score, on every non-terminal
candidate. Same root cause as the 2026-06-11 generation-side
separator bug (finding below), but on the scoring side — a
distortion rather than a collapse.

**Decision:** add a shared static helper, `PRM._split_steps`,
that strips the trailing separator before splitting
(`answer.removesuffix("\n\n").split("\n\n")`); both subclasses
call it instead of splitting directly. No-op for terminal
candidates.

**Verified:** live against both loaded PRMs (`unittests/
examine_prm_scores_qwenprm_v1.ipynb`,
`examine_prm_scores_rlhflowprm_v1.ipynb`), reproducing the
pre-fix behavior via a temporary `unittest.mock.patch.object`
(auto-restoring, no source file touched). Both PRMs' bogus score
reads as a holistic trajectory-level P(correct) rather than a
per-step judgment — but whether it can *mask* a bad branch is
PRM-specific: **QwenPRM tracks** a just-failed step tightly (cut
right after a bad step, bogus 0.0115 vs the bad step's own
0.0103 — no masking); **RLHFlowPRM masks it** (bogus 0.8130 vs
the bad step's own 0.2394 — a bad branch scored healthy at
exactly the point search should prune it). So the bug substituted
trajectory-level value for last-step value on every internal
search node, for both PRMs in the codebase — real in magnitude
and broad in blast radius, and for RLHFlowPRM specifically, not
bounded in direction either. Full writeup:
[prm-step-split-trailing-separator.md](findings/coding-findings/prm-step-split-trailing-separator.md).

**Revisit if:** a ds_alpha or model-family comparison result that
used `agg_strategy="last"` comes under question — check whether
it predates this fix, with extra scrutiny for any RLHFlowPRM
result given the masking risk above.
This entry is part of a larger PRM-scoring architecture thread; see
[decisions/prm-scoring-design.md](decisions/prm-scoring-design.md).

## 2026-07-06 — Search: sem-mcts gets the strip-and-reappend separator guard, applied in place

**Context:** the 2026-06-13 "use native chat templates" decision
(below) fixed prompt corruption for `mcts_cnt_search_v01_00_00`
and `mcts_bl_cnt_search_v01/v02_00_00` by stripping the trailing
`\n\n` step separator before `apply_chat_template` and
re-appending it after, making the separator's survival
independent of the template/transformers version. That
migration never reached `mcts_sem_search_v01/v02_00_00` — their
`_generate_candidates` templates `current_text` directly, with
`removesuffix("\n\n")` applied only to the embed/score copy of
candidates, never the generation prompt.

**Bug:** Llama's native template trims a trailing `\n\n`; without
the guard, the model sees a finished-looking message and emits
EOS immediately, collapsing the search tree to 1-step stubs
(same failure class as
[library-version-trajectory-completeness.md](findings/coding-findings/library-version-trajectory-completeness.md)).
Nothing broke in practice because the 2026-06-19 per-family
default (below) keeps Llama on the custom, whitespace-preserving
template — configuration was masking a missing code guard, not
correctness.

**Decision:** port the identical strip-and-reappend block
(`mcts_cnt_search_v01_00_00:263-273`) into both
`mcts_sem_search_v01_00_00` and `mcts_sem_search_v02_00_00`,
**applied in place, no version/method-string bump**. Normally a
core-file behavior change needs a new `search.method` label
(config hash includes it, so old and new code would otherwise
collide on the same result dir) — but every currently recorded
sem run uses a template that already preserves the separator
(Llama+custom or Qwen+native), so the fix reproduces
byte-identical prompts at every existing hash, and zero
Llama+native sem runs existed before this fix. There is no prior
data at the one hash this changes behavior for.

**Verified:** smoke-tested Llama3.2-1B + native template + sem-v02
before and after. Before: 0/26 nodes reached a final answer, 77%
were 1-step stubs. After: 32/39 (82%) final-answer, 2.6% stubs —
in line with the healthy controls (Llama+custom 8/8, Qwen+native
99.7% over a full trial). Recorded qwen sem-v02 results were
never affected (native-Qwen preserves the separator).

**Revisit if:** a future sem search file is added that copies the
old un-guarded pattern — check for the strip-and-reappend block
whenever cloning `_generate_candidates` into a new version.
Full mechanism + current coverage across all 5 MCTS variants:
[decisions/strip-and-reappend-separator.md](decisions/strip-and-reappend-separator.md).

## 2026-06-24 — Experiments: read run_id BEFORE the first write_manifest (resume-fragmentation bug)

**Context:** the three launchers
([generate_mcts_cnt.py](../generate_mcts_cnt.py),
[generate_mcts_sem.py](../generate_mcts_sem.py),
[generate_mcts_bl_cnt_v01.py](../generate_mcts_bl_cnt_v01.py))
write `manifest.json` twice per run — once before `wandb.init`
and once after (the run-id lifecycle from the 2026-06-21 "fold
run-id into manifest" decision below). The original ordering
was: `write_manifest(cfg)` (no run_id) → `load_wandb_run_id` →
`wandb.init(id=run_id, resume="allow")` → `write_manifest(cfg,
run_id=wandb_run.id)`.

**Bug:** the first `write_manifest(cfg)` passed `run_id=None`,
and `write_manifest` writes the *whole* payload (atomic
replace) — so it **overwrote the saved `run_id` with null
before `load_wandb_run_id` ran one line later**. Every resume
therefore loaded `None`, `wandb.init(id=None)` minted a *fresh*
run, and the original run was orphaned. Observed live: a
stalled `mfs5klyg` resumed as `aum658fp` (and `7ccy14de` →
`lzqhvfj6`), fragmenting one logical run across multiple empty
W&B runs and leaving any doc/ledger citation of the old id
dangling — the same failure class as the deleted-`ctmgmcrp`
citation the recorder caught earlier.

**Decision:** read `load_wandb_run_id` **before** the first
`write_manifest`, and pass it through:
`run_id = load_wandb_run_id(result_dir)` then
`write_manifest(result_dir, cfg, run_id=run_id)`. The
pre-`init` write now *preserves* an existing id instead of
nulling it.
**Why:** restores the invariant the 2026-06-21 fold-decision
assumed — run_id is "set-once-then-frozen," written twice but
never *cleared*. Fresh runs are unchanged (`load_` returns
None → `wandb.init` mints one, as intended); resumes keep the
id and `wandb.init(id=<old>, resume="allow")` reattaches to the
same W&B run. `run_id` is not part of `config_identity`/the
hash, so this touches nothing `status.py` reconciles
(`status.py --verify` stayed green across the change).
**Verified:** re-running the two stalled configs kept their
original ids (`mfs5klyg`, `7ccy14de`) in the manifest instead
of minting new ones; the two orphan runs were deleted from W&B
(both empty, uncited).
**Revisit if:** `write_manifest` ever gains a caller that
legitimately needs to *clear* run_id — then the "first write
preserves" assumption would need an explicit flag rather than
relying on the loaded value.
Full lineage (this entry plus the 2026-06-17/06-21 entries it
builds on): [decisions/manifest-runid-resume-design.md](decisions/manifest-runid-resume-design.md).

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
Part of the manifest/run-id lifecycle thread:
[decisions/manifest-runid-resume-design.md](decisions/manifest-runid-resume-design.md).

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
Part of the manifest/run-id lifecycle thread:
[decisions/manifest-runid-resume-design.md](decisions/manifest-runid-resume-design.md).

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
Full lineage (this entry plus the 2026-06-17/06-18/06-20 entries that
led here): [decisions/config-name-design.md](decisions/config-name-design.md).

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
Part of the PRM-scoring architecture thread:
[decisions/prm-scoring-design.md](decisions/prm-scoring-design.md).

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
(It did — see the 2026-06-21 entry above.) Full lineage:
[decisions/config-name-design.md](decisions/config-name-design.md).

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
This entry covers only the value spelling; for the algorithm itself
(what `"exact"` vs `"sm"` actually do, and a real divergence between
v01's and v02's `"sm"` implementations) see
[decisions/sherman-morrison-covariance-update.md](decisions/sherman-morrison-covariance-update.md).

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
Full current-state writeup:
[decisions/chat-template-per-family.md](decisions/chat-template-per-family.md).

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
**Status (2026-07-07):** the `centering_mode` flag and online-mean
mechanism described above are not yet implemented — only fixed-mean
centering exists in code today; online is planned. Full status and
design across all centering modes:
[decisions/embeds-centering-design.md](decisions/embeds-centering-design.md).

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
Full current-state writeup:
[decisions/sparse-random-projection.md](decisions/sparse-random-projection.md).

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
Full lineage: [decisions/config-name-design.md](decisions/config-name-design.md).

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
Full lineage: [decisions/manifest-runid-resume-design.md](decisions/manifest-runid-resume-design.md).

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
Full lineage: [decisions/manifest-runid-resume-design.md](decisions/manifest-runid-resume-design.md).

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
Full lineage: [decisions/config-name-design.md](decisions/config-name-design.md).

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
Part of the PRM-scoring architecture thread:
[decisions/prm-scoring-design.md](decisions/prm-scoring-design.md).

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
Part of the PRM-scoring architecture thread:
[decisions/prm-scoring-design.md](decisions/prm-scoring-design.md).

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
Superseded/refined by the 2026-06-19 entry below (this decision read
as "native for everyone"; the actual, current, per-family split is
Llama=custom / Qwen=native). Full current-state writeup:
[decisions/chat-template-per-family.md](decisions/chat-template-per-family.md).
For the strip-and-reappend mechanism itself (introduced here) and its
current coverage across all MCTS variants:
[decisions/strip-and-reappend-separator.md](decisions/strip-and-reappend-separator.md).

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
Full writeup: [decisions/set-gen-budget-for-mcts-search.md](decisions/set-gen-budget-for-mcts-search.md).

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
