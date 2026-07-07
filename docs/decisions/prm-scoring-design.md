# PRM scoring: how it's architected, invoked, and where a real bug lived

Four log entries trace how PRM scoring got its current shape: vendored
in-repo with an MCTS-vs-BoN invocation split (2026-06-16), its own
batch-size knobs separated from search hyperparameters (2026-06-16), a
throughput-only default lowered under memory pressure (2026-06-20),
and a real step-splitting bug shared across both PRM implementations
(2026-07-06). This doc covers the thread in order;
[decisions-log.md](../decisions-log.md) has each entry's full original
text.

## Current architecture (verified 2026-07-07)

**Vendored, not upstream.** `core/scoring.py` + `core/qwen_math_parser.py`
own PRM-reward aggregation, answer parsing, and weighted/majority/naive
prediction — no dependency on the external `sal` library for this path.
`build_scored_dataset` turns a trial's raw results into a per-question
HF dataset method-agnostically (it auto-attaches whatever per-question
stats a method emitted, so the same function serves tree-stats methods
and completion-stats methods alike).

**MCTS scores in-loop; BoN scores standalone — a deliberate GPU-residency
choice, not an inconsistency.** MCTS launchers (`generate_mcts_cnt.py`
and siblings) call `build_scored_dataset` immediately after each trial,
wrapped in `try`/`except` so a scoring failure never loses the raw
generation (`generate_mcts_cnt.py:198-205`). BoN deliberately stays
raw-only at generation time; `prepare_scored_dataset.py` is the
standalone process that scores it afterward, and also serves as the
general re-scoring entry point for any method
(`prepare_scored_dataset.py` calls the identical `build_scored_dataset`
that MCTS runs inline). The asymmetry exists because MCTS already holds
the PRM co-resident on the GPU throughout the search (it interleaves
generation and per-step scoring by construction), so in-loop scoring is
free; a large-n BoN (e.g. n=256) scored beside the generative vLLM
engine risks OOM, so its scoring is decoupled to a separate process
where the PRM can own the GPU alone.

**Three distinct batch-size knobs, not one overloaded field:**
- `search.batch_size` — MCTS expansion candidates per generation call.
  Nothing to do with PRM scoring.
- `search.prm_batch_size` (default **1**, both `MCTSCntConfig` and
  `MCTSSemV01Config`/`v02` inherits it, `conf/search/mcts_cnt.yaml`,
  `mcts_sem_v01.yaml`, `mcts_sem_v02.yaml` all set `1`) — the PRM
  forward-pass micro-batch **inside the search loop**, scoring
  candidates as they're generated.
- `prm.score_batch_size` (default **8**, `PRMConfig`) — the PRM
  forward-pass micro-batch for the **standalone post-hoc** scoring
  path (`build_scored_dataset`/`prepare_scored_dataset`), independent
  of anything search-related.

A fourth, `run.num_proc` (default 1), is the CPU-side answer-parsing/
sympy map's process count — unrelated to any PRM forward pass, listed
here only because it was decided alongside `score_batch_size` for the
same de-conflation reason (see below).

**Step splitting is shared and separator-safe.** `PRM._split_steps`
(`core/reward_models.py:76`) strips a trailing `"\n\n"` before
splitting an answer into steps
(`answer.removesuffix("\n\n").split("\n\n")`); both `QwenPRM._build_prompt`
(line 236) and `RLHFlowPRM._build_conversations` (line 409) call it
rather than splitting directly. See "The `_split_steps` bug" below for
why this exists.

## How it got here

### 2026-06-16 — vendor scoring in-repo, split MCTS-in-loop vs. BoN-standalone

**Context:** scoring lived in the external `sal` library; the project
wanted to own its generate→score→dataset path, and MCTS and BoN have
different GPU-memory profiles during generation. **Decision:** vendor
into `core/scoring.py` + `core/qwen_math_parser.py` (verified
byte-identical to `sal` on a 128-row reference at the time), with
MCTS auto-scoring in-loop (raw dumped first, scoring wrapped in
try/except) and BoN staying raw-only, scored by the standalone
`prepare_scored_dataset` pass. **Why:** dropping the `sal` dependency
removes an upstream coupling; the MCTS-vs-BoN split is a deliberate
co-residency choice (MCTS already holds the PRM resident so in-loop
scoring is free; large-n BoN risks OOM if scored beside the generative
engine, so it's decoupled to a process where the PRM can own the GPU).
**Revisit if:** BoN's `n` shrinks enough to co-reside with the PRM
(fold its scoring in-loop too), or scoring needs to diverge from
`sal`'s parser semantics (the byte-identical guarantee no longer
applies then).

### 2026-06-16 — separate PRM scoring batch and CPU procs from search batch_size

**Context:** `build_scored_dataset` was using `cfg.search.batch_size`
(the MCTS expansion-candidate count) as the PRM scoring micro-batch —
the same name-overload the 2026-06-11 batch-size decision had already
warned about for generation. On large-n BoN this forced ~4096
sequential 8B forward passes. **Decision:** add `prm.score_batch_size`
(default 8) for the standalone-scoring PRM micro-batch, and
`run.num_proc` (default 1) for CPU-side answer-parsing/sympy maps;
`search.batch_size` reverts to meaning only "candidates per expansion."
**Why:** extends the 2026-06-11 principle (BoN `n` / MCTS `batch_size`
/ PRM `prm_batch_size` are distinct quantities) to the post-hoc scoring
path specifically, which had quietly reused the search batch and
coupled PRM throughput to an unrelated hyperparameter.

### 2026-06-20 — lower the in-loop `prm_batch_size` default, 2 → 1

**Context:** `prm_batch_size` (the *in-loop* PRM micro-batch — distinct
from the `score_batch_size` the entry above introduced for the
*standalone* path) defaulted to 2 across `MCTSCntConfig`,
`MCTSSemV01Config` (inherited by v02), and both sem YAMLs.
**Decision:** lower the default to 1 in all four places. **Why:**
purely a throughput knob — it does not change which candidates get
scored or their scores, only how they're batched into forward passes —
lowered specifically to ease PRM memory pressure on the V100S once
larger co-resident PRMs were in the mix. Result dirs now tag
`--prmbs-1`; pre-existing `--prmbs-2/4` runs remain valid and
comparable on pass@gb, since this never affected accuracy. **Revisit
if:** PRM scoring becomes the wall-clock bottleneck and memory allows
raising it back up.

### 2026-07-06 — the `_split_steps` bug: a bogus trailing step corrupted `agg_strategy="last"`

**Context:** both `QwenPRM._build_prompt` and
`RLHFlowPRM._build_conversations` split a candidate answer into steps
via a plain `answer.split("\n\n")`. vLLM's
`include_stop_str_in_output=True` with `stop=["\n\n"]` means
**non-terminal** candidates (generation cut mid-search by the stop
string, not by EOS/length) keep a trailing `"\n\n"` — splitting on
that produces a bogus, empty final "step," which then gets its own
scored position (`<extra_0>` for QwenPRM, a separator-marker position
for RLHFlowPRM).

**The bug:** under `agg_strategy="last"` — the repo-wide default
(`core/scoring.py::aggregate_scores`) — that bogus final step's score
silently *replaced* the trajectory's true last-step score, on every
non-terminal candidate scored anywhere in the codebase. This is the
scoring-side twin of the 2026-06-11 generation-side separator bug (the
same trailing-`"\n\n"` root cause, but here a *distortion* of an
existing score rather than a *collapse* of the trajectory).

**Fix:** a shared static helper, `PRM._split_steps`, strips the
trailing separator before splitting; both PRM subclasses call it
instead of splitting directly. No-op for terminal candidates (which
never carry the trailing separator in the first place).

**Verified — and PRM-specific severity, not just "bugged for both":**
tested live against both loaded PRMs, reproducing the pre-fix behavior
via a temporary `unittest.mock.patch.object` (auto-restoring, no source
touched). Both PRMs' bogus score reads as a holistic
trajectory-level judgment rather than a genuine per-step one, but
whether that can *mask* a bad branch differs by PRM: **QwenPRM tracks**
a just-failed step tightly (bogus 0.0115 vs. the bad step's own 0.0103
— essentially no masking); **RLHFlowPRM masks it** (bogus 0.8130 vs.
the bad step's own 0.2394 — a genuinely bad branch reads as healthy at
exactly the point search should have pruned it). Full measured writeup:
[findings/coding-findings/prm-step-split-trailing-separator.md](../findings/coding-findings/prm-step-split-trailing-separator.md).

**Blast radius:** every internal search node scored under
`agg_strategy="last"` with either PRM, before this fix, substituted
trajectory-level value for true last-step value — real in magnitude,
broad in reach, and for RLHFlowPRM specifically, unbounded in
direction (it can make a bad branch look *better*, not just noisier).
**Revisit if:** a `ds_alpha`, model-family, or agg_strategy comparison
result under question predates 2026-07-06 and used
`agg_strategy="last"` — check its date against the fix, with extra
scrutiny for any RLHFlowPRM result given the masking risk.

## Revisit if (thread-level)

Any future PRM subclass is added: it must call `PRM._split_steps`
rather than splitting `"\n\n"` directly, or it reintroduces the exact
bug the 2026-07-06 fix closed. Any future scoring-batch knob should be
named to avoid re-conflating with `search.batch_size` or
`prm_batch_size` — that specific naming collision has already caused
one real slowdown (2026-06-16) and the fix is the standing convention,
not a one-off.
