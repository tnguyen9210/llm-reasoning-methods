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

## 2026-08-14 — Search, Config: `q_beta` exploitation weight + first-visit q-argmax branch for cnt-mcts

**Context:** cnt-mcts exposed only `cpuct` (exploration weight),
while sem-mcts-v02 exposes the full pair `ds_beta*score +
ds_alpha*diversity`. That asymmetry made the two methods'
INFINITE-exploration limits incomparable, and forced the ugly
`cpuct=1e18` stand-in for the `∞` arm (literal `inf` NaNs out via
`inf*0` when `log(N_parent)=0`; 1e18 works only because float64
spacing swallows q).
**Decision:** new `search.q_beta` on `MCTSCntConfig`; selection
after a node's first visit maximizes `q_beta*q + cpuct*u`. Plus
an explicit FIRST-VISIT branch in `select_child` — raw q argmax
at `node.visit_count()==1` — mirroring sem-mcts-v02's
`_select_by_q_value` dispatch at the same condition. Default
`q_beta=1.0` is registered in `_HASH_EXCLUDE_IF_DEFAULT`, so
every recorded cnt hash is unchanged (verified: baseline
llama-1b cpuct=2.0 composes to `0f003563` with and without an
explicit `q_beta=1.0`; `q_beta=0.0` is hash-visible as
`bf21290d`). Pinned by `unittests/check_cnt_q_beta.py` (12
checks).
**Why the first-visit branch:** at `q_beta=1.0` it is a no-op —
`log(parent_visits)=0` already zeroes u at visit 1 and
`1.0*q + 0.0` is q exactly, so trajectories replay bit-for-bit
(pinned, RNG stream included). It is load-bearing at
`q_beta=0`: without it every first-descent value scales to 0 and
the pick degenerates to a coin flip, whereas sem at `ds_beta=0`
still does a q argmax there. With the branch, the two methods'
first-visit behavior is identical (same dispatch condition, same
raw-q argmax, same `tie_tol` band).
**Payoff:** `q_beta=0` is an EXACT pure-exploration arm at any
`cpuct>0` — no float-spacing trick, no NaN hazard — and it
reproduces the `cpuct=1e18` stand-in pick-for-pick (test C3), so
it supersedes `1e18` as the sweep's `∞` row.
**Known limit (do not overclaim):** `q_beta=0` is NOT a
cross-implementation replica of sem's `ds_beta=0` the way
`cpuct=0` replicates `ds_alpha=0`. After the first visit cnt
ranks by `1/sqrt(n_i)` (count novelty) while sem ranks by
`sqrt(u_i' V_n^-1 u_i)` (direction novelty in the local
covariance). They coincide only if sibling vectors are
orthogonal with equal norms — then `V_n^-1` is diagonal and the
bonus collapses to `(lam + n_i c^2)^-1/2`, a decreasing function
of the count. Real siblings are correlated and unequal-norm, so
the limits differ: sem's folds shrink correlated siblings too,
and it ranks by displacement magnitude before any fold.

## 2026-08-14 — Search, Config: `tie_tol` select_child tie band for cnt-mcts (sem-aligned ties, hash-neutral default)

**Context:** cnt-mcts v01 broke selection ties by exact float
equality while sem-mcts-v02 randomizes uniformly within a 1e-4
band, so the two methods' zero-exploration limits (cpuct=0 vs
ds_alpha=0) — otherwise verified identical (same q definition,
phase loop, dedupe, scoring, seeding) — differed exactly on
near-ties (q gap ≤ 1e-4, which saturated PRM scores do produce).
**Decision:** new `search.tie_tol` knob on `MCTSCntConfig`;
`select_child` picks uniformly among children within `tie_tol`
of the max PUCT value. Default 0.0 reproduces the historical
behavior bit-for-bit (RNG stream included — one random.choice
per call either way) and is registered in
`_HASH_EXCLUDE_IF_DEFAULT`, so every recorded cnt config_hash
is unchanged; `search.tie_tol=1e-4` is a hash-visible variant
matching sem's band. Pinned by
`unittests/check_cnt_tie_tol.py`.
Mirrored the same day on sem-mcts-v02: `MCTSSemV02Config.tie_tol`
(default 1e-4 = the historical hardcoded band,
neutral-registered; 0.0 = exact ties, cnt-aligned), threaded
through `_select_by_q_value` / `_diverse_select`. Because the
two families' neutrals differ (0.0 cnt, 1e-4 sem),
`_HASH_EXCLUDE_IF_DEFAULT` entries may now be method-prefix
dicts resolved by `_neutral_for` on `search.method`. Pinned by
`unittests/check_sem_tie_tol.py`; the cov_scope RNG-equivalence
check still passes unchanged.
**Why:** hardcoding the band would have silently changed the
policy behind every existing cnt hash, breaking the hash ⇔
experiment invariant; the flag route (house pattern — cf. the
2026-07-28 cov_scope/embeds_ref entry) makes alignment an
explicit, hash-tracked choice. Payoff: a `cpuct=0 tie_tol=1e-4`
run is distribution-identical to sem-v02's `w_eff=0` arm, so it
can serve as a cross-implementation anchor — e.g. for the open
tbl-375fa0 (.2724) vs tbl-ba6b11 (.2388) `w_eff=0` dispute.
**Follow-on (same day, Tuan's call):** all five level-5 cpuct
sweep tables (`tbl-e0b779` / `ed4e96` / `c93854` / `0f6c1a` /
`57b084`) adopt `tie_tol=1e-4` on EVERY row, not just
`cpuct=0`. The band is a sweep-wide policy choice, not a
`cpuct=0` special case: equal-visit siblings share an identical
bonus `cpuct*sqrt(log N_parent / n)`, so their PUCT gap is
exactly their q gap at any cpuct. (An earlier draft of the
doc preamble claimed the band was inert for cpuct>0 — wrong;
it is inert only between children with DIFFERENT visit
counts.) Accepted cost: the `cpuct=2.0` rows stop being the
shared `tbl-afdda0` model-family runs (`tie_tol=0.0`) and
become 5 net-new cells, so each sweep is internally consistent
but no longer directly comparable to the doc's `tie_tol=0.0`
cnt-mcts baseline — the sweep's own `2.0` row is its
reference. Three in-flight cells (qwen-3b `1.0`/`10`,
llama-3b `1.0`) were killed and relaunched from trial 0,
discarding ~10.7 GPU-h.

## 2026-08-12 — Workflow, Docs: trial-count continuations reuse the entry; n=4 tables live in the same doc, n=2 tables freeze (undecided)

**Context:** extending the `cov_scope=local` `embeds_ref=relative`
sweeps from 2 to 4 trials, with separate tables for the n=2 and n=4
readings; the level-5 doc is ~4,300 lines, so a new file was on the
table. **Status: undecided** — proposal recorded, nothing executed.
**Proposed:** a continuation is an in-place edit of the existing
ledger entry (`trials: 2→4`, `status: scored→inqueue`) — never a new
entry, since `run.num_trials` is hash-excluded and resume skips
finished trials. The n=4 tables go in the SAME doc under a mirrored
`## Tuning tables [trials=4]` section with fresh tbl-ids; each n=2
table freezes (⚠ "frozen at n=2, superseded by tbl-XXXXXX") and its
entry's `feeds` key MOVES to the n=4 table, because `--sync-doc`
derives every fed row's status and would otherwise flip the frozen
snapshot back to `running`. Prose that needs no status cells may
move to untracked `docs/analysis-*.md`; tables stay synced.
**Why:** the doc↔ledger map is hard-coded 1:1 (`status.py` DOC_MAP)
and hash uniqueness is global, so a second synced doc needs code
changes or a ledger split that would orphan shared-feed comparison
tables; dual-feeding un-freezes the snapshot; a dummy-override new
hash re-runs trials 0–1 from scratch. Caveat recorded: compute_stats
refreshes W&B `eval/*` in place, so frozen tables become the only
n=2 record after the rescore.
Full writeup: [decisions/trials-4-continuation-tables.md](decisions/trials-4-continuation-tables.md).

## 2026-07-28 — Search, Config: per-node diversity covariance (`cov_scope` / `embeds_ref`) — one file, not two

**Context:** asked for a variant of `mcts_sem_v02` in which the
diversity covariance `V` is maintained per node rather than once per
tree. **Decision:** two orthogonal flags on the existing
`MCTSSemV02Config` — `cov_scope: "global" | "local"` (where `V`
lives) and `embeds_ref: "absolute" | "relative"` (whether a child is
represented by its own embedding or by its displacement from its
parent) — both pinned in `_HASH_EXCLUDE_IF_DEFAULT` at the
pre-existing behavior, so all 316 scored sem_v02 hashes are
untouched (verified 204/204 on level 5). **This reverses a
same-day decision** to build it as a standalone
`mcts_sem_search_v02_01_00.py` + `MCTSSemV02LocalConfig`: the
hash-stability argument that justified the split does not hold, since
`_HASH_EXCLUDE_IF_DEFAULT` already exists for exactly this and had
been used twice before (`cov_dtype`, `embeds_center_mode`). Merging
also *dissolves* the verification problem the split created — with
one file, `cov_scope="global"` is not "equivalent to" the old
behavior, it **is** the old code path. The two files had differed by
~80 lines of real code out of 1122, and two defects found in review
already existed in both copies. The merge had a deadline: the
variant had zero ledger entries, and the method string bakes into
result-dir names, so file-vs-flag must be decided **before the first
run**. **Measured, not asserted:** local scope holds 644 MiB of
covariance per question at `embeds_dim=512`/fp64/`gen_budget=320`,
and `del agent` frees *none* of it (parent<->children reference
cycles), peaking at 2.19x across four questions until an explicit
`gc.collect()` brought it to 1.01x; `embeds_proj="none"` (a
documented config that forces `embeds_dim=4096`) would need ~40 GiB
per question and die as a silent cgroup OOM, so `MCTS.__init__` now
refuses it above a 4 GiB cap. **Verified by trace diff, not code
review:** a scripted-generator stub drives the real selection loop
on CPU, pinning RNG consumption as well as arithmetic — a GPU run
was rejected because vLLM's own nondeterminism would confound the
comparison. `global`/`absolute` reproduced the pre-merge file's
selection traces and `gen_cnt` exactly across 6 configurations; 42
checks pass. **Consequence for tuning:** the global operating point
does **not** transfer — locally `k` in `1/sqrt(lam+k)` is a node's
own fold count (~3) rather than the run's total selections (~300),
a 10x stronger bonus for the same `ds_alpha`, so local's optimum
should sit near `w_eff ≈ 1` against global's measured `w_eff = 10`,
and sweeping local on the global grid would produce a
uniformly-over-diversified artifact. **Doc organization is
independent:** local scope gets its own `###` section with its own
sweeps and zero code changes, because `--sync-doc` matches tables by
`feeds:` tbl-id and never reads `group`. Full writeup, including the
naming alternatives and the known-unfixed list, in
[decisions/local-covariance-scope.md](decisions/local-covariance-scope.md).

## 2026-07-28 — Evaluation, Experiments: slice MATH by `level`, never by `subject` — subject is a post-hoc analysis axis

**Context:** asked whether experiments should run on MATH subjects
instead of levels. **Decision:** no subject-filtered runs, ever;
`level` stays the launch-time slice and subject-wise numbers come
from re-slicing already-scored runs. Four reasons: (1) the axes are
not symmetric — `data.level` filters at launch so a level slice
costs a run, while every level run already spans all 7 subjects, so
a subject filter would only shrink `n` at the same per-problem cost;
(2) every scored trial record already carries `subject`, `level`,
and `unique_id` next to `pred_*@gb`, so the breakdown is a groupby,
not a GPU job; (3) power — subject cells inside a level are 10-36
problems (SE ~.08-.15), which resolves the .1-.3 model-family gaps
but not the ~.05 `w_eff` sweep effects, and repeated trials do not
help because every trial reuses the same fixed question set; (4)
`level` is a difficulty proxy that maps onto the hardness axis the
FBMCTS bounds are stated in, while `subject` supports only a
robustness claim, never a mechanism claim. Subject *is* worth using
for failure-mode diagnosis (completion length by subject bears
directly on the open `mml=8000` question in
[decisions/context-length-overflow-guard.md](decisions/context-length-overflow-guard.md)),
one model-family robustness table at level 5, and grading-flake
checks. Verified not confounded: every subject spans all five
levels and the L4+L5 share ranges only 45%-66%. **Considered and
kept open:** dropping `data.level` and running the full 500, then
deriving both the level tables and the subject tables from one run
— strictly more informative (filtering by level reproduces the
same question sets, so existing tables stay comparable) and it
lifts subject cells to 38-124 (SE .045-.081). Estimated cost is
1.6-1.8x the current L4+L5 pair, not 3.7x, since two level runs
are already paid for; at b=320 one trial is ~42-46 h so 4 trials
outrun a 3-day allocation, at b=80 it is ~6-12 hr/trial.
Recommendation if pursued: pilot at b=80 on one model-family
comparison, and never report the pooled 500 number (48% of the set
is levels 1-3, which compresses the hard-instance signal). Full
cross-tab, per-cell counts, and follow-ups in
[decisions/math-subject-vs-level-slicing.md](decisions/math-subject-vs-level-slicing.md).

## 2026-07-20 — Search, Instrumentation: record six exploration-diagnostic keys in every bl_* results dict

**Context:** the results dict recorded outcomes but almost nothing
about *where* in the tree the fixed expansion budget was spent, so
reproducing the depth-allocation finding
([findings/exp-findings/bl-frontier-depth-allocation.md](findings/exp-findings/bl-frontier-depth-allocation.md))
required the special `unittests/examine_driver.py` tree dump on a
handful of questions. **Decision:** add six per-question keys —
`phase_selected_depth` / `phase_selected_q` / `phase_selected_score`
(per-phase arrays) and `q_nodes_total` / `q_nodes_terminal` /
`q_nodes_completed` (scalars) — to all 8 bl cores with identical names
(key-parity is load-bearing; verified byte-identical key sets and a
uniform 14-tuple `mcts_search` return across the 8). The winning
frontier score is captured by stashing it on `agent.last_selected_score`
inside each selector (a declared field, like `cnt_node_max_depth`), so
no selector signature changes. Dropped a proposed `phase_frontier_size`
(grows near-linearly for all methods; its slope is a downstream
consequence of selected-depth + terminal counts, not independent).
Smoke-tested all 4 families offline: keys populate, and
`phase_selected_score` carries the correct per-family scale
(PUCT/density/depth-density/diversity-value). Tree-*structure* keys
(depth histogram, branching-by-depth, q-by-depth, visit Gini) discussed
and deferred to a follow-up. Full writeup:
[decisions/bl-search-tree-diagnostics.md](decisions/bl-search-tree-diagnostics.md).

## 2026-07-20 — Search, Refactor: `mcts_bl_sem_v01` loop reordered to generate→expand→select, matching v01's shape

**Context:** Tuan asked for `mcts_bl_sem_search_v01_00_00` to align
with `mcts_bl_cnt_search_v01_00_00`, the same request already applied
to `mcts_bl_kube_v01`/`mcts_bl_kdepth_v01` on 2026-07-17. sem's loop
still read select-first (pick the globally best frontier leaf via the
diversity-adjusted value, then expand or backprop it) — the one BL
sibling not yet rotated to v01's generate → expand → select order.
**Decision:** reorder only, same as the prior two siblings: each
iteration now expands (or backprops) `current_node`, adds children to
the frontier, then selects the next node globally across the whole
frontier — `current_node` initialized to `agent.root` rather than
root sitting in `leaf_nodes` awaiting its own first selection. No
config field changed; every existing `config_hash` is untouched.
**Why this one needed an extra check the other two didn't:** sem's
"global" `ds_alpha_schedule` reads a selection-count clock `t` at
each call to `select_leaf_from_list`, previously fed by the loop
index `p` directly (one selection per iteration under the old shape).
Under the reorder, `t` had to become its own counter, incremented
once per selection immediately before the call — get this wrong and
every "global"-schedule cell's diversity bonus silently shifts by one
selection's worth of `sqrt(log t)` growth for the rest of the run.
Verified with a standalone state-machine harness (7 cases mirroring
the kube/kdepth harness's coverage — generous budget, budget cutoff
mid-expand, early terminals, root-at-max-depth, phase-cap binding,
fast-draining frontier, wide branching) asserting not just that the
same nodes get selected in the same order, but that `t` at each
selection call is bit-for-bit identical between the two loop shapes.
All 7 pass. `select_leaf_from_list`'s singleton-frontier fast path
(previously commented as the "root-only first iteration" case) no
longer fires on the root specifically — root is consumed directly as
`current_node` at the first iteration without going through
selection at all — so the comment was corrected to describe what the
branch actually guards (any singleton frontier, which can still recur
later in a search); the guarded behavior itself is unchanged, since
root's `embeds` field was always `None` and `_fold_covariance`
already no-ops on it, so the removed call was a no-op either way.

## 2026-07-19 — Search, Feature: `mcts_bl_cnt_v02` gains a selectable frontier score (`score_mode`): one-hop parent blend vs. full-path decayed subtree value

**Context:** Tuan plans one sweep directly comparing the two
candidate path-aware scores — the shipped one-hop parent blend
(`alpha`) and an AlphaZero-style full-path decayed subtree value —
and expects to keep only the winner for future development.
**Decision:** both modes live in the same file behind one config
knob, `score_mode: "parent_blend" | "path_decay"` (default
`parent_blend` = shipped behavior), rather than as separate version
files — with a joint sweep planned, one method string with
`config_hash` separating arms beats two files' worth of
launcher/YAML/group boilerplate (the `kube_schedule` precedent).
Designed for deletion: the two scorers (`path_aware_puct`,
`path_decay_score`) share NO code and are joined only by the new
`MCTS.frontier_score` dispatcher — the single point that reads the
mode — so pruning the loser is a pure-deletion diff touching zero
lines of the survivor; no scorer registry or strategy classes (that
would be machinery half-deleted immediately after the sweep). The
post-sweep pruning becomes a v03 (v02 will have scored runs by
then, so it can't be edited in place). `path_decay`'s formula:
`q_path = sum_k gamma^k q(ancestor_k) / sum_k gamma^k` (leaf to
root) plus `cpuct*sqrt(N_parent)/(1+N_leaf)` — the AlphaZero
exploration shape, NOT v01's UCB1, so **cpuct is not comparable
across modes** (sweep it per mode); the only exact-v01 control arm
remains `parent_blend, alpha=1.0`. **P(x) intentionally omitted:**
canonical AlphaZero PUCT is
`P(x)*cpuct*sqrt(N_parent)/(1+N_leaf)`, with `P(x)` the policy
network's prior over a fixed action set. This repo has no policy
head and no per-child prior — the "actions" are free-form
LLM-generated steps, not an enumerable set with a scalar `P` each
— so the code uses the AlphaZero-shaped bonus *without* the prior
factor (effectively `P(x)`≡const, folded into `cpuct`). Hence
"AlphaZero-shaped," not "AlphaZero," in the code docstrings. This
is Option 2, not the prior-carrying Option 3 of
[decisions/bl-cnt-path-aware-frontier-score-design.md](decisions/bl-cnt-path-aware-frontier-score-design.md)
§4.5 — Option 3 (a `prior_p` field fed from normalized sibling
scores) was considered there and deliberately not built; if a
usable per-child prior (e.g. normalized sibling log-probs) is
ever added, it slots into the `P(x)` position and only then is
the term genuinely PUCT. Log labels in the selection
methods were made mode-neutral (`score =`, not `puct =`).
`config_hash` for existing v02 YAMLs changed (two new fields) —
safe, zero recorded v02 runs. The earlier design-space discussion
doc is explicitly NOT a dependency of this implementation (Tuan's
call, 2026-07-19: temporary artifact, not source of truth).
**Verified:** ast-extraction harness (dispatcher==method==
pre-refactor transcription for parent_blend across the edge-case
battery; path_decay against a hand-computed 3-level tree; gamma
limits 1.0/0.0 + zero-visit and root guards; unknown mode raises);
compile/DeprecationWarning/line-length clean; Hydra composes the
default and a sweep-shaped override arm to distinct hashes;
`status.py --verify` unchanged (same 4 pre-existing `mcts_cnt`
drift issues).

**Same day — `mcts_bl_kube_v02` aligned to the same two
score_modes.** Kube gets the identical config surface
(`score_mode`, `alpha`, `gamma`), the identical
designed-for-deletion structure (self-contained
`path_decay_kube_density`, no shared code with
`path_aware_kube_density`, single `MCTS.frontier_score(node, t)`
dispatcher), and the identical formula composed with kube's own
structure: `path_decay` = gamma-decayed path value + the AZ-shaped
bonus `kube_c*sqrt(clock)/(1+N)`, all divided by `cost`. The one
kube-specific design call: the AZ bonus takes the SCHEDULE'S clock
(`N_parent` under `"parent"` — making kube-parent path_decay
exactly bl_cnt v02's `path_decay_score / cost`, harness-proven —
and `1+t` under `"global"`), per the established "schedules differ
only in the bonus's clock" principle. `kube_c` is not comparable
across modes; `parent_blend alpha=1.0` remains the only exact-v01
arm. In-place edit again (zero kube-v02 runs; default reproduces
shipped behavior, harness-proven no-op). Verified: 4-part
ast-extraction harness incl. the cross-file alignment identity,
Hydra composition of default + a global/path_decay arm, `--verify`
unchanged.

## 2026-07-18 — Search, Feature: v02 of `mcts_bl_cnt`, `mcts_bl_kube`, `mcts_bl_kdepth` — eager terminal backprop, implemented per each family's §7 verdict

**Context:** Tuan asked to "create a v02 versions for these three
methods, that will implement eager terminal backprop with additional
options (maybe options mentioned above but determined later)" — the
direct implementation follow-up to the 2026-07-16 design-only
entries above. **Decision:** each family got a genuinely different
treatment, per
[decisions/bl-cnt-path-aware-frontier-score-design.md](decisions/bl-cnt-path-aware-frontier-score-design.md)
§7's own per-variant verdicts, resolved via a short scoped-questions
pass rather than one mechanism copy-pasted three times:
`mcts_bl_cnt_v02` gets the terminal-split + eager backprop plus
Option 1's parent-blended PUCT (`alpha` knob, `alpha=1.0` recovers
v01 exactly — verified numerically identical across 6+ edge cases);
`mcts_bl_kube_v02` gets the same terminal-split (both
`kube_schedule` values — fixes a real defect where dead-ends were
*permanently* stuck at `cost≤0`/`-inf`, silently disabling the
`kube_affordable` fallback for the rest of any run) plus the
identical parent-blend, but **only** under `kube_schedule="parent"`
(the `"global"` bonus has no per-node channel to blend into, so it
gets hygiene only, no invented formula); `mcts_bl_kdepth_v02` gets
**hygiene only** — `depth_density()` is byte-identical to v01's
(diff-verified), since no visit-count/parent-q channel exists there
at all for a blend to hook into.

**Correction, same day:** the initial implementation backpropped a
terminal child *immediately*, inline during expand, before that
step's own selection ran. Tuan caught the conceptual problem: this
lets a same-batch terminal sibling's outcome — produced by the exact
`_generate_candidates` call that also produced the candidates NOW
being ranked — leak into the selection choosing among them, which
isn't genuinely prior evidence. Fixed to **delayed-eager**: newly
generated terminal children are queued, the selection that was
already going to happen runs first using pre-existing state, then the
queue flushes — so a terminal's effect reaches every selection
strictly after the one immediately following its own creation, never
that one. Live in `mcts_bl_cnt_v02` and `mcts_bl_kube_v02` (both read
`parent.q_value()`/`parent.visit_count()` — exactly the channel a
same-batch leak would corrupt); behaviorally inert but structurally
applied for consistency in `mcts_bl_kdepth_v02` (`depth_density`
reads neither). Re-verified with a dedicated harness exercising the
real `path_aware_puct` formula, plus a re-run of the Hydra-composition
and `status.py --verify` checks — same clean result. A `` `\S` ``
transcription artifact (should have been the literal `§` character)
found and fixed across all three core files, `utils/configs.py`, and
this doc during the same pass.

**Second same-day revision — kube blend extended to `"global"`:**
the scoping answer "parent-blend only under `kube_schedule="parent"`"
was reversed later the same day, after Tuan asked whether `"global"`
could have the parent-blended value term too. The original "no
per-node channel" reasoning was correct about the *bonus* term
(`"global"`'s clock is the shared `t`; backprop can't move it) but
was wrongly transferred to the *value* term, which Option 1 blends
and which exists identically under both schedules. On analysis,
global+blend is the **cleaner** one-variable test of the blend: under
`"parent"`, a backprop moves two entangled channels (bonus clock
`N(P)` — the count-attraction burst — and blended value `q(P)`);
under `"global"` the bonus can't burst, so the blend is the *only*
ancestor channel — pure discouragement of failed neighborhoods with
no attraction side channel. Implemented by hoisting the blend out of
the schedule branch (the branch now picks only the bonus clock);
edited v02 **in place, no v03 bump**, because zero kube-v02 runs
exist and `alpha=1.0` reproduces the pre-reversal `"global"` behavior
exactly (strict generalization). Delayed-eager flush timing is
thereby live under both schedules now. Verified via a new harness
(`verify_kube_global_blend.py`, method source extracted from the
real file via ast): parent-schedule hoist is a no-op, global
`alpha=1.0` equals shipped behavior, global `alpha<1` moves by
exactly the blend delta. Sweep note: global sweeps need their own
`alpha=1.0` control arm. Full rationale in
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md)
§3.5.

All three verified end-to-end: Hydra composition, distinct
`config_hash` from each v01 sibling, and `status.py --verify` showing
zero new problems (same 4 pre-existing `mcts_cnt` drift issues as
every prior run this session). Each v02 got its own `status.py`
group label (`cnt-mcts-bl-v02` etc.) rather than sharing its v01's —
considered the `mcts_sem_v01`/`mcts_sem_v02` shared-group precedent,
decided against it: the algorithm change here is large enough to
warrant its own `docs/exp-comp-*.md` table rather than a same-table
row next to v01. **Why:** full per-file mechanics, the four scoping
questions and their answers, the delayed-vs-immediate design-space
table, and the verification detail are in
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md)
(see §1.5 for the correction).

## 2026-07-17 — Search, Refactor: `mcts_bl_cnt_v03` renamed to `mcts_bl_kdepth_v01`, its own algorithm family; result dirs, manifests, and ledger migrated

**Context:** Tuan asked to rename `mcts_bl_cnt_search_v03_00_00` to
match the naming convention of the previous day's `mcts_bl_cnt_v02` →
`mcts_bl_kube_v01` rename, and asked what name would best describe a
protocol combining "knapsack-based allocation with depth-based
exploration." **Decision:** the same family-scope question from the
KUBE rename was raised and answered the same way — this variant's own
docstring already argued it was "a deliberately different theoretical
basis... not a bugfix or refinement" of anything in bl_cnt/bl_kube,
and "cnt" specifically denotes count-based (visit-count) exploration,
which this variant has none of at all. New family: `mcts_bl_kdepth`
(v01 of that family). Tuan proposed the name directly — `kdepth`
(knapsack + depth), mirroring `kube`'s shape and answering both the
naming and family-scope questions in one coinage. Same migration
mechanics as the KUBE rename: `config_hash` includes `search.method`
and `level_dir` derives the result-dir name from it, so all 5 existing
scored result directories were physically renamed, their manifests'
`config_name`/`config_hash`/`config_identity.search.method` rewritten
with a faithfully recomputed hash (reproducing `config_hash()`'s exact
algorithm, independently spot-verified), and W&B `run_id` links
preserved. `experiments.yaml`'s 5 corresponding entries had
`config_root`, `group` (`cnt-mcts-bl` → `kdepth-mcts-bl`), and their
`note:`'s dir/hash citation updated. Checked whether v01's (`bl_cnt`)
and bl_kube v01's own sibling docstrings reference v03 by name —
neither does, so no edits were needed there. Two unrelated pre-existing
typos ("knapbe" for "knapsack," appearing twice in the module
docstring, pointing at a filename that never existed) were fixed
opportunistically while touching the file anyway. Every code, config,
and doc reference across the repo was updated to match, except
genuinely historical `decisions-log.md` entries, which keep their
original "v03" terminology per the append-only convention. (The
standalone migration-record doc this entry originally pointed to was
later removed, 2026-07-19 — Tuan's call that this class of doc
doesn't need to be kept; this log entry stands as the record.)

## 2026-07-16 — Search, Refactor: `mcts_bl_cnt_v02` renamed to `mcts_bl_kube_v01`, its own algorithm family; result dirs, manifests, and ledger migrated

**Context:** Tuan asked to rename
`mcts_bl_cnt_search_v02_00_00` to `mcts_bl_kube_search_v01_00_00`. Two
scoping questions were asked and confirmed before touching anything:
whether the config `method`/`algo` string changes too (not just
file/class names), and whether sibling files' (v01, v03) own
docstrings should be updated to the new name. **Decision:** yes to
both. The variant becomes its own family, `mcts_bl_kube` (v01 of that
family, not v02 — first member of a new lineage) rather than a
same-family sibling of `mcts_bl_cnt`'s PUCT variant, since fractional
KUBE is a distinct selection theory (cost-normalized knapsack density),
not a PUCT variant. `mcts_bl_cnt_v03` (depth-shaping) was deliberately
NOT renamed or moved into the new family — it already frames itself
as "a deliberately different theoretical basis... not a refinement" of
KUBE and has no visit-count/confidence-bound term at all, closer to a
heuristic bl_cnt variant than to KUBE's bandit lineage. Because
`config_hash` hashes the full `search` dict (including `method`) and
`level_dir` derives the result-dir name from `method`, changing the
string forced physical migration, not just a code change: all 5
existing scored result directories were renamed, their manifests'
`config_name`/`config_hash`/`config_identity.search.method` rewritten
with a faithfully recomputed hash (reproducing `config_hash()`'s exact
algorithm against the mutated identity, independently spot-verified),
and W&B `run_id` links preserved. `experiments.yaml`'s 5 corresponding
entries had `config_root`, `group` (`cnt-mcts-bl` → `kube-mcts-bl`),
and their `note:`'s dir/hash citation updated. Every code, config, and
doc reference across the repo was updated to match, except genuinely
historical `decisions-log.md` entries (2026-07-09, predating this
rename), which keep their original "v02" terminology per the
append-only convention — only one stale internal link among them was
fixed. (The standalone migration-record doc this entry originally
pointed to was later removed, 2026-07-19 — Tuan's call that this
class of doc doesn't need to be kept; this log entry stands as the
record.)

## 2026-07-16 — Search, Design: eager-terminal-backprop proposal extended to v02 (KUBE) and v03 (depth-shaping); different verdict per variant; a v02/v03 docstring discrepancy surfaced

**Context:** the eager-terminal-backprop proposal below (see the
following entry) was evaluated against v01 only. Tuan asked how the
same proposal — terminal candidates split out of the frontier and
backpropped at creation, no scoring change — would land on the other
two active `bl_cnt` variants,
`mcts_bl_cnt_search_v02_00_00` (fractional KUBE — renamed later the
same day to `mcts_bl_kube_search_v01_00_00`, see the 2026-07-16
rename entry above) and
`mcts_bl_cnt_search_v03_00_00` (depth-shaping knapsack), which each
replace PUCT with a different selection criterion. **Decision:** no
code changed; the analysis was extended and recorded. v02: max-depth
dead-ends currently score `−inf` (`cost ≤ 0`) and so are *never*
selected under the present lazy scheme — not delayed, permanently
stuck — which also means they permanently satisfy the `kube_affordable`
feasibility filter's "always eligible" terminal clause, silently
preventing that filter's empty-set fallback from ever relaxing to the
full frontier as designed. The terminal-split fix genuinely repairs
this, but under `kube_schedule="parent"` it also introduces a new
failure mode identical to v01's (count-driven attraction toward failed
siblings); under `"global"` it's pure hygiene with no such side
effect. v03's selection criterion reads no visit counts at all, so
backprop is write-only on every channel there — the fix is a
near-pure refactor and the stated goal (faster negative-feedback
propagation) is unreachable in v03 through any backprop-timing change.
Separately, both v02's and v03's module docstrings describe a
terminal-split frontier policy that neither file's code actually
implements — stale drift, not a live bug, but left uncorrected.
**Why:** full per-variant mechanics, the `kube_affordable` walkthrough,
and the comparison table are in §7 of
[decisions/bl-cnt-path-aware-frontier-score-design.md](decisions/bl-cnt-path-aware-frontier-score-design.md).

**Update 2026-07-18:** implemented, per this entry's own §7 verdicts
— v02s of all three families now exist. See the 2026-07-18 entry
below and
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md).
This section is left as written for historical accuracy (it correctly
describes the pre-implementation analysis), not corrected in place.

## 2026-07-16 — Search, Design: bl_cnt eager terminal backprop alone doesn't propagate dead-end signal; two path-aware scoring designs proposed instead

**Context:** Tuan proposed splitting terminal candidates out of
`leaf_nodes` and backpropping them eagerly at creation (instead of
waiting for a later selection to trigger it), aiming to make bad
directions discourage the search sooner. **Decision:** the proposal's
mechanics are sound but don't reach the stated goal in bl_cnt's
frontier-selection shape — `select_child_from_list` never reads an
internal node's `q`, only a leaf's own `q`/`N` and its parent's *visit
count*, so a backpropped value is write-only regardless of how early
it lands; the only live effect is `N(ancestor) += 1`, which increases
exploration pull toward the failed path's siblings rather than
discouraging it. Two alternative designs that would actually work were
sketched instead — not implemented. **Why:** full analysis, the
magnitude argument, both formulas, and the recommendation (try the
one-hop blend first, in a new version file, before the full
discounted-path variant) are in
[decisions/bl-cnt-path-aware-frontier-score-design.md](decisions/bl-cnt-path-aware-frontier-score-design.md).

**Update 2026-07-18:** the one-hop blend (Option 1) shipped as
`mcts_bl_cnt_v02` — see the 2026-07-18 entry below and
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md).
This section is left as written for historical accuracy, not
corrected in place.

## 2026-07-16 — Search, Refactor: mcts_bl_cnt_v01 loop reordered to generate→expand→select; selection scope stays global

**Context:** `mcts_bl_cnt_search_v01_00_00`'s loop read select-first
(pick the globally best frontier leaf, then expand it), while
`mcts_cnt_search_v01_00_00`'s walk step reads generate → expand →
select. Tuan asked for the two to follow the same
generation/expansion/selection order. "Selection among the expanded
children" is ambiguous in the frontier setting: it can mean the fresh
children merely *compete* in the next selection (a reorder), or that
selection is *restricted* to them (a per-parent greedy descent,
~max_depth generations per frontier pick — a different algorithm).
**Decision:** reorder only, confirmed explicitly: each iteration now
expands (or backprops) the current node, adds the children to the
frontier, then selects the next node globally across the whole
frontier. The per-parent-scope descent variant was deliberately not
built. No config field changed, so every existing `config_hash` is
untouched.
**Why:** the reorder is a pure rotation of the old loop — the same
nodes are expanded in the same order (traces can drift only via
RNG-draw ordering on tie-breaks), so scored `cnt-mcts-bl-v01` cells
remain comparable, while the two sibling files now read step-for-step
parallel. The descent variant would have changed the search shape
fundamentally and, per the two-tier convention in
[algorithms.md](algorithms.md), belongs in a new version file if ever
wanted — not an in-place edit of v01.

## 2026-07-15 — Search: mean-centering recommendation for mcts_bl_sem_v01 — fixed stays primary, local is a stricter ablation than in v02

**Context:** `cov_dtype` and `embeds_center_mode` ("fixed"/"local")
were ported into `mcts_bl_sem_search_v01_00_00` from
`mcts_sem_search_v02_00_00` (same day, see the config-porting entry
below) so bl_sem could run either mode at all. Porting the mechanism
answers "can it run local mode," not "should it, here" — bl_sem's
best-first frontier selection accumulates `V` differently from v02's
tree walk (global sequential folding of one winner per selection,
each frontier node expanded exactly once, vs. v02's per-parent
sibling comparison with possible revisits), so the mode recommendation
needed its own pass rather than inheriting v02's.
**Decision:** fixed held-out centering stays the recommended default
for bl_sem, same as v02. `local` mode is available (parity with v02,
already verified to run) but scoped strictly as an ablation arm, with
a **stronger** incoherence caveat than v02 carries: v02's per-parent
sibling groups are at least locally comparable at selection time
(same parent, same expansion); bl_sem's global frontier selection
compares candidates whose *groups* were centered at different points
in the run against each other directly, so the affine-offset mismatch
that already made local mode an ablation in v02
([embeds-centering-design.md](decisions/embeds-centering-design.md#local-mean-sibling-group--built-2026-07-1415-v02-only))
is structurally worse here. No design for a coherent adaptive/online
mode in bl_sem yet — sketched only informally: it would need
selection-time centering (not expansion-time Welford, since frontier
comparison is global and cross-branch) and `V` rebuilt from raw
embedding history whenever the center moves, which is a materially
different shape than the Welford-at-expansion-time design already
scoped for v02's online mode.
**Why:** recommending fixed-by-default without flagging local as
*more* fragile here than in v02 would understate the risk — bl_sem's
frontier structurally removes the one thing that kept v02's local
mode locally coherent (comparison scoped to one parent's siblings).
Full writeup:
[embeds-centering-design.md](decisions/embeds-centering-design.md#local-mean-sibling-group-in-frontier-selection--bl_sem-caveat).

## 2026-07-15 — Search, Configs: cov_dtype + embeds_center_mode — reimplemented after a pydantic field-declaration bug

**Context:** both flags were built and verified 2026-07-14 (hash
stability, ledger sweep), then launched live via the idle-GPU
orchestrator. Every launched job crashed instantly: `ValueError:
"MCTS" object has no field "cov_dtype"`. Root cause: `MCTS`
(`core/mcts_sem_search_v02_00_00.py`) is a **pydantic `BaseModel`**,
which raises on `self.attr = value` for any attribute not declared
as a class-level field — the first implementation only assigned
`self.cov_dtype` inside `__init__`, never declaring it alongside
`V`/`V_inv`/`completed_nodes`. Reverted both flags entirely
(`git restore`) rather than patch live, then reimplemented.
**Decision:** identical design to the 2026-07-14 entries below, plus
one fix: `cov_dtype: Any = np.float64` is now declared as a
class-level pydantic field on `MCTS`, matching the existing
`V`/`V_inv` pattern. Also added a shared predicate `_is_local_center
(sc)` so `_extract_embeds`'s defer decision and
`_maybe_center_local`'s gate can't silently diverge (a hazard flagged
in a code review of the first implementation, fixed opportunistically
during the redo). Both hash-verified unchanged from the first pass
(`cfg-c371341f` baseline stable) AND, this time, verified with a live
end-to-end smoke test on an idle GPU (`WANDB_MODE=offline`,
1q/1trial) for each flag before considering it done — the step that
was skipped before shipping the buggy first version. Full writeups:
[decisions/covariance-precision.md](decisions/covariance-precision.md),
[decisions/embeds-centering-design.md](decisions/embeds-centering-design.md#local-mean-sibling-group--built-2026-07-1415-v02-only).

## 2026-07-14 — Docs, Naming: exp-comparison*.md renamed to exp-comp-{dataname}-{level}.md

**Context:** three tracking docs existed under inconsistent names —
`exp-comparison.md` (PRM800K level-4, no dataset/level in the name),
`exp-comparison-level5.md` (PRM800K level-5, no dataset in the name),
`exp-comparison-gsm8k.md` (GSM8K, correctly dataset-named). Adding a
fourth doc for AIME (itself split across `aime2024`/`aime2025`
dataset names, per the same-day `data.name` decision) made the
inconsistency untenable — there was no naming slot for "which
dataset, which level" that generalized.

**Decision:** rename all three to `exp-comp-{dataname}-{level}.md`,
omitting `{level}` when the dataset has none:
- `exp-comparison.md` → `docs/exp-comp-prm800k-level4.md`
- `exp-comparison-level5.md` → `docs/exp-comp-prm800k-level5.md`
- `exp-comparison-gsm8k.md` → `docs/exp-comp-gsm8k.md`

All cross-references updated in the same commit: `experiments.yaml`
`note:` fields (~90 occurrences), `status.py` comment, doc
cross-links (`exp-comp-prm800k-level5.md`'s and `exp-comp-gsm8k.md`'s
own provenance notes), `docs/benchmarks.md`, and both
`.claude/skills/exp-new-comparison-table/SKILL.md` and
`.claude/skills/exp-record-results/SKILL.md`. Historical mentions of
the old names elsewhere in this log (e.g. the 2026-07-07 entry below)
are left as-is — they describe what the file was called *at the
time*, which this rename doesn't retroactively change.

**Why:** future dataset additions (AIME and beyond) need a naming
slot that scales without inventing a new ad hoc suffix each time.

## 2026-07-10 — Infra: idle-GPU experiment orchestration — 15-min cycle, queue.yaml + auto-refreshed jobs.yaml (designed, not yet armed)

**Context/decision:** recurring system that launches queued
experiments onto idle GPUs inside existing SLURM jupyter
allocations, generalizing the manual flow validated today
(`srun --jobid=<id> --overlap nvidia-smi` idle probe →
`nohup srun --overlap <launcher> &` launch; first run: level-5
llama-1b cnt-mcts, W&B `05lky8bc`). Three files under
`orchestration/`: `queue.yaml` (thin user-curated queue, drained
top-first; orchestrator only flips planned→running + writes a
launch block; deliberately NOT `experiments.yaml`, which stays
the append-only ledger), `jobs.yaml` (auto-refreshed each cycle
from squeue, with a persistent `exclude:` list as the manual
override), and `log.md` (append-only audit). Idle ⇔ 0% util AND
0 MiB. Walltime guard via `expected_hr`. Explicitly out of scope
per Tuan: success/failure monitoring, retries, completion
marking (manual verification; W&B is the run log). Scheduler:
CronCreate `*/15` if it runs locally, else system crontab +
headless `claude -p`. Supersedes the SSH-based multi-node
orchestrator vault todo for SLURM clusters. Full writeup:
[decisions/hpc-idle-gpu-orchestration.md](decisions/hpc-idle-gpu-orchestration.md).

## 2026-07-10 — Search: context-length overflow = terminality — guard decided after llama-3b bl_sem crashes (implementation pending)

**Context/decision:** both `mcts_bl_sem_v01` llama-3b fp16 cells
(w_eff=100 `0f06296f`/`2goolnzd` 0/2 trials; w_eff=10
`3ca318f6`/`yf562ig8` 1/2) crashed with vLLM's "decoder prompt
(length 5000) ... longer than the maximum model length of 5000" —
a deep frontier path filled `max_model_len` and nothing in the
stack checks prompt length (`generate_k_steps` submits unchecked;
no per-question try/except), so one over-long path killed each
trial. Token-length analysis across the w_eff=10 grid showed the
exposure is Llama-specific (5–7 near-cap questions per trial for
llama-1b/-3b vs ≤1 for every qwen model) and amplified by the
frontier protocol (the phase-based `mcts_sem_v02` counterpart
completed 2/2 with one near-cap question). Decided: treat context
exhaustion exactly like `max_depth` — a terminality condition
(backprop + drop from frontier) in every search variant — plus a
per-question try/except in the launcher as containment; rejected
raising `max_model_len` (moves the cliff, costs V100 KV memory,
silently changes the experiment). Not yet implemented; both
llama-3b cells rerun after it lands. Full writeup:
[decisions/context-length-overflow-guard.md](decisions/context-length-overflow-guard.md).

## 2026-07-09 — Search: new `mcts_bl_cnt_v03` — depth-shaping knapsack bonus, no UCB/exploration term

**Context/decision:** proposed replacing v02's UCB bonus with a
fixed depth-preference function, `f_a(z)=1-z^alpha`, under the same
fractional-knapsack objective/cost constraint — the same shape that
was in the original pre-rewrite v02, removed earlier today for
lacking a UCB/visit-count term. As written the exponent was indexed
on the wrong fraction: `z=(d_max-d_i)/d_max` (cost fraction) gives
`f_a=0` at the root and `f_a=1` at max depth, rewarding deep nodes —
inverted from the stated goal of encouraging shallow exploration.
Fixed by indexing `f_a` on the depth fraction `d_i/d_max` instead (0
at root → max bonus, 1 at max depth → no bonus). Implemented as a
new sibling, `mcts_bl_cnt_search_v03_00_00.py`
(`BLMCTSCntV03Config`: `depth_beta`, `depth_alpha`,
`kube_affordable`), not folded into v02 — `f_a` is a deterministic,
evidence-blind function of tree position with no confidence-bound
guarantee, so mixing it into v02's PUCT-vs-KUBE ablation would
muddy what that comparison tests. v01/v02/v03 now form a clean
three-way comparison (PUCT / evidence-based UCB bonus / fixed
depth-shaping bonus) sharing the same cost mapping and affordability
restriction. Full writeup:
[decisions/bl-kdepth-knapsack-bonus.md](decisions/bl-kdepth-knapsack-bonus.md).

## 2026-07-09 — Search: KUBE alignment audit — `(q+bonus)/cost` confirmed against paper + budget-mab; affordability restriction added (`kube_affordable`)

**Context/decision:** audited `kube_density` against the paper's
Eq. 9 (read from `budget-mab/paper.pdf` directly) and
`budget-mab/src/algorithms.py::FractionalKUBE`. The `(q+bonus)/cost`
form is confirmed correct — the paper divides both the mean and the
confidence bonus by cost. Two gaps surfaced: (1) the paper's
`sqrt(2·ln t/n)` constant is folded into `kube_c` (documented as
fine — same convention as v01's `cpuct`); (2) the feasibility step
was missing — the paper/reference restrict the argmax to arms
affordable under the residual budget *before* ranking, we ranked the
whole frontier. Fixed via set-restriction in
`select_child_from_list`, gated by new
`BLMCTSCntV02Config.kube_affordable` (default `true`): terminal
nodes always eligible (they cost no generations), empty affordable
set relaxes to the full frontier (cost is a worst-case bound; EOS
can finish early), `false` kept as the middle arm isolating cost
normalization from feasibility filtering. Full audit and rationale
in
[decisions/kube-affordability-restriction.md](decisions/kube-affordability-restriction.md).

## 2026-07-09 — Search: `mcts_bl_cnt_v02` bonus clock made configurable (`kube_schedule`), default switched from global `t` to parent visits

**Context/decision:** the entry below chose a global-clock UCB bonus
for v02 by analogy to `mcts_bl_sem_v01`'s `ds_alpha_schedule=
"global"`; re-deriving it for KUBE's specific frontier lifecycle
showed the analogy doesn't hold (every node has `visit_count()==1` at
comparison time, so the global-clock bonus is a frontier-wide
constant with no per-node discrimination). Reframing as a tree-search
question — UCT is UCB1 re-instantiated per parent, not a different
bonus — showed the fix is to index the clock by `parent_visits`
instead, which also restores real per-node discrimination under
short-tree/frequent-terminal dynamics and turns the v01-vs-v02
comparison into a single-factor (cost-normalization-only) ablation.
New `BLMCTSCntV02Config.kube_schedule: "parent"|"global"`, default
`"parent"`; `"global"` kept as an explicit ablation arm. (Also fixed
in passing: the zero-visit case used `bonus=inf`, forcing exhaustive
exploration of every new node before any q_value-informed comparison
— unaffordable under a small `gen_budget`; now `bonus=0.0`, matching
v01's `puct()`.) Full discussion and derivation in
[decisions/bl-kube-bonus-schedule.md](decisions/bl-kube-bonus-schedule.md)
(renamed 2026-07-16 from `kube-bonus-schedule.md`).

## 2026-07-09 — Search, Refactor: `mcts_bl_cnt_v02` rewritten to match budget-mab's actual Fractional KUBE; infra aligned with v01

**Context:** `core/mcts_bl_cnt_search_v02_00_00.py` predated the
infra-alignment pass v01 got in commit `646629c` (flat `config.*`
access instead of `config.search.*`/`config.gen.*`, mismatched output
key names, hardcoded `prm.score(..., batch_size=4)`, a launcher with
no `manifest.json`/`.done` markers/resume/timing-sidecar/scoring
step). Its "KUBE" selection was also a hand-rolled static depth-decay
heuristic — `(q_value + beta*(1-((max_depth-depth)/max_depth)**alpha))
/ (max_depth-depth)` — with no UCB/exploration-bonus term and no
visit-count dependence at all, which does not match Tran-Thanh et
al.'s Fractional KUBE (arXiv:1204.1909 sec. 3.3) or the reference
implementation in the sibling `budget-mab` repo
(`src/algorithms.py::FractionalKUBE`: `density = (mean + UCB
confidence bonus) / cost`, restricted to affordable arms).

**Decision:** rewrote `mcts_bl_cnt_search_v02_00_00.py`,
`generate_mcts_bl_cnt_v02.py`, and the config
(`utils/configs.py::BLMCTSCntV02Config`, `conf/search/
mcts_bl_cnt_v02.yaml`, `conf/mcts_bl_cnt_v02_prm800k.yaml`) to mirror
v01's infra exactly, changing only the selection formula:

    density(x) = (q_value(x) + kube_c*sqrt(log(1+t)/visits(x)))
                 / cost(x)
    cost(x) = max_depth - depth(x)

- **Cost** stays `max_depth - depth` (the existing intuition: shallower
  nodes have more remaining generations to reach the depth limit — the
  MCTS analogue of an arm's fixed pull price in the bandit
  abstraction). Confirmed with Tuan rather than switching to a
  cost-per-expansion=1 mapping (which would have collapsed
  FractionalKUBE to plain UCB1 over the frontier).
- **UCB bonus** added with the **global-time schedule**
  (`t` = frontier selections so far, shared by every node), the same
  choice as `mcts_bl_sem_v01`'s `ds_alpha_schedule="global"` — the
  frontier is a flat, globally-shared arm set in both algorithms, so
  the same schedule reasoning applies (see
  [decisions/global-vs-local-exploration-schedule.md](decisions/global-vs-local-exploration-schedule.md)).
- The exploration coefficient is named `kube_c`, not reusing bl_cnt
  v01's `cpuct`, so config diffs/tables don't imply the two formulas
  are the same (density-with-UCB-then-divide-by-cost vs. additive
  PUCT bonus) when they aren't.
- `kube_beta`/`kube_alpha`/the `f_a(z)` depth-decay term are dropped
  entirely — no longer part of the design, not just renamed.
- Smoke-tested (llama-1b, qwen PRM, 2 questions, b=80): ran clean,
  33 and 45 completions respectively — a strong contrast with v01's
  0-completion result on the same budget/question-1 config (see
  `docs/benchmarks.md`, 2026-07-09 trace comparison), though this is
  one smoke run, not a scored comparison.

**Why:** the goal was for v02 to actually be "the fractional-KUBE
variant, following the design from the mab-budget repo" — before this
rewrite it was neither infra-aligned with its own sibling nor a
faithful KUBE implementation by its own reference.

## 2026-07-08 — Search: PUCT's local per-parent clock is not a fairness gap to fix, in either direction

**Context:** follow-on to the `mcts_bl_sem_v01` entry below. Two
related "should X use a global clock instead, for fairness" questions
came up: (1) should `mcts_bl_cnt_v01` adopt `mcts_bl_sem_v01`'s new
`ds_alpha_schedule="global"` default; (2) should `mcts_bl_cnt_v01`'s
PUCT bonus use a global visit-count clock instead of `parent_visits`,
for fairness against `mcts_cnt_v01`.

**Decision (no code change; findings only):** both answered no, for
different reasons — full derivation in
[decisions/global-vs-local-exploration-schedule.md](decisions/global-vs-local-exploration-schedule.md).
(1) doesn't apply because PUCT has no `ds_alpha`/schedule concept at
all — its bonus is the count-based degenerate case of the diversity
term's linear-bandit width, not an approximation needing the same fix.
(2) doesn't apply because `mcts_cnt_v01::MCTSNode.puct` and
`mcts_bl_cnt_v01::MCTSNode.puct` are byte-identical already — there is
no existing asymmetry between them to correct. Both real reasons trace
back to the same fact: PUCT's `parent_visits`/`visits` are exact local
counts that make `sqrt(log N/n)` a valid per-node confidence bound;
swapping in a global counter would compare mismatched quantities
(elapsed global time vs. one node's own visits), not "generalize" the
bonus. A genuine "make PUCT's exploration term account for global
tree state" idea would need its own bandit-theoretic justification and
would apply identically to `mcts_cnt_v01` and `mcts_bl_cnt_v01` (same
formula) — open, unexplored, not pursued here.

## 2026-07-08 — Search: `mcts_bl_sem_v01` composes frontier selection with semantic diversity; the alpha schedule is exposed, not hardcoded

**Context:** the frontier (best-first) and semantic-diversity search
families existed only as separate variants: `mcts_bl_cnt_v01` (global
`leaf_nodes` frontier, PUCT selection) and `mcts_sem_v02` (phase-based
root-to-leaf walks, `ds_beta*q + ds_alpha*sqrt(x^T V^-1 x)` child
selection). A combined variant — frontier selection with the
diversity-adjusted value — needed three composition decisions where
the parents disagree.

**Decision:** new `core/mcts_bl_sem_search_v01_00_00.py`
(`method=mcts_bl_sem_v01`, `BLMCTSSemConfig`), run from
`generate_mcts_sem.py` (same `_search` signature as sem — no new
launcher; bl_cnt only got its own because its signature differs).

1. **ds_alpha schedule is a config knob (`ds_alpha_schedule`), not a
   hardcoded transplant.** sem_v02 scales the diversity weight by
   `sqrt(log(1+parent_visits))` per selection — a local clock. On a
   global frontier the natural analog is ambiguous, and the choice is
   a real design axis, so all three candidates are implemented:
   `global` (default; `sqrt(log(1+t))`, t = frontier selections —
   the frontier is a flat arm set and the diversity term is the
   LinUCB confidence width, so the global clock is the OFUL-standard
   form and keeps effective-alpha magnitudes comparable to sem_v02's),
   `parent` (literal transplant), `none` (constant); full derivation
   (plus why the same reasoning does NOT extend to `mcts_bl_cnt_v01`'s
   PUCT bonus) in
   [decisions/global-vs-local-exploration-schedule.md](decisions/global-vs-local-exploration-schedule.md).
   The knob is in
   the config hash, so schedule ablations get distinct run
   identities. Expected to be empirically minor at `ds_alpha≥10`
   (the multiplier only spans ~0.8–2x and the plateau finding says
   that range is flat) but it matters for interpretability and for
   small-alpha sweeps.
2. **No first-visit q-only special case.** sem_v02 selects by q alone
   on the first descent through a newly expanded node. That concept
   is per-parent and has no analog under global selection; fresh
   children compete by `q + diversity` immediately, and since none of
   them are in `V` yet their widths start near-equal — q
   differentiates them anyway, which is what the special case
   approximated.
3. **Covariance folds on every selection, root excluded.** Same
   semantics as sem_v02 (every committed direction enters `V`,
   including terminal picks). The root has no embedding and is only
   ever selected alone on the first iteration, so it is skipped —
   mirroring sem, where the root's embedding never enters `V` either.

Also dropped from the schema: `cpuct` (no PUCT) and `revisit_policy`
(a frontier node is expanded at most once by construction).
`BLMCTSSemConfig` is a fresh `SearchConfig` subclass rather than a
`MCTSSemV02Config` child so it can't silently inherit knobs that mean
nothing here. Index entry: docs/algorithms.md ("BL-Sem-MCTS").

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
[generate_mcts_bl_cnt.py](../generate_mcts_bl_cnt.py) — at the
time of this entry, `generate_mcts_bl_cnt_v01.py`; merged with
its v02/v03 siblings into one launcher 2026-07-09, see that
day's "launchers merged" entry)
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
