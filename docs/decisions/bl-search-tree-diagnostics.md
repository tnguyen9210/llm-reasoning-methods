# Recording exploration diagnostics in the bl_* results dicts

*2026-07-20*

Records the design discussion behind adding six per-question
exploration-diagnostic keys to every `mcts_bl_*` search core's results
dict, so the depth-allocation analysis (see
[../findings/exp-findings/bl-frontier-depth-allocation.md](../findings/exp-findings/bl-frontier-depth-allocation.md))
becomes reproducible from **normal launcher output over the whole
dataset**, not just from the special tree-dump driver on a handful of
hand-picked questions.

Files changed (all 8 bl cores): `mcts_bl_cnt_search_v0{1,2}_00_00.py`,
`mcts_bl_kube_search_v0{1,2}_00_00.py`,
`mcts_bl_kdepth_search_v0{1,2}_00_00.py`,
`mcts_bl_sem_search_v0{1,2}_00_00.py`.

## 0. The problem

The results dict recorded **outcomes** (`completions`, `comp_depth`,
`comp_phase`, `q_nodes_max_depth`, …) but almost nothing about the
**process** — where in the tree the fixed 80-expansion budget was
spent. Reproducing the depth-allocation finding required a separate
`unittests/examine_driver.py` run that dumps the full per-node tree
JSON, and only on 5 questions. Recording a few cheap per-phase / per-
question quantities inside the loop makes that analysis a byproduct of
every real run.

## 1. What is recorded (six keys, uniform across all 8 cores)

Scope prefixes follow the existing convention (`phase_*` = per-question
array over phases; `q_*` = per-question scalar).

| key | scope | meaning |
|---|---|---|
| `phase_selected_depth` | per-q list | depth of the node chosen for expansion each phase — the shallow-vs-deep signal |
| `phase_selected_q` | per-q list | that node's q-value |
| `phase_selected_score` | per-q list | the winning frontier score (per-family: PUCT / density / depth-density / diversity-adjusted value) |
| `q_nodes_total` | per-q scalar | total nodes created (one iterative tree walk) |
| `q_nodes_terminal` | per-q scalar | # terminal nodes = completed + max-depth dead-ends |
| `q_nodes_completed` | per-q scalar | # EOS/length-completed nodes |

**Key-name parity is load-bearing.** The results-dict comment in every
core states key names match across all variants so downstream
scoring/metrics read every algorithm identically. All six keys were
added with identical names to all 8 cores; verified post-hoc that the
`results["..."]` key set is byte-identical across the 8, and that
`mcts_search` returns a uniform 14-tuple everywhere.

## 2. Decisions made (asked or reasoned, not guessed)

Two choices went to `AskUserQuestion`; the rest were reasoned from the
code.

1. **Include `phase_selected_score`?** — its cross-family meaning
   differs (PUCT vs. density vs. diversity value; not comparable across
   methods, a within-method diagnostic only), and capturing it is the
   most invasive part. **Answer: include it.**
2. **How to capture the winning score without breaking 4 different
   selector signatures?** — **Answer: stash on an instance attribute.**
   Each selector sets `self.last_selected_score` just before returning
   (a declared field, exactly like the existing `cnt_node_max_depth`);
   the loop reads it after the call. Selector signatures stay
   **unchanged** — the lowest-risk option, and uniform across all
   families. (The rejected alternative — returning `(node, score)` —
   touches 4 distinct signatures, every call site, and each singleton
   fast-path, with more chance of a subtle break.) For the sem family,
   `_diverse_select` (a module-level function with a single caller) was
   extended to return `(best_idx, best_score)`; the cnt/kube/kdepth
   selectors already track `best_value`/`best density` locally, so the
   stash is a one-line addition there.
3. **Verification.** **Answer: smoke-test all 4 families** (offline,
   `results/smoketest/`), plus static parity checks. All four confirmed
   the six keys populate with sensible values, and — the important
   cross-check — `phase_selected_score` carries the *right per-family
   scale*: PUCT ~1 (cnt), density ~0.05 (kube), depth-density ~0.15
   (kdepth), diversity value ~600 (sem, `ds_alpha=100`). This proves
   `last_selected_score` captures each family's own frontier score, not
   a generic placeholder.

## 3. Dropped: `phase_frontier_size`

An earlier proposal included recording the live frontier size
(`len(leaf_nodes)`) per phase, framed as a fan-out-vs-tunnel signal.
**Dropped after analysis.** Because a node is expanded exactly once and
each expansion nets roughly `+(batch_size − dedup − terminal children)`
frontier nodes, the frontier grows near-linearly for **all** methods —
there is no shrinking/tunnel regime to detect. The ~2× slope difference
that does exist (sem ~2.9 vs. kdepth ~1.5 net non-terminal children per
expansion) is a **downstream consequence** of where each method expands:
kdepth expands deep nodes whose children hit the max-depth wall and
become terminals (not frontier), so it adds fewer. That mechanism is
already captured by `phase_selected_depth` (#1) + the terminal counts
(`q_nodes_terminal`/`q_nodes_completed`) — the terminals are the actual
throttle; frontier size is just the visible symptom. So it was
redundant, not independent, and left out.

## 4. What `phase_selected_depth` supports directly

The "% shallow vs. deep nodes selected for exploration" split is
computable from this one list:

- **Fixed-threshold** (`% depth ≤ K`): self-contained, just count.
- **Relative "deep-half"** (`% depth > max_depth/2`): needs a max-depth
  scalar. `max(phase_selected_depth)` under-reports the true tree max
  by **exactly 1** (verified across all 20 finding runs) — the deepest
  nodes are always terminal leaves, which are never *selected* for
  expansion, so the deepest *selected* node sits one level above the
  deepest node. Harmless for the deep-half split (a ±0.5 shift in the
  midpoint), or use `q_nodes_max_depth`-adjacent state for an exact
  value.

Note this is "where the method *chose to explore*" (denominator = the
~80 expanded nodes), distinct from "depth distribution of *all* nodes"
(which a static tree histogram would give and which is dominated by
terminal leaves the method never acted on). For the exploration
question, the selected-depth denominator is the correct one.

## 5. Deferred: tree-*structure* characterization keys

The six keys characterize **exploration allocation** (where effort
went). A fuller characterization of the generated **tree** itself was
discussed and deferred to a follow-up, to keep this change self-
contained and its history clean:

- **`q_tree_depth_hist`** — node-count-per-depth histogram (the static
  tree profile, vs. the dynamic `phase_selected_depth`).
- **`q_branching_by_depth`** — mean post-dedup children per expanded
  node at each depth (where the tree fans out vs. narrows).
- **`q_by_depth`** — mean/max q at each depth (does going deep find
  better nodes? — directly tests whether kdepth's depth-seeking pays
  off).
- visit concentration (`q_visit_gini` or raw `q_visit_counts`) — the
  one finding metric not yet recorded; the Gini scalar is cheap, raw
  counts are `O(nodes)`.

Recommendation recorded for the follow-up: add `q_tree_depth_hist` +
`q_by_depth` first — together with the six keys here they reproduce the
depth-allocation analysis AND answer "does depth pay off," from normal
launcher output.

## Connections

- [../findings/exp-findings/bl-frontier-depth-allocation.md](../findings/exp-findings/bl-frontier-depth-allocation.md)
  — the finding these keys make reproducible at dataset scale; its
  diagnostics (depth histogram, depth/visit Gini, dead-end counts) are
  exactly what keys #1/#4/#5/#6 feed.
- [../../unittests/examine_driver.py](../../unittests/examine_driver.py)
  — the tree-dump driver these keys partly obviate (it still gives the
  full per-node structure the deferred §5 keys would summarize).
