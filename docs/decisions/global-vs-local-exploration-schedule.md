# Global vs. local exploration-bonus schedule under frontier selection

*2026-07-08*

Records the reasoning behind `mcts_bl_sem_v01`'s
`ds_alpha_schedule="global"` default (see `docs/decisions-log.md`
2026-07-08 entry for the shorter summary), then works through why the
same question does *not* apply to `mcts_bl_cnt_v01` — a point that
came up as a natural follow-on ("would it be fairer for bl_cnt to use
the same global schedule?") and is worth having answered in one place
so it doesn't get re-litigated.

## The question for mcts_bl_sem_v01

`mcts_sem_search_v02_00_00.py::select_child` scales the diversity term
by a **local** clock:

```python
log_nvisit_parent = np.sqrt(np.log(1 + node.visit_count()))
best_idx = _diverse_select(
    self.V_inv, embeds, q_values,
    self.config.search.ds_alpha * log_nvisit_parent,
    self.config.search.ds_beta,
)
```

— `node` is the *parent* being descended through; the schedule is
local to wherever the phase-based walk currently is. `mcts_bl_sem_v01`
selects globally over an explicit `leaf_nodes` frontier instead (no
current node to anchor a "parent" on), so the schedule had to be
redesigned, not just copy-pasted. Three candidates were considered
(`BLMCTSSemConfig.ds_alpha_schedule`, all implemented and selectable):

- **`global`** (chosen default) — `ds_alpha * sqrt(log(1+t))`, `t` =
  number of frontier selections so far (a single counter shared by
  the whole tree, incremented once per iteration of the main loop in
  `mcts_bl_sem_search_v01_00_00.py::mcts_search`).
- **`parent`** — `ds_alpha * sqrt(log(1+node.parent.visit_count()))`
  per frontier node: the literal transplant of sem_v02's formula onto
  each node individually.
- **`none`** — constant `ds_alpha` (no schedule).

### Why `global` is the theoretically correct choice here

The frontier is, at every iteration, one flat set of arms compared
directly against each other — exactly the structure a linear-bandit
selection rule (`sqrt(x^T V^-1 x)` is the standard LinUCB/OFUL
confidence width for a shared covariance `V`) assumes. In that family
the width's growing multiplier is indexed by **global elapsed time**
`t` (`sqrt(log t)` up to constants), not by a per-arm or per-node
counter — because the confidence guarantee being encoded is about `V`
accumulating evidence across the *whole* process, and `V` in this
algorithm already is global (folded from every selection anywhere in
the tree, not scoped per parent). Using `t` = frontier-selection count
matches that: the multiplier is shared by all nodes at a given
iteration, so it can never distort the *ranking* between two frontier
nodes (only `q` and the `V^-1` geometry do that) — it only tilts the
global explore/exploit balance over the run, which is its intended
job.

`parent` would break that cleanly-factored property: a node in the
`0.2.*` subtree gets a different multiplier from a sibling one level
up in `0.4.*`, purely from tree position, at the *same* iteration —
i.e. the schedule would leak into node comparisons it has no business
affecting, on top of doing its intended job. That's a real cost of
`parent`, not just an aesthetic one.

### Why the choice is expected to be near-invisible empirically

Two independent reasons the schedule choice likely won't move numbers
much at typical operating points, both established before this
variant existed:

1. **The multiplier's dynamic range is narrow.**
   `sqrt(log(1+n))` ranges roughly 0.83 (n=1) to 1.34 (n=5) to 1.74
   (n=20) to 2.15 (n=100) — a 0.8-2x band, not orders of magnitude.
2. **`ds_alpha` itself is a switch, not a dial** past a threshold —
   [ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
   found `ds_alpha≈10` already saturates a plateau that `100`/`1000`
   sit flat on. A 2x wiggle in the effective multiplier at the
   project's usual `ds_alpha=100` operating point sits well inside
   that flat region.

So `global` vs. `parent` vs. `none` should be near-indistinguishable
at `ds_alpha≥10` — the choice matters more for (a) interpretability
of ds_alpha sweeps (keeping units comparable to sem_v02's) and (b) any
future sweep that probes small `ds_alpha` near the on/off transition,
where the 0.8-2x shift moves you along the sensitive part of the
curve. This is *why* all three were implemented as a config knob
rather than deciding one and hardcoding it — the empirical question
is open, only the theoretical default is settled.

## Does the same argument extend to mcts_bl_cnt_v01?

The follow-on question was posed two ways in this session: first as
"would it be fairer for `mcts_bl_cnt` to use the global schedule too
(vs. `mcts_bl_sem`)," then clarified to the sharper version — "should
`mcts_bl_cnt`'s PUCT bonus itself switch to a global visit-count clock
instead of `parent_visits`, for fairness against `mcts_cnt`." The
second framing is actually the cleaner one to check, because
`mcts_cnt_search_v01_00_00.py::MCTSNode.puct` and
`mcts_bl_cnt_search_v01_00_00.py::MCTSNode.puct` are **byte-identical**
— both `q_value() + cpuct * sqrt(log(parent_visits) / visits)`, same
local per-parent clock. So there is no existing bl_cnt-vs-cnt
asymmetry to fix: whatever argument would justify globalizing PUCT's
clock in bl_cnt applies equally to cnt, since they run the exact same
formula. This is a "should PUCT ever use a global clock" question,
independent of frontier-vs-phase-based selection — not a fairness gap
between the two count-based variants.

Answered no, for the reasons below. `mcts_bl_cnt_v01`'s selection rule
is PUCT (`mcts_bl_cnt_search_v01_00_00.py::MCTSNode.puct`):

```python
def puct(self, cpuct=2) -> float:
    ...
    parent_visits = self.parent.visit_count()
    visits = self.visit_count()
    u = cpuct * np.sqrt(np.log(parent_visits) / visits)
    return q + u
```

This has no `ds_alpha`, no `V`/`V_inv`, and no analog of a "schedule"
knob at all — the exploration term is a **per-node** function of that
node's own `visit_count()` and its immediate parent's, full stop.
Two structural reasons this doesn't generalize to a global-`t` version
the way sem's diversity term did:

1. **PUCT's bonus is already the count-based special case of the
   linear-bandit width, not an approximation to it.** In the
   diversity term's own derivation
   ([tuning-semantic-score-weights-and-lambda.md](tuning-semantic-score-weights-and-lambda.md)),
   `sqrt(x^T V^-1 x)` reduces to `1/sqrt(n_x)` (up to a constant) in
   the limit of mutually orthogonal embeddings — i.e. PUCT's
   `1/sqrt(visits)` bonus IS what the diversity term degenerates to
   when there is no shared geometry to exploit. There is no further
   "more global" form to move to: PUCT doesn't have a `V` to make
   global, because it never had the cross-node geometry sem's `V`
   encodes in the first place.
2. **Local (per-parent) is what makes PUCT count-consistent.**
   PUCT's whole guarantee is that `visits`/`parent_visits` are exact
   counts of how many times *this specific node* and *its exact
   parent* were chosen — that's what makes `sqrt(log N/n)` a genuine
   confidence bound on that node's own value estimate. Swapping
   `parent_visits` for a frontier-wide selection counter would
   compare `log(t)` (time elapsed in the whole search) against `n`
   (visits to one specific node) — dimensionally mismatched
   quantities that no longer form a valid confidence-bound ratio.
   `bl_sem`'s `global` schedule doesn't have this problem because the
   thing being globalized (the covariance `V`) was already global by
   construction; PUCT's per-node visit count is not.

So the "fairness" framing doesn't quite apply, in either direction:

- **bl_cnt vs. bl_sem**: `bl_sem`'s `global` default isn't a
  stylistic preference imposed for symmetry with PUCT — it's the
  correct instantiation of a *different* selection rule (linear-bandit
  width) that happens to already need a global quantity. There is no
  parallel argument for PUCT to import.
- **bl_cnt vs. cnt**: since the two already run byte-identical PUCT
  code, there is no asymmetry between them to correct — "fairness"
  would require an actual difference in the first place.

Retrofitting a "global-t" PUCT variant (for either comparison) would
be inventing a non-standard bonus with no bandit-theoretic
justification, not applying an existing fix. If there's a real "should
PUCT's exploration be more global" question worth asking, it would
have to be posed on its own terms — e.g. "should `parent_visits` be
replaced by a frontier-wide (or tree-wide) visit total as a
heuristic" — which is an open empirical question, not a
theory-mandated one, and applies equally to `mcts_cnt_v01` and
`mcts_bl_cnt_v01` since they'd be changing the same formula. It hasn't
been explored.

## Connections

- `docs/decisions-log.md`, 2026-07-08 entry — short-form record of
  the `mcts_bl_sem_v01` composition decisions (this schedule plus
  two others).
- [tuning-semantic-score-weights-and-lambda.md](tuning-semantic-score-weights-and-lambda.md)
  — derives the `1/sqrt(n)` degeneracy of the diversity term used in
  point 1 of the bl_cnt section above.
- [ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
  — the empirical plateau that bounds how much the schedule choice
  can matter in practice.
- `core/mcts_bl_sem_search_v01_00_00.py::select_leaf_from_list` — the
  `global`/`parent`/`none` implementation.
- `core/mcts_bl_cnt_search_v01_00_00.py::MCTSNode.puct` — the PUCT
  bonus discussed in the second half.
