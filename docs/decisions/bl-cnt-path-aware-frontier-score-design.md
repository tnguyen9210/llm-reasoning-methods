# bl_cnt: path-aware frontier scoring — eager terminal backprop and
# two candidate leaf-score designs

*2026-07-16*

Records a design discussion, not yet implemented. Tuan asked whether
`mcts_bl_cnt_search_v01_00_00` should backprop terminal candidates
(especially max-depth dead-ends) *eagerly*, at creation, instead of
waiting for them to be selected off the frontier — the goal being to
propagate a bad direction's negative signal into the tree faster, so
the search stops circling low-scoring dead-ends. Evaluating the
proposal surfaced that eager backprop alone doesn't reach that goal in
v01, and two concrete alternatives were sketched that would (§§1-6
below). The same question was then extended across all three active
`bl_cnt` variants — v01 (PUCT), v02 (fractional KUBE), v03
(depth-shaping knapsack) — with a different verdict for each (§7). This
file is the pointer target for the shorter
[decisions-log.md](../decisions-log.md) entry; see there for the
one-paragraph version.

## 1. The proposal as stated

Split candidate generation output into two lists instead of one:
terminal candidates (EOS / max-depth) go straight to
`agent.backprop(...)` and `completed_nodes`, non-terminal candidates go
into `leaf_nodes` as today. Currently *all* children — terminal or not
— enter `leaf_nodes` ([mcts_bl_cnt_search_v01_00_00.py:220-224,
423-425](../../core/mcts_bl_cnt_search_v01_00_00.py#L220)); a terminal
one only backprops once it's later selected off the frontier
(`current_node.is_terminal` branch at the top of the next loop
iteration,
[mcts_bl_cnt_search_v01_00_00.py:413-416](../../core/mcts_bl_cnt_search_v01_00_00.py#L413)).

## 2. Why eager backprop alone doesn't propagate the intended signal (v01)

`select_child_from_list` ([mcts_bl_cnt_search_v01_00_00.py:257-282](../../core/mcts_bl_cnt_search_v01_00_00.py#L257))
scores each frontier leaf `x` as:

```
puct(x) = q(x) + cpuct · sqrt( ln N(parent(x)) / N(x) )
```

Three structural facts make backpropagated *values* invisible to this
scoring, regardless of how early backprop runs:

1. **A frontier leaf's own `q` and `N` never change before it is
   selected.** Backprop walks terminal → ancestors → root
   (`update_recursive`); a frontier leaf is childless, so it is never
   on anyone else's backprop path. Every leaf sits at `N=1`, `q =` its
   own PRM score, from creation until it is selected (at which point
   it leaves the frontier for good).
2. **Internal-node `q` is never read by `select_child_from_list`.**
   The formula above reads only a leaf's own `q`/`N` and its parent's
   *visit count* — never the parent's `q`. So when backprop folds a
   dead-end's value into every ancestor's `value_sum`
   (`MCTSNode.update`, [mcts_bl_cnt_search_v01_00_00.py:112-114](../../core/mcts_bl_cnt_search_v01_00_00.py#L112)),
   that number is write-only: nothing in the bl loop ever consults an
   internal node's q again. (Contrast `mcts_cnt_search_v01_00_00`,
   where descent re-selects among internal children by their q every
   pass — there, eager backprop and value-reading are both native, and
   the proposal would work as intended unmodified.)
3. **The only live effect of any backprop is `N(ancestor) += 1`** —
   which *increases* the `u` term for every not-yet-selected sibling
   along the backprop path, since parent-visit count sits in the
   numerator. Counts attract in this UCB formula; the discouraging
   force in a value-based sense would need to come through `q`, and no
   `q` change ever reaches a live frontier leaf.

### Magnitude: the count effect dominates and is value-blind

With `cpuct = 2.0` (both `mcts_cnt.yaml` and `mcts_bl_cnt_v01.yaml`)
and PRM scores in `(0, 1)`:

- A leaf whose parent has never been backpropped through has
  `N(parent) = 1`, so `u = 2·sqrt(ln 1) = 0` — pure-`q` selection.
- One backprop through that parent: `N = 2`,
  `u = 2·sqrt(ln 2) ≈ 1.67` — already larger than any possible `q`.
  Two backprops: `u ≈ 2.10`.

So a single count-increment through a parent catapults all of its
remaining frontier children past every fresh leaf in the tree,
regardless of score. Under the *current* lazy scheme this burst
dynamic already exists but is gated: a terminal must first win a
selection, and a `q=0` dead-end under quiet ancestors essentially never
wins one (its puct is 0 while any fresh leaf offers its own PRM score).
Eager backprop removes that gate — every max-depth dead-end fires the
burst immediately on creation, and the burst pulls toward the
dead-end's siblings with the same strength regardless of whether the
terminal that triggered it scored 0.0 or 0.95. That is close to the
opposite of "discourage low-scoring directions."

### What in the proposal is right regardless

- The lazy scheme has a real defect: a terminal that never wins
  selection contributes nothing, ever — dead information. And each
  terminal that *does* win selection burns a phase iteration doing no
  generation, which is pure overhead relative to evaluating it at
  creation.
- Eager backprop is the standard-MCTS ordering (evaluate, then back up
  immediately) and is a clean simplification either way.
- Splitting terminals out of `leaf_nodes` is correct housekeeping —
  they are not expandable, so today they only occupy the frontier as
  delayed-backprop triggers.

Both of the designs below keep the terminal-split + eager-backprop
half of the proposal; they differ in how the frontier's leaf score is
changed so that the now-meaningful ancestor state is actually read.

## 3. Option 1 — blend leaf score with parent's running mean (one hop)

Read one extra number that already exists and is already correctly
maintained: the immediate parent's running-mean `q`.

```python
def path_aware_score(leaf, alpha, cpuct):
    parent = leaf.parent
    q_parent = parent.q_value() if parent.visit_count() > 0 else leaf.q_value()
    blended_q = alpha * leaf.q_value() + (1 - alpha) * q_parent

    parent_visits = parent.visit_count()
    u = cpuct * np.sqrt(np.log(parent_visits) / 1) if parent_visits > 0 else 0.0
    return blended_q + u
```

Worked example: node `P` has two fresh children, `A` (PRM score 0.8)
and `B`, a max-depth dead-end (score 0, backpropped eagerly). After
`B`'s backprop, `P.q_value()` drops. When `A` is scored,
`blended_q = α·0.8 + (1−α)·q(P)` — `A`'s score is pulled down by its
sibling's failure, in proportion to `(1−α)`, without touching `u` at
all.

**Tuning knob** `α ∈ [0, 1]`: how much a leaf trusts its own PRM score
vs. its local neighborhood.
- `α = 1` → exactly today's behavior (parent info ignored) — the
  natural control arm for a sweep.
- `α = 0` → leaf's own score discarded entirely, pure path-history —
  likely too aggressive, since per-candidate PRM scoring is the whole
  point of scoring candidates individually rather than just the path.
- `α ≈ 0.6–0.8` is the reasonable starting range.

**Reach:** one hop only. A dead-end's effect on `q(P)` doesn't
propagate to `P`'s parent `G` or to leaves elsewhere in the tree; to
reach further this would need to recurse the blend (fold in
`q(grandparent)` too, geometrically discounted) — deferred until
one-hop is shown to be insufficient.

**Cost:** O(1) extra per leaf per selection — `q(parent)` and
`N(parent)` already exist and are already correctly maintained by
`update_recursive`; no new state.

**Known softening case:** if `P` has many children and only one is a
dead-end, `q(P)` averages over all of them, so a single 0 barely moves
it when `P` already has several successful branches. This is
appropriately conservative (one bad sibling shouldn't tank a
proven-good neighborhood) but means the intended effect is weakest
exactly where the neighborhood is otherwise strong — worth knowing,
not necessarily worth over-correcting for.

## 4. Option 2 — AlphaZero-style subtree value, full path, decayed

Read the *entire* path to root, not just one hop, and change the
exploration term's shape to match:

```python
def path_aware_score(leaf, cpuct, gamma=0.8):
    node, acc, norm, depth = leaf, 0.0, 0.0, 0
    while node.parent is not None:
        acc += (gamma ** depth) * node.q_value()
        norm += gamma ** depth
        depth += 1
        node = node.parent
    q_path = acc / norm if norm > 0 else leaf.q_value()

    parent_visits = leaf.parent.visit_count()
    u = cpuct * np.sqrt(parent_visits) / (1 + leaf.visit_count())
    return q_path + u
```

Two differences from Option 1:

1. **Full-path reach.** A dead-end anywhere along the lineage
    contributes to `q_path`, discounted by `gamma ** distance` so a
    failure many levels up counts for less than an immediate parent's
    failure. `gamma ≈ 0.8` is a reasonable starting point; `gamma → 1`
    recovers a plain path average, `gamma → 0` recovers Option 1's
    one-hop blend (with `norm` renormalizing instead of a fixed `α`).
2. **Exploration-term shape changes, not just its inputs.** Current
    bl_cnt uses `sqrt(ln N_parent / N_leaf)` (UCB1; the log damps
    growth in `N_parent`). The formula above uses AlphaZero's
    `sqrt(N_parent) / (1 + N_leaf)`, which grows in `N_parent` roughly
    like `sqrt`, not `log(sqrt)` — a *much* stronger pull toward
    less-visited parents. Given the magnitude problem above (a single
    backprop already blows past every `q`), the literal AlphaZero form
    pulls in the wrong direction for this goal unless `cpuct` is
    lowered substantially alongside it — real signal is now flowing
    through `q_path`, so the bonus term doesn't need to dominate the
    way it currently does. **This needs explicit retuning, not a
    drop-in swap of the exploration formula.**

**Cost:** `O(depth)` per leaf per selection instead of `O(1)` — with
`select_child_from_list` scoring every frontier leaf every iteration,
total cost per phase is `O(frontier_size · depth)`. With
`max_depth = 20` and a frontier that can grow into the hundreds, this
is a real (if not prohibitive) cost difference from Option 1.

## 5. Comparison (Option 1 vs. Option 2, v01)

| | reach | new state read | exploration term | cost/selection | main risk |
|---|---|---|---|---|---|
| Option 1 | one hop (immediate parent) | `parent.q_value()` (already exists) | unchanged (`ln`-UCB1) | O(1)/leaf | weak effect in mixed-quality neighborhoods |
| Option 2 | full path, `gamma`-discounted | whole ancestor chain's `q_value()` | changed to AlphaZero `sqrt(N)/(1+N)` form | O(depth)/leaf | two things change at once (depth *and* term shape); needs `cpuct` retuned or it overshoots harder than today |

Option 1 isolates one variable (does neighborhood-blending help at
all) with the smaller footprint; Option 2 is the more "textbook"
design (proper discounted subtree value) but confounds two changes and
actively fights the stated goal unless `cpuct` is retuned alongside it.

## 6. Recommendation (v01)

Try Option 1 first, in a **new version file** (`v04` — `v02` is KUBE
and `v03` is depth-shaping knapsack, see §7), sweeping
`α ∈ {1.0, 0.8, 0.6}` with `α = 1.0` as the current-behavior control
arm. Reach for Option 2's full path-walk only if one-hop blending is
shown insufficient. Either option is a new version file, not an edit
to `v01_00_00`, per the two-tier convention in
[algorithms.md](../algorithms.md) — this changes search behavior, so
existing scored `bl_cnt` cells must stay attributable to the old
formula.

## 7. Extending the same question to v02/bl_kube (KUBE) and v03 (depth-shaping)

*Note: the v02 module referenced throughout this section was renamed
2026-07-16, shortly after this analysis was written, from
`mcts_bl_cnt_search_v02_00_00.py` / `mcts_bl_cnt_v02` to
`mcts_bl_kube_search_v01_00_00.py` / `mcts_bl_kube_v01` — see
[bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md). File-path
citations below have been updated to the current name and line
numbers; the analysis and "v02" shorthand in the prose are otherwise
unchanged.*

*Note: the v03 module referenced throughout this section was renamed
2026-07-17, the day after this analysis was written, from
`mcts_bl_cnt_search_v03_00_00.py` / `mcts_bl_cnt_v03` to
`mcts_bl_kdepth_search_v01_00_00.py` / `mcts_bl_kdepth_v01` — see
[bl-cnt-to-bl-kdepth-rename.md](bl-cnt-to-bl-kdepth-rename.md).
File-path citations below have been updated to the current name and
line numbers; the analysis and "v03" shorthand in the prose are
otherwise unchanged.*

Tuan asked how the *original* proposal (terminal-split + eager
backprop, no path-aware scoring change) would land on
`mcts_bl_kube_search_v01_00_00` (fractional KUBE) and
`mcts_bl_kdepth_search_v01_00_00` (depth-shaping knapsack), not just
v01. Both share v01's frontier bookkeeping (all children, terminal or not,
enter `leaf_nodes`; a terminal only backprops if later selected) but
each replaces PUCT with a different selection criterion — so each
reads backpropagated state differently, and the verdict is different
in each.

### 7.1 v02 — dead-ends are permanently stuck, so this is a real fix

`kube_density(x) = (q(x) + bonus(x)) / cost(x)`, `cost = max_depth −
depth`, and **`cost ≤ 0 → density = −inf`**
([mcts_bl_kube_search_v01_00_00.py:201-203](../../core/mcts_bl_kube_search_v01_00_00.py#L201)).

A max-depth dead-end always has `cost ≤ 0`, so it scores `−inf` and is
never selected while *any* finite-density node remains — meaning
under the current lazy scheme, **dead-ends don't just backprop late,
they never backprop at all.** They sit in the frontier permanently,
scanned every selection round for the rest of the search.

This interacts badly with the `kube_affordable` feasibility filter
(`select_child_from_list`,
[mcts_bl_kube_search_v01_00_00.py:339-346](../../core/mcts_bl_kube_search_v01_00_00.py#L339)),
walked through in full below. Short version: a stuck dead-end silently
disables a fallback the code was explicitly built to have.

**What `kube_affordable` does.** It's a feasibility pre-filter that
runs before the density argmax:

```python
if self.config.search.kube_affordable:
    affordable = [
        node for node in nodes
        if node.is_terminal
        or max_depth - node.depth <= residual
    ]
    if affordable:
        nodes = affordable
```

`residual` is generations left in the budget; `max_depth − depth` is a
node's *cost* — the worst-case number of expansions to walk it out to
`max_depth`. The filter's intent: before ranking candidates, first
shrink the pool to only those that can still be *afforded* to finish
within the remaining budget (terminals are free to select — no
expansion, just a backprop — so they're always in-pool). The
`if affordable:` line is the escape hatch: if the affordable set comes
back empty (nothing fits), don't filter at all — fall back to ranking
every frontier node unfiltered. The stated reason (docstring,
[mcts_bl_kube_search_v01_00_00.py:324-333](../../core/mcts_bl_kube_search_v01_00_00.py#L324)):
`cost` is a worst-case bound, so a technically "unaffordable" node
might still finish cheaply via early EOS — worth considering rather
than stopping cold.

**Why a stuck dead-end breaks the escape hatch.** `node.is_terminal`
alone satisfies the affordability check — a dead-end always counts as
"affordable," unconditionally, regardless of budget. Combine that with
§7.1's finding that dead-ends never leave the frontier under the
current scheme: **the instant one dead-end exists in the frontier, the
`affordable` list can never be empty again**, no matter how expensive
or budget-starved every *actual* (expandable) candidate becomes. The
`if affordable:` fallback is written to catch "nothing fits the
budget" — but it can only ever see "nothing fits" if `affordable`
comes back `[]`, and one inert dead-end guarantees it never does. So a
fallback the code explicitly has a reason to want (relax filtering
near the end of budget, since cost is only a worst-case estimate) sits
unreachable for the rest of any run that has accumulated even a single
dead-end — which, given how common max-depth truncation is, is most
runs past their first few phases.

Splitting terminals out of the frontier at creation (the original
proposal, no scoring change needed) fixes this directly: dead-ends
backprop and leave immediately, so they can no longer prop up
`affordable`'s non-emptiness, and the fallback becomes reachable again
exactly as designed.

**Does the newly-happening backprop help or hurt, once it fires?**
Depends on `kube_schedule`:
- **`"parent"`** (default) — same channel as v01's PUCT bonus, scaled
  by `1/cost`. The v01 analysis (§2) carries over unchanged: values
  stay write-only (`kube_density` never reads a *parent's* q, only the
  leaf's own frozen q plus the parent's visit count), and each
  dead-end's backprop now fires a fresh count-burst that *raises* the
  bonus of that dead-end's still-live siblings — attraction toward the
  failed region, not discouragement. Since these bursts currently
  never fire for dead-ends in v02 (they're stuck at `−inf`), this is
  new behavior the fix introduces, not merely an earlier version of
  something already happening.
- **`"global"`** — the bonus clock is the frontier counter `t`, shared
  by every node; with all frontier leaves at `visits == 1`, the bonus
  is a frontier-wide constant. Backprop has **zero** effect on
  ranking under this schedule — the fix is pure hygiene (frontier
  stops accumulating dead clutter, `affordable` fallback restored),
  with no new attraction-toward-failure side effect. One second-order
  wrinkle: `t` advances once per loop iteration; today's
  terminal-selection iterations advance `t` without spending budget,
  so removing them slightly slows `t` relative to `gen_cnt`, which
  very slightly weakens the depth-tilt at any fixed budget point
  (`√log(1+t)` grows the bonus weight over the run). Small, but traces
  will differ.

### 7.2 v03 — no channel exists at all; the goal is unreachable by construction

`depth_density(x) = (q(x) + depth_beta·(1 − depth_frac(x)^depth_alpha))
/ cost(x)` reads the leaf's own frozen q, its depth, and two constants
— **no visit counts anywhere**, by explicit design
([mcts_bl_kdepth_search_v01_00_00.py:94-98](../../core/mcts_bl_kdepth_search_v01_00_00.py#L94):
"No visit_count/parent_visit/global-clock term anywhere"). Backprop in
v03 is therefore write-only on *both* channels that mattered
elsewhere — not just the value channel (as in v01/v02), but the count
channel too, since nothing in `depth_density` ever reads a visit
count.

Consequence: in v03, no backprop timing — eager, lazy, or never — can
change which non-terminal node gets expanded next. The original
proposal reduces to a near-pure refactor here: identical non-terminal
ranking, identical completions, minus wasted phases spent re-selecting
already-known EOS terminals, minus the same permanent dead-end
clutter and affordability-fallback suppression described in §7.1 (v03
reuses the identical `kube_affordable` filter). **The stated goal —
propagate negative feedback faster — is unreachable in v03 through any
backprop-timing change**; reaching it here would require adding a
value-reading term to `depth_density` itself, which is a different
and larger change than anything discussed in this file.

### 7.3 A docstring/code discrepancy surfaced along the way

Both v02's and v03's module docstrings' Algorithm blocks say **"Add
non-terminal children to leaf_nodes"**
([mcts_bl_kube_search_v01_00_00.py:46](../../core/mcts_bl_kube_search_v01_00_00.py#L46),
[mcts_bl_kdepth_search_v01_00_00.py:64](../../core/mcts_bl_kdepth_search_v01_00_00.py#L64))
— i.e., they already describe the terminal-split half of this
proposal — while both code bodies append *all* children, terminal or
not. Both docstrings are internally inconsistent on top of that: they
also list "If selected_node.is_terminal: Backprop" as a loop step,
which would be dead code if terminals genuinely never entered the
frontier. The code's actual behavior is unambiguous (the
terminal-selection branch and "terminals always eligible" affordability
rule both depend on terminals being in the frontier), so this reads as
stale drift from an earlier draft rather than a live bug — but it
means this exact design question already left an unresolved mark in
the docs before this discussion. Whichever way the terminal-split
question is ultimately decided for v02/v03, both docstrings' Algorithm
blocks need a fix to match — either correct the prose (append *all*
children) or implement the split described (making the docstring
correct and closing this gap for real).

### 7.4 Summary across all three variants

| | selection reads | backprop → selection channel | dead-ends today | effect of terminal-split + eager backprop |
|---|---|---|---|---|
| v01 (PUCT) | leaf q, parent N | counts only — attract | lazily backpropped when selected | bursts fire earlier, incl. from dead-ends: counterproductive for the stated goal (§2) |
| v02 `parent` | leaf q, parent N, cost | counts only, cost-scaled — attract | **never backpropped** (stuck at `−inf`); permanently props up `affordable` | real fix (frontier hygiene + fallback restored) but *introduces* new attraction bursts from dead-ends, same objection as v01 |
| v02 `global` | leaf q, t, cost | none | same stuck state | pure hygiene fix, no new side effect; goal itself still unreachable through backprop timing alone |
| v03 | leaf q, depth, cost | **none — no counts read either** | same stuck state, same `affordable` issue | provably no ranking change; pure cleanup; goal unreachable by construction |

Across all three, this is behavior-changing at the trace level at
minimum (RNG-draw ordering shifts once terminal selections are
removed; the `phase_depths`/`q_last_phase` metrics change meaning,
since there are no more backprop-only phases to record) → **new
version files** in every case, per the two-tier convention in
`algorithms.md`; existing scored cells for v01/v02/v03 stay put and
comparable to what's on disk today.

## Open / unresolved

- Whether `negative_reward = 0` (current default,
  `mcts_bl_cnt_v01.yaml`) is discouraging *enough* once a dead-end's
  value is actually read by selection — 0 against PRM scores in
  `(0, 1)` is mild. Worth revisiting once either option is implemented
  and if dead-ends still don't seem sufficiently avoided.
- Neither Option 1 nor Option 2 (v01) has been implemented or
  benchmarked; this file records the design space only.
- The v02/v03 docstring-vs-code discrepancy (§7.3) is unresolved —
  neither file's Algorithm block has been corrected yet.
- Whether v02/v03 should actually get the terminal-split fix (as
  hygiene, independent of the v01 path-aware-scoring question) hasn't
  been decided — §7 only establishes what the change would and
  wouldn't do in each.

## Revisit if

- Option 1 (v01, §3) ships and a sweep shows it's too weak — go to
  Option 2 (§4), and retune `cpuct` alongside the exploration-term-shape
  change rather than swapping the formula in isolation.
- The terminal-split fix is implemented for v02/v03 — remember it
  *introduces* new dead-end-driven attraction bursts under v02's
  `parent` schedule (§7.1), which is a new failure mode, not a
  pre-existing one being merely surfaced earlier.
- The same question ("does backprop reach frontier selection")
  resurfaces for `mcts_bl_sem_v01` — its selection criterion is a
  diversity term over embeddings, not PUCT/KUBE/depth-shaping, and
  this analysis has not been verified to transfer there.
