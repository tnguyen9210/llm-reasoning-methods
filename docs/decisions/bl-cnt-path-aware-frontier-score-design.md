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

## 4.5. Option 3 — PUCT-proper prior term on the exploration bonus
*(added 2026-07-18, discussed after v02/Option 1 shipped)*

Options 1 and 2 both fix the same defect: a backpropagated *value*
never reaches selection. Neither touches a separate, second defect
named in §"Magnitude" above — the exploration term `u` is not just
inert to backprop, it is **structurally blind to score, period**,
backprop or no backprop. `u = cpuct · sqrt(ln N_parent / N_leaf)`
is built entirely from counts; a fresh dead-end that scored 0.0 and
one that scored 0.95 produce byte-identical `u`, so a count-burst
through a shared parent pulls toward a dead-end's siblings "with the
same strength regardless of whether the terminal that triggered it
scored 0.0 or 0.95" — the passage's own words, quoted in full above.
Option 1's `alpha` blend softens this indirectly (a bad sibling drags
down `q(parent)`, which leaks into `blended_q`) but the *exploration*
term itself still can't see the score that triggered its own burst.

Standard PUCT (AlphaZero, and the source of the "P" in "PUCT") already
has a slot for exactly this — a prior probability weighting the
exploration bonus itself, not just the value term:

```
PUCT(x) = Q(x) + c · P(x) · sqrt(N_parent) / (1 + N_x)
```

`P(x)` is normally a policy network's output. This repo has no policy
network, but it already computes something that can stand in for one:
the PRM's `candidate_score` at expansion time (`create_child`'s
`new_node.update(candidate_score)` call —
[mcts_bl_cnt_search_v01_00_00.py:221](../../core/mcts_bl_cnt_search_v01_00_00.py#L221),
`core/mcts_bl_cnt_search_v02_00_00.py:310` in the v02 sibling). That
value already seeds a fresh node's first `q_value()`; Option 3 would
additionally store it
(or a batch-normalized version of it) as a separate, never-updated
`prior_p` field, then read `prior_p` inside the exploration term:

```python
u = cpuct * node.prior_p * sqrt(N_parent) / (1 + N_leaf)
```

`prior_p` needs normalizing across the sibling batch created by the
same expansion call — a raw PRM score in `(0, 1)` is one node's
absolute quality, not a probability distribution over its siblings.
Two natural normalizations:

```
P(x) = score(x) / sum(score(sibling) for sibling in batch)          # ratio
P(x) = softmax(score(x) / tau over batch)                            # temperature
```

Softmax with high `tau` degenerates to `P(x) = 1/len(batch)` for every
sibling — recovering something close to today's score-blind `u` as a
limiting case, which is a useful sanity check that this is a strict
generalization, not a different formula family.

**Why this is a fix for the "Magnitude" defect specifically, not a
restatement of Option 1**: a low-scoring dead-end now gets a small
`prior_p`, so *its own* burst through `u` shrinks in proportion to how
bad it scored — the exploration term finally tracks the value that
triggered it. Options 1/2 route backpropagated value through `Q(x)`;
Option 3 routes the *originating* score through `u` directly, and
requires no backprop timing decision at all (`prior_p` is fixed once,
at creation, from data already computed then) — it composes with
either Option 1 or Option 2's value-term change rather than competing
with it.

**Cost / reach**: O(1) per leaf per selection, same as Option 1 —
`prior_p` is read, not walked. Reach is local to the sibling batch
that produced it (no cross-batch or cross-depth propagation, unlike
Option 2's full path).

**Not implemented; not benchmarked.** Recorded here as a design-space
entry only, per Tuan's request (2026-07-18) to document it alongside
Options 1/2 rather than leave it in conversation only.

## 4.6. How these three options compare to real AlphaZero
*(added 2026-07-18, researched against primary/near-primary sources —
Silver et al. 2017 Nature, the ELF OpenGo reimplementation paper
(Tian et al. 2019, arXiv:1902.04522), and the Leela Chess Zero
implementation notes — after Tuan asked what AlphaZero itself
actually does for Q/bonus computation.)*

### The real mechanism: backup touches every ancestor, every
simulation, unconditionally

AlphaZero's MCTS runs many independent root-to-leaf simulations per
move (800-1,600 in the primary sources). Each simulation ends with a
leaf evaluation `v` (the value network's output — an estimated win
probability, not a terminal game outcome, since most evaluated leaves
are *not* terminal game states in AlphaZero's setting, only in bl_cnt's
generation-until-terminal setting). The **backup** step then
unconditionally updates *every node on that simulation's root-to-leaf
path* — not just the leaf, not just its parent, all of them:

```
for node in path_from_leaf_to_root:
    node.N += 1
    node.W += v
    node.Q = node.W / node.N
```

Because many simulations share the same upper portion of the tree (all
of them pass through the root; most pass through the first few plies),
an internal node's `Q` becomes, by construction, "the mean evaluation
of every simulation that ever passed through here" — i.e. a live
summary of its entire explored subtree, updated continuously, with
zero extra computation needed at *selection* time. Selection then
reads this already-aggregated `Q` directly:

```
a* = argmax_a  Q(s,a) + c_puct · P(s,a) · sqrt(ΣN(s,b)) / (1 + N(s,a))
```

### Why this doesn't map cleanly onto the bl_cnt codebase as-is

bl_cnt's search is a best-first frontier search, not a
simulation-based tree search: `agent.backprop(node)` only ever fires
when a **terminal** node is created (or, in lazy v01, selected) —
there is no notion of "every root-to-leaf walk backs up its own
path," because there is no repeated independent walk at all, only a
single frontier that grows breadth-first-ish via best-first
selection. Consequently, in bl_cnt today, a node like `A` only
accumulates contributions from whichever of *its own descendants'*
terminals happen to have backpropped by a given point in the search —
not "everything below `A`," and not on any fixed cadence. This is a
real, structural difference from AlphaZero's per-simulation backup,
not just a smaller-scale version of it.

### Where Options 1/2/3 land relative to the real mechanism

| | reach | prior term `P` | exploration-term shape | backup discount |
|---|---|---|---|---|
| **Real AlphaZero** | full path, every simulation, unconditional | policy-net softmax; root gets Dirichlet noise (`ε=0.25`, `Dir(0.03)`) | `c_puct · P · sqrt(ΣN_b) / (1+N_a)` | **none** — undiscounted, uniform across depth |
| Option 1 (§3) | one hop (parent only) | none | unchanged bl_cnt UCB1, `sqrt(ln N_parent / N_leaf)` | n/a — only the value term is touched |
| Option 2 (§4) | full path (matches AZ's *reach*) | none | changed to AZ's `sqrt(N_parent)/(1+N_leaf)` shape (matches) | `gamma`-discounted — **does not match**; real AlphaZero has no depth discount at all (see below) |
| Option 3 (§4.5) | one batch (siblings at creation) | PRM-derived, batch-normalized — matches AZ's *role* (a prior weight on `u`), different *source* (no policy network here) | needs AZ's `P`-weighted shape to have a slot for `prior_p` at all; UCB1 alone has none | n/a |

**None of the three is a full reproduction of AlphaZero, and none was
trying to be** — each isolates a different piece of the mechanism on
purpose (§3, §4's own text). But two specific, easy-to-miss points are
worth recording plainly:

1. **Option 2's `gamma`-discount is not part of real AlphaZero.** The
   primary-source backup applies the *same* `v`, undiscounted, to
   every ancestor regardless of distance from the leaf — a failure
   twenty plies up counts exactly as much as one plie up, because it's
   the identical simulation's `v` being added at every stop.
   Option 2's `gamma ** depth` decay (§4) is a reasonable, deliberate
   modification given bl_cnt's stated goal ("a failure many levels up
   counts for less than an immediate parent's failure") — but it
   should be described as *AlphaZero-inspired*, not
   *AlphaZero-matching*, when writing this up anywhere more formal
   (a paper, a related-work paragraph). Real AlphaZero's implicit
   `gamma` is 1.0.
2. **Option 3 is structurally the closest of the three to what "PUCT"
   originally names** — a *prior-weighted* exploration bonus — even
   though its reach (one sibling batch) is the shallowest of the
   three. Option 2 is closest in *reach* (full path, matching AZ's
   backup scope) but diverges on the discount. There is no reason
   Options 2 and 3 couldn't be combined (full-path value + prior-
   weighted bonus, `gamma` dropped to 1.0) as a fourth, more literal
   reproduction — not currently a planned direction, recorded here
   only as a named possibility in case the comparison becomes useful
   later (e.g. if a reviewer asks "why doesn't this match AlphaZero
   exactly").

### A clarifying note on "why Option 2 is closer to the real
mechanism than Option 1, in plain terms"

Option 1's `path_aware_puct` reads `self.parent.q_value()` — a single
number that, in bl_cnt's terminal-triggered backprop scheme, reflects
"whatever has backpropped through this parent so far via its terminal
descendants," which is a real but partial and update-cadence-dependent
signal (see "Why this doesn't map cleanly" above). Option 2's
full-path walk is an attempt to approximate AlphaZero's
"every-ancestor-aggregates-its-whole-subtree" property directly at
*read* time (walking and discounting the path on every selection
call), precisely because bl_cnt's *write* step (backprop) doesn't
already maintain that property the way AlphaZero's per-simulation
backup does for free. Same end goal — let a node's score summarize
its subtree — reached by different means, because the two systems
don't share a backup trigger.

**Sources** (primary/near-primary, checked directly rather than
recalled): Silver, D. et al. "Mastering the game of Go without human
knowledge," *Nature* 550, 354-359 (2017) — selection formula, backup
rule (`W += v`, `Q = W/N`), Dirichlet noise at root. Tian, Y. et al.
"ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero,"
ICML 2019 (arXiv:1902.04522) — `c_puct = 1.5` and virtual-loss-constant
`= 1.0` as empirically swept values (both left unspecified in the
original AGZ/AZ papers, per that paper's own Table S1). Leela Chess
Zero implementation notes (lczero.org/dev/lc0/search/alphazero) —
independent confirmation of the same formula and backup shape.

## 5. Comparison (Option 1 vs. Option 2 vs. Option 3, v01)

| | reach | new state read | which term changes | cost/selection | main risk |
|---|---|---|---|---|---|
| Option 1 | one hop (immediate parent) | `parent.q_value()` (already exists) | value term only (`ln`-UCB1 exploration unchanged) | O(1)/leaf | weak effect in mixed-quality neighborhoods |
| Option 2 | full path, `gamma`-discounted | whole ancestor chain's `q_value()` | both — value term *and* exploration term reshaped to AlphaZero `sqrt(N)/(1+N)` | O(depth)/leaf | two things change at once (depth *and* term shape); needs `cpuct` retuned or it overshoots harder than today |
| Option 3 | one batch (siblings at creation) | new `prior_p` field, batch-normalized PRM score | exploration term only (value term untouched) | O(1)/leaf | needs a normalization/temperature choice (`tau`); no backprop-timing interaction to reason about, unlike 1/2 |

Option 1 isolates one variable (does neighborhood-blending help at
all) with the smaller footprint; Option 2 is the more "textbook"
design (proper discounted subtree value) but confounds two changes and
actively fights the stated goal unless `cpuct` is retuned alongside
it; Option 3 targets a *different* defect (exploration-term
value-blindness, not backprop-reachability) and is orthogonal enough
to combine with either 1 or 2 rather than replace them.

## 6. Recommendation (v01)

Try Option 1 first, in a **new version file** (`v04` — `v02` is KUBE
and `v03` is depth-shaping knapsack, see §7), sweeping
`α ∈ {1.0, 0.8, 0.6}` with `α = 1.0` as the current-behavior control
arm. Reach for Option 2's full path-walk only if one-hop blending is
shown insufficient. Option 3 is a candidate to layer on top of
whichever of Option 1/2 ships, once the `alpha` (or `gamma`) sweep
gives a baseline to compare against — it changes a different term, so
sequencing it strictly after rather than sweeping all three at once
keeps each ablation attributable to one variable. Each option (and
Option 3, if it proceeds) is a new version file, not an edit
to `v01_00_00`, per the two-tier convention in
[algorithms.md](../algorithms.md) — this changes search behavior, so
existing scored `bl_cnt` cells must stay attributable to the old
formula.

## 7. Extending the same question to v02/bl_kube (KUBE) and v03 (depth-shaping)

*Note: "v02" and "v03" in this section's prose are this analysis's
original shorthand for what are now the separate `mcts_bl_kube_v01`
and `mcts_bl_kdepth_v01` families; file-path citations use the
current module names.*

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

  *Update 2026-07-18:* this "zero effect" verdict describes the
  UNMODIFIED `"global"` formula and remains true of it — but the
  shipped `mcts_bl_kube_v02` later extended Option 1's value-term
  blend to `"global"` too (same day it first shipped
  `"parent"`-only), which creates exactly the read channel this
  bullet says the unmodified formula lacks. Notably, global+blend
  gets the discouragement channel *without* this section's
  count-burst objection to `"parent"` (the shared-clock bonus
  cannot burst). See
  [bl-cnt-v02-eager-backprop-path-aware.md](bl-cnt-v02-eager-backprop-path-aware.md)
  §3.5 for the reversal rationale.

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
- None of Option 1, Option 2, or Option 3 (v01) has been implemented
  or benchmarked; this file records the design space only. (Option 1
  *has* shipped as `mcts_bl_cnt_search_v02_00_00.py` — see
  [bl-cnt-v02-eager-backprop-path-aware.md](bl-cnt-v02-eager-backprop-path-aware.md)
  — but that implementation predates and is untracked by this file's
  own §3-§6 recommendation text, which still calls it "a new version
  file (`v04`)"; a renumbering note wasn't added retroactively since
  §3-§6 remain an accurate description of the design regardless of
  what it was eventually named.)
- The v02/v03 docstring-vs-code discrepancy (§7.3) is unresolved —
  neither file's Algorithm block has been corrected yet.
- Whether v02/v03 should actually get the terminal-split fix (as
  hygiene, independent of the v01 path-aware-scoring question) hasn't
  been decided — §7 only establishes what the change would and
  wouldn't do in each.

## Revisit if

- Option 1 (v01, §3; shipped as `mcts_bl_cnt_search_v02_00_00.py`,
  see [bl-cnt-v02-eager-backprop-path-aware.md](bl-cnt-v02-eager-backprop-path-aware.md))
  is swept and shows either (a) too weak an effect — go to Option 2
  (§4), retuning `cpuct` alongside the exploration-term-shape change
  rather than swapping the formula in isolation — or (b) the specific
  "count-burst attracts toward a dead-end's siblings regardless of its
  score" failure mode from the "Magnitude" section is still visible in
  traces even with `alpha < 1` — go to Option 3 (§4.5) instead, since
  that's the defect it targets directly and Option 2 doesn't fix it
  either (Option 2's exploration term is reshaped for a different
  reason, not made score-aware).
- The terminal-split fix is implemented for v02/v03 — remember it
  *introduces* new dead-end-driven attraction bursts under v02's
  `parent` schedule (§7.1), which is a new failure mode, not a
  pre-existing one being merely surfaced earlier.
- The same question ("does backprop reach frontier selection")
  resurfaces for `mcts_bl_sem_v01` — its selection criterion is a
  diversity term over embeddings, not PUCT/KUBE/depth-shaping, and
  this analysis has not been verified to transfer there.
