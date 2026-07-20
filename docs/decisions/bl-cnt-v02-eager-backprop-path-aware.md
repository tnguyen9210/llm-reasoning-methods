# v02 of bl_cnt, bl_kube, bl_kdepth: eager terminal backprop, implemented per the §7 verdicts

*2026-07-18*

Records the implementation that follows directly from
[bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
(2026-07-16), which analyzed but did not implement the design space.
Tuan asked to "create a v02 versions for these three methods, that
will implement eager terminal backprop with additional options
(maybe options mentioned above but determined later)" — the options
were resolved via a short series of scoped questions (§0 below)
rather than guessed, since the design doc's own §7.4 summary table
shows each of the three files needs a genuinely different treatment,
not one mechanism copy-pasted three times.

New files: `core/mcts_bl_cnt_search_v02_00_00.py`,
`core/mcts_bl_kube_search_v02_00_00.py`,
`core/mcts_bl_kdepth_search_v02_00_00.py`. Each is v02 of its
existing family (not a new family) — v01/v02 share the identical
selection theory in every case; only backprop timing (and, for two
of the three, a path-aware blend) changes.

## 0. Scoping decisions (asked, not guessed)

Four questions were asked via `AskUserQuestion` before writing any
code, each because the design doc's own analysis left a genuine
choice rather than a single correct answer:

1. **v01 mechanism** — Option 1 (parent-blend) vs. Option 2
   (AlphaZero-style full path) vs. both. **Answer: Option 1**, per
   the design doc's own §6 recommendation (smaller footprint,
   isolates one variable, existing state only — Option 2 changes the
   exploration-term shape too and needs explicit `cpuct` retuning,
   deferred until Option 1 is shown insufficient).
2. **kube mechanism** — apply the parent-blend under both
   `kube_schedule` values, or only `"parent"` (the one the design
   doc's §7.1 actually supports, since `"global"`'s bonus has no
   per-node channel to blend into)? **Answer: `"parent"` only**;
   `"global"` gets the terminal-split fix as pure hygiene, no
   invented formula. **[Reversed same day, later on 2026-07-18 —
   see §3.5 below: the "no channel" reasoning is true of the BONUS
   term, but Option 1 blends the VALUE term, which exists
   identically under both schedules; the blend now applies under
   both, with `alpha=1.0` reproducing the pre-reversal `"global"`
   behavior exactly. Original answer kept here as the record of
   what was decided at scoping time.]**
3. **kdepth scope** — hygiene-only (matching what §7.2 actually
   established: no channel exists at all), or design a brand-new
   value-reading term for `depth_density` as part of this same task?
   **Answer: hygiene-only.** Inventing a new formula is real,
   undesigned work explicitly out of scope of the existing analysis
   — a later, separate decision if ever pursued.
4. **Naming** — straight v02 bump within each existing family
   (`mcts_bl_cnt_v02`, `mcts_bl_kube_v02`, `mcts_bl_kdepth_v02`), per
   the two-tier convention and the design doc's own §6 note ("new
   version file... not an edit to v01_00_00"). **Confirmed as-is** —
   no naming ambiguity here, unlike the kube/kdepth renames (this is
   a same-family version bump, not a family-scope question).

A fifth question came up mid-implementation, after the group-label
choice was first drafted analogously to the `mcts_sem_v01`/
`mcts_sem_v02` precedent (both share the `"sem-mcts"` group in
`status.py`):

5. **`status.py` group label** — should each v02 share its v01's
   `_METHOD_TO_GROUP` label (rows in the same `docs/exp-comp-*.md`
   table, matching the sem precedent) or get its own (a separate
   subsection/table)? **Answer: separate group per v02** —
   `cnt-mcts-bl-v02`, `kube-mcts-bl-v02`, `kdepth-mcts-bl-v02` — the
   sem precedent was considered and explicitly not followed here;
   Tuan judged the algorithm change (backprop timing + blend) large
   enough to warrant its own table rather than a same-table row next
   to v01.

Also mid-implementation: Tuan interjected **"actually I'd like not
to have the eager terminal backprop for these files"** while the
*previous* session's loop-reshape work (aligning kube/kdepth's loop
to v01's generate→expand→select shape) was in progress — that
reshape had been scoped, via `AskUserQuestion`, as being "in service
of" eager backprop, and the interjection dropped that framing
entirely (the reshape shipped as pure structural alignment, no
eager-backprop content, in commit `61dd578`). This file's v02s are a
**separate, later request** ("I'd like to create a v02 versions...")
made the next day, not a continuation of the dropped thread — eager
backprop genuinely is being implemented now, just not as a
byproduct of that unrelated reshape.

## 1. What's the same in all three v02s

- **Terminal split + delayed-eager backprop**, unconditionally, in
  all three: `expand_node` now returns the batch of newly created
  children (previously the caller re-read `current_node.children`
  after the call); the caller splits that batch by `is_terminal` —
  a terminal child is QUEUED (`pending_terminal_backprops.append`),
  a non-terminal child is appended to `leaf_nodes`. The queue is
  flushed (`agent.backprop(child)` for every queued node) right
  after the very next `select_child_from_list` call resolves, or
  right before either of the loop's early-exit points (`gen_budget`
  exhausted, `leaf_nodes` empty) if the loop is about to break before
  reaching a selection. A terminal never enters the frontier at all
  under v02. See §1.5 for why this is delayed rather than immediate
  — the distinction was raised and resolved mid-implementation.
- The defensive `current_node.is_terminal` check at the top of the
  loop (root boundary case, `max_depth == 0`) is kept in all three,
  now effectively dead in the common case — same treatment as v01's
  own such guard.
- `completed_nodes` membership and every completion's recorded
  `(text, depth, phase, gen_cnt)` are **provably unaffected**: they
  are set inside `create_child` at node-creation time, before any
  split/backprop-timing logic runs at all — verified by inspection
  (grep confirms `self.completed_nodes.append(new_node)` lives
  inside `create_child`, called once per candidate regardless of
  what happens to the node afterward).
- Candidate generation, expansion mechanics, node classes, output
  shape, the loop's generate→expand→select ordering — all
  byte-identical to each file's v01.

## 1.5. Delayed vs. immediate flush timing (decided mid-implementation)

The first pass of all three files backpropped a terminal child
**immediately**, inline inside the expand step, before that step's
own selection ran. Tuan flagged the conceptual issue directly: doing
so lets a same-batch terminal sibling's outcome — produced by the
exact same `_generate_candidates` call that produced the OTHER
candidates now being ranked — influence the selection choosing among
those very candidates. That's not genuinely "prior" evidence; it's
information concurrent with the decision reading it. Tuan proposed
the fix precisely: collect newly generated terminal nodes into a
temporary list, run the selection that was already going to happen
using pre-existing state, THEN flush (backprop) the queued nodes —
so a terminal's effect reaches every selection strictly AFTER the one
immediately following its own creation, never that one.

This is a genuine third point in the design space, not a
re-derivation of either extreme:

| | v01 (lazy) | immediate-eager (first pass) | delayed-eager (final) |
|---|---|---|---|
| When does a terminal backprop? | Only if/when selected off the frontier — possibly never | Same step it's created in | One step after the step it's created in |
| Worst-case latency until first influencing ANY selection | Unbounded | Zero | One step |
| Same-batch sibling selection sees it? | No | **Yes** | No |

Confirmed live where it matters: bl_cnt's `path_aware_puct` and
bl_kube's `path_aware_kube_density` (originally `"parent"` schedule
only; both schedules since §3.5's global-blend extension) read
`parent.q_value()` (and, under kube's `"parent"` schedule,
`parent.visit_count()` as the bonus clock) — the exact channel a
same-batch terminal sibling's backprop would write through — so the
distinction is not academic there. Re-verified numerically after the
rewrite (see §5, updated): a leaf's PUCT score during the selection
immediately following its own creation is now provably unaffected by
a terminal sibling created in that same batch, while the very next
selection *does* see the flushed effect.

bl_kdepth is the one file where this distinction is **inert**:
`depth_density` reads neither `parent.q_value()` nor
`parent.visit_count()`, so delayed vs. immediate changes nothing
observable there. The delayed-flush queue pattern was still applied
to `mcts_bl_kdepth_search_v02_00_00.py` for structural consistency
with its two siblings (one invariant to reason about across all three
files), documented explicitly in that file's docstring as inert
rather than left silently inconsistent with the other two.

A transcription bug was caught and fixed during this same pass:
`` `\S` `` (an escaped-backslash artifact, not a valid Python string
escape — Python emits a `DeprecationWarning` on it, silently
swallowed by default) had crept into section-reference citations
(intended as `§`) across all three core files, `utils/configs.py`,
and this doc's own earlier draft. Fixed globally to the literal `§`
character; re-verified zero `DeprecationWarning`s remain on any
touched file.

## 2. bl_cnt v02 — path-aware PUCT (Option 1)

`core/mcts_bl_cnt_search_v02_00_00.py`. New `MCTSNode.path_aware_puct
(cpuct, alpha)`:

```python
blended_q = alpha * q_value(leaf) + (1 - alpha) * q_value(parent)
path_aware_puct = blended_q + cpuct * sqrt(log(N_parent) / N_leaf)
```

`alpha` guards the zero-visit-parent case by falling back to the
leaf's own q (matches v01's own zero-guard pattern, not a new
convention). `alpha=1.0` is the built-in control arm — the parent
term's weight drops to zero, making `path_aware_puct(cpuct, 1.0)`
**numerically identical** to v01's `puct(cpuct)` for the same node
state, verified below.

New config field: `BLMCTSCntV02Config.alpha: float = 0.8` (the
design doc's §3 recommended starting range is 0.6–0.8).

## 3. bl_kube v02 — path-aware KUBE density (both schedules, as of
the §3.5 reversal; originally `"parent"` only)

`core/mcts_bl_kube_search_v02_00_00.py`. New
`MCTSNode.path_aware_kube_density(max_depth, kube_c, t, schedule,
alpha)` — current form, after §3.5's same-day extension:

```
blended_q = alpha*q_value(x) + (1-alpha)*q_value(parent(x))
    # both schedules; alpha=1.0 recovers v01 exactly under either
bonus = kube_c * sqrt(log(clock(x)) / N(x))    # each schedule's v01
    # bonus, unchanged:
    # clock = N_parent(x)  under schedule == "parent"
    # clock = 1 + t        under schedule == "global"
density = (blended_q + bonus) / cost(x)
```

Per the design doc's §7.1: `"parent"`'s bonus term "is exactly
bl_cnt v01's PUCT bonus," so the blend applies to the same `q`
position PUCT's `q` occupies, before the (unchanged) bonus and cost
division. As first shipped, `"global"` was excluded from the blend
(see §0 item 2's original answer and §3.5 for the reversal).

New config field: `BLMCTSKubeV02Config.alpha: float = 0.8` (same
semantics and default as bl_cnt's; live under both schedules since
§3.5).

**Also fixed by the terminal-split** (not new formula work, a
byproduct of the mechanical fix, both schedules): the design doc's
§7.1 defect where a max-depth dead-end (`cost <= 0`, density
`-inf`) was *permanently stuck* in `leaf_nodes` under v01, and its
permanent `is_terminal==True` membership permanently satisfied the
`kube_affordable` filter's "always eligible" clause — silently
disabling that filter's own empty-set fallback for the rest of any
run past its first few dead-ends. Under v02 a dead-end is queued and
exits the frontier at expand time (never re-entering it), so it can
no longer prop up `affordable`'s non-emptiness — this hygiene fix
holds regardless of the delayed-vs-immediate flush timing (§1.5),
since it depends only on the dead-end never being IN `leaf_nodes`,
not on exactly when its value gets backpropped. The
`node.is_terminal` disjunct in `select_child_from_list`'s
affordability filter is now dead in practice (a terminal never
reaches that method's `nodes` argument at all) but was kept, not
deleted — see the module docstring's "Note on the now-dead
is_terminal clause" for why (the method is a general-purpose helper
mirroring v01's exact signature, not special-cased to its one current
caller).

## 3.5. Global-blend extension (same-day scoping reversal, 2026-07-18)

Later the same day, Tuan asked whether it would be OK to give the
`"global"` schedule the parent-blended value term too, and whether
the blend mechanism should be aligned with the `"parent"` schedule's.
Working through it reversed §0 item 2's original scoping answer:

- **The original "no per-node channel" reasoning conflated the two
  terms.** It is true of the BONUS term: `"global"`'s bonus reads
  only the shared clock `t`, so backprop can never move it — that
  part of §7.1's analysis stands. But Option 1 never touches the
  bonus term anywhere (that was its entire design, per the design
  doc's §3: change the value term, leave exploration alone). The
  VALUE term `q_value(x)` exists identically under both schedules,
  and `parent.q_value()` is just as readable from either. There was
  never a structural obstacle — only an unexamined transfer of the
  bonus-term analysis onto the value term.
- **Global+blend is actually the cleaner one-variable test of the
  blend idea.** Under `"parent"`, a backprop through parent `P`
  moves two entangled channels at once: the bonus clock `N(P)` (the
  count-attraction burst §7.1 flags as pulling toward failed
  regions regardless of score) AND the blended value `q(P)` (the
  discouraging signal the blend wants). The blend's effect can never
  be observed in isolation there. Under `"global"`, the bonus cannot
  burst — it reads only `t` — so the blend is the ONLY ancestor
  channel: pure value-based discouragement of failed neighborhoods,
  no counterproductive count-attraction side channel. §7.1's main
  objection to the `"parent"`-schedule fix does not apply to
  global+blend.
- **Implementation**: the blend was hoisted out of the schedule
  branch in `path_aware_kube_density` — computed once, shared
  formula, shared `alpha`, shared parent-unvisited fallback — and
  the branch now decides only the bonus's clock (`N_parent` vs.
  `1 + t`), making the code structure mirror the concept ("the
  schedules differ in their exploration clock, not their value
  estimate"). `alpha` means the same thing under both schedules, so
  sweeps are comparable across them.
- **Versioning: edited v02 in place, no v03 bump**, on two facts:
  (a) zero kube-v02 runs had been launched or scored, so no result
  cell's attributability breaks; (b) `alpha=1.0` under `"global"`
  reproduces the pre-reversal behavior exactly (the blend term
  vanishes identically), so the change is a strict generalization
  with the old behavior reachable by config. The conservative
  reading of the two-tier convention would have bumped anyway; the
  in-place edit was a deliberate call given (a)+(b), recorded here.
- **Consequences propagated**: the delayed-eager flush timing
  (§1.5) is now live under `"global"` too, not just `"parent"` —
  the blend reads `parent.q_value()` under both schedules, so the
  same-batch-leak argument applies to both (the queue-and-flush
  code was already schedule-agnostic; only documentation claims of
  `"global"` inertness needed updating). Sweep design note: global
  sweeps need their own `alpha=1.0` no-blend control arm, same as
  parent-schedule sweeps (recorded in the YAML comment and config
  docstring so a future sweep doesn't omit its baseline).

## 4. bl_kdepth v02 — hygiene only, formula untouched

`core/mcts_bl_kdepth_search_v02_00_00.py`. `depth_density()` is
**byte-identical** to v01's — verified by diffing the method bodies
directly (empty diff). Per the design doc's §7.2: `depth_density`
reads only a leaf's own frozen `q_value`, its own `depth`, and two
constants — no visit-count or parent-q channel exists at all, so "no
backprop timing — eager, lazy, or never — can change which
non-terminal node gets expanded next" (§7.2, verbatim). No `alpha`
field on `BLMCTSKdepthV02Config` — confirmed by direct attribute
check (`hasattr(cfg, 'alpha') == False`) rather than merely leaving
it unset, since there is genuinely nothing designed for it to blend.

What v02 changes here is exactly the same terminal-split hygiene
fix described in §3 for bl_kube (identical `kube_affordable`
filter, identical dead-end-permanence defect, identical fix), using
the same delayed-flush queue structure as its two siblings (§1.5) —
though the delay itself is inert here, since `depth_density` reads
neither `parent.q_value()` nor `parent.visit_count()`. Ranking among
non-terminal nodes is **provably unchanged**: every non-terminal
node that would have been in `leaf_nodes` under v01 is in
`leaf_nodes` under v02 too, and `depth_density`'s inputs
(`depth`/`q_value`/constants) are untouched by the split for a
non-terminal node. The only removed events are terminal-selection
phases — a phase that, under v01, does nothing but re-select an
already-known dead-end and backprop it. Pure overhead removed, not a
ranking change.

## 5. Verification performed

Not just "it compiles" — three independent checks per claim, mirroring
the rigor used for the 2026-07-17 loop-reshape's behavior-preservation
proof:

1. **Numerical equivalence at the control-arm setting**, via a
   standalone harness
   (scratchpad `verify_v02_cnt.py` / `verify_v02_kube.py`) comparing
   the new formulas against hand-transcribed copies of v01's, across
   6-and-more edge cases each (normal, zero-visit child, zero-visit
   parent, `cost<=0` boundary, no-parent/root case, `parent_visits=1`
   log-zero case):
   - bl_cnt: `path_aware_puct(cpuct, alpha=1.0)` == v01's
     `puct(cpuct)` exactly, all cases.
   - bl_kube `"parent"`: `path_aware_kube_density(..., "parent",
     alpha=1.0)` == v01's `kube_density(..., "parent")` exactly, all
     cases including the `-inf` boundary.
   - bl_kube `"global"`: `path_aware_kube_density(..., "global",
     alpha)` == v01's `kube_density(..., "global")` exactly for
     **every** `alpha in {1.0, 0.8, 0.5, 0.0}` — proves the "no
     channel to blend" claim is actually implemented, not just
     asserted in the docstring. *(Historical: this verified the
     original alpha-ignoring `"global"` branch, superseded the same
     day by §3.5's global-blend extension — see item 7 below for
     the post-extension re-verification. Under the current code this
     equality holds at `alpha=1.0` only, by design.)*
   - Sanity check the reverse direction too: at `alpha < 1.0`
     (bl_cnt and bl_kube `"parent"`), a leaf's score DOES change when
     its parent's q_value changes — confirms the blend is live, not
     an accidental no-op.
2. **`depth_density` code-body diff** (bl_kdepth): direct `diff` of
   the method's source between v01 and v02 — empty, confirming
   byte-identical formula, not merely "looks the same."
3. **End-to-end Hydra composition** for all three v02 root configs:
   composes cleanly, `cfg.algo` resolves through the real
   `algo_dict` to the correct new core module, `cfg.search.alpha`
   present/absent exactly as designed per file, and each v02's
   `config_hash` is confirmed **distinct** from its v01 sibling's
   (so a v02 run will never collide with existing v01 result dirs).
4. **`status.py --verify`** (full, unscoped): reports the exact same
   4 pre-existing, unrelated `mcts_cnt` hash-drift problems as every
   prior run this session — zero new problems from the v02 work; also
   re-run after the delayed-eager rewrite (§1.5), same result.
5. Line-length sweep on all three new core files: every violation
   found (`create_child`/`_generate_candidates` region) matches a
   verbatim pre-existing line already present in the corresponding
   v01 file — none introduced by this change.
6. **Delayed-eager flush timing** (scratchpad
   `verify_delayed_eager.py`), added for the §1.5 rewrite, exercising
   the real `path_aware_puct` formula (not a mock): (a) a same-batch
   terminal sibling's backprop is confirmed invisible to the selection
   immediately following its own creation — the selection is shown to
   use the parent's pre-backprop `(visits, q)` state, by construction;
   (b) the very next selection IS shown to read the flushed,
   post-backprop state — a real, distinct value; (c) a phase whose
   entire newly-created batch is terminal (so `leaf_nodes` stays empty
   and the loop is about to break) is confirmed to still flush its
   pending queue before that break, via a hand-computed expected final
   `(visits, q)` on the shared parent that matches exactly.
   Re-verified against the actual current file content (not just the
   standalone harness) via the Hydra-composition and `status.py
   --verify` checks in items 3–4, run again after the rewrite.
7. **Global-blend extension** (scratchpad
   `verify_kube_global_blend.py`), added for §3.5's same-day
   reversal: (a) `"parent"`-schedule densities are byte-identical
   before vs. after the blend-hoist refactor, across every edge case
   and every `alpha` — the hoist is a pure restructuring for that
   schedule; (b) `"global"` at `alpha=1.0` equals the pre-reversal
   (alpha-ignoring) `"global"` branch exactly, all cases — the
   control arm reproduces shipped-v02 behavior; (c) `"global"` at
   `alpha<1.0` DOES respond to a parent's q_value change (and is
   confirmed to differ from the pre-reversal branch by exactly the
   blend delta) — the new channel is live, not an accidental no-op.

## 6. What was deliberately left alone

- **Option 2** (AlphaZero-style full-path, discounted) for bl_cnt —
  not implemented; the design doc's §6 defers it until Option 1 is
  shown insufficient by a sweep.
- **Option 3** (PUCT-proper prior term `P(x)` on the exploration
  bonus, batch-normalized from the PRM's `candidate_score`) for
  bl_cnt — added to the design doc's §4.5 on 2026-07-18, after this
  v02 shipped; not implemented. Targets a different defect than
  Options 1/2: the exploration term `u` is score-blind by
  construction (a 0.0-scoring and a 0.95-scoring dead-end produce
  identical `u`), which `alpha`-blending only softens indirectly via
  `Q(x)`. Orthogonal enough to layer on top of `alpha` rather than
  compete with it — see the design doc's §6 "Revisit if" for the
  condition under which to reach for it (score-blind attraction
  bursts still visible in traces even with `alpha < 1`).
- **A new value-reading formula for bl_kdepth** — explicitly
  out-of-scope per §0 item 3; would be a separate, later decision.
- ~~**A new blend formula for `kube_schedule="global"`**~~ —
  originally out-of-scope per §0 item 2; **reversed same day by
  §3.5** (no *new* formula was invented — the existing Option 1
  value-term blend was extended unchanged, once it was recognized
  the "no channel" reasoning applied only to the bonus term).
- **The `node.is_terminal` disjunct** in both bl_kube's and
  bl_kdepth's `kube_affordable` filter — now dead code in practice
  given the terminal-split, kept rather than deleted (general-purpose
  helper, not special-cased to its current caller).
- **No runs launched yet** — these are net-new methods with zero
  scored data; nothing to migrate, no result-dir/manifest/ledger
  work like the two family renames required.

## Connections

- [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  — the analysis this implements; §7's per-variant verdict table is
  the direct source for §0's scoping answers.
- `core/mcts_bl_cnt_search_v01_00_00.py` /
  `core/mcts_bl_kube_search_v01_00_00.py` /
  `core/mcts_bl_kdepth_search_v01_00_00.py` — each v02's sibling and
  control-arm baseline (`alpha=1.0` recovers v01 exactly, verified
  §5).
- `docs/decisions-log.md`, 2026-07-18 entry (top of file) — the
  short-form pointer to this file.
