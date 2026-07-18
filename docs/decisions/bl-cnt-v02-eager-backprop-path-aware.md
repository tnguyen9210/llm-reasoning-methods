# v02 of bl_cnt, bl_kube, bl_kdepth: eager terminal backprop, implemented per the `\S`7 verdicts

*2026-07-18*

Records the implementation that follows directly from
[bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
(2026-07-16), which analyzed but did not implement the design space.
Tuan asked to "create a v02 versions for these three methods, that
will implement eager terminal backprop with additional options
(maybe options mentioned above but determined later)" — the options
were resolved via a short series of scoped questions (`\S`0 below)
rather than guessed, since the design doc's own `\S`7.4 summary table
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
   the design doc's own `\S`6 recommendation (smaller footprint,
   isolates one variable, existing state only — Option 2 changes the
   exploration-term shape too and needs explicit `cpuct` retuning,
   deferred until Option 1 is shown insufficient).
2. **kube mechanism** — apply the parent-blend under both
   `kube_schedule` values, or only `"parent"` (the one the design
   doc's `\S`7.1 actually supports, since `"global"`'s bonus has no
   per-node channel to blend into)? **Answer: `"parent"` only**;
   `"global"` gets the terminal-split fix as pure hygiene, no
   invented formula.
3. **kdepth scope** — hygiene-only (matching what `\S`7.2 actually
   established: no channel exists at all), or design a brand-new
   value-reading term for `depth_density` as part of this same task?
   **Answer: hygiene-only.** Inventing a new formula is real,
   undesigned work explicitly out of scope of the existing analysis
   — a later, separate decision if ever pursued.
4. **Naming** — straight v02 bump within each existing family
   (`mcts_bl_cnt_v02`, `mcts_bl_kube_v02`, `mcts_bl_kdepth_v02`), per
   the two-tier convention and the design doc's own `\S`6 note ("new
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

- **Terminal split + eager backprop**, unconditionally, in all
  three: `expand_node` now returns the batch of newly created
  children (previously the caller re-read `current_node.children`
  after the call); the caller splits that batch by `is_terminal` —
  a terminal child calls `agent.backprop(child)` immediately, a
  non-terminal child is appended to `leaf_nodes`. A terminal never
  enters the frontier at all under v02.
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
design doc's `\S`3 recommended starting range is 0.6–0.8).

## 3. bl_kube v02 — path-aware KUBE density, `"parent"` schedule only

`core/mcts_bl_kube_search_v02_00_00.py`. New
`MCTSNode.path_aware_kube_density(max_depth, kube_c, t, schedule,
alpha)`:

```
schedule == "parent":
    blended_q = alpha*q_value(x) + (1-alpha)*q_value(parent(x))
    bonus = kube_c * sqrt(log(N_parent(x)) / N(x))     # v01's bonus, unchanged
    density = (blended_q + bonus) / cost(x)

schedule == "global":
    density = (q_value(x) + bonus) / cost(x)           # identical to v01
    # alpha is read from config but unused in this branch
```

Per the design doc's `\S`7.1: `"parent"`'s bonus term "is exactly
bl_cnt v01's PUCT bonus," so the blend applies to the same `q`
position PUCT's `q` occupies, before the (unchanged) bonus and cost
division. `"global"`'s bonus is a frontier-wide constant when every
frontier node sits at `visits==1` — there is no per-node parent-q
channel to fold in beyond what `q_value(x)` already is, so `"global"`
gets **only** the terminal-split fix, with no claim of reading
propagated values differently than v01.

New config field: `BLMCTSKubeV02Config.alpha: float = 0.8` (same
semantics and default as bl_cnt's; unused under `"global"`).

**Also fixed by the terminal-split** (not new formula work, a
byproduct of the mechanical fix, both schedules): the design doc's
`\S`7.1 defect where a max-depth dead-end (`cost <= 0`, density
`-inf`) was *permanently stuck* in `leaf_nodes` under v01, and its
permanent `is_terminal==True` membership permanently satisfied the
`kube_affordable` filter's "always eligible" clause — silently
disabling that filter's own empty-set fallback for the rest of any
run past its first few dead-ends. Under v02 a dead-end backprops and
exits the frontier immediately, so it can no longer prop up
`affordable`'s non-emptiness. The `node.is_terminal` disjunct in
`select_child_from_list`'s affordability filter is now dead in
practice (a terminal never reaches that method's `nodes` argument at
all) but was kept, not deleted — see the module docstring's "Note on
the now-dead is_terminal clause" for why (the method is a
general-purpose helper mirroring v01's exact signature, not
special-cased to its one current caller).

## 4. bl_kdepth v02 — hygiene only, formula untouched

`core/mcts_bl_kdepth_search_v02_00_00.py`. `depth_density()` is
**byte-identical** to v01's — verified by diffing the method bodies
directly (empty diff). Per the design doc's `\S`7.2: `depth_density`
reads only a leaf's own frozen `q_value`, its own `depth`, and two
constants — no visit-count or parent-q channel exists at all, so "no
backprop timing — eager, lazy, or never — can change which
non-terminal node gets expanded next" (`\S`7.2, verbatim). No `alpha`
field on `BLMCTSKdepthV02Config` — confirmed by direct attribute
check (`hasattr(cfg, 'alpha') == False`) rather than merely leaving
it unset, since there is genuinely nothing designed for it to blend.

What v02 changes here is exactly the same terminal-split hygiene
fix described in `\S`3 for bl_kube (identical `kube_affordable`
filter, identical dead-end-permanence defect, identical fix) — ranking
among non-terminal nodes is **provably unchanged**: every non-terminal
node that would have been in `leaf_nodes` under v01 is in
`leaf_nodes` under v02 too, and `depth_density`'s inputs
(`depth`/`q_value`/constants) are untouched by the split for a
non-terminal node. The only removed events are terminal-selection
phases — a phase that, under v01, does nothing but re-select an
already-known dead-end and immediately backprop it. Pure overhead
removed, not a ranking change.

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
     asserted in the docstring.
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
   prior run this session — zero new problems from the v02 work.
5. Line-length sweep on all three new core files: every violation
   found (`create_child`/`_generate_candidates` region) matches a
   verbatim pre-existing line already present in the corresponding
   v01 file — none introduced by this change.

## 6. What was deliberately left alone

- **Option 2** (AlphaZero-style full-path, discounted) for bl_cnt —
  not implemented; the design doc's `\S`6 defers it until Option 1 is
  shown insufficient by a sweep.
- **A new value-reading formula for bl_kdepth** — explicitly
  out-of-scope per `\S`0 item 3; would be a separate, later decision.
- **A new blend formula for `kube_schedule="global"`** — explicitly
  out-of-scope per `\S`0 item 2; the design doc doesn't support one
  and inventing one wasn't part of this task.
- **The `node.is_terminal` disjunct** in both bl_kube's and
  bl_kdepth's `kube_affordable` filter — now dead code in practice
  given the terminal-split, kept rather than deleted (general-purpose
  helper, not special-cased to its current caller).
- **No runs launched yet** — these are net-new methods with zero
  scored data; nothing to migrate, no result-dir/manifest/ledger
  work like the two family renames required.

## Connections

- [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  — the analysis this implements; `\S`7's per-variant verdict table is
  the direct source for `\S`0's scoping answers.
- [bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md),
  [bl-cnt-to-bl-kdepth-rename.md](bl-cnt-to-bl-kdepth-rename.md) —
  the family renames that gave bl_kube/bl_kdepth their current names;
  this file's v02s are versions *within* those already-renamed
  families, not further renames.
- `core/mcts_bl_cnt_search_v01_00_00.py` /
  `core/mcts_bl_kube_search_v01_00_00.py` /
  `core/mcts_bl_kdepth_search_v01_00_00.py` — each v02's sibling and
  control-arm baseline (`alpha=1.0` recovers v01 exactly, verified
  `\S`5).
- `docs/decisions-log.md`, 2026-07-18 entry (top of file) — the
  short-form pointer to this file.
