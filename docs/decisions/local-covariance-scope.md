# Per-node diversity covariance: `cov_scope` and `embeds_ref`

*Built 2026-07-28 —
[decisions-log.md #2026-07-28](../decisions-log.md#2026-07-28--search-config-per-node-diversity-covariance-cov_scope--embeds_ref--one-file-not-two).*

Two orthogonal knobs added to `MCTSSemV02Config`, both read only by
the covariance machinery in `core/mcts_sem_search_v02_00_00.py`:

```yaml
cov_scope:  global      # "global" | "local"    -- WHERE V lives
embeds_ref: absolute    # "absolute" | "relative" -- WHAT V accumulates
```

Both are pinned in `_HASH_EXCLUDE_IF_DEFAULT` at the pre-existing
behavior, so every `config_hash` recorded before this change is
unaffected.

## 1. `cov_scope` — where the covariance lives

**`global`** (unchanged, the default): one `V` for the whole search
tree. Every selection anywhere folds its chosen child's embedding
into that single `V`, and every diversity bonus reads the same
`V^-1`.

**`local`**: one `V` per node, over the children *that node* has
selected. Selecting among node `n`'s children reads `n`'s own
`V_n^-1` and folds the chosen child back into `n` alone. Sibling
subtrees never see each other's folds. The bonus at `n` becomes
"which child points somewhere `n` has not committed to yet?"
instead of "...somewhere the entire search has not visited?".

### Why local scope was worth building

Three independent arguments, all pre-existing in the repo's own
notes:

1. **Coherence with local centering.**
   `embeds_center_mode="local"` (built 2026-07-14) centers each
   expansion batch on its own sibling mean, but a global `V` then
   accumulates those group-centered vectors across the whole
   search — vectors carrying group-dependent offsets pile into one
   covariance. [embeds-centering-design.md](embeds-centering-design.md)
   records the caveat;
   [rep-exp-elliptical-bonus-review.md](rep-exp-elliptical-bonus-review.md)
   notes that rep_exp pairs local centering with a per-group *fresh*
   covariance, which the accumulated `V` deliberately is not. Local
   scope is that missing pairing.
2. **Coherence with the alpha schedule.** The diversity weight is
   already scaled by `sqrt(log(1 + parent_visits))` — a per-node
   clock. [global-vs-local-exploration-schedule.md](global-vs-local-exploration-schedule.md)
   argues a node-indexed multiplier against a globally accumulated
   `V` mixes two clocks. With `V_n` local, the two clocks match.
3. **Bonus dynamic range.** With L2-normalized embeddings and
   `V = lam*I + sum u u^T`, a direction already covered `k` times
   scores `~1/sqrt(lam + k)`. Globally `k` grows with the run's
   *total* selections (hundreds to thousands), so late selections
   see a compressed bonus everywhere; locally `k` is the node's own
   fold count (typically 1–5), so `ds_alpha` means roughly the same
   thing at the root and at depth 15.

### Consequence: the global operating point does not transfer

This is the single most actionable finding, and it governs how the
local tuning tables were designed.

`w_eff = ds_alpha/sqrt(lam)` is the diversity weight **at the ridge
init**, before any folds. Under global scope `V` saturates quickly,
so the *actual* weight decays far below `w_eff`. Under local scope
`k` stays small, so the actual weight stays *near* `w_eff` for the
whole run. Concretely at `lam=0.01`:

| scope | typical `k` | bonus `1/sqrt(lam+k)` |
|---|---|---|
| local | 3 | **0.576** |
| global | 300 | **0.058** |

A **10x stronger diversity push for the same `ds_alpha`**. So local's
optimum should sit near `w_eff ≈ 1`, roughly an order of magnitude
below global's measured optimum of `w_eff = 10`.

Sweeping local on the global grid ({1, 10, 100, 1000}) would place
every cell at or above the predicted optimum and make local look
uniformly over-diversified — a measurement artifact that would
plausibly be read as "local scope doesn't work". The level-5 local
sweeps therefore use `w_eff` ∈ {0, 0.1, 0.3, 1, 3, 10}, shifted down
and denser at the low end.

## 2. `embeds_ref` — what vector represents a child

**`absolute`** (unchanged, the default): the child's own pooled
embedding `x_c`.

**`relative`**: the displacement `x_c - x_n`, i.e. the child
expressed in coordinates centred on **its parent**, renormalized.

### Why

A child's embedding is pooled over its **whole text prefix** —
question plus every step so far — not just the new step. Siblings
under one node therefore share a long common prefix, and after L2
normalization they sit in a tight cluster around the parent's
direction. Measured on a synthetic parent with three children
differing by one 10%-magnitude step:

```
ABSOLUTE  cos(child, parent)     0.995  0.995  0.995
          cos(child_i, child_j)  0.990  0.990  0.991   <- near-identical

RELATIVE  cos(disp, parent)     -0.050 -0.050 -0.049
          cos(disp_i, disp_j)   -0.008 -0.081  0.058   <- near-orthogonal
```

So `V` currently spends its capacity modelling the shared prefix
while the sibling differences — the only thing selection can act
on — are a ~1% perturbation. Subtracting the parent deletes the
shared component and leaves the *steps*, which are essentially
orthogonal. The repo's unit suite asserts the same effect on
clustered vectors: mean `|cos|` 0.982 -> 0.070.

### `relative` always means parent-relative

The value names a *kind* of measurement, not a reference point, so
"relative to what" is not self-evident — especially since
`embeds_center_mode="local"` is *also* a relative transform
(relative to the sibling-group mean). The parent is the only
reference implemented, and that is pinned in the config comment, the
`_cov_vec` docstring, the module docstring and the YAML. A second
reference (root, grandparent) would arrive as **its own knob**, not
as another value of `embeds_ref`.

### Three design decisions inside `relative`

1. **The root is embedded explicitly.** `embeds` is otherwise
   assigned in exactly one place — `create_child` — so the root
   would have none, and its children would have nothing to subtract.
   The first implementation fell back to `absolute` at depth 0; a
   review measured what that costs and it was fixed by adding
   `_embed_root`, which pools the question with an **empty answer**
   through `_embed_candidates` (so scope / pooling / projection /
   centering / normalize are byte-identical to what every child
   gets — a root vector built any other way would not live in the
   same space as the children subtracted from it).

   What the fallback cost, measured on a 100-fold descent:

   | scope | with fallback | consequence |
   |---|---|---|
   | `global` | 25% of folds absolute, 75% displacements | one `V` fitted to a bimodal mixture; the bonus means different things at different depths |
   | `local` | root's `V` gets only absolute vectors | depth 0 keeps the clustered-sibling problem `relative` exists to remove — and depth 0 is where branching is widest |

   With `_embed_root`, `relative` applies at every depth and
   `global`+`relative` genuinely accumulates only displacements.
   The `node.embeds is None` branch in `_cov_vec` survives as a
   defensive fallback for trees driven outside `mcts_search` (unit
   tests), not as the depth-0 path of a real run.

   This depends on the `relative` + `embeds_center_mode="local"`
   guard: a one-element batch centred on its own group mean is
   exactly the zero vector, which would make every child's
   displacement equal to its own embedding. That combination is
   rejected in `__init__`, so `_embed_root` can never be reached
   with it.
2. **Renormalize the difference.** Raw `||x_c - x_n||` measured
   ~0.10 against `||x_c|| = 1.0`, and the bonus scales linearly in
   `||x||`, so raw displacements would shrink it ~10x and make
   `ds_alpha` mean something different from every existing sweep.
   Renormalization follows the existing `embeds_normalize` flag
   rather than adding a knob. The cost is discarding step
   *magnitude* ("this was a big move"), which is real information —
   a raw-magnitude variant is a reasonable later addition. Note that
   even renormalized the geometry changes (normalized displacements
   fill `V` far more isotropically than clustered absolutes), so
   `ds_alpha` may still want re-tuning, just not by an order of
   magnitude.
3. **Read and fold must use the same vector.** Not tidiness — a
   correctness requirement. If the bonus scored `x_c - x_n` while
   `V_n` accumulated `x_c`, `V_n` would live in a different space
   than the queries and the bonus would be meaningless. One helper,
   `MCTS._cov_vec`, is called by both `_select_by_diversity` and
   `select_child`, so the two cannot drift.

### Mutually exclusive with `embeds_center_mode="local"`

Using both double-centers: the sibling-group mean is subtracted,
then the parent on top of it, so the vectors fed to `V` are
differences of already-differenced quantities and neither knob means
what its name says. `MCTS.__init__` raises. They are two answers to
the same question ("remove the offset siblings share"), and under
`cov_scope="local"` **parent-relative is the more coherent of the
two**: the parent is a fixed reference for the node's entire
lifetime, whereas the group mean is recomputed at every expansion,
so under `revisit_policy="regenerate"` one node's `V` would
accumulate vectors measured from different origins.

## 3. Architecture: one file, not two — a decision that was reversed

**Initially built as a separate module** (`mcts_sem_search_v02_01_00.py`,
method `mcts_sem_v02_01`, schema `MCTSSemV02LocalConfig`), on two
stated grounds:

1. *Hash stability* — adding a field to `MCTSSemV02Config` would
   re-hash every existing sem_v02 config (316 scored level-4 +
   level-5 entries).
2. *No silent no-ops* — if the knob lived on v02's schema, a config
   setting `cov_scope="local"` with `method=mcts_sem_v02` would run
   global and record "local" in its manifest.

**Both grounds were wrong, and the file was merged back on the same
day.** Reason 1 does not hold: `_HASH_EXCLUDE_IF_DEFAULT` drops a
field from the hash whenever it equals a pinned neutral, and the
repo had already used exactly that mechanism twice — for `cov_dtype`
("fp64") and `embeds_center_mode` ("fixed"). Adding
`"cov_scope": "global"` and `"embeds_ref": "absolute"` to that dict
is a three-word edit that leaves every pre-existing hash untouched
(verified: 204/204 level-5 entries still compose to their recorded
hash). Reason 2 dissolves once the knob is read by the one file that
implements it.

### Why merging is actively better

- **It dissolves the verification problem instead of solving it.**
  The separate file kept `cov_scope="global"` as a "verification
  lever" — the claim being that global reproduces v02 exactly, so a
  difference under global is a porting bug and a difference under
  local is the ablation. In two files that is a claim needing a test
  that must keep passing forever. In one file it is true **by
  construction**: `global` is not "equivalent to" the old behavior,
  it *is* the old code path, same lines, same RNG draws.
- **The duplication was already rotting.** The two files differed by
  ~80 lines of real code out of 1122; everything else was a copy.
  Two defects found in review lived in *both* copies (a
  `select_child`-returns-`None` crash path, and every `logging.fatal`
  diagnostic suppressed by `basicConfig(level=FATAL+1)`). A fix to
  either in one file would silently miss the other.
- **The repo's own convention says flag, not file.** `cov_update`
  (exact vs Sherman-Morrison), `cov_dtype` (fp32 vs fp64),
  `embeds_center_mode` (fixed vs local) and `revisit_policy` (reuse
  vs regenerate) are all binary algorithm switches living in one
  file. The line the repo actually draws is: **new file when the
  data flow changes** (v01 -> v02 swapped the embedding source from
  policy to PRM — different model, different pipeline); **flag when
  the same pipeline computes something differently**. `cov_scope`
  moves no data; it moves where one matrix is stored.
- **`embeds_ref` made it decisive.** The second knob is orthogonal
  to the first, so two files would mean implementing the relative
  path twice and keeping both in sync; four combinations would mean
  four files.

### The decision had a deadline

`mcts_sem_v02_01` had **zero ledger entries** when it was merged.
That mattered: the method string is baked into result directory
names (`mcts_sem_v02_01--level-5--...`) and manifests, so merging
after the first run would have meant orphaning those results or
writing a migration. Recorded here because the same reasoning
applies to any future variant: **decide file-vs-flag before the
first run, not after.**

## 4. Memory — the real cost of local scope

Under global scope there is exactly one `d x d` matrix per search.
Under local scope there is one per *selected-through* node, bounded
by `gen_budget`. Same `d`, `gen_budget` times the cost.

Measured at the b=320 settings (`embeds_dim=512`, fp64,
`cov_update=sm`, 320 folded nodes):

```
baseline RSS                           928.4 MiB
after 1 question (320 nodes)          1572.9 MiB   (+644.5)
after `del agent`, NO gc.collect()    1572.9 MiB   (+644.5 still held)
after gc.collect()                     929.0 MiB   (+0.6)
```

**`del agent` frees nothing.** Every `MCTSNode` points at its parent
and the parent at its children, so a tree is one large reference
cycle: refcounting cannot reclaim it, and the generational GC
triggers on allocation *counts*, not bytes, so a few hundred
multi-MiB arrays can sit unreclaimed for a long time. Without an
explicit collect, four sequential questions peaked at **2.19x** one
question's worth (~1.4 GiB). `_search` now does `del agent;
gc.collect()` per question, which brings the peak to **1.01x**
(re-measured).

This was harmless under global scope (one 2 MiB matrix per tree),
which is why it went unnoticed for the file's whole prior life.

### The `embeds_dim` guard

`embeds_proj="none"` *forces* `embeds_dim` to the raw PRM hidden
size (4096) — a combination the v02 YAML documents as legitimate
("Set 4096 + embeds_proj=none to feed the raw PRM embeds straight
in, as before"). Under global that costs 128 MiB once. Under local:

| `embeds_dim` | per node | per question @ b=320 |
|---|---|---|
| 512 (shipped) | 2.0 MiB | 0.62 GiB |
| 2048 | 32 MiB | 10.0 GiB |
| 4096 | 128 MiB | **40.0 GiB** |

At 4096, with the GC lag, peak approaches 90 GiB — over the job's
100 GB, and the failure mode is a **silent cgroup OOM kill with no
traceback** (the same failure seen on 2026-07-22 when 21 of 45
concurrent scoring processes vanished). A legal, documented config
would have died with no diagnosable cause.

`MCTS.__init__` now refuses `cov_scope="local"` when the worst-case
footprint exceeds `_LOCAL_COV_MAX_BYTES` (4 GiB), with the
arithmetic in the message. Global scope is unaffected — `d=4096`
still runs there.

## 5. Verification strategy

Four properties needed checking. The interesting question was *how*.

| # | property | how |
|---|---|---|
| 1 | `global` reproduces the pre-merge behavior exactly | scripted-generator trace diff vs the git-extracted old file |
| 2 | `ds_alpha=0` makes local and global identical | scripted-generator trace diff |
| 3 | `V_n^-1 == inv(lam*I + sum u u^T)` over the node's own folds | direct numpy, brute-force inverse |
| 4 | folding at A leaves B untouched | direct numpy |

### Why not a GPU run for #1

The obvious version — run both on a GPU with the same seed and diff
the completions — has a confound: two separate vLLM processes with
the same seed usually generate identical text, but batching and
scheduling introduce their own nondeterminism. A diff could show a
difference that is vLLM's, not the code's, and the test would prove
nothing either way.

### What was done instead

`unittests/check_cov_scope_embeds_ref.py` replaces the generator and
PRM with a **scripted stub**: candidate text, embedding and score are
pure functions of `(node text, index)`. Generation is then
deterministic by construction, so the only thing that can differ
between two runs is the selection logic — exactly the code that
changed. It runs on CPU in seconds.

Crucially this pins **RNG consumption**, not just arithmetic. Tie-breaks
use `random.choice`; one extra draw anywhere and two runs diverge with
completely correct math on both sides. Code review is poor at catching
that; a trace diff is not.

Result: `cov_scope="global", embeds_ref="absolute"` produced
**identical selection traces and identical `gen_cnt`** against the
pre-merge file across 6 configurations (sm/exact x `ds_alpha`
1/10/100 x seeds x depths). 42 checks pass in total, including the
guards and all four `cov_scope` x `embeds_ref` cells being distinct.

A GPU-based confirmation remains available but is now optional
rather than the primary evidence.

## 6. Naming

`embeds_ref: "absolute" | "relative"`. Alternatives considered:

| option | verdict |
|---|---|
| `embeds_ref: absolute \| relative` | **chosen** — conventional word pair; matches the familiar absolute-vs-relative position framing |
| `embeds_ref: absolute \| parent` | built first; names the reference *point*, so extensible for free (`root`, `sibling_mean` slot in as values) but mixes word kinds — `absolute` names a kind, `parent` names a point |
| `embeds_ref: origin \| parent` | fully consistent (both name reference points) but "origin" is odd for L2-normalized vectors on a unit sphere |
| `cov_coords: absolute \| relative` | arguably the best *family* — the knob is read only by `_cov_vec`, never changes `node.embeds`, and pairs verbally with `cov_scope` ("local scope, relative coordinates"). Not chosen; `embeds_ref` was kept for continuity |

The accepted cost of `relative`: it does not say relative to *what*,
so every definition site and every table preamble that mentions it
must pin "parent-relative", and a second reference needs a second
knob rather than a third value.

## 7. Doc organization is independent of code organization

A related question arose: local scope needs its **own section** in
`docs/exp-comp-prm800k-level5.md` (its own `lam`/`ds_alpha` sweeps,
because §1's scaling argument says the global optimum does not
transfer). Does that argue for un-merging the code?

**No.** `--sync-doc` never reads a ledger entry's `group` field;
table matching is entirely by the `tbl-xxxxxx` id in `feeds:`
(see [stable-table-ids.md](stable-table-ids.md)). `_METHOD_TO_GROUP`
feeds only the `--backfill` default, the `--running` table's "family"
column, a `--group` CLI filter, and a display label. So `###`
sections are free-form: local scope gets its own section, its own
tables and its own tuning grid with **zero code changes**.

The level-5 doc now carries `### sem-mcts-v02 [cov_scope=local]`,
whose heading names the flag deliberately — it is the same file, not
a second implementation, and the preamble says so in its first
sentence.

The one cosmetic consequence: local runs show `sem-mcts` in the
occupancy table's family column, same as global ones. Deriving the
group from `(method, cov_scope)` would fix that; not done, since it
is display-only.

## 8. Known-unfixed and follow-ups

- ~~**`select_child` returning `None` crashes the next loop
  iteration**~~ — **fixed 2026-07-28.** The descent now captures the
  result in a temp, backprops the node it was standing on, and ends
  the phase. Two corrections to the original entry: the crash was
  *conditional* (a `gen_cnt >= gen_budget` break on the same
  iteration got there first), and it was **not** reachable via "an
  expansion yields zero candidates" — `generate_k_steps` returns one
  Beam per prompt unconditionally, independent of what vLLM returns,
  and every config pins `batch_size: 4`. The only live route would be
  a future length mismatch in `expand_node`'s three-way `zip`, which
  truncates to the shortest. So the guard is insurance, not a bug
  fix; the trace-identity argument for leaving it alone was sound but
  the cost of the guard is zero (every non-`None` path is
  unchanged). Covered by Part C of
  `unittests/check_cov_scope_embeds_ref.py`, which drives the real
  `mcts_search` with generation stubbed to return zero candidates.
- ~~**`cnt_cov_nodes` is unobservable at runtime**~~ — **fixed
  2026-07-28.** Surfaced per question as `results["q_cov_nodes"]`,
  **gated on `cov_scope="local"`**;
  `core.scoring.build_scored_dataset` auto-attaches any per-question
  list as a dataset column, so no reader changed. Correction to the
  original rationale: schema stability was never at risk for the
  *stats* path — `utils.metrics.evaluate_correctness` returns a fixed
  8-key tuple, so extra dataset columns never reach `_load_trials`.
  The gate is kept for two better reasons: under `global` no node
  ever allocates a covariance, so the column would be all zeros; and
  global runs' scored JSONL stays schema-identical to the runs
  already on disk. Per question this reads the memory multiplier
  directly — peak local covariance bytes ≈ `q_cov_nodes · d² ·
  itemsize` (×2 under `cov_update="exact"`), i.e. it measures what §4
  could only bound.
- **`_cov_read`'s unfolded branch is defensive, not hot.** A node
  only reaches `_select_by_diversity` at `visit_count >= 2`, which
  implies it was already selected through, which implies `_cov_fold`
  already allocated. The branch is kept so the accessor is total.
- **Raw-magnitude `relative`** (no renormalization) is an unbuilt
  variant; see §2.
- **`w_eff ≈ 1` is a prediction, not a measurement.** The level-5
  local sweeps (`tbl-375fa0`, `tbl-898c25`, `tbl-fa65d4`) are the
  test. The `embeds_ref` and model-family tables in that section are
  authored at the predicted point and explicitly gated on the sweeps.
