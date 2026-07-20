# Algorithm registry

Map of active search algorithm variants: which core file implements each,
which launcher runs it, and which config drives it. Algorithm descriptions
live in the module docstrings of the core files — this file is the index,
not the spec. Cross-cutting design decisions: [decisions-log.md](decisions-log.md).
Empirical observations about repo behavior: [findings/](findings/README.md).

## Terminology

- **max_depth**: depth limit of the search tree and max reasoning steps
  per trajectory; nodes at this depth are forced terminal.
- **batch_size**: number of next-step candidates generated per generation
  (MCTS expansion). Maps to `config.batch_size`; distinct from SAL's
  `Config.n` and from PRM scoring batches.
- **gen_budget**: total number of generations allowed across the full
  run; each generation produces `batch_size` candidates; the algorithm
  terminates when `gen_budget` is exhausted.
  - *MCTS*: set directly as the primary budget parameter.
  - *BoB*: distributed evenly across depths for fair comparison — each
    depth receives `gen_budget / max_depth` generations.
- **n** (BoN only): number of candidate completions to generate and
  select from. BoN keeps SAL's `config.n = cfg.n`; do not rename to
  `batch_size`.
- **prm_batch_size**: batch size for PRM scoring calls; independent of
  generation `batch_size`.

## Active variants

| Algorithm | Core file | Launcher | Config |
|---|---|---|---|
| CNT-MCTS (PUCT), `method=mcts_cnt_v01` | `core/mcts_cnt_search_v01_00_00.py` | `generate_mcts_cnt.py` | `conf/mcts_cnt_prm800k.yaml` |
| Semantic-MCTS v01 (`mcts_sem_v01`, policy embeds) | `core/mcts_sem_search_v01_00_00.py` | `generate_mcts_sem.py` | `conf/mcts_sem_v01_prm800k.yaml` |
| Semantic-MCTS v02 (`mcts_sem_v02`, PRM embeds) | `core/mcts_sem_search_v02_00_00.py` | `generate_mcts_sem.py` | `conf/mcts_sem_v02_prm800k.yaml` |
| BL-Sem-MCTS v01 (`mcts_bl_sem_v01`, best-first + PRM embeds) | `core/mcts_bl_sem_search_v01_00_00.py` | `generate_mcts_sem.py` | `conf/mcts_bl_sem_v01_prm800k.yaml` |
| BL-MCTS v01 (PUCT, best-first) | `core/mcts_bl_cnt_search_v01_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_cnt_v01_prm800k.yaml` |
| BL-MCTS v02 (`mcts_bl_cnt_v02`, delayed-eager terminal backprop + selectable path-aware score: parent-blend or path-decay) | `core/mcts_bl_cnt_search_v02_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_cnt_v02_prm800k.yaml` |
| BL-KUBE-MCTS v01 (`mcts_bl_kube_v01`, fractional KUBE) | `core/mcts_bl_kube_search_v01_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_kube_v01_prm800k.yaml` |
| BL-KUBE-MCTS v02 (`mcts_bl_kube_v02`, + delayed-eager terminal backprop + selectable path-aware score: parent-blend or path-decay, both schedules) | `core/mcts_bl_kube_search_v02_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_kube_v02_prm800k.yaml` |
| BL-KDEPTH-MCTS v01 (`mcts_bl_kdepth_v01`, knapsack + depth-shaping) | `core/mcts_bl_kdepth_search_v01_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_kdepth_v01_prm800k.yaml` |
| BL-KDEPTH-MCTS v02 (`mcts_bl_kdepth_v02`, + eager terminal backprop only) | `core/mcts_bl_kdepth_search_v02_00_00.py` | `generate_mcts_bl_cnt.py` | `conf/mcts_bl_kdepth_v02_prm800k.yaml` |
| BoN | `core/bon_search_v01_0_0.py` | `generate_bon.py` | `conf/bon_prm800k.yaml` (+ gsm8k, aime2025) |
| BoB | `core/bob_search_v03_0_0.py` | `generate_bob_prm800k_v0101.py` | none (params hardcoded in launcher) |

CNT-MCTS routes under two `method=`/`algo=` labels onto the same core
file: `mcts_cnt_v01` (default, current) and `mcts_cnt` (retained only so
older result dirs remain queryable). New experiments always use
`mcts_cnt_v01`.

BL-MCTS v01, BL-KUBE-MCTS v01, and BL-KDEPTH-MCTS v01 share one
launcher, `generate_mcts_bl_cnt.py` (merged 2026-07-09 — the three
`_search` signatures and launcher bodies were already identical; only
`cfg.algo` differs, same pattern `generate_mcts_sem.py` already uses
for its three variants). Select the variant via `--config-name`
(`conf/mcts_bl_cnt_v01_prm800k.yaml`, `conf/mcts_bl_kube_v01_prm800k.
yaml`, `conf/mcts_bl_kdepth_v01_prm800k.yaml` each set `algo:`
accordingly), same as sem's `algo=mcts_sem_v01|v02|mcts_bl_sem_v01`.
The old per-variant launchers (`generate_mcts_bl_cnt_v01.py` etc.) no
longer exist; historical `experiments.yaml` entries recorded under
the old launcher names are left as-is (they record what was actually
run, not a forward-looking pointer). The KUBE and depth-shaping
variants are each their own algorithm family (`mcts_bl_kube_v01`,
`mcts_bl_kdepth_v01`), independent of the launcher-merge history
above.

## CNT-MCTS

Phase-based root-to-leaf walks: each of `num_phases` outer iterations
descends from the root via PUCT selection, expanding the first
unexpanded node it reaches. Only expansions charge against
`gen_budget`.

## BL-MCTS

Budget-limited best-first MCTS: an explicit `leaf_nodes` frontier with
global leaf selection, instead of CNT-MCTS's phase-based root-to-leaf
walks.
- **v01** — PUCT leaf selection.
- **v02** (`mcts_bl_cnt_v02`) — v01 plus two changes:
  (1) terminal candidates are queued at creation and backpropped
  right after the immediately-following selection resolves —
  delayed one step, not instantaneous, so a terminal never
  influences the selection choosing among its own same-batch
  siblings (see
  [decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md)
  §1.5) — instead of waiting to win a later frontier selection as
  v01 does; (2) a **selectable path-aware frontier score**
  (`score_mode`, added 2026-07-19): `"parent_blend"` (default)
  blends PUCT's `q` term with the leaf's immediate parent's
  `q_value` by `alpha` before adding the unchanged UCB1 exploration
  term (`alpha=1.0` recovers v01 exactly — the only exact-v01
  control arm); `"path_decay"` replaces the value term with a
  `gamma`-decayed average of `q_value` along the full leaf-to-root
  path, paired with an AlphaZero-shaped exploration term
  (`cpuct*sqrt(N_parent)/(1+N_leaf)` — so `cpuct` is **not
  comparable across modes**; sweep it per mode). The reason for
  value-aware scoring in either mode: backprop alone doesn't change
  v01's ranking (PUCT never reads a parent's `q`, only its own
  frozen `q` plus the parent's visit count — a backpropagated value
  is otherwise write-only). The two modes are arms of one planned
  sweep; the loser is expected to be deleted afterward (a pure
  deletion — the scorers share no code, joined only by the
  `MCTS.frontier_score` dispatcher). Config:
  `utils/configs.py::BLMCTSCntV02Config` (`score_mode`, `alpha`,
  `gamma`). See `docs/decisions-log.md` (2026-07-19) and
  [decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md).

BL-MCTS v01/v02, BL-KUBE-MCTS v01/v02, and BL-KDEPTH-MCTS v01/v02 are
maintained in parallel for a PUCT / evidence-based-UCB /
fixed-depth-shaping comparison at matched gen_budget, even though
BL-KUBE-MCTS and BL-KDEPTH-MCTS now each live in their own algorithm
family (see below) rather than as same-family sibling versions.

## BL-KUBE-MCTS

Its own algorithm family, `mcts_bl_kube_v01` — a distinct selection
criterion from BL-MCTS's PUCT, not a same-family variant of it.

Same budget-limited best-first frontier skeleton as BL-MCTS (an
explicit `leaf_nodes` frontier with global leaf selection), but with
PUCT replaced by Fractional KUBE density-based leaf selection,
following Tran-Thanh et al. arXiv:1204.1909 sec. 3.3 (reference
implementation: the sibling `budget-mab` repo's
`src/algorithms.py::FractionalKUBE`).
`density(x) = (q_value(x) + bonus(x)) / cost(x)`, `cost(x) =
max_depth - depth(x)` (the MCTS analogue of an arm's fixed pull
price). The bonus clock is configurable via `kube_schedule`:
`"parent"` (default) uses `kube_c*sqrt(log(parent_visits)/visits)` —
UCT-style local clock, identical to BL-MCTS v01's PUCT bonus, so this
differs from BL-MCTS v01 only by the cost division (single-factor
ablation); `"global"` uses `kube_c*sqrt(log(1+t)/visits)` with `t` =
frontier selections so far — faithful to KUBE's flat-bandit clock,
but since frontier nodes keep `visits == 1` for life it is a
frontier-wide constant with no per-node discrimination (kept as an
ablation arm). Selection mirrors KUBE's feasibility step
(`kube_affordable`, default true): the argmax is restricted to nodes
whose cost fits the remaining generation budget (terminal nodes
always eligible; empty set relaxes to the full frontier). Config:
`utils/configs.py::BLMCTSKubeV01Config` (`kube_c`, default 2.0;
`kube_schedule`, default `parent`; `kube_affordable`, default true).
Rewritten 2026-07-09 to match budget-mab's actual FractionalKUBE (a
UCB index over cost) — an earlier version used a static depth-decay
bonus with no UCB/visit-count term at all; see `docs/decisions-log.md`
(2026-07-09, three entries).

**v02** (`mcts_bl_kube_v02`) — v01 plus terminal-split, delayed-eager
backprop (queued at creation, flushed right after the
immediately-following selection resolves — not instantaneous, so a
terminal never leaks into the selection ranking its own same-batch
siblings; both `kube_schedule` values: fixes a real defect where a
max-depth dead-end's `cost≤0`/`-inf` density meant it was
*permanently* stuck in the frontier, silently disabling the
`kube_affordable` filter's empty-set fallback for the rest of any
run) and the same **selectable path-aware score** BL-MCTS v02 has
(`score_mode`, aligned 2026-07-19), under **both** `kube_schedule`
values: `"parent_blend"` (default) blends the leaf's `q` with its
parent's by `alpha` before the schedule's unchanged log-form bonus
(`alpha=1.0` = exact v01, the only v01 control arm; originally
shipped `"parent"`-schedule-only, reversed same day 2026-07-18 —
under `"global"` the blend is the *only* ancestor channel, since
the shared-clock bonus cannot count-burst); `"path_decay"` replaces
the value term with a `gamma`-decayed leaf-to-root average and the
bonus with the AlphaZero shape `kube_c·sqrt(clock)/(1+N)` — the
schedule's clock substituted as always (`N_parent` / `1+t`), making
kube-parent path_decay exactly BL-MCTS v02's `path_decay_score`
divided by `cost`. `kube_c` is **not comparable across modes**. The
sweep's losing mode is slated for deletion (scorers share no code;
single `frontier_score` dispatcher). Config:
`utils/configs.py::BLMCTSKubeV02Config` (`score_mode`, `alpha`,
`gamma`). See
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md)
§3.5 and `docs/decisions-log.md` (2026-07-19).

## BL-KDEPTH-MCTS

Its own algorithm family, `mcts_bl_kdepth_v01` — "kdepth" = knapsack
cost normalization + deterministic depth-shaping. Same reasoning as
BL-KUBE-MCTS: it has no visit-count term of any kind, a deliberately
different theoretical basis from BL-MCTS/BL-KUBE-MCTS rather than a
refinement of either.

Same budget-limited best-first frontier skeleton, cost mapping
(`max_depth - depth`), and `kube_affordable` feasibility step as
BL-KUBE-MCTS, but the UCB confidence bonus is replaced with a fixed
depth-preference function:
`density(x) = (q_value(x) + depth_beta*f_a(depth_frac(x))) / cost(x)`,
`depth_frac(x) = depth(x)/max_depth`, `f_a(z) = 1 - z**depth_alpha`
(1 at the root, 0 at max depth — monotonically favors shallower
nodes). No visit-count/parent-visit/global-clock term at all, so no
confidence-bound/regret guarantee — a deliberately different
theoretical basis from BL-KUBE-MCTS, not a refinement of it. Config:
`utils/configs.py::BLMCTSKdepthV01Config` (`depth_beta`, default 2.0;
`depth_alpha`, default 1.0; `kube_affordable`, default true). See
[decisions/bl-kdepth-knapsack-bonus.md](decisions/bl-kdepth-knapsack-bonus.md).

**v02** (`mcts_bl_kdepth_v02`) — v01 plus the same terminal-split,
delayed-eager backprop queue structure as BL-KUBE-MCTS v02 (identical
`kube_affordable` defect, identical fix), and **nothing else**:
`depth_density()` is byte-identical to v01's. The delayed-vs-immediate
flush timing that matters for BL-MCTS/BL-KUBE-MCTS's `"parent"`
schedule is inert here, kept only for structural consistency — no
formula change and no `alpha` knob, because `depth_density` reads
only a leaf's own frozen `q_value`, its own depth, and two constants
— no visit-count or parent-`q` channel exists at all for a blend to
hook into. Ranking among non-terminal nodes is provably
unchanged; the only removed events are terminal-selection phases that
did nothing but re-select an already-known dead-end. Config:
`utils/configs.py::BLMCTSKdepthV02Config` (same fields as v01, no new
ones). See
[decisions/bl-cnt-v02-eager-backprop-path-aware.md](decisions/bl-cnt-v02-eager-backprop-path-aware.md).

## Semantic-MCTS

Adds an embedding-based diversity bonus to child selection: `q_val =
ds_beta*q_value + ds_alpha*sqrt(x^T V^-1 x)`, where the second term
grows for candidates whose pooled embedding points in a direction the
running covariance `V` has seen little of. One launcher
(`generate_mcts_sem.py`, `algo=mcts_sem_v01|v02`) serves both variants;
they differ only in the SOURCE of the diversity embeddings:

- **v01** (`embeds_source=policy`) — a second vLLM engine (`runner=
  "pooling"`) on the generator supplies per-token hidden states.
- **v02** (`embeds_source=prm`, the config default) — embeddings come
  from a dedicated `prm.embed(...)` forward pass over the plain
  candidate chat (last-layer hidden states by default, `prm_embeds_layer`
  selects others), separate from the judge-transcript pass
  `prm.score(...)` runs. Both reuse the already-loaded PRM, so no
  second engine is needed.

Both variants pool a candidate's embedding through the same
`_extract_embeds` pipeline (scope → pool → project → center →
normalize), so the v01-vs-v02 comparison isolates the embedding model
alone. Config flags (`utils/configs.py::MCTSSemV01Config` /
`MCTSSemV02Config`):

- `embeds_strategy`: `"last"` | `"avg"` — pooling over the scoped tokens.
- `embeds_scope`: `"full"` | `"response"` — which tokens are pooled.
  `"response"` is unimplemented for `embeds_source="prm"` (the
  generator-tokenizer `response_start_idx` doesn't apply to the PRM's
  own tokenization) — see `docs/decisions-log.md` (2026-07-07).
- `embeds_center` / `embeds_mean`: optional held-out mean subtraction.
- `cov_update`: `"exact"` | `"sm"` (Sherman-Morrison) — how `V^-1` is
  maintained across selections; `"sm"` is the default fast path.
- `revisit_policy`: `"reuse"` | `"regenerate"` — whether a revisited
  node's candidates are cached or regenerated.
- `ds_alpha` / `ds_beta`: diversity-bonus weight vs. q-value weight.
  `ds_alpha` needs to be roughly 100x `ds_beta` since the diversity
  term's scale at initialization (`1/sqrt(lam)` ≈10 at the default
  `lam=0.01`) sits far above the PRM score's `[0,1]` range, and `lam`
  itself is coupled to `ds_alpha` (not an independent knob) — see
  [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md).
  Empirically, turning the bonus on matters; its magnitude past
  `ds_alpha≈10` does not — see
  [findings/exp-findings/ds-alpha-diversity-bonus-plateau.md](findings/exp-findings/ds-alpha-diversity-bonus-plateau.md).

## BL-Sem-MCTS

Frontier counterpart of Semantic-MCTS v02, exactly as BL-MCTS v01 is
to CNT-MCTS: an explicit `leaf_nodes` frontier with global best-first
selection, where the sem family's diversity-adjusted value replaces
BL-MCTS's PUCT:

    q_val = ds_beta*q + ds_alpha*sched*sqrt(x^T V^-1 x)

computed fresh over the whole frontier each iteration; the selected
leaf's embedding is folded into `V` (rank-1) on every selection. Runs
from the same launcher (`generate_mcts_sem.py`,
`algo=mcts_bl_sem_v01`); embeddings default to the PRM source.

`sched` is the `ds_alpha_schedule` knob (`BLMCTSSemConfig`), exposed
because on a global frontier the schedule is a real design axis
(sem_v02 hardcodes the per-parent form):

- `global` (default) — `sqrt(log(1+t))`, `t` = frontier selections so
  far. The frontier is a flat arm set and `sqrt(x^T V^-1 x)` is the
  LinUCB confidence width, so the global clock is the OFUL-standard
  schedule; the multiplier is shared across the frontier, so per-node
  differentiation comes only from `q` and the `V^-1` geometry.
- `parent` — `sqrt(log(1+parent_visits))` per node, the literal
  sem_v02 transplant (tree-position-dependent scales).
- `none` — constant `ds_alpha`.

Differences from sem_v02 beyond the frontier: no first-visit q-only
special case (per-parent concept, no global analog) and no
`revisit_policy` (frontier nodes expand at most once). See the module
docstring and `docs/decisions-log.md` (2026-07-08).

Both sem-mcts variants (and CNT-MCTS) carry the `\n\n` step-separator
strip-and-reappend guard in candidate-generation templating (some chat
templates trim or crash on a trailing `\n\n`, but the model needs to see
it to continue instead of emitting EOS) — see
[findings/coding-findings/library-version-trajectory-completeness.md](findings/coding-findings/library-version-trajectory-completeness.md).
All PRM scoring relies on `PRM._split_steps` (`core/reward_models.py`)
so `agg_strategy="last"` scoring isn't corrupted by a bogus trailing
empty step from vLLM's `include_stop_str_in_output=True` — see
[findings/coding-findings/prm-step-split-trailing-separator.md](findings/coding-findings/prm-step-split-trailing-separator.md).
