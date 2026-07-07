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
| BL-MCTS v01 (PUCT, best-first) | `core/mcts_bl_cnt_search_v01_00_00.py` | `generate_mcts_bl_cnt_v01.py` | `conf/mcts_bl_cnt_v01_prm800k.yaml` |
| BL-MCTS v02 (KUBE) | `core/mcts_bl_cnt_search_v02_00_00.py` | `generate_mcts_bl_cnt_v02.py` | `conf/mcts_bl_cnt_v02_prm800k.yaml` |
| BoN | `core/bon_search_v01_0_0.py` | `generate_bon.py` | `conf/bon_prm800k.yaml` (+ gsm8k, aime2025) |
| BoB | `core/bob_search_v03_0_0.py` | `generate_bob_prm800k_v0101.py` | none (params hardcoded in launcher) |

CNT-MCTS routes under two `method=`/`algo=` labels onto the same core
file: `mcts_cnt_v01` (default, current) and `mcts_cnt` (retained only so
older result dirs remain queryable). New experiments always use
`mcts_cnt_v01`.

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
- **v02** — KUBE (fractional-knapsack) density-based leaf selection.

Both are maintained in parallel for a PUCT-vs-KUBE comparison.

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
  term's scale at initialization (~10, from `V_inv=(1/lam)*I`) sits far
  above the PRM score's `[0,1]` range — see
  [findings/coding-findings/ds-alpha-ds-beta-scale.md](findings/coding-findings/ds-alpha-ds-beta-scale.md).
  Empirically, turning the bonus on matters; its magnitude past
  `ds_alpha≈10` does not — see
  [findings/exp-findings/ds-alpha-diversity-bonus-plateau.md](findings/exp-findings/ds-alpha-diversity-bonus-plateau.md).

Both sem-mcts variants (and CNT-MCTS) carry the `\n\n` step-separator
strip-and-reappend guard in candidate-generation templating (some chat
templates trim or crash on a trailing `\n\n`, but the model needs to see
it to continue instead of emitting EOS) — see
[findings/coding-findings/library-version-trajectory-completeness.md](findings/coding-findings/library-version-trajectory-completeness.md).
All PRM scoring relies on `PRM._split_steps` (`core/reward_models.py`)
so `agg_strategy="last"` scoring isn't corrupted by a bogus trailing
empty step from vLLM's `include_stop_str_in_output=True` — see
[findings/coding-findings/prm-step-split-trailing-separator.md](findings/coding-findings/prm-step-split-trailing-separator.md).
