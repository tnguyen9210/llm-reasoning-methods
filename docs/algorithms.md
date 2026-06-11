# Algorithm registry

Map of active search algorithm variants: which core file implements each,
which launcher runs it, and which config drives it. Algorithm descriptions
live in the module docstrings of the core files — this file is the index,
not the spec. Cross-cutting design decisions: [decisions.md](decisions.md).

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
| CNT-MCTS (PUCT) | `core/mcts_cnt_search_v05_00_00.py` | `generate_mcts_cnt.py` | `conf/mcts_cnt_prm800k.yaml` |
| BL-MCTS v01 (PUCT, best-first) | `core/mcts_bl_cnt_search_v01_00_00.py` | `generate_mcts_bl_cnt_v01.py` | `conf/mcts_bl_cnt_v01_prm800k.yaml` |
| BL-MCTS v02 (KUBE) | `core/mcts_bl_cnt_search_v02_00_00.py` | `generate_mcts_bl_cnt_v02.py` | `conf/mcts_bl_cnt_v02_prm800k.yaml` |
| Semantic-MCTS (embeds) | `core/mcts_embeds_search_v05_00_00.py` | `generate_mcts_embeds.py` | `conf/mcts_embeds_prm800k.yaml` |
| Semantic-MCTS v03 (legacy launcher) | `core/mcts_embeds_search_v03_02_00.py` | `generate_mcts_embeds_v03.py` | `conf/mcts_embeds_v03_prm800k.yaml` |
| BoN | `core/bon_search_v01_0_0.py` | `generate_bon.py` | `conf/bon_prm800k.yaml` (+ gsm8k, aime2025) |
| BoB | `core/bob_search_v03_0_0.py` | `generate_bob_prm800k_v0101.py` | none (params hardcoded in launcher) |

## Lineage

### CNT-MCTS
- `v03_01_00` — baseline (rStar-Math-derived). Superseded; pending
  archive to `core/olds/`.
- `v05_00_00` — reorganized to match `mcts_embeds_search_v05_00_00`
  structure; no behavior changes. Canonical.

### BL-MCTS
Budget-limited best-first MCTS: explicit `leaf_nodes` frontier with
global leaf selection, instead of CNT-MCTS's phase-based root-to-leaf
walks.
- `v01_00_00` — PUCT leaf selection. Active.
- `v02_00_00` — KUBE (fractional knapsack) density-based leaf
  selection; otherwise identical to v01. Active.

Both variants are maintained in parallel for PUCT-vs-KUBE comparison.

### Semantic-MCTS
`v05_00_00` consolidates all earlier files; variant behavior is gated
behind config flags (defaults reproduce `v03_01_00`).
- `v03_01_00` — baseline.
- `v03_02_00` — +mean-centering (`embeds_center`, `embeds_mean`).
- `v03_02_01` — docstring claimed Sherman-Morrison update; not
  implemented (now a real flag in v05: `cov_update`).
- `v03_02_02` — last vs avg pooling (`embeds_strategy`).
- `v03_02_03` — response-only token scope (`embeds_scope`).
- `v04_01_00` — docstring claimed regenerate-on-revisit; not
  implemented (now a real flag in v05: `revisit_policy`).

### Older exploratory files
`core/mcts_search_extra_v*`, `core/diverse_reward_search_v*`,
`core/bon_search_v1.py`, `core/bob_search_v01_*` predate the current
naming scheme; superseded or exploratory, pending archive.
