# Algorithm registry

Map of active search algorithm variants: which core file implements each,
which launcher runs it, and which config drives it. Algorithm descriptions
live in the module docstrings of the core files — this file is the index,
not the spec. Cross-cutting design decisions: [decisions.md](decisions.md).
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
| CNT-MCTS (PUCT) | `core/mcts_cnt_search_v01_00_00.py` | `generate_mcts_cnt.py` | `conf/mcts_cnt_prm800k.yaml` |
| BL-MCTS v01 (PUCT, best-first) | `core/mcts_bl_cnt_search_v01_00_00.py` | `generate_mcts_bl_cnt_v01.py` | `conf/mcts_bl_cnt_v01_prm800k.yaml` |
| BL-MCTS v02 (KUBE) | `core/mcts_bl_cnt_search_v02_00_00.py` | `generate_mcts_bl_cnt_v02.py` | `conf/mcts_bl_cnt_v02_prm800k.yaml` |
| Semantic-MCTS v01 (`mcts_sem_v01`, policy embeds) | `core/mcts_sem_search_v01_00_00.py` | `generate_mcts_sem.py` | `conf/mcts_sem_v01_prm800k.yaml` |
| Semantic-MCTS v02 (`mcts_sem_v02`, PRM embeds) | `core/mcts_sem_search_v02_00_00.py` | `generate_mcts_sem.py` | `conf/mcts_sem_v02_prm800k.yaml` |
| BoN | `core/bon_search_v01_0_0.py` | `generate_bon.py` | `conf/bon_prm800k.yaml` (+ gsm8k, aime2025) |
| BoB | `core/bob_search_v03_0_0.py` | `generate_bob_prm800k_v0101.py` | none (params hardcoded in launcher) |

Archived (pre-rename `mcts_embeds` lineage and old `prm800k_*`
launchers, superseded by the rows above): moved to
`archive/core/`, `archive/generate/`, `archive/conf/`.

## Lineage

### CNT-MCTS
- `v03_01_00` — baseline (rStar-Math-derived). Superseded; archived
  to `archive/core/`.
- `v01_00_00` (renumbered from `v05_00_00`, which had reorganized
  to match the semantic-MCTS file structure with no behavior
  changes) — Canonical.

### BL-MCTS
Budget-limited best-first MCTS: explicit `leaf_nodes` frontier with
global leaf selection, instead of CNT-MCTS's phase-based root-to-leaf
walks.
- `v01_00_00` — PUCT leaf selection. Active.
- `v02_00_00` — KUBE (fractional knapsack) density-based leaf
  selection; otherwise identical to v01. Active.

Both variants are maintained in parallel for PUCT-vs-KUBE comparison.

### Semantic-MCTS
Fresh lineage, renumbered from `v01_00_00` (was `v05_00_00` under the
old `mcts_embeds` numbering). The two active variants differ only in
the SOURCE of the diversity embeddings and are maintained in parallel
for an embedding-source ablation; one launcher (`generate_mcts_sem.py`,
`algo=mcts_sem_v01|v02`) serves both, building the second vLLM pooling
engine only when `search.embeds_source == "policy"`.
- `v01_00_00` — policy embeds. A second vLLM engine (`runner=
  "pooling"`) on the generator supplies per-token hidden states.
  Variant behavior gated behind config flags (defaults reproduce the
  old `v03_01_00` baseline). Canonical baseline.
- `v02_00_00` — PRM embeds (`embeds_source=prm`). Same algorithm and
  the same `embeds_*` knobs as v01, but embeddings come from the PRM's
  last-layer hidden states (folded into the in-loop `prm.score`
  forward pass), so no pooling engine is loaded. *Scaffolded; the
  embedding-source mechanism is not yet implemented — see the module
  docstring.*

Earlier `mcts_embeds_search_v03_*` / `v04_01_00` files are archived,
untouched (not consolidated into the v01 file). The flags that supersede
them live in v01:
- `v03_01_00` — baseline.
- `v03_02_00` — +mean-centering (`embeds_center`, `embeds_mean`).
- `v03_02_01` — docstring claimed Sherman-Morrison update; not
  implemented (now a real flag: `cov_update`).
- `v03_02_02` — last vs avg pooling (`embeds_strategy`).
- `v03_02_03` — response-only token scope (`embeds_scope`).
- `v04_01_00` — docstring claimed regenerate-on-revisit; not
  implemented (now a real flag: `revisit_policy`).

### Older exploratory files
`core/bon_search_v1.py`, `core/bob_search_v01_*` predate the
current naming scheme; superseded or exploratory, pending archive.
`core/diverse_reward_search_v*` and `core/mcts_search_extra_v21/
v61/v72/v73/v81.py` moved to `archive/core/` (`v71` was renamed to
`core/mcts_search_extra.py`, kept in place).
