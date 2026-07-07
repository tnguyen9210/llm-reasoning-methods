# Embedding centering: design and status by mode

Umbrella doc for `embeds_center`-gated mean-centering in sem-mcts's
diversity embeddings (`_extract_embeds`,
`core/mcts_sem_search_v02_00_00.py`) — one section per mode, covering
what's built, what's planned, and the reasoning behind each. Update
this file (add a mode section, update a status line) rather than
spinning up a new decisions file per mode.

Originating log entry:
[decisions-log.md #2026-06-18](../decisions-log.md#2026-06-18--search-configs-online-vs-fixed-centering-mean-is-a-flag-not-a-version).

## Mechanism shared by every mode

`MCTSSemV01Config` (`utils/configs.py`, inherited by v02):
`embeds_center: bool = False` is the on/off master switch, independent
of which mean is used when it's on. `_extract_embeds` subtracts
whatever mean is active from the pooled-and-projected embedding, with
a shape guard that raises if the mean's dimension doesn't match the
(post-projection) embedding dimension — so a raw-space mean can never
silently be subtracted from a projected vector, regardless of mode.

## Fixed mean — **built, current default**

`embeds_mean_dir: str = ""` names a `results/`-relative path prefix
for a precomputed `.npy` file. `embeds_mean: Optional[Any] = None` is
**not set from YAML** — it's populated once, at startup, by the
launcher (`generate_mcts_sem.py`):
`cfg.search.embeds_mean = np.load(f"{root_dir}/results/
{cfg.search.embeds_mean_dir}.npy").flatten()`. This single fixed
vector is then subtracted from every embedding for the entire run.

**Why fixed, not adaptive:** the diversity covariance `V = λI + Σ uuᵀ`
accumulates across the whole run, and is only coherent if every folded-in
vector lives in the same feature space. A mean that changed mid-run
would put earlier and later embeddings in different bases, making the
accumulated `V⁻¹` meaningless — the same correctness argument that
fixes the sparse-projection matrix
([decisions/sparse-random-projection.md](sparse-random-projection.md)).
A held-out, precomputed mean sidesteps this entirely: it's decided
before the run starts and never moves.

**Verified (2026-07-07):** confirmed live via `grep` — `embeds_center`,
`embeds_mean_dir`, `embeds_mean` all present and wired exactly as
above; this is the only mode with a working implementation today.

## Online mean (Welford running update) — **planned, not yet built**

The original 2026-06-18 log entry scoped a second mode: a
`centering_mode: "fixed" | "online"` flag (with `embeds_center` staying
the on/off switch and `centering_mode` choosing which mean when it's
on), a per-question mutable Welford `_mean`/`_count` running-average
state living on the `MCTS` instance, an optional running-state argument
threaded through `_extract_embeds` → `_embed_candidates` →
`_generate_candidates`, and a `config_name` tag
(`--center-{fixed|online}`, shown only when `embeds_center` is true).

**Status as of 2026-07-07:** none of this exists in the codebase yet —
confirmed via `grep` across `utils/configs.py` and both sem-mcts core
files (`mcts_sem_search_v01_00_00.py`, `mcts_sem_search_v02_00_00.py`):
no `centering_mode` field, no Welford state, no historical trace in
tracked files either (not a removed feature, a not-yet-built one).
This is tracked here as a planned addition, per 2026-07-07 discussion —
implement when prioritized.

**Why it's worth building (the original ablation rationale):** the
online mode is deliberately intended as the *theoretically-unsound*
comparison arm, not a hoped-for improvement — a drifting μ_t (updated
with each new embedding, mid-run) makes the feature map non-stationary
and the accumulated covariance `V` incoherent, exactly the failure mode
the fixed mean is designed to avoid. Building it lets that theoretical
argument be checked empirically (does fixed-mean centering actually
outperform online, or does it not matter in practice) rather than
resting on the argument alone. Until it's built, the fixed-mean choice
is unverified by experiment.

**Implementation shape, when built:** as scoped above — a
`centering_mode` flag, Welford state on `MCTS`, threaded through the
embed-candidates call chain, gated `config_name` tag. No new `vNN` file
needed (same lineage/algorithm/embedding source, only how μ is produced
differs — the two-tier convention in `docs/algorithms.md` reserves a
config flag for exactly this shape of variant, not a new version).

## Adding a future mode

Add a new `##` section here (mode name, built/planned status, current
config surface if any, why it's worth having, implementation shape).
Keep the "mechanism shared by every mode" section as the one place the
common `embeds_center`/shape-guard behavior is described, so per-mode
sections only need to cover what's specific to them.
