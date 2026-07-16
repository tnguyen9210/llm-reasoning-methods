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

## Local mean (sibling-group) — **built 2026-07-14/15, v02 only**

`embeds_center_mode: "fixed" | "local"` (`MCTSSemV01Config`; default
`"fixed"` preserves the original behavior — `embeds_center` stays the
on/off master switch, the mode picks which mean when it's on). With
`"local"`, each expansion's candidates are centered on **their own
group's mean** — the mean of the `batch_size` siblings generated
together — recomputed fresh at every expansion and never carried
forward. No precomputed `.npy` is needed; the launcher skips the
`embeds_mean` load in this mode.

**Provenance:** direct transplant of rep_exp's local centering
(Tuyls et al., arXiv 2510.11686 — see
[rep-exp-elliptical-bonus-review.md](rep-exp-elliptical-bonus-review.md),
follow-up #3), which centers each prompt's n=8 GRPO rollouts on the
group mean before the covariance fold. Our group is the exact
structural analog: `bs=4` same-prompt sibling candidates at one MCTS
node.

**Implementation shape:** the group mean needs the whole sibling
batch, so it can't live in per-vector `_extract_embeds`. In local
mode `_extract_embeds` defers BOTH its center and normalize steps;
`_maybe_center_local` (called from `_embed_candidates` once the group
is assembled, both embed sources) subtracts the group mean and then
L2-normalizes — preserving the pipeline invariant that centering
happens in the linear space before the non-linear normalize. Both
functions share one predicate, `_is_local_center(sc)`, so the defer
decision and the centering gate can never silently disagree.
`batch_size=1` edge: the centered vector is exactly 0 → zero bonus,
and the Sherman-Morrison fold of a zero vector is a no-op.
**Ported to bl_sem 2026-07-15** (`mcts_bl_sem_search_v01_00_00`,
`BLMCTSSemConfig`) — same mechanism, `_center_and_normalize` shared
line-for-line with v02. **v01 (`mcts_sem_search_v01_00_00`) still
ignores the flag** — not yet ported.

**Hash handling (reuses the mechanism `cov_dtype` also uses):**
adding any field to the `search` group would change every existing
sem-mcts `config_hash`. `_HASH_EXCLUDE_IF_DEFAULT`
(`utils/configs.py`) drops the field from the identity iff it equals
the pinned neutral value `"fixed"` — so every pre-existing config
hashes exactly as before (verified 2026-07-15: `cfg-c371341f`
recomputes unchanged; full `status.py --group sem-mcts` sweep shows
0 orphans), while `local` runs get a distinct identity (verified:
`cfg-1a54038d`). The pinned value is frozen forever, independent of
the dataclass default.

**Verified (2026-07-15, live smoke test):** `1q/1trial`,
`results_subdir=smoketest`, `embeds_center=true
embeds_center_mode=local`, `WANDB_MODE=offline` — ran end-to-end
(model load → full search → scoring → scored dataset written) with
no crash.

**Coherence caveat (why this is an ablation arm, not a clean
transplant):** rep_exp pairs local centering with a per-group *fresh*
covariance (`persist_covariance=False`); our `V` accumulates across
the whole search. Local centering + accumulated `V` means each
group's vectors enter `V` with a different affine offset — the same
incoherence concern raised for the online mode below. The faithful
transplant (fresh `V` per expansion) would be a separate, larger
change; this mode lets the cheap half be tested first.

## Local mean (sibling-group) in frontier selection — bl_sem caveat

Ported mechanism ≠ ported recommendation. bl_sem
(`mcts_bl_sem_search_v01_00_00`) accumulates `V` by best-first
**global** selection over a flat, ever-growing leaf frontier: each
selection folds in the one winning candidate's embedding, and a
frontier node is expanded at most once by construction (no
per-parent revisits, no tree walk — see the module docstring's
"Differences from mcts_sem_v02"). v02 instead compares a node's
diversity bonus against its own siblings under one parent, walking
the tree, with a node sometimes revisited.

That structural difference matters for local centering specifically.
v02's local mode centers each expansion's `bs=4` siblings on their
own group mean, then compares diversity bonuses *within that same
group* at selection time — the comparison is at least locally
coherent, even though the accumulated `V` mixes offsets across
groups (the caveat already on record above). bl_sem's global
frontier selection instead compares candidates from *different*
groups, centered at *different* points in the run, directly against
each other in one selection step — there is no "local" frame left in
which the comparison is coherent, only the incoherent accumulated-`V`
one. Local mode remains available in bl_sem (ported, verified to
run) but should be read strictly as a v02-parity ablation arm, not a
mode to prefer over fixed centering here.

**Recommendation (2026-07-15):** fixed held-out centering stays the
default for bl_sem, same as v02, and for a stronger reason than
v02's — v02 at least has locally-coherent per-parent comparisons to
fall back on; bl_sem's frontier does not.

**If a coherent adaptive mode is ever wanted for bl_sem:** the
Welford-at-expansion-time shape scoped for v02's online mode (below)
doesn't transfer as-is, because bl_sem's selection is global and
cross-branch rather than scoped to one parent's siblings at a time.
A coherent version would need centering decided **at selection time**
(the mean used to compare frontier candidates would have to be
current as of that global comparison, not fixed at each candidate's
own expansion time), and `V` would need to be rebuilt from raw
embedding history whenever the center moves, rather than folded
incrementally — since a shifting center invalidates the affine basis
every previously-folded vector was expressed in. This is sketched
informally only; no implementation, no config surface, not scoped as
a concrete task.

## Online mean (Welford running update) — **planned, not yet built**

The original 2026-06-18 log entry scoped a second mode: a mode flag
(provisionally named `centering_mode`; **now built as
`embeds_center_mode`** — see the local-mean section above — so online
would be a third value on that existing flag, not a new field), a
per-question mutable Welford `_mean`/`_count` running-average state
living on the `MCTS` instance, an optional running-state argument
threaded through `_extract_embeds` → `_embed_candidates` →
`_generate_candidates`, and a `config_name` tag
(`--center-{fixed|online}`, shown only when `embeds_center` is true —
NOT adopted by the local-mode implementation, which relies on the
config hash alone; revisit if readable dir names start mattering for
centering sweeps).

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
