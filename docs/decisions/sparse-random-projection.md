# Sparse random projection shrinks the PRM-embedding covariance, with a fixed matrix

*Decided 2026-06-18 —
[decisions-log.md #2026-06-18](../decisions-log.md#2026-06-18--search-sparse-random-projection-of-prm-embeds-fixed-matrix-poolprojectcenternormalize)*

## What

`embeds_proj: "none" | "sparse"` lives on `MCTSSemV01Config`
(`utils/configs.py`, inherited by v02) and is read in
`_extract_embeds` (`core/mcts_sem_search_v02_00_00.py`). The real
config (`conf/search/mcts_sem_v02.yaml`) runs `embeds_proj: sparse`,
`embeds_dim: 512`. `embeds_dim` means the size of the diversity
covariance `V` — with projection on, it's the *post*-projection
dimension; the *raw* pooled dimension (4096 for Llama3.1-8B-PRM,
3584 for the Qwen-Math-7B PRM) is read off the pooled tensor at
runtime, not configured.

When `embeds_proj == "sparse"`, the pooled embedding is projected via
sklearn's `SparseRandomProjection(density="auto")` (JL-optimal
sparsity `1/√d`) to `embeds_dim`. The projection matrix is **fixed for
the whole run** — built once via `_get_sparse_projector(in_dim,
out_dim, seed)` and cached in a module-level dict
(`_PROJECTOR_CACHE`) keyed by `(in_dim, out_dim, seed)`. The seed is
**not** a config knob: it's the hardcoded internal constant
`_PROJ_SEED = 0`. Empirically verified (this session, 2026-07-07):
`SparseRandomProjection` with a fixed `random_state` and a
data-independent `fit()` call produces byte-identical component
matrices across separate construction calls — so a resumed run
rebuilds the exact same projector deterministically.

`_extract_embeds`'s five-step pipeline order is **scope → pool →
project → center → normalize**. Projection happens on the numpy side,
after pooling and before any optional centering; the centering
mean's shape is checked against the *projected* dim, with a guard
that raises if `embeds_mean`'s dim doesn't match — so a raw-space mean
can't silently be subtracted from a projected vector.

## Why fix the matrix, not let it drift or adapt

The diversity covariance is `V = λI + Σ uuᵀ`, accumulated across
selections over the whole run. This accumulation is only meaningful
if every `u` folded into `V` lives in the *same* feature space —
`V⁻¹` needs a stable basis to mean anything as "which directions have
been seen." A projection matrix that changed mid-run (recomputed,
refit online, or seeded differently per resume) would put earlier and
later embeddings in different bases, making the accumulated `V⁻¹`
incoherent — comparing apples-and-oranges vectors as if they were the
same kind of thing. This is also the reason an *online*-adapted
projection or online-updated centering mean is rejected as a design:
correctness here requires a fixed linear map, not a better-fitting one.

A **random** (not learned/data-adaptive, e.g. not PCA) projection is
the right kind of fixed map because the Johnson-Lindenstrauss lemma
guarantees it preserves pairwise distances (w.h.p.) for *any* fixed
seed, uniformly over all possible input vectors — it doesn't need to
see the data to be a good map. That's why the seed choice is
empirically irrelevant and can be safely hardcoded rather than
exposed as a tunable: any seed gives the same near-isometry guarantee,
so pinning one internally (rather than making it a config knob) loses
nothing while guaranteeing a resumed run reconstructs the identical
matrix without needing to persist it to disk.

The **pool → project → center → normalize** ordering is forced by
what's linear vs. non-linear: the projection is (approximately)
linear, so it composes cleanly with the mean subtraction that follows
— the mean must be subtracted in the same (projected) linear space it
was computed in, not before projection. Normalization is inherently
non-linear (a projected-then-centered vector's norm isn't the sum of
its parts' norms), so it has to be the last step regardless of
whether centering is on.

## Verified

At the time of the decision (2026-06-18): the projector's fixed/seeded
behavior, JL distance-ratio preservation (~1.00±0.03), and every
pool/project/center/normalize path plus its shape guards were unit
tested. A GPU end-to-end run over real 4096-dim PRM embeddings with
`V` sized to 512 completed cleanly. A notable side effect: the
projected run measured **~2.5× faster** than the unprojected v02
baseline (~147 vs ~362 s/question) — `_diverse_select`'s per-selection
matrix inverse is O(d³), and d dropped 8× (4096→512), while the PRM
forward-pass cost is unchanged; the entire speedup is in the
covariance arithmetic, not generation or scoring.

Re-verified this session (2026-07-07): the fixed-seed determinism
claim holds under direct construction-vs-construction comparison
(`SparseRandomProjection(random_state=0).fit(zeros)` run twice
produces `np.array_equal` component matrices).

## Revisit if

The projected dimension (512) proves too small to preserve the
diversity signal — either raise it, or set `embeds_proj=none` with
`embeds_dim` matching the raw PRM hidden size (4096 for Llama-PRM,
3584 for Qwen-Math-PRM) to feed raw, unprojected embeddings. A
data-adaptive subspace (e.g. PCA fit on PRM embeddings) is a
different, separate experiment — not a mutation of this fixed random
matrix, since a learned subspace reintroduces exactly the
basis-drift-across-time risk the fixed random map was chosen to avoid.
