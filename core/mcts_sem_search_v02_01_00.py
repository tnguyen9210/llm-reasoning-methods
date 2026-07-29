"""
Semantic MCTS v02.01: per-node (local) diversity covariance.

The "mcts_sem_v02_01" method. Identical to mcts_sem_v02
(mcts_sem_search_v02_00_00) in every respect — PRM-sourced
embeddings, the same pooling / projection / centering / normalize
pipeline, the same q-value machinery, the same budget accounting —
except for WHERE the covariance `V` lives:

    v02        one V for the whole search tree. Every selection
               anywhere in the tree folds its chosen child's
               embedding into that single V, and every diversity
               bonus is read off the same V^-1.

    v02.01     one V per node. Selecting among node `n`'s children
               reads `n`'s own V_n^-1, and folds the chosen child's
               embedding back into `n` alone. Sibling subtrees never
               see each other's folds.

So the diversity term at node `n` answers "which child points
somewhere `n` has not committed to yet?" instead of "which child
points somewhere the entire search has not visited yet".

Why this variant exists
    1. Coherence with local centering. `embeds_center_mode="local"`
       (built 2026-07-14) centers each expansion batch on its own
       sibling mean, but v02 then folds those group-centered vectors
       into a globally accumulated V — vectors carrying
       group-dependent offsets pile into one covariance. The caveat
       is recorded in docs/decisions/embeds-centering-design.md, and
       docs/decisions/rep-exp-elliptical-bonus-review.md notes that
       rep_exp pairs local centering with a per-group FRESH
       covariance, which our accumulated V deliberately is not.
       local centering + local V is that missing pairing.
    2. Coherence with the alpha schedule. The diversity weight is
       already scaled by sqrt(log(1 + parent_visits)) — a per-node
       clock. docs/decisions/global-vs-local-exploration-schedule.md
       argues a node-indexed multiplier against a globally
       accumulated V mixes two clocks; with V_n local, the local
       clock is the matching one.
    3. Bonus dynamic range. With L2-normalized embeddings and
       V = lam*I + sum u u^T, a direction already covered k times
       scores ~1/sqrt(lam + k). Globally k grows with total
       selections across the run (thousands), so late selections see
       a compressed bonus everywhere; locally k is the node's own
       visit count (tens), so `ds_alpha` means roughly the same
       thing at the root and at depth 15.

Config flags
    Every mcts_sem_v02 flag, unchanged, plus:

    cov_scope : "local" | "global"                  (default: "local")
        "local"  — per-node V as described above.
        "global" — one tree-level V; behaviorally IDENTICAL to
                   mcts_sem_v02. Kept in this file on purpose: it is
                   the verification lever. Any v02-vs-v02.01
                   difference observed with cov_scope="global" is a
                   porting bug, not the ablation (see
                   "Verifying against v02" below).

    cov_update ("exact" | "sm") and cov_dtype ("fp32" | "fp64") keep
    their v02 meanings and apply per node under local scope.

Cost, relative to v02
    Compute: unchanged per selection — one O(d^2) Sherman-Morrison
    update (or one O(d^3) re-inversion under "exact"), just applied
    to a smaller matrix set. Locality does not add work; it
    partitions the same work.

    Memory: this is the real cost. Each node that is ever selected
    through allocates its own d x d matrix (two, under "exact").
    Only nodes with children allocate, so the count is bounded by
    the number of expansions, i.e. by `gen_budget`. At d=512 / fp64
    that is 2 MiB per node, so a gen_budget=320 question can hold
    ~640 MiB of covariance at peak (~320 MiB under cov_dtype=fp32),
    freed when the per-question agent is dropped. See
    `cnt_cov_nodes` on the agent for the measured count.

Verifying against v02 (what the tests should assert)
    1. cov_scope="global" must reproduce mcts_sem_v02 exactly —
       same completions, same gen_cnt — for any seed.
    2. ds_alpha=0 must make cov_scope="local" and "global" identical:
       the covariance is multiplied by zero, so the two can only
       diverge through a bug outside the bonus.
    3. max_depth=1 must make "local" and "global" identical: every
       phase performs exactly one selection, always at the root, so
       the root's local V and the tree's global V receive the same
       folds in the same order.
    4. Algebraically, V_n^-1 must equal inv(lam*I + sum_j u_j u_j^T)
       over exactly the embeddings of the children selected at n —
       and folding at node A must leave node B's V untouched (the
       defining property of locality).
    Beyond those, "local" and "global" are EXPECTED to diverge; the
    comparison is empirical, not an equivalence.

Variant lineage: docs/algorithms.md.
"""

import random
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Optional, Any, Type, Dict, List

import numpy as np
import torch
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from sklearn.random_projection import SparseRandomProjection

from vllm import SamplingParams

from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv, generate_k_steps


logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)

# config.search.cov_dtype -> numpy dtype for V/V_inv and everything
# multiplied against them (see MCTS.__init__, select_child,
# _diverse_select).
_COV_DTYPES = {"fp32": np.float32, "fp64": np.float64}

# config.search.cov_scope -> where the covariance lives.
_COV_SCOPES = ("local", "global")


# --------------------------------------------------------------------- #
# Diversity selection                                                   #
# --------------------------------------------------------------------- #

def _diverse_select(
    V_inv, q_embeds, q_scores, ds_alpha, ds_beta, cov_dtype=np.float64,
):
    """Pick the one arm maximizing `beta*score + alpha*diversity`.

    Unchanged from v02 — a pure decision function. It neither knows
    nor cares whether the `V_inv` handed to it is the tree's or one
    node's; that choice is entirely `MCTS._cov_read`'s.

    The diversity term for arm `x` is `sqrt(x^T V^-1 x)` — large when
    `x` points in a direction V has seen little of, small when V has
    accumulated many similar vectors.

    `cov_dtype` (config.search.cov_dtype) fixes the precision
    `q_embeds` is cast to before multiplying against `V_inv`, so the
    einsum below runs at a controlled precision rather than whatever
    NumPy's mixed-dtype promotion happens to pick.

    Ties (within `tol`) are broken by uniform random sampling, which
    avoids the systematic bias of picking the first argmax.
    """
    q_embeds = np.asarray(q_embeds, dtype=cov_dtype)
    q_scores = np.asarray(q_scores)
    tol = 1e-4

    # Diversity bonus per arm: sqrt(x^T V^-1 x). Implemented as a
    # single einsum to avoid a Python-level loop over arms.
    q_diversity = np.sqrt(
        np.einsum('ij,jk,ik->i', q_embeds, V_inv, q_embeds)
    )
    q_vals = ds_beta * q_scores + ds_alpha * q_diversity

    # Argmax with uniform random tie-breaking (within tol).
    max_val = q_vals.max()
    candidates = [
        i for i, v in enumerate(q_vals) if abs(max_val - v) <= tol
    ]
    best_idx = random.choice(candidates)

    logging.fatal(f"q_diversity = {q_diversity}")
    logging.fatal(f"alpha*q_diversity = {ds_alpha * q_diversity}")
    logging.fatal(f"beta*q_values = {ds_beta * q_scores}")
    logging.fatal(f"vals = {q_vals}")
    logging.fatal(f"candidate_idxes = {[i + 1 for i in candidates]}")
    logging.fatal(f"best_idx = {best_idx + 1}")

    return best_idx


# --------------------------------------------------------------------- #
# Embedding extraction                                                  #
# --------------------------------------------------------------------- #

# Fixed seed for the sparse-projection matrix. Not a config knob: JL
# holds w.h.p. for any seed, so the choice doesn't matter empirically;
# it's pinned only so a resumed run rebuilds the IDENTICAL matrix (a
# drifting map would put past/present vectors in different bases and
# make V^-1 meaningless). Hardcoded so it can't accidentally vary.
_PROJ_SEED = 0

# Fixed sparse-projection matrices, one per (in_dim, out_dim, seed).
# The matrix MUST stay fixed for a whole run (see _PROJ_SEED). Caching
# here makes "one fixed matrix per (dim, seed) in the process"
# automatic.
_PROJECTOR_CACHE: Dict[tuple, np.ndarray] = {}


def _get_sparse_projector(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    """Return a fixed (in_dim, out_dim) sparse-random-projection matrix.

    Built with sklearn's SparseRandomProjection (Achlioptas / JL):
    density="auto" gives the JL-optimal sparsity 1/sqrt(in_dim). The
    fit is data-INDEPENDENT — it reads only the input dim from the
    dummy array to size the components — so a zeros row suffices. We
    store the matrix densely as (in_dim, out_dim) float32 (a 4096x512
    matmul is trivial) so a pooled (in_dim,) vector projects by
    `pooled @ R`.
    """
    key = (in_dim, out_dim, seed)
    cached = _PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    proj = SparseRandomProjection(
        n_components=out_dim, density="auto", random_state=seed
    )
    # fit() only needs the input dimensionality; values are unused.
    proj.fit(np.zeros((1, in_dim), dtype=np.float32))
    # components_ is (out_dim, in_dim), possibly sparse; densify and
    # transpose to (in_dim, out_dim) for a right-multiply.
    components = proj.components_
    if hasattr(components, "toarray"):
        components = components.toarray()
    R = np.asarray(components, dtype=np.float32).T
    _PROJECTOR_CACHE[key] = R
    return R


def _extract_embeds(raw, config, response_start_idx):
    """Turn per-token embeddings into a single pooled vector.

    Unchanged from v02. `raw` is a (seq_len, dim) tensor of per-token
    hidden states; in v02.01 it always comes from the PRM
    (`prm.embed(...)`, one row per candidate).

    Three steps, in this order:
      1. scope     pick which tokens contribute (full sequence or only
                   the assistant response).
      2. pool      reduce to (dim,): last token or mean over scope.
      3. project   optional sparse random projection to embeds_dim
                   (JL near-isometry; fixed matrix per run). Linear, so
                   it composes cleanly with the centering that follows.

    Centering (fixed-mean OR local-group-mean) and the final L2
    normalize are NOT done here — both need to see the whole sibling
    batch at once, so they're `_center_and_normalize`'s job, called
    once per expansion from `_embed_candidates`.
    """
    sc = config.search
    # 1. Scope.
    if sc.embeds_scope == "response":
        raw = raw[response_start_idx:, :]
    elif sc.embeds_scope != "full":
        raise ValueError(f"unknown embeds_scope: {sc.embeds_scope!r}")

    # 2. Pool.
    if sc.embeds_strategy == "last":
        pooled = raw[-1]
    elif sc.embeds_strategy == "avg":
        pooled = raw.mean(dim=0)
    else:
        raise ValueError(
            f"unknown embeds_strategy: {sc.embeds_strategy!r}"
        )

    pooled = pooled.detach().cpu().float().numpy()

    # 3. Project (optional). Shrinks the raw pooled dim to embeds_dim
    # via a fixed JL sparse matrix; the raw dim is read off the vector.
    embeds_proj = getattr(sc, "embeds_proj", "none")
    if embeds_proj == "sparse":
        R = _get_sparse_projector(
            pooled.shape[-1], sc.embeds_dim, _PROJ_SEED
        )
        pooled = pooled @ R
    elif embeds_proj != "none":
        raise ValueError(f"unknown embeds_proj: {embeds_proj!r}")

    return pooled


def _center_and_normalize(embeds, sc):
    """Center (optional, mode-dependent) then L2-normalize (optional)
    an expansion's whole sibling batch of pooled/projected embeddings.

    Unchanged from v02, but note that under `cov_scope="local"` the
    coherence caveat v02 carries here is RESOLVED for
    `embeds_center_mode="local"`: each expansion group is centered on
    its own mean AND folded into its own parent's covariance, which is
    the rep_exp pairing (docs/decisions/
    rep-exp-elliptical-bonus-review.md). Under `cov_scope="global"`
    the original caveat still applies verbatim: groups centered at
    their own means pile into one accumulated V, so folded vectors
    carry group-dependent offsets.

      embeds_center=False        : pass through; only normalize below.
      embeds_center_mode="fixed" : subtract a held-out, precomputed
                                    mean (search.embeds_mean) — same
                                    constant for every vector, every
                                    expansion, the whole run. The mean
                                    lives in the post-projection space,
                                    so its shape must match embeds_dim.
      embeds_center_mode="local" : subtract the mean of THIS
                                    expansion's own sibling group,
                                    recomputed fresh every expansion,
                                    never carried forward.
                                    batch_size=1 edge: the centered
                                    vector is exactly 0 — zero
                                    diversity bonus, and the Sherman-
                                    Morrison fold of a zero vector is
                                    a no-op. Harmless.
    """
    if sc.embeds_center and sc.embeds_center_mode not in (
        "fixed", "local",
    ):
        raise ValueError(
            f"unknown embeds_center_mode: {sc.embeds_center_mode!r}"
        )

    stacked = np.stack(embeds)  # (batch_size, embeds_dim)

    if sc.embeds_center and sc.embeds_center_mode == "local":
        mean = stacked.mean(axis=0)
    elif sc.embeds_center:
        if sc.embeds_mean is None:
            raise ValueError("embeds_center=True requires search.embeds_mean")
        mean = np.asarray(sc.embeds_mean)
        if mean.shape[-1] != stacked.shape[-1]:
            raise ValueError(
                "embeds_mean dim "
                f"{mean.shape[-1]} != embedding dim {stacked.shape[-1]}; "
                "when embeds_proj='sparse' the mean must be computed in "
                "the post-projection space (same fixed projection)."
            )
    else:
        mean = None

    if mean is not None:
        stacked = stacked - mean
    if sc.embeds_normalize:
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        stacked = np.divide(
            stacked, norms, out=np.zeros_like(stacked), where=norms > 0,
        )
    return list(stacked)


def _embed_candidates(
    question, candidate_texts, config, tokenizer,
    llm_vllm_embeds, prm, response_start_idx,
):
    """Pool a diversity embedding for each candidate, from the source
    selected by config.search.embeds_source. Unchanged from v02.

    Returns a list of pooled (dim,) vectors aligned with
    candidate_texts.

      policy : per-candidate vLLM pooling engine on the generator
               (the v01 path). Requires llm_vllm_embeds.
      prm    : one batched PRM forward pass over the plain candidate
               chat (system / user / assistant), last-layer hidden
               states. No second engine; the PRM is already loaded.
    """
    sc = config.search
    source = getattr(sc, "embeds_source", "policy")

    if source == "policy":
        if llm_vllm_embeds is None:
            raise ValueError(
                "embeds_source='policy' needs the pooling engine, but "
                "llm_vllm_embeds is None. Launch with the v01 config "
                "(it builds the pooling engine)."
            )
        embeds = []
        for cand_text in candidate_texts:
            cand_convs = [
                build_conv(question, cand_text, config.gen.system_prompt)
            ]
            cand_templated = tokenizer.apply_chat_template(
                cand_convs,
                add_generation_prompt=False,
                continue_final_message=True,
                date_string=config.gen.date_string,
                tokenize=False,
            )
            outputs = llm_vllm_embeds.encode(
                cand_templated, pooling_task="token_embed", use_tqdm=False
            )
            embeds.append(
                _extract_embeds(
                    outputs[0].outputs.data, config, response_start_idx
                )
            )
        return _center_and_normalize(embeds, sc)

    if source == "prm":
        # response_start_idx was computed with the GENERATOR tokenizer
        # and is invalid for the PRM's tokenizer/template. Until a
        # PRM-side start index is wired, only "full" scope is correct.
        if sc.embeds_scope != "full":
            raise NotImplementedError(
                "embeds_source='prm' currently supports embeds_scope="
                "'full' only; 'response' needs a PRM-tokenizer start "
                "index (the generator's response_start_idx doesn't "
                "apply to the PRM sequence)."
            )
        raw_embeds = prm.embed(
            [question], [candidate_texts],
            system_prompt=config.gen.system_prompt,
            batch_size=sc.prm_batch_size,
            layer=getattr(sc, "prm_embeds_layer", -1),
        )[0]
        embeds = [
            _extract_embeds(raw, config, response_start_idx)
            for raw in raw_embeds
        ]
        return _center_and_normalize(embeds, sc)

    raise ValueError(f"unknown embeds_source: {source!r}")


# --------------------------------------------------------------------- #
# Node and tree classes                                                 #
# --------------------------------------------------------------------- #

@dataclass(slots=True)
class BaseNode:
    """Generic tree node. Carries the running text, an optional
    pooled embedding for diversity, and bookkeeping for depth /
    terminal status. `MCTSNode` extends with q-value + visit counts.
    """
    state: Dict[str, str] = field(
        default_factory=lambda: {"text": "", "step": "", "extra_info": ""}
    )
    parent: Optional[Any] = None
    children: List[Any] = field(default_factory=list)
    embeds: NDArray[Shape["2048"], np.float32] = None

    tag: str = "0"                # dotted lineage, e.g. "0.1.2"
    depth: int = 0
    phase: int = 0                # which `num_phases` outer loop made this
    gen_cnt: int = 0             # gen_budget value at creation time
    is_terminal: bool = False     # EOS / max-depth / empty completion
    is_completed: bool = False    # specifically: ended via EOS / length

    def has_children(self) -> bool:
        return self.children != []


@dataclass(slots=True)
class MCTSNode(BaseNode):
    # Name-mangled to _MCTSNode__visit_count etc.; access through the
    # methods below. Keeping them "private" is the original code's way
    # of making the q-value invariant explicit: only `update` may mutate.
    __visit_count: int = 0
    __value_sum: float = 0.0

    # ---- local covariance (cov_scope="local" only) ------------------
    # V_local / V_inv_local hold THIS node's own ridge covariance over
    # the embeddings of the children it has selected, and are the whole
    # difference from v02. They stay None until the first fold at this
    # node (MCTS._cov_fold allocates), so nodes that are created but
    # never selected through — the vast majority of leaves — cost
    # nothing. Under cov_scope="global" they stay None forever and the
    # tree-level MCTS.V / MCTS.V_inv are used instead.
    #
    # Which of the two is populated follows cov_update, exactly as at
    # tree level: "sm" keeps only V_inv_local; "exact" keeps both.
    # cov_n_folds counts folds at this node — the local analogue of
    # "how much evidence V has seen", useful for diagnostics and for
    # the algebraic test in the module docstring.
    V_local: Optional[Any] = None
    V_inv_local: Optional[Any] = None
    cov_n_folds: int = 0

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0.0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update(self, value: float) -> None:
        """Single-step value update. Called once per (a) child creation
        with the PRM score, and (b) traversal through this node.
        """
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value, start_node) -> None:
        """Backprop: update self, then ancestors, until we hit
        `start_node` (typically the root). The `start_node.tag`
        equality is the stopping condition.
        """
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def __repr__(self):
        return (
            f"MCTSNode(state={self.state}, "
            f"is_terminal={self.is_terminal}, nvisits={self.__visit_count})"
        )


class BaseTree(BaseModel):
    """Root holder. The actual search algorithm lives on `MCTS`."""
    config: Any
    question: Optional[str] = None
    root: Optional[Type[BaseNode]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = self.create_root()
        # Seed the root with an update so visit_count >= 1 from the
        # start; otherwise the visit_count==1 special case in
        # `select_child` triggers on the wrong iteration.
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with embedding-based diversity selection, per-node V.

    Same as v02's MCTS except that the covariance is addressed through
    `_cov_read` / `_cov_fold`, which dispatch on `cov_scope`:

      "local"  — read/write the covariance stored ON THE NODE whose
                 children are being compared (MCTSNode.V_inv_local).
                 MCTS.V / MCTS.V_inv stay None.
      "global" — read/write the single tree-level MCTS.V / MCTS.V_inv,
                 exactly as v02 does.

    Nothing else in the class knows which scope is active; `select_child`
    and `_select_by_diversity` are written against the two accessors.
    """

    # Pydantic field declarations; the actual values are allocated
    # per-instance in __init__ (the shape hints are documentation, and
    # completed_nodes must be a fresh list per question — a class-level
    # [] default is shared state waiting to happen).
    completed_nodes: List[Type[BaseNode]] = None
    # Tree-level covariance: populated under cov_scope="global" only.
    V: NDArray[Shape["2048, 2048"], np.float32] = None
    V_inv: NDArray[Shape["2048, 2048"], np.float32] = None
    # Count of nodes that hit max_depth (mirrors mcts_cnt_search_v05's
    # cnt_node_max_depth); incremented in create_child.
    cnt_node_max_depth: int = 0
    # How many nodes have allocated a local covariance. Under
    # cov_scope="local" this is the memory multiplier: peak local
    # covariance bytes ~= cnt_cov_nodes * d^2 * itemsize (x2 for
    # cov_update="exact", which keeps V alongside V_inv). Bounded by
    # the number of expansions, i.e. by gen_budget. Diagnostic only —
    # deliberately NOT added to the results dict, so v02.01 scored
    # datasets stay schema-identical to v02's.
    cnt_cov_nodes: int = 0
    # Precision for V/V_inv (config.search.cov_dtype), resolved to a
    # numpy dtype in __init__. MUST be declared here — MCTS is a
    # pydantic BaseModel, which raises on `self.attr = ...` for any
    # attribute not declared as a field.
    cov_dtype: Any = np.float64
    # Resolved config.search.cov_scope; see _cov_read / _cov_fold.
    cov_scope: str = "local"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-question state: a fresh completed-node list every run.
        self.completed_nodes = []
        sc = self.config.search

        cov_dtype_cfg = sc.cov_dtype
        if cov_dtype_cfg not in _COV_DTYPES:
            raise ValueError(f"unknown cov_dtype: {cov_dtype_cfg!r}")
        self.cov_dtype = _COV_DTYPES[cov_dtype_cfg]

        cov_scope = getattr(sc, "cov_scope", "local")
        if cov_scope not in _COV_SCOPES:
            raise ValueError(f"unknown cov_scope: {cov_scope!r}")
        self.cov_scope = cov_scope

        if cov_scope == "global":
            # Tree-level V, allocated exactly as v02 does. Under
            # "local" nothing is allocated here — each node allocates
            # its own on first fold (see _cov_fold).
            self.V, self.V_inv = self._new_cov()

    # ----- Covariance plumbing --------------------------------------- #

    def _new_cov(self):
        """Allocate a fresh (V, V_inv) pair at the ridge init.

        The start state is V_0 = lam*I, so V_0^-1 = (1/lam)*I in closed
        form — no inverse call, and the initial diversity term is
        uniform across arms. Under cov_update="sm" only the inverse is
        maintained, so V is left None (v02 does the same at tree
        level); "exact" keeps both because it re-inverts V each fold.

        This is called once per run under cov_scope="global", and once
        per selected-through NODE under "local" — which is exactly the
        memory story in the module docstring: the d x d allocation is
        per node, not per search.
        """
        sc = self.config.search
        embeds_dim = sc.embeds_dim
        lam = sc.lam
        V_inv = (1.0 / lam) * np.eye(embeds_dim, dtype=self.cov_dtype)
        if sc.cov_update == "sm":
            return None, V_inv
        V = lam * np.eye(embeds_dim, dtype=self.cov_dtype)
        return V, V_inv

    def _cov_read(self, node):
        """The V^-1 that governs selection among `node`'s children.

        Under "global" that is the tree's single V_inv. Under "local"
        it is the node's own — and if the node has never been folded
        into, its covariance is still the ridge init, so return the
        closed-form (1/lam)*I WITHOUT allocating: an unfolded node
        gives every child the same bonus, so materializing a d x d
        identity just to read a uniform bonus off it would be pure
        waste (this is the common case for freshly expanded nodes).
        """
        if self.cov_scope == "global":
            return self.V_inv
        if node.V_inv_local is None:
            lam = self.config.search.lam
            return (1.0 / lam) * np.eye(
                self.config.search.embeds_dim, dtype=self.cov_dtype
            )
        return node.V_inv_local

    def _cov_fold(self, node, u):
        """Fold the direction `u` into the covariance owned by `node`.

        `u` is the selected child's embedding as a (d, 1) column, cast
        to cov_dtype by the caller. Under "global" this mutates the
        tree-level V/V_inv exactly as v02 does; under "local" it
        mutates only `node`'s, allocating on first use — so sibling
        subtrees never see each other's folds, which IS the ablation.

        The two cov_update paths are v02's, verbatim in their math:
          "sm"    persistent rank-1 inverse update, O(d^2):
                  (V + uu^T)^-1 = V^-1 - (V^-1 u)(V^-1 u)^T
                                         / (1 + u^T V^-1 u)
                  then symmetrize, to stop floating-point asymmetry
                  compounding over the run.
          "exact" accumulate V and re-solve, O(d^3). solve(V, I) over
                  inv(V): same cost, slightly better-conditioned.
        """
        if self.cov_scope == "global":
            V, V_inv = self.V, self.V_inv
        else:
            if node.V_inv_local is None:
                node.V_local, node.V_inv_local = self._new_cov()
                self.cnt_cov_nodes += 1
            V, V_inv = node.V_local, node.V_inv_local

        if self.config.search.cov_update == "sm":
            Vu = V_inv @ u
            denom = 1.0 + float(u.T @ Vu)
            V_inv = V_inv - (Vu @ Vu.T) / denom
            V_inv = 0.5 * (V_inv + V_inv.T)
        else:
            V = V + u @ u.T
            V_inv = np.linalg.solve(
                V, np.eye(V.shape[0], dtype=self.cov_dtype)
            )

        if self.cov_scope == "global":
            self.V, self.V_inv = V, V_inv
        else:
            node.V_local, node.V_inv_local = V, V_inv
            node.cov_n_folds += 1

    def create_node(self, parent=None):
        return MCTSNode(parent=parent)

    # ----- Expansion ------------------------------------------------- #

    def create_child(
        self, current_node, candidate_info, candidate_embeds,
        candidate_score, phase, gen_cnt,
    ):
        """Append a single child to `current_node`. Marks terminal if
        the underlying vLLM generation hit EOS / length, OR if this
        would push depth past `config.search.max_depth` — in which case
        the score is overwritten with `config.search.negative_reward` so
        the backprop discourages re-entering this branch.
        """
        new_node = self.create_node(parent=current_node)
        parent_child_count = len(current_node.children)
        new_node.tag = f"{current_node.tag}.{parent_child_count + 1}"
        new_node.depth = current_node.depth + 1
        new_node.embeds = candidate_embeds
        new_node.phase = phase
        new_node.gen_cnt = gen_cnt

        new_node.state["text"] = (
            current_node.state["text"] + candidate_info.next_texts[0]
        )
        new_node.state["step"] = candidate_info.next_texts[0]

        stop_reason = candidate_info.stop_reasons[0]
        if stop_reason in ("EOS", "length") or candidate_info.next_texts[0] == "":
            new_node.is_completed = True
            new_node.is_terminal = True
            self.completed_nodes.append(new_node)

        if (not new_node.is_terminal
                and new_node.depth >= self.config.search.max_depth):
            new_node.is_terminal = True
            candidate_score = self.config.search.negative_reward
            self.cnt_node_max_depth += 1

        new_node.update(candidate_score)
        current_node.children.append(new_node)

    def expand_node(
        self, current_node, infos, embeds, scores, phase, gen_cnt,
    ):
        """Append one child per (info, embedding, score) tuple."""
        for info, emb, score in zip(infos, embeds, scores):
            self.create_child(current_node, info, emb, score, phase, gen_cnt)

    # ----- Selection ------------------------------------------------- #

    def _select_by_q_value(self, node):
        """First-visit selection: pick the highest q-value child,
        uniform-random tie-break. Unchanged from v02.

        On the very first descent through a newly-expanded node, every
        child has visit_count == 1 and a q-value equal to its
        PRM-derived candidate_score. The diversity bonus isn't
        informative yet — and under cov_scope="local" it is exactly
        uniform by construction, since the node's covariance is still
        the ridge init — so a plain q-value argmax gives cleaner signal.
        """
        qs = [ch.q_value() for ch in node.children]
        for ch in node.children:
            logging.fatal(f"{ch.tag}")
            logging.fatal(f"   q-value = {ch.q_value():0.4f}")
            logging.fatal(f"   nvisit = {ch.visit_count():0.2f}")
            logging.fatal(f"   parent.nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {ch.is_terminal}")
        if not qs:
            return None

        tol = 1e-4
        best_q = max(qs)
        best_childs = [
            ch for ch, q in zip(node.children, qs)
            if abs(best_q - q) <= tol
        ]
        return random.choice(best_childs) if best_childs else None

    def _select_by_diversity(self, node):
        """Subsequent-visit selection: combine q-value with the
        diversity term via `_diverse_select`, reading the covariance
        through `_cov_read` — the node's own under "local", the tree's
        under "global".

        The alpha weight is scaled by sqrt(log(1 + parent_visits)).
        Under "local" that per-node clock and the per-node covariance
        are finally the same clock (docs/decisions/
        global-vs-local-exploration-schedule.md); under "global" the
        two are mixed, exactly as in v02.
        """
        q_values: List[float] = []
        embeds: List[Any] = []
        _children: List[Any] = []
        for ch in node.children:
            q_values.append(ch.q_value())
            embeds.append(ch.embeds)
            _children.append(ch)
            logging.fatal(f"{ch.tag}")
            logging.fatal(f"   nvisit = {ch.visit_count():0.2f}")
            logging.fatal(f"   parent.nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {ch.is_terminal}")
        if not q_values:
            return None

        log_nvisit_parent = np.sqrt(np.log(1 + node.visit_count()))
        best_idx = _diverse_select(
            self._cov_read(node), embeds, q_values,
            self.config.search.ds_alpha * log_nvisit_parent,
            self.config.search.ds_beta,
            cov_dtype=self.cov_dtype,
        )
        return _children[best_idx] if _children else None

    def select_child(self, node):
        """Dispatcher: q-value-only on first visit, q + diversity on
        subsequent visits. Either way, fold the selected child's
        embedding into the covariance that governs THIS node.
        """
        if node.visit_count() == 1:
            selected_node = self._select_by_q_value(node)
        else:
            selected_node = self._select_by_diversity(node)

        if selected_node is None:
            return None

        # Covariance update — UNCONDITIONAL: it runs on BOTH branches,
        # including the first-visit q-value path that never reads the
        # covariance. That path still commits to a child, so its
        # direction must enter the covariance or V_inv would go stale
        # (no longer equal inv(V)) and later diversity bonuses would be
        # wrong. Cast to cov_dtype so u's precision matches V/V_inv
        # exactly — otherwise NumPy's mixed-dtype promotion decides
        # silently.
        u = selected_node.embeds.reshape(-1, 1).astype(self.cov_dtype)
        self._cov_fold(node, u)
        logging.fatal(f"selected_node = {selected_node.tag}")
        return selected_node

    # ----- Backprop -------------------------------------------------- #

    def backprop(self, node):
        """Recursive q-value backprop up to (and including) the root."""
        node.update_recursive(node.q_value(), self.root)


# --------------------------------------------------------------------- #
# Main search loop                                                      #
# --------------------------------------------------------------------- #

def _compute_response_start_idx(question, config, tokenizer) -> int:
    """Token index where the assistant response begins. Used only when
    `config.search.embeds_scope == "response"` to slice off the system /
    user prefix tokens before pooling.
    """
    prefix_convs = [build_conv(question, "", config.gen.system_prompt)]
    prefix_texts = tokenizer.apply_chat_template(
        prefix_convs,
        add_generation_prompt=True,
        continue_final_message=False,
        date_string=config.gen.date_string,
        tokenize=False,
    )
    prefix_inputs = tokenizer(prefix_texts)
    return len(prefix_inputs["input_ids"][0])


def _generate_candidates(
    question, current_node, d, p, config,
    tokenizer, llm_vllm, llm_vllm_embeds, prm,
    response_start_idx, sampling_params,
):
    """Generate, dedupe, embed and score next-step candidates branching
    off `current_node`. Returns (candidate_infos, embeds, scores).
    Unchanged from v02 — the variant changes selection, not
    generation.
    """
    current_text = current_node.state["text"]
    logging.error(f"current_text = {current_text}")

    # Build the prompt as a chat conversation. `add_generation_prompt`
    # is True only on the first step (depth==0) because at depth>0 we
    # want vLLM to continue the existing assistant turn rather than
    # start a fresh one.
    #
    # Strip the step terminator before templating, then re-append it to
    # the templated string: some templates / transformers versions trim
    # or crash on trailing "\n\n" inside apply_chat_template, but the
    # model must see the separator to continue with a next step instead
    # of emitting EOS (docs/findings/coding-findings/
    # library-version-trajectory-completeness.md, 2026-06-11).
    current_text_clean = current_text.removesuffix("\n\n")
    current_convs = [
        build_conv(question, current_text_clean, config.gen.system_prompt)
    ]
    current_templated = tokenizer.apply_chat_template(
        current_convs,
        add_generation_prompt=current_node.depth == 0,
        continue_final_message=current_node.depth > 0,
        date_string=config.gen.date_string,
        tokenize=False,
    )
    if current_text.endswith("\n\n"):
        current_templated = [t + "\n\n" for t in current_templated]
    # Replicate the same prompt config.search.batch_size times —
    # `generate_k_steps` uses each copy as an independent sampling
    # slot. Sampling differs across copies because SamplingParams sets
    # n=1; the variance comes from temperature, not from n.
    current_templated = current_templated * config.search.batch_size

    # `lookahead` controls how far ahead generate_k_steps peeks past
    # the "\n\n" step terminator. On the last possible depth we don't
    # look ahead (no room to continue), otherwise use the configured
    # lookahead budget.
    lookahead = (
        0 if d == config.search.max_depth - 1 else config.search.lookahead
    )
    llm_outputs = generate_k_steps(
        current_templated, lookahead, llm_vllm, sampling_params, 1
    )
    logging.error("llm_outputs")
    logging.error(llm_outputs)

    # Dedupe by next-step text, keeping the first occurrence. With
    # temperature > 0 and config.search.batch_size > 1 we often see
    # duplicates, especially on short next-steps.
    seen: Dict[str, int] = {}
    for idx, output in enumerate(llm_outputs):
        seen.setdefault(output.next_texts[0], idx)
    candidate_infos = [llm_outputs[idx] for idx in seen.values()]

    # Collect the candidate texts. All candidates branch off the same
    # question, so candidate_texts is the flat answer list for that one
    # question (embedded / scored as [question], [candidate_texts]).
    candidate_texts: List[str] = []
    for output in candidate_infos:
        cand_text = current_text + output.next_texts[0]
        # Strip the trailing step-terminator so it doesn't bleed into
        # the embedding / score.
        candidate_texts.append(cand_text.removesuffix("\n\n"))

    candidate_embeds = _embed_candidates(
        question, candidate_texts, config, tokenizer,
        llm_vllm_embeds, prm, response_start_idx,
    )

    candidate_scores = prm.score(
        [question], [candidate_texts],
        batch_size=config.search.prm_batch_size,
    )
    # score returns [question][answer][step]; one question here, so
    # candidate_scores[0] is a list of candidates, each a per-step
    # score list. Aggregate each candidate's step list to a scalar.
    candidate_scores = [
        aggregate_scores(cand_scores, config.gen.agg_strategy)
        for cand_scores in candidate_scores[0]
    ]
    logging.error(f"candidate_scores = {candidate_scores}")

    return candidate_infos, candidate_embeds, candidate_scores


def mcts_search(question, agent, config, llm_vllm, llm_vllm_embeds, prm):
    """Run MCTS-with-diversity on a single `question`.

    Identical to v02's loop — the variant lives entirely inside
    `agent.select_child`. Outer loop: `config.search.num_phases`
    independent descents from the root, each up to
    `config.search.max_depth` levels deep or until a terminal node.

    Budget: only expansions (not selections) charge against
    `config.search.gen_budget`.
    """
    tokenizer = llm_vllm.get_tokenizer()
    # Template selection (mirrors generate_mcts_cnt): default is the
    # model's NATIVE chat template — its own in-distribution format,
    # avoiding the cross-model confound (docs/decisions/
    # chat-template-per-family.md).
    if config.llm.use_custom_template:
        tokenizer.chat_template = config.gen.custom_chat_template

    sampling_params = SamplingParams(
        temperature=config.gen.temperature,
        max_tokens=config.gen.max_tokens,
        top_p=config.gen.top_p,
        stop=["\n\n"],                       # step terminator
        include_stop_str_in_output=True,
        n=1,
    )

    response_start_idx = _compute_response_start_idx(
        question, config, tokenizer
    )
    revisit_policy = config.search.revisit_policy

    gen_cnt = 0
    p = 0
    d = 0
    phase_depths: List[int] = []
    for p in range(config.search.num_phases):
        logging.fatal(f"\n-> p = {p}")
        current_node = agent.root

        for d in range(config.search.max_depth + 1):
            logging.fatal(f"\n-> d = {d}")

            if current_node.is_terminal:
                logging.fatal(f"current_node.is_terminal = True")
                agent.backprop(current_node)
                break

            # Expand if this node has never been expanded, OR if the
            # revisit policy says to regenerate fresh candidates
            # every time we land here.
            should_expand = (
                not current_node.has_children()
                or revisit_policy == "regenerate"
            )
            if should_expand:
                gen_cnt += 1
                infos, embeds, scores = _generate_candidates(
                    question, current_node, d, p, config,
                    tokenizer, llm_vllm, llm_vllm_embeds, prm,
                    response_start_idx, sampling_params,
                )
                agent.expand_node(
                    current_node, infos, embeds, scores, p, gen_cnt
                )

            current_node = agent.select_child(current_node)
            logging.fatal(f"gen_cnt = {gen_cnt}")
            if gen_cnt >= config.search.gen_budget:
                break

        phase_depths.append(d)
        if gen_cnt >= config.search.gen_budget:
            logging.fatal("run out of budget!")
            break

    logging.fatal(f"cov nodes allocated = {agent.cnt_cov_nodes}")

    # Collect unique completed-node completions. Multiple completed
    # nodes can share the same text (e.g. when regenerate_policy
    # produces near-duplicates from different paths); dedupe by text.
    seen: Dict[str, int] = {}
    for idx, node in enumerate(agent.completed_nodes):
        seen.setdefault(node.state["text"], idx)

    completions: List[str] = []
    comp_depth: List[int] = []
    comp_phase: List[int] = []
    comp_gen: List[int] = []
    for i in seen.values():
        node = agent.completed_nodes[i]
        completions.append(node.state["text"])
        comp_depth.append(node.depth)
        comp_phase.append(node.phase)
        comp_gen.append(node.gen_cnt)

    return (
        completions, comp_depth, comp_phase, comp_gen,
        gen_cnt, p, phase_depths, agent.cnt_node_max_depth,
    )


def _search(
    batch_of_questions, config, trial_idx,
    llm_vllm, llm_vllm_embeds, prm,
):
    """Run `mcts_search` on each question in the batch sequentially.

    Per-question deterministic seed: 100_000 + trial_idx. This means
    every question within a trial sees the *same* seed, so the only
    thing varying within a trial is the question — fine for
    comparing methods across trials, slightly weird semantically.
    Don't change without re-running baselines.

    Returns a defaultdict of per-question lists, all aligned to the
    same `q_idx`. The key set is byte-identical to v02's: the local
    covariance is diagnostics on the agent, never a results key, so
    v02.01 scored datasets read through utils.metrics unchanged.
    """
    n = len(batch_of_questions)
    batch_completions = [[] for _ in range(n)]
    batch_comp_depth = [[] for _ in range(n)]
    batch_comp_phase = [[] for _ in range(n)]
    batch_comp_gen = [[] for _ in range(n)]
    batch_q_total_gens = [[] for _ in range(n)]
    batch_q_last_phase = [[] for _ in range(n)]
    batch_phase_depths = [[] for _ in range(n)]
    batch_q_nodes_max_depth = [[] for _ in range(n)]

    for q_idx, question in enumerate(batch_of_questions):
        seed = 100_000 + trial_idx
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        agent = MCTS(config=config, question=question)
        (
            completions, comp_depth, comp_phase, comp_gen,
            q_total_gens, q_last_phase, phase_depths,
            q_nodes_max_depth,
        ) = mcts_search(
            question, agent, config, llm_vllm, llm_vllm_embeds, prm
        )

        batch_completions[q_idx] = completions
        batch_comp_depth[q_idx] = comp_depth
        batch_comp_phase[q_idx] = comp_phase
        batch_comp_gen[q_idx] = comp_gen
        batch_q_total_gens[q_idx] = q_total_gens
        batch_q_last_phase[q_idx] = q_last_phase
        batch_phase_depths[q_idx] = phase_depths
        batch_q_nodes_max_depth[q_idx] = q_nodes_max_depth

    # Key names match mcts_cnt_search_v01_00_00's results dict
    # (comp_depth/comp_phase/comp_gen/q_total_gens/q_last_phase/
    # phase_depths/q_nodes_max_depth) so utils.metrics.evaluate_
    # correctness reads both algorithms' scored datasets identically.
    results: Dict[str, Any] = defaultdict(list)
    results["completions"] = batch_completions
    results["comp_depth"] = batch_comp_depth
    results["comp_phase"] = batch_comp_phase
    results["comp_gen"] = batch_comp_gen
    results["q_total_gens"] = batch_q_total_gens
    results["q_last_phase"] = batch_q_last_phase
    results["phase_depths"] = batch_phase_depths
    results["q_nodes_max_depth"] = batch_q_nodes_max_depth
    return results
