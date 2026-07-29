"""
Semantic MCTS v02: embedding-based diversity selection, PRM-sourced
embeddings.

The "mcts_sem_v02" method. Same diversity-selection algorithm as
v01 (mcts_sem_search_v01_00_00) and the same embeds_* knobs, but the
pooled embeddings are read from the PRM's last-layer hidden states
instead of a second vLLM pooling engine on the policy. So the
embedding space shifts policy -> reward-model; pooling / normalize /
center / the covariance bonus are all identical, making v01-vs-v02 a
clean ablation of the embedding SOURCE alone. The launcher passes
llm_vllm_embeds=None for this variant (no pooling engine).

Mechanism (vs v01). The only behavioral difference is in candidate
expansion (_embed_candidates): where v01 calls
llm_vllm_embeds.encode(...), v02 calls prm.embed(...) — one batched
PRM forward pass (output_hidden_states=True) over the SAME plain
candidate chat v01 embeds (system / user(question) / assistant(answer),
config.gen.system_prompt), then pools the per-token hidden states with
the SAME _extract_embeds. The PRM is the one already loaded for
scoring, so no second model is added; the score and embed passes are
separate (the score pass runs the judge transcript, which is a
different sequence) but both reuse the loaded PRM. embeds_dim must be
the PRM's hidden size (4096 for Llama3.1-8B-PRM, not the generator's
2048) since it sizes the covariance V — set in the v02 YAML.

Caveat: embeds_scope="response" is NOT yet supported for the PRM
source (the response_start_idx is computed with the generator's
tokenizer and doesn't apply to the PRM sequence); _embed_candidates
raises for that combination. embeds_scope="full" is fully supported.

v01 itself replaces the v03_01_00, v03_02_00, v03_02_01, v03_02_02,
v03_02_03, and v04_01_00 files (all named mcts_embeds_search_* before
the rename). Variant behavior is selected via config flags; defaults
reproduce v03_01_00 (modulo the two features whose docstrings
v03_02_01 / v04_01_00 claimed but never actually landed — both now
implemented and gated behind opt-in flags).

Config flags (read off config.search.*; see MCTSSemV02Config in
utils/configs.py for defaults)
    embeds_source   : "policy" | "prm"             (default: "prm")
    embeds_strategy : "last" | "avg"               (default: "last")
    embeds_scope    : "full" | "response"          (default: "full")
    embeds_normalize: bool                         (default: True)
    embeds_center   : bool                         (default: False)
    embeds_center_mode: "fixed" | "local" (default: "fixed")
    embeds_mean     : np.ndarray | None    (required if center=True
                                            and mode="fixed")
    embeds_dim      : int                  (default: 2048; set 4096 for PRM)
    prm_embeds_layer: int                          (default: -1 = last)
    cov_update      : "exact" | "sm" (Sherman-Morrison) (default: "exact")
    cov_dtype       : "fp32" | "fp64"              (default: "fp64")
    cov_scope       : "global" | "local"           (default: "global")
    embeds_ref      : "absolute" | "relative"      (default: "absolute")
    revisit_policy  : "reuse" | "regenerate"       (default: "reuse")
    prm_batch_size  : int                          (default: 4)

The diversity covariance: two orthogonal axes (merged 2026-07-28,
absorbing the former standalone mcts_sem_v02_01 file)

    cov_scope — WHERE V lives.
        "global"  one V for the whole tree. Every selection anywhere
                  folds into it and every bonus reads it. This is the
                  original behavior and the pinned hash neutral, so
                  every run recorded before the merge is unaffected.
        "local"   one V per node, over the children THAT node has
                  selected. Sibling subtrees never see each other's
                  folds, so the bonus at n asks "which child points
                  somewhere n has not committed to yet?" rather than
                  "...somewhere the entire search has not visited?".
                  Motivation: (a) the alpha schedule is already a
                  per-node clock (sqrt(log(1+parent_visits))), so a
                  node-indexed multiplier against a globally
                  accumulated V mixes two clocks; (b) with L2-normalized
                  embeddings a direction covered k times scores
                  ~1/sqrt(lam+k), and globally k grows with total
                  selections (thousands), so ds_alpha means something
                  different at the root than at depth 15 — locally k is
                  the node's own visit count (tens).
                  COSTS one d x d matrix per selected-through node,
                  bounded by gen_budget; see the guard in __init__.

    embeds_ref — WHAT vector represents a child (see _cov_vec).
        "absolute" the child's own pooled embedding (original).
        "relative" the displacement x_child - x_parent, i.e. the
                   child expressed in coordinates centred on ITS
                   PARENT. ("relative" always means parent-relative
                   here; it is the only reference implemented. A
                   second one — root, grandparent — would arrive as
                   its own knob, not another value.) Embeddings are
                   pooled over the whole text prefix, so siblings share
                   a long prefix and cluster tightly around the parent's
                   direction; that shared component dominates
                   sqrt(x^T V^-1 x) and leaves the sibling differences —
                   the thing being selected on — as a small
                   perturbation. Subtracting the parent scores step
                   DIRECTIONS instead of absolute positions.
                   The root is embedded explicitly (_embed_root) so
                   this applies at EVERY depth including depth 0.
                   Without that the root has no embedding, its
                   children fall back to absolute vectors, and under
                   cov_scope="global" one V accumulates a MIXTURE of
                   absolute positions and displacements — measured at
                   25% / 75% before _embed_root existed. With it,
                   global+relative genuinely does accumulate
                   frame-free step directions across the whole tree.
                   Rejected together with embeds_center_mode="local"
                   — that double-centers, and it would also make the
                   one-element root batch center to zero.

    Under cov_scope="local", embeds_ref="relative" is the coherent
    pairing: the parent is a fixed reference for the node's whole
    lifetime, whereas embeds_center_mode="local"'s group mean is
    recomputed at every expansion, so under revisit_policy=
    "regenerate" one node's V would accumulate vectors measured from
    different origins.

    Verification: unittests/check_cov_scope_embeds_ref.py drives the
    real selection loop from a scripted generator (no GPU) and pins
    both the algebra and the RNG consumption. cov_scope="global" +
    embeds_ref="absolute" was checked trace-for-trace against the
    pre-merge file across 6 configurations.

Algorithm sketch
    For each of `num_phases` phases:
      Walk from the root down to a terminal node. At each step:
        - If `current_node` has no children (or `revisit_policy ==
          "regenerate"`), generate `config.search.batch_size`
          next-step continuations
          via vLLM, dedupe by text, pool the per-token embeddings of
          each candidate, score them with the PRM, and add them as
          children. This is the only operation that charges against
          `gen_budget`.
        - Select one child:
            * On the first visit after expansion, pick the highest
              q-value child (random tie-break).
            * On subsequent visits, pick by
                  beta * q + alpha * sqrt(x^T V^-1 x)
              with alpha scaled by sqrt(log(1 + parent_visits)).
        - Update V += x x^T with the chosen child's embedding.
      Backprop the terminal node's q-value to the root.

Old flag names are not aliased — they were renamed intentionally
(`embeds_normalizing` -> `embeds_normalize`, `embeds_centering` ->
`embeds_center`). Update old configs at the call site rather than
papering over with shims.

Variant lineage: docs/algorithms.md.
"""

import gc
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

# config.search.cov_scope -> where the covariance lives (MCTS._cov_read
# / _cov_fold); config.search.embeds_ref -> what vector represents a
# child in its parent's selection problem (MCTS._cov_vec).
_COV_SCOPES = ("global", "local")
_EMBEDS_REFS = ("absolute", "relative")

# Ceiling on the worst-case per-question local-covariance footprint
# (cov_scope="local" allocates one d x d matrix per selected-through
# node, so cost scales with gen_budget). Enforced in MCTS.__init__;
# see the guard there for why a legal config can otherwise reach tens
# of GiB and die as a silent cgroup OOM kill.
_LOCAL_COV_MAX_BYTES = 4 * 2**30      # 4 GiB


# --------------------------------------------------------------------- #
# Diversity selection                                                   #
# --------------------------------------------------------------------- #

def _diverse_select(
    V_inv, q_embeds, q_scores, ds_alpha, ds_beta, cov_dtype=np.float64,
):
    """Pick the one arm maximizing `beta*score + alpha*diversity`.

    The diversity term for arm `x` is `sqrt(x^T V^-1 x)` — large when
    `x` points in a direction V has seen little of, small when V has
    accumulated many similar vectors. `V_inv` is the inverse covariance
    the caller maintains across selections (see MCTS.select_child); this
    function is a pure decision — it neither inverts nor mutates state.

    `cov_dtype` (config.search.cov_dtype) fixes the precision `q_embeds`
    is cast to before multiplying against `V_inv`, so the einsum below
    runs at a controlled precision rather than whatever NumPy's mixed-
    dtype promotion happens to pick.

    Ties (within `tol`) are broken by uniform random sampling, which
    avoids the systematic bias of picking the first argmax.

    (Single-arm selection: the caller always picks one child at a time.
    A K>1 batch variant would need a sequential within-call update of
    V_inv between picks — Sherman-Morrison, not a re-inversion — but no
    caller needs that today.)
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

    `raw` is a (seq_len, dim) tensor of per-token hidden states. In v02
    it comes from the PRM (`prm.embed(...)`, one row per candidate); in
    the policy path it's `llm_vllm_embeds.encode(..., "token_embed")`.
    Both feed this same function so the two sources pool identically.

    Three steps, in this order:
      1. scope     pick which tokens contribute (full sequence or only
                   the assistant response).
      2. pool      reduce to (dim,): last token or mean over scope.
      3. project   optional sparse random projection to embeds_dim
                   (JL near-isometry; fixed matrix per run). Linear, so
                   it composes cleanly with the centering that follows.

    Centering (fixed-mean OR local-group-mean) and the final L2
    normalize are NOT done here — both need to see the whole sibling
    batch at once (fixed mode's shape guard lives alongside local
    mode's group-mean math for one reason to read), so they're
    entirely `_center_and_normalize`'s job, called once per expansion
    from `_embed_candidates` after every candidate has been pooled.

    Step 3 runs on the numpy side; pooling/scope on the torch tensor.
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

    Owns BOTH gated steps `_extract_embeds`'s docstring used to number
    4 and 5 — moved here because "fixed" mode's shape guard and
    "local" mode's group-mean math read better side by side than
    split across two functions, and because "local" mode strictly
    needs the whole batch (see below), so per-vector `_extract_embeds`
    could never do it anyway. Centering always happens in the linear
    space, before the non-linear normalize — this function is the one
    place that ordering has to be kept correct now.

      embeds_center=False        : pass through; only normalize below.
      embeds_center_mode="fixed" : subtract a held-out, precomputed
                                    mean (search.embeds_mean) — same
                                    constant for every vector, every
                                    expansion, the whole run. The mean
                                    lives in the post-projection space,
                                    so its shape must match embeds_dim
                                    (guard: a raw-space mean can never
                                    be silently subtracted from a
                                    projected vector).
      embeds_center_mode="local" : subtract the mean of THIS
                                    expansion's own sibling group,
                                    recomputed fresh every expansion,
                                    never carried forward (rep_exp-
                                    style local centering:
                                    docs/decisions/
                                    rep-exp-elliptical-bonus-review.md).
                                    batch_size=1 edge: the centered
                                    vector is exactly 0 — zero
                                    diversity bonus, and the Sherman-
                                    Morrison fold of a zero vector is
                                    a no-op. Harmless.

    Coherence caveat for local mode (recorded in
    docs/decisions/embeds-centering-design.md): each group is centered
    at its own mean while V accumulates across the whole search, so
    folded vectors carry group-dependent offsets. This is an empirical
    ablation arm, not a coherence-preserving transform — rep_exp pairs
    local centering with a per-group FRESH covariance, which our
    accumulated V deliberately is not.
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
    selected by config.search.embeds_source.

    Returns a list of pooled (dim,) vectors aligned with
    candidate_texts. Both sources feed the SAME _extract_embeds, so
    v01-vs-v02 differs only in which model produced the hidden states.

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
        # Forward pass(es) over all candidates of this question, chunked
        # by sc.prm_batch_size (see PRM.embed). prm.embed returns
        # [question][answer] -> (seq_len, dim) tensors; one question
        # here, so [0] is the per-candidate list. The PRM builds the
        # plain candidate chat with config.gen.system_prompt so the
        # embedded text matches the policy path's.
        raw_embeds = prm.embed(
            [question], [candidate_texts],
            system_prompt=config.gen.system_prompt,
            batch_size=sc.prm_batch_size,
            layer=getattr(sc, "prm_embeds_layer", -1),
        )[0]
        # response_start_idx is unused under "full" scope; pass it
        # through so _extract_embeds keeps one signature.
        embeds = [
            _extract_embeds(raw, config, response_start_idx)
            for raw in raw_embeds
        ]
        return _center_and_normalize(embeds, sc)

    raise ValueError(f"unknown embeds_source: {source!r}")


def _embed_root(
    question, config, tokenizer, llm_vllm_embeds, prm,
    response_start_idx,
):
    """The root's diversity embedding: the question with an EMPTY
    answer, pooled through the same pipeline as every candidate.

    Only computed under embeds_ref="relative", where each child is
    represented by its displacement from its parent and the root
    would otherwise have nothing to subtract. Without it the root's
    children fall back to absolute vectors, which costs differently
    under each scope:

      cov_scope="global"  one V accumulates a MIXTURE of absolute
                          positions (root's children) and
                          displacements (everywhere else) — measured
                          at 25% / 75% on a 100-fold descent. V^-1 is
                          then fitted to a bimodal mixture and the
                          bonus means different things at different
                          depths.
      cov_scope="local"   each V stays internally consistent, but the
                          ROOT's V collects the tightly-clustered
                          absolute sibling embeddings that relative
                          mode exists to decorrelate — so depth 0,
                          where the branching factor is widest, keeps
                          the exact problem the knob was built to fix.

    Implemented as _embed_candidates with one empty candidate, so
    scope / pooling / projection / centering / normalize are
    byte-identical to what every child gets — a root embedding built
    any other way would not live in the same space as the children it
    is subtracted from. Costs one extra PRM forward pass per
    question, against `gen_budget` batched passes for the search.

    Depends on the embeds_ref="relative" + embeds_center_mode="local"
    guard in MCTS.__init__: a one-element batch centred on its own
    group mean is exactly the zero vector, which would make every
    child's displacement equal to its own embedding. That combination
    is rejected there, so this function can never be reached with it.
    """
    return _embed_candidates(
        question, [""], config, tokenizer, llm_vllm_embeds, prm,
        response_start_idx,
    )[0]


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

    # ---- per-node covariance (cov_scope="local" only) ---------------
    # V_local / V_inv_local hold THIS node's own ridge covariance over
    # the vectors of the children it has selected. They stay None
    # until the first fold at this node (MCTS._cov_fold allocates), so
    # nodes that are created but never selected through — the vast
    # majority of leaves — cost nothing. Under cov_scope="global" they
    # stay None forever and the tree-level MCTS.V / MCTS.V_inv are
    # used instead.
    #
    # Which of the two is populated follows cov_update, exactly as at
    # tree level: "sm" keeps only V_inv_local; "exact" keeps both.
    # cov_n_folds counts folds at this node — the local analogue of
    # "how much evidence V has seen".
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
    """MCTS with embedding-based diversity selection.

    Inherits the root/question machinery from BaseTree and adds:
      - V / V_inv: the covariance and its inverse, fed to the diversity
        bonus; maintained per cov_update (see __init__ / select_child).
      - completed_nodes: nodes that ended via EOS or length stop.
      - the algorithm methods (`expand_node`, `select_child`,
        `backprop`).

    Note: in the v03_*/v04_* files there used to be a `BS` class
    sitting between `BaseTree` and `MCTS` that held `V` and
    `completed_nodes`. It was never imported anywhere outside its
    own file, so it's been flattened in here.
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
    # covariance bytes ~= cnt_cov_nodes * d^2 * itemsize (x2 under
    # cov_update="exact", which keeps V alongside V_inv). Bounded by
    # the number of expansions, i.e. by gen_budget. Surfaced per
    # question as results["q_cov_nodes"] under cov_scope="local"
    # ONLY — see the gate at the end of _search for why global runs
    # must not gain the key.
    cnt_cov_nodes: int = 0
    # Precision for V/V_inv (config.search.cov_dtype), resolved to a
    # numpy dtype in __init__. MUST be declared here — MCTS is a
    # pydantic BaseModel, which raises on `self.attr = ...` for any
    # attribute not declared as a field (unlike V/V_inv above).
    cov_dtype: Any = np.float64
    # Resolved config.search.cov_scope / embeds_ref; see _cov_read,
    # _cov_fold, _cov_vec.
    cov_scope: str = "global"
    embeds_ref: str = "absolute"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-question state: a fresh completed-node list every run.
        self.completed_nodes = []
        # Ridge-regularized covariance V = lam*I + sum u u^T. The
        # diversity term sqrt(x^T V^-1 x) needs V^-1; how it's
        # maintained across selections depends on cov_update (see
        # select_child):
        #   "exact"            keep V; recompute V^-1 = inv(V) each
        #                      selection. O(d^3) per selection.
        #   "sm"               keep V^-1 directly and rank-1 update it
        #                      per selection (O(d^2)); V isn't needed.
        # Either way the start state is V_0 = lam*I, so
        # V_0^-1 = (1/lam)*I in closed form (no inverse call), and the
        # initial diversity term is uniform across arms.
        sc = self.config.search
        embeds_dim = sc.embeds_dim
        # cov_dtype fixes V/V_inv's precision explicitly (default
        # "fp64" matches the previous implicit behavior: np.eye() with
        # no dtype= already defaulted to float64).
        cov_dtype_cfg = sc.cov_dtype
        if cov_dtype_cfg not in _COV_DTYPES:
            raise ValueError(f"unknown cov_dtype: {cov_dtype_cfg!r}")
        self.cov_dtype = _COV_DTYPES[cov_dtype_cfg]

        cov_scope = sc.cov_scope
        if cov_scope not in _COV_SCOPES:
            raise ValueError(f"unknown cov_scope: {cov_scope!r}")
        self.cov_scope = cov_scope

        embeds_ref = sc.embeds_ref
        if embeds_ref not in _EMBEDS_REFS:
            raise ValueError(f"unknown embeds_ref: {embeds_ref!r}")
        self.embeds_ref = embeds_ref

        # Double-centering guard. embeds_center_mode="local" already
        # subtracts the sibling-group mean; embeds_ref="relative"
        # then
        # subtracts the parent on top of that, so the vectors fed to V
        # are differences of already-differenced quantities and
        # neither knob means what its name says. They are two answers
        # to the same question ("remove the offset siblings share"),
        # so pick one. parent-relative is the more coherent of the two
        # under cov_scope="local": the parent is a fixed reference for
        # the node's whole lifetime, whereas the group mean is
        # recomputed at every expansion (see _center_and_normalize).
        if (embeds_ref == "relative" and sc.embeds_center
                and sc.embeds_center_mode == "local"):
            raise ValueError(
                "embeds_ref='relative' with "
                "embeds_center_mode='local' "
                "double-centers: the group mean is subtracted, then "
                "the parent. Choose one — set embeds_center=false, "
                "or embeds_ref='absolute'."
            )

        # Memory guard for the per-node covariance. Under "global"
        # there is exactly one d x d matrix no matter what d is;
        # under "local" there is one per selected-through node, so the
        # same d costs gen_budget times as much. embeds_proj="none"
        # forces embeds_dim to the raw PRM hidden size (4096), which
        # is 128 MiB per node — a documented, legal v02 config that
        # would need ~40 GiB per question here and die as a silent
        # cgroup OOM kill with no traceback. Fail loudly instead.
        if cov_scope == "local":
            itemsize = np.dtype(self.cov_dtype).itemsize
            per_node = embeds_dim * embeds_dim * itemsize
            if sc.cov_update != "sm":
                per_node *= 2      # "exact" keeps V beside V_inv
            worst = per_node * sc.gen_budget
            if worst > _LOCAL_COV_MAX_BYTES:
                raise ValueError(
                    f"cov_scope='local' would hold up to "
                    f"{worst / 2**30:.1f} GiB of covariance per "
                    f"question (embeds_dim={embeds_dim}, "
                    f"gen_budget={sc.gen_budget}, cov_update="
                    f"{sc.cov_update!r} -> {per_node / 2**20:.0f} MiB "
                    f"per node), over the "
                    f"{_LOCAL_COV_MAX_BYTES / 2**30:.0f} GiB cap. Use "
                    f"embeds_proj='sparse' with a smaller embeds_dim, "
                    f"or cov_scope='global'."
                )
        else:
            # Tree-level V, allocated exactly as before. Under "local"
            # nothing is allocated here — each node allocates its own
            # on first fold (see _cov_fold).
            self.V, self.V_inv = self._new_cov()

    # ----- Covariance plumbing --------------------------------------- #

    def _new_cov(self):
        """Allocate a fresh (V, V_inv) pair at the ridge init.

        Ridge-regularized covariance V = lam*I + sum u u^T. The
        diversity term sqrt(x^T V^-1 x) needs V^-1; how it is
        maintained depends on cov_update:
          "exact"  keep V; recompute V^-1 = inv(V) each fold, O(d^3).
          "sm"     keep V^-1 directly and rank-1 update it per fold
                   (O(d^2)); V isn't needed, so it stays None.
        Either way the start state is V_0 = lam*I, so
        V_0^-1 = (1/lam)*I in closed form (no inverse call), and the
        initial diversity term is uniform across arms.

        Called once per run under cov_scope="global", and once per
        selected-through NODE under "local".
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
        it is the node's own.

        The `V_inv_local is None` fallback below is defensive, not a
        hot path: a node only reaches _select_by_diversity once its
        visit_count exceeds 1, which means it was already selected
        through once, which means _cov_fold already allocated. It is
        kept so the accessor is total — a caller that reads before
        any fold gets the mathematically correct ridge init rather
        than an AttributeError.
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

        `u` is the selected child's vector (see _cov_vec) as a (d, 1)
        column, cast to cov_dtype by the caller. Under "global" this
        mutates the tree-level V/V_inv; under "local" it mutates only
        `node`'s, allocating on first use — so sibling subtrees never
        see each other's folds, which IS the local-scope ablation.

        The two cov_update paths:
          "sm"    persistent rank-1 inverse update, O(d^2):
                  (V + uu^T)^-1 = V^-1 - (V^-1 u)(V^-1 u)^T
                                         / (1 + u^T V^-1 u)
                  then symmetrize, to stop floating-point asymmetry
                  compounding over the run.
          "exact" accumulate V and re-solve, O(d^3). solve(V, I) over
                  inv(V): same cost, slightly better-conditioned
                  (avoids explicitly forming the inverse via a less
                  stable routine).
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

    def _cov_vec(self, node, child):
        """`child` as represented inside `node`'s selection problem.

        This is BOTH what the diversity bonus scores and what the
        fold accumulates — one function so the two can never diverge.
        That is a correctness requirement, not tidiness: if the bonus
        scored x_c - x_n while V_n accumulated x_c, V_n would live in
        a different space than the queries and the bonus would be
        meaningless.

          embeds_ref="absolute" : the child's own pooled embedding
                                  (the original v02 behavior).
          embeds_ref="relative" : the DISPLACEMENT x_c - x_n, i.e.
                                  the child in PARENT-CENTRED
                                  coordinates ("relative" always
                                  means parent-relative here). A
                                  child's embedding is pooled over its
                                  whole text prefix, so siblings share
                                  a long common prefix and their
                                  absolute embeddings cluster tightly
                                  around the parent's direction. That
                                  shared component dominates
                                  sqrt(x^T V^-1 x), leaving the
                                  sibling differences — the only thing
                                  diversity selection cares about — as
                                  a small perturbation. Subtracting
                                  the parent removes it, so the bonus
                                  is over step DIRECTIONS.

        The root: `embeds` is otherwise set only in create_child, so
        mcts_search calls _embed_root to give the root the question
        pooled with an empty answer — the natural origin for a first
        step. The `node.embeds is None` branch below is therefore a
        DEFENSIVE fallback (a tree driven without that setup, e.g.
        from a unit test), not the depth-0 path of a real run.

        Renormalization follows embeds_normalize rather than adding
        its own knob. ||x_c - x_n|| is well below 1 for clustered
        siblings and the bonus scales linearly in ||x||, so raw
        differences would shrink it several-fold and make ds_alpha
        mean something different than it does in every existing
        sweep. A zero difference passes through as the zero vector:
        zero bonus, and a Sherman-Morrison fold of zero is a no-op
        (denom == 1).
        """
        x = child.embeds
        if self.embeds_ref == "absolute" or node.embeds is None:
            return x
        x = x - node.embeds
        if self.config.search.embeds_normalize:
            norm = float(np.linalg.norm(x))
            if norm > 0:
                x = x / norm
        return x

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
        uniform-random tie-break.

        Rationale for the split: on the very first descent through a
        newly-expanded node, every child has visit_count == 1 and a
        q-value equal to its PRM-derived candidate_score. Diversity
        based on V^-1 isn't informative yet (V hasn't accumulated
        anything from these children), so a plain q-value argmax
        gives cleaner signal than mixing in noise from V.

        Ties (within `tol`) are broken by uniform random sampling —
        the same tolerance-based scheme `_diverse_select` uses, so both
        selection paths handle near-equal values consistently (exact
        float `==` would miss ties differing by ~1e-16 and reintroduce
        a first-argmax bias).
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
        diversity term via `_diverse_select`. The alpha weight is
        scaled by sqrt(log(1 + parent_visits)) so diversity matters
        more after we've sunk visits into a node (classic UCB-style
        exploration schedule).
        """
        q_values: List[float] = []
        embeds: List[Any] = []
        _children: List[Any] = []
        for ch in node.children:
            q_values.append(ch.q_value())
            embeds.append(self._cov_vec(node, ch))
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
        embedding into the covariance so future selections see it.
        """
        if node.visit_count() == 1:
            selected_node = self._select_by_q_value(node)
        else:
            selected_node = self._select_by_diversity(node)

        if selected_node is None:
            return None

        # Covariance update — UNCONDITIONAL: it runs on BOTH branches,
        # including the first-visit q-value path that never reads
        # V_inv. That path still commits to a child, so its direction
        # must enter the covariance or V_inv would go stale (no longer
        # equal inv(V)) and later diversity bonuses would be wrong.
        # _cov_vec builds the SAME representation _select_by_diversity
        # scored; cast to cov_dtype so u's precision matches V/V_inv
        # exactly, rather than letting NumPy's mixed-dtype promotion
        # decide silently.
        u = self._cov_vec(node, selected_node)
        u = u.reshape(-1, 1).astype(self.cov_dtype)
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

    Built by rendering the chat template with an empty assistant
    message and counting tokens up to the assistant turn marker.
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

    Three vLLM calls per invocation:
      1. `generate_k_steps` produces `config.search.batch_size`
         continuations.
      2. After dedup-by-text, `llm_vllm_embeds.encode(...)` runs the
         embedding model on each unique candidate prefix.
      3. `prm.score(...)` runs the process reward model on the same
         (question, prefix) pairs.
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
    # duplicates,
    # especially on short next-steps.
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

    # Embed each candidate. v02 differs from v01 ONLY here, by source:
    #   policy — the second vLLM pooling engine on the generator
    #            (identical to v01).
    #   prm    — the PRM's last-layer hidden states over the same plain
    #            candidate chat, one batched forward pass (no 2nd
    #            engine). _extract_embeds pools both the same way, so
    #            the comparison isolates the embedding model alone.
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

    Outer loop: `config.search.num_phases` independent descents from
    the root. Each descent goes up to `config.search.max_depth` levels
    deep or until it hits a terminal node.

    Budget: only expansions (not selections) charge against
    `config.search.gen_budget`. With `revisit_policy="reuse"` a node is
    expanded exactly once per phase. With `revisit_policy="regenerate"`
    a node is re-expanded every time we revisit it, which lets the
    tree grow children at hot nodes but also burns through the
    budget faster.
    """
    tokenizer = llm_vllm.get_tokenizer()
    # Template selection (mirrors generate_mcts_cnt): default is the
    # model's NATIVE chat template — its own in-distribution format,
    # avoiding the cross-model confound (docs/decisions/
    # chat-template-per-family.md).
    # llm.use_custom_template defaults True (custom) for Llama; Qwen
    # YAML groups set it False (native) — see
    # LLMConfig.use_custom_template. The trailing "\n\n" step
    # separator is preserved by the strip-and-reappend in
    # _generate_candidates.
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
    # Under embeds_ref="relative" the root needs a real embedding or
    # its children have nothing to measure a displacement from (see
    # _embed_root for what the fallback costs under each scope).
    # Skipped under "absolute", where it would never be read and would
    # buy a PRM forward pass for nothing — so this leaves every
    # pre-existing config's behavior untouched.
    if agent.embeds_ref == "relative":
        agent.root.embeds = _embed_root(
            question, config, tokenizer, llm_vllm_embeds, prm,
            response_start_idx,
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

            # select_child returns None only when the node ended up
            # with zero children -- unreachable today (expand_node
            # always appends batch_size>=1 candidates), but guarded so
            # a length mismatch in its zip() degrades to ending the
            # phase instead of an AttributeError one iteration later.
            # Assign via a temp: writing None straight into
            # current_node would lose the node we still need to
            # backprop from.
            nxt = agent.select_child(current_node)
            if nxt is None:
                agent.backprop(current_node)
                break
            current_node = nxt
            logging.fatal(f"gen_cnt = {gen_cnt}")
            if gen_cnt >= config.search.gen_budget:
                break

        phase_depths.append(d)
        if gen_cnt >= config.search.gen_budget:
            logging.fatal("run out of budget!")
            break

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
        agent.cnt_cov_nodes,
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
    same `q_idx`.
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
    batch_q_cov_nodes = [[] for _ in range(n)]

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
            q_nodes_max_depth, q_cov_nodes,
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
        batch_q_cov_nodes[q_idx] = q_cov_nodes

        # Drop the tree before building the next one. Every MCTSNode
        # points at its parent and the parent at its children, so the
        # whole tree is one big reference cycle: refcounting alone
        # reclaims NOTHING when `agent` goes out of scope, and the
        # generational GC triggers on allocation COUNTS, not bytes, so
        # a few hundred multi-MiB arrays can sit unreclaimed for a
        # long time. Harmless under cov_scope="global" (one d x d
        # matrix per tree) but not under "local": measured 644 MiB
        # held per question at d=512/fp64/gen_budget=320, and a
        # 2.19x high-water mark across four questions without this
        # call. With it, the peak is one question's worth.
        del agent
        gc.collect()

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

    # Local-scope memory diagnostic: how many nodes allocated their
    # own covariance, per question. Peak local covariance bytes ~=
    # max(q_cov_nodes) * d^2 * itemsize (x2 under cov_update=
    # "exact", which keeps V alongside V_inv). core.scoring.
    # build_scored_dataset auto-attaches any per-question list as a
    # dataset column, so nothing downstream needs changing.
    #
    # Gated on cov_scope="local" for two reasons. (1) Under "global"
    # no node ever allocates a covariance, so the column would be
    # all zeros — noise, not data. (2) It keeps every global run's
    # scored JSONL schema identical to the runs already on disk, so
    # a mixed-vintage set of trial files still unions cleanly in any
    # ad-hoc load. Note the stats path is NOT at risk either way:
    # utils.metrics.evaluate_correctness returns a fixed key tuple,
    # so extra dataset columns never reach _load_trials.
    if config.search.cov_scope == "local":
        results["q_cov_nodes"] = batch_q_cov_nodes
    return results
