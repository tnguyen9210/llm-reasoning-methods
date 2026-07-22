"""
Budget-Limited Semantic MCTS v01: best-first frontier selection with
embedding-based diversity (the "mcts_bl_sem_v01" method).

Frontier counterpart of mcts_sem_search_v02_00_00, exactly as
mcts_bl_cnt_search_v01_00_00 is the frontier counterpart of
mcts_cnt_search_v01_00_00: instead of phase-based root-to-leaf walks,
maintain an explicit `leaf_nodes` frontier and select globally across
all current leaves each iteration. bl_cnt's PUCT criterion is replaced
by the sem family's diversity-adjusted value.

Loop shape shared with mcts_bl_cnt_search_v01_00_00.py, including the
generate -> expand -> select ordering (aligned 2026-07-20 — see
docs/decisions-log.md; this file previously used an older select-first
loop shape, matching mcts_bl_kube/kdepth_v01 before their own 2026-07-17
alignment — the algorithm and every selection is unchanged, only the
reading order of the loop body moved to match v01's).

Algorithm
    Initialize leaf_nodes = [], gen_cnt = 0, current = root
    While gen_cnt < gen_budget:
        If current.is_terminal:
            Backprop: update_recursive(current.q_value(), root)
        Else:
            Expand: generate batch_size next-step continuations,
                    dedupe, embed (PRM hidden states by default),
                    score with PRM, attach as current's children
            gen_cnt += 1
            Add children to leaf_nodes
        t += 1  (selections so far, 1-indexed; first selection is t=1)
        current = argmax_{x in leaf_nodes}
            ds_beta*q(x) + ds_alpha*sched(t,x)*sqrt(x^T V^-1 x);
        remove current from leaf_nodes
        Fold current's embedding into V (rank-1; skipped for the
        root, which has no embedding and is only ever selected alone
        on the first iteration)

    The loop body is ordered generate -> expand -> select to mirror
    mcts_bl_cnt_search_v01_00_00.py's walk step; the selection SCOPE
    stays global (the whole leaf frontier — the children just
    expanded compete against every older leaf), not per-parent. This
    is a rotation of the older select-first loop: the same nodes are
    selected in the same order by the same diversity-adjusted value,
    AND the `t` fed to the "global" ds_alpha schedule at each
    selection is unchanged (verified behavior-preserving with a
    state-machine harness — the reorder only changes when in the loop
    body t is computed and where current_node is read from, not what
    gets selected, what t equals at each selection, or when
    gen_cnt/phase caps trigger).

ds_alpha schedule (config.search.ds_alpha_schedule)
    "global" (default) — sched = sqrt(log(1 + t)), t = frontier
        selections so far. The frontier is a flat arm set and
        sqrt(x^T V^-1 x) is the LinUCB confidence width, so the
        global-clock sqrt(log t) growth is the OFUL-standard
        schedule. The multiplier is shared by every frontier node at
        a given iteration — per-node differentiation comes only from
        q and the V^-1 geometry.
    "parent" — sched = sqrt(log(1 + x.parent.visit_count())) per
        node: the literal transplant of mcts_sem_v02's per-parent
        schedule. Nodes get tree-position-dependent scales.
    "none" — sched = 1 (constant ds_alpha).

Differences from mcts_sem_v02 beyond the frontier:
  - no first-visit q-only special case: it is defined per-parent
    ("first descent through a newly expanded node") and has no analog
    under global selection. Fresh children compete by q + diversity
    immediately; since none of them are in V yet their widths start
    near-equal, so q differentiates them anyway.
  - no revisit_policy: a frontier node is expanded at most once by
    construction (removed from the frontier when selected, never
    re-added).

Embedding machinery (_diverse_select, _extract_embeds,
_center_and_normalize, _embed_candidates, sparse projector) is
identical to mcts_sem_search_v02_00_00's; both embeds sources ("prm"
default, "policy" via the pooling engine) are supported, as are both
embeds_center_mode values ("fixed" | "local") and both cov_dtype
values ("fp32" | "fp64") — see MCTSSemV01Config's docstrings in
utils/configs.py for the semantics; BLMCTSSemConfig carries its own
copies of these fields (a fresh SearchConfig subclass, not inherited)
with matching defaults.

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
# multiplied against them (see MCTS.__init__, _fold_covariance,
# select_leaf_from_list).
_COV_DTYPES = {"fp32": np.float32, "fp64": np.float64}


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
    the caller maintains across selections (see
    MCTS.select_leaf_from_list); this function is a pure decision — it
    neither inverts nor mutates state.

    `ds_alpha` may be a scalar (uniform weight, the "global"/"none"
    schedules) or a per-arm array (the "parent" schedule) — numpy
    broadcasting handles both identically.

    `cov_dtype` (config.search.cov_dtype) fixes the precision
    `q_embeds` is cast to before multiplying against `V_inv`, so the
    einsum below runs at a controlled precision rather than whatever
    NumPy's mixed-dtype promotion happens to pick.

    Ties (within `tol`) are broken by uniform random sampling, which
    avoids the systematic bias of picking the first argmax.

    Returns `(best_idx, best_score)` — the winning arm's index and its
    combined `beta*score + alpha*diversity` value (the latter for the
    caller to log as phase_selected_score).
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

    # Also return the winning combined score so the caller can record
    # it (phase_selected_score) without recomputing the diversity term.
    return best_idx, float(q_vals[best_idx])


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

    `raw` is a (seq_len, dim) tensor of per-token hidden states. With
    embeds_source="prm" it comes from the PRM (`prm.embed(...)`, one
    row per candidate); in the policy path it's
    `llm_vllm_embeds.encode(..., "token_embed")`. Both feed this same
    function so the two sources pool identically.

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
                                    diversity bonus, and folding a zero
                                    vector into V is a no-op. Harmless.

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
    the comparison isolates the embedding model alone.

      policy : per-candidate vLLM pooling engine on the generator.
               Requires llm_vllm_embeds.
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
                "llm_vllm_embeds is None. The launcher builds it only "
                "when embeds_source == 'policy'."
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
    # methods below. Keeping them "private" makes the q-value invariant
    # explicit: only `update` may mutate them.
    __visit_count: int = 0
    __value_sum: float = 0.0

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0.0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update(self, value: float) -> None:
        """Single-step value update. Called once per (a) child creation
        with the PRM score, and (b) backprop through this node.
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
        # start; keeps the count convention aligned with the sibling
        # variants (every node starts at visit_count 1).
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with best-first frontier selection and embedding-based
    diversity.

    Inherits the root/question machinery from BaseTree and adds:
      - V / V_inv: the covariance and its inverse, fed to the diversity
        bonus; maintained per cov_update (see __init__ /
        select_leaf_from_list).
      - completed_nodes: nodes that ended via EOS or length stop.
      - the algorithm methods (`expand_node`, `select_leaf_from_list`,
        `backprop`).
    """

    # Pydantic field declarations; the actual values are allocated
    # per-instance in __init__ (the shape hints are documentation, and
    # completed_nodes must be a fresh list per question — a class-level
    # [] default is shared state waiting to happen).
    completed_nodes: List[Type[BaseNode]] = None
    V: NDArray[Shape["2048, 2048"], np.float32] = None
    V_inv: NDArray[Shape["2048, 2048"], np.float32] = None
    # Count of nodes that hit max_depth (mirrors the sibling variants'
    # cnt_node_max_depth); incremented in create_child.
    cnt_node_max_depth: int = 0
    # Winning frontier score of the most recent select_leaf_from_list
    # call; stashed rather than returned so the selector signature is
    # unchanged. Read each phase into phase_selected_score. NaN until
    # the first selection.
    last_selected_score: float = float("nan")
    # Precision for V/V_inv (config.search.cov_dtype), resolved to a
    # numpy dtype in __init__. MUST be declared here — MCTS is a
    # pydantic BaseModel, which raises on `self.attr = ...` for any
    # attribute not declared as a field (unlike V/V_inv above).
    cov_dtype: Any = np.float64

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-question state: a fresh completed-node list every run.
        self.completed_nodes = []
        # Ridge-regularized covariance V = lam*I + sum u u^T. The
        # diversity term sqrt(x^T V^-1 x) needs V^-1; how it's
        # maintained across selections depends on cov_update (see
        # select_leaf_from_list):
        #   "exact"            keep V; recompute V^-1 = inv(V) each
        #                      selection. O(d^3) per selection.
        #   "sm"               keep V^-1 directly and rank-1 update it
        #                      per selection (O(d^2)); V isn't needed.
        # Either way the start state is V_0 = lam*I, so
        # V_0^-1 = (1/lam)*I in closed form (no inverse call), and the
        # initial diversity term is uniform across arms.
        embeds_dim = self.config.search.embeds_dim
        lam = self.config.search.lam
        # cov_dtype fixes V/V_inv's precision explicitly (default
        # "fp64" matches the previous implicit behavior: np.eye() with
        # no dtype= already defaulted to float64).
        cov_dtype_cfg = self.config.search.cov_dtype
        if cov_dtype_cfg not in _COV_DTYPES:
            raise ValueError(f"unknown cov_dtype: {cov_dtype_cfg!r}")
        self.cov_dtype = _COV_DTYPES[cov_dtype_cfg]
        if self.config.search.cov_update == "sm":
            self.V = None
            self.V_inv = (1.0 / lam) * np.eye(
                embeds_dim, dtype=self.cov_dtype
            )
        else:
            self.V = lam * np.eye(embeds_dim, dtype=self.cov_dtype)
            self.V_inv = (1.0 / lam) * np.eye(
                embeds_dim, dtype=self.cov_dtype
            )

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

    def _fold_covariance(self, selected_node):
        """Fold the selected node's embedding into the covariance so
        future selections see it. Runs on EVERY selection (including
        terminal picks — the search committed a selection to that
        direction), skipped only when the node has no embedding: the
        root, which is selected alone on the first iteration before
        anything is in V.

        Cast to cov_dtype so u's precision matches V/V_inv exactly —
        otherwise NumPy's mixed-dtype promotion decides silently.
        """
        if selected_node.embeds is None:
            return
        u = selected_node.embeds.reshape(-1, 1).astype(self.cov_dtype)
        if self.config.search.cov_update == "sm":
            # Persistent rank-1 inverse update (O(d^2)), then symmetrize
            # to stop floating-point asymmetry compounding over the run.
            #   (V + uu^T)^-1 = V^-1 - (V^-1 u)(V^-1 u)^T / (1 + u^T V^-1 u)
            Vu = self.V_inv @ u
            denom = 1.0 + float(u.T @ Vu)
            self.V_inv = self.V_inv - (Vu @ Vu.T) / denom
            self.V_inv = 0.5 * (self.V_inv + self.V_inv.T)
        else:
            # Exact: accumulate V, recompute its inverse from scratch.
            # solve(V, I) over inv(V): same O(d^3) cost, slightly
            # better-conditioned (avoids explicitly forming the inverse
            # via a less stable routine).
            self.V = self.V + u @ u.T
            self.V_inv = np.linalg.solve(
                self.V, np.eye(self.V.shape[0], dtype=self.cov_dtype)
            )

    def select_leaf_from_list(self, nodes: List[Any], t: int):
        """Pick the frontier node maximizing
        ds_beta*q + ds_alpha*sched*sqrt(x^T V^-1 x), then fold the
        winner's embedding into the covariance. Operates on the
        BL-MCTS global leaf frontier — the sem analog of
        mcts_bl_cnt's select_child_from_list.

        `t` is the global iteration counter (frontier selections so
        far); only the "global" schedule reads it. Every value is
        recomputed fresh each call: the schedule multiplier is shared
        across the frontier under "global"/"none", so per-node
        differentiation comes from q and the V^-1 geometry alone.
        """
        # Singleton frontier: nothing to compare against — select
        # directly. Common on the first real selection if root's
        # expansion deduped to one child; can also recur later
        # whenever the frontier drains to a single leaf before the
        # next expansion refills it.
        if len(nodes) == 1:
            selected_node = nodes[0]
            # No comparison happened; record the node's own q as the
            # "winning score" (the diversity term is undefined without
            # a competing set).
            self.last_selected_score = float(selected_node.q_value())
            self._fold_covariance(selected_node)
            logging.fatal(f"selected_leaf = {selected_node.tag}")
            return selected_node

        q_values: List[float] = []
        embeds: List[Any] = []
        for node in nodes:
            q_values.append(node.q_value())
            embeds.append(node.embeds)
            logging.fatal(f"{node.tag}")
            logging.fatal(f"   q-value = {node.q_value():0.4f}")
            logging.fatal(f"   nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {node.is_terminal}")

        schedule = self.config.search.ds_alpha_schedule
        if schedule == "global":
            # Shared clock: one scalar for the whole frontier.
            sched = np.sqrt(np.log(1 + t))
        elif schedule == "parent":
            # Literal sem_v02 transplant: per-node parent visits.
            sched = np.sqrt(np.log(1 + np.array(
                [node.parent.visit_count() for node in nodes]
            )))
        elif schedule == "none":
            sched = 1.0
        else:
            raise ValueError(
                f"unknown ds_alpha_schedule: {schedule!r}"
            )
        logging.fatal(f"t = {t}  sched = {sched}")

        best_idx, best_score = _diverse_select(
            self.V_inv, embeds, q_values,
            self.config.search.ds_alpha * sched,
            self.config.search.ds_beta,
            cov_dtype=self.cov_dtype,
        )
        self.last_selected_score = best_score
        selected_node = nodes[best_idx]
        self._fold_covariance(selected_node)
        logging.fatal(f"selected_leaf = {selected_node.tag}")
        return selected_node

    # ----- Backprop -------------------------------------------------- #

    def backprop(self, node):
        """Recursive q-value backprop up to (and including) the root."""
        node.update_recursive(node.q_value(), self.root)


# --------------------------------------------------------------------- #
# Candidate generation                                                  #
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

    Model calls per invocation:
      1. `generate_k_steps` produces `config.search.batch_size`
         continuations.
      2. After dedup-by-text, `_embed_candidates` pools a diversity
         embedding per unique candidate (PRM forward pass by default).
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

    # Embed each candidate from the configured source (prm default —
    # the PRM's last-layer hidden states over the plain candidate
    # chat; or policy — the second vLLM pooling engine).
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


# --------------------------------------------------------------------- #
# Main search loop                                                      #
# --------------------------------------------------------------------- #

def _count_nodes(root) -> int:
    """Total nodes in the tree rooted at `root` (iterative)."""
    total, stack = 0, [root]
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def mcts_search(question, agent, config, llm_vllm, llm_vllm_embeds, prm):
    """Run budget-limited best-first semantic MCTS on one `question`.

    Outer loop: `config.search.num_phases` iterations (safety cap).
    Each iteration expands the current node (or backprops it, if
    terminal), then selects the next node by diversity-adjusted value
    globally across the leaf frontier — generate -> expand -> select,
    mirroring mcts_bl_cnt_search_v01_00_00.py's walk step, with the
    freshly expanded children competing against every older leaf at
    that selection. Only expansions charge gen_cnt. `t` (selections
    so far, 1-indexed) feeds the "global" ds_alpha schedule.
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

    gen_cnt = 0
    p = 0
    t = 0
    phase_depths: List[int] = []
    # Per-phase exploration diagnostics (one entry per selection). See
    # the results-dict assembly in _search for scope/meaning.
    phase_selected_depth: List[int] = []
    phase_selected_q: List[float] = []
    phase_selected_score: List[float] = []
    leaf_nodes: List[Any] = []
    current_node = agent.root

    for p in range(config.search.num_phases):
        logging.fatal(f"\n-> p = {p}")

        if current_node.is_terminal:
            logging.fatal(f"current_node.is_terminal = True")
            agent.backprop(current_node)
            phase_depths.append(current_node.depth)
        else:
            gen_cnt += 1
            infos, embeds, scores = _generate_candidates(
                question, current_node, current_node.depth, p, config,
                tokenizer, llm_vllm, llm_vllm_embeds, prm,
                response_start_idx, sampling_params,
            )
            agent.expand_node(current_node, infos, embeds, scores, p, gen_cnt)
            for child in current_node.children:
                leaf_nodes.append(child)

        logging.fatal(f"gen_cnt = {gen_cnt}")
        if gen_cnt >= config.search.gen_budget:
            logging.fatal("run out of budget!")
            break

        if not leaf_nodes:
            logging.fatal("leaf_nodes is empty — stopping.")
            break

        # Select the next node with highest diversity-adjusted value
        # across the entire frontier — the children expanded above
        # compete against every older leaf here. t only feeds the
        # "global" ds_alpha schedule; the winner's embedding is folded
        # into V inside the call.
        t += 1
        current_node = agent.select_leaf_from_list(leaf_nodes, t)
        leaf_nodes.remove(current_node)

        # Record which node this phase chose to expand: depth (the
        # shallow-vs-deep signal), q-value, and the winning frontier
        # score stashed inside the selector.
        phase_selected_depth.append(current_node.depth)
        phase_selected_q.append(current_node.q_value())
        phase_selected_score.append(agent.last_selected_score)

        logging.fatal(
            f"selected = {current_node.tag}  "
            f"q={current_node.q_value():.4f}  "
            f"nvisit={current_node.visit_count()}"
        )

    # Collect unique completions from completed nodes.
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

    # Tree-shape scalars (see docs/findings/exp-findings/
    # bl-frontier-depth-allocation.md): completed = EOS/length
    # terminals; terminal = completed + max-depth dead-ends; total =
    # every node created.
    q_nodes_completed = len(agent.completed_nodes)
    q_nodes_terminal = q_nodes_completed + agent.cnt_node_max_depth
    q_nodes_total = _count_nodes(agent.root)

    return (
        completions, comp_depth, comp_phase, comp_gen,
        gen_cnt, p, phase_depths, agent.cnt_node_max_depth,
        phase_selected_depth, phase_selected_q, phase_selected_score,
        q_nodes_total, q_nodes_terminal, q_nodes_completed,
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
    batch_phase_selected_depth = [[] for _ in range(n)]
    batch_phase_selected_q = [[] for _ in range(n)]
    batch_phase_selected_score = [[] for _ in range(n)]
    batch_q_nodes_total = [[] for _ in range(n)]
    batch_q_nodes_terminal = [[] for _ in range(n)]
    batch_q_nodes_completed = [[] for _ in range(n)]

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
            phase_selected_depth, phase_selected_q, phase_selected_score,
            q_nodes_total, q_nodes_terminal, q_nodes_completed,
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
        batch_phase_selected_depth[q_idx] = phase_selected_depth
        batch_phase_selected_q[q_idx] = phase_selected_q
        batch_phase_selected_score[q_idx] = phase_selected_score
        batch_q_nodes_total[q_idx] = q_nodes_total
        batch_q_nodes_terminal[q_idx] = q_nodes_terminal
        batch_q_nodes_completed[q_idx] = q_nodes_completed

    # Output keys use scope prefixes: comp_* = per completion,
    # q_* = per-question scalar, phase_* = per-question array
    # over phases. Key names match the sibling variants' results dict
    # so downstream scoring/metrics read all algorithms identically.
    results: Dict[str, Any] = defaultdict(list)
    results["completions"] = batch_completions
    # comp_depth: per completion, tree depth at which it finished.
    results["comp_depth"] = batch_comp_depth
    # comp_phase: per completion, MCTS phase it finished in.
    results["comp_phase"] = batch_comp_phase
    # comp_gen: per completion, generation count when it finished.
    results["comp_gen"] = batch_comp_gen
    # q_total_gens: per question, total generations used (budget).
    results["q_total_gens"] = batch_q_total_gens
    # q_last_phase: per question, final iteration index reached.
    results["q_last_phase"] = batch_q_last_phase
    # phase_depths: per question, depth of the selected leaf on each
    # iteration that backprops (not appended on an expand iteration —
    # this frontier-based search has no fixed root-to-leaf walk).
    results["phase_depths"] = batch_phase_depths
    # q_nodes_max_depth: per question, # nodes that hit max depth.
    results["q_nodes_max_depth"] = batch_q_nodes_max_depth
    # --- exploration diagnostics (added 2026-07-20; see docs/findings/
    # exp-findings/bl-frontier-depth-allocation.md). Same keys across
    # all bl_* variants so downstream reads them uniformly. ---
    # phase_selected_depth: per question, per-phase depth of the node
    # chosen for expansion — the shallow-vs-deep exploration signal.
    results["phase_selected_depth"] = batch_phase_selected_depth
    # phase_selected_q: per question, per-phase q-value of that node.
    results["phase_selected_q"] = batch_phase_selected_q
    # phase_selected_score: per question, per-phase WINNING frontier
    # score (per-family; within-method diagnostic, NOT cross-comparable).
    results["phase_selected_score"] = batch_phase_selected_score
    # q_nodes_total: per question, total nodes created.
    results["q_nodes_total"] = batch_q_nodes_total
    # q_nodes_terminal: per question, # terminal nodes (completed +
    # max-depth dead-ends).
    results["q_nodes_terminal"] = batch_q_nodes_terminal
    # q_nodes_completed: per question, # EOS/length-completed nodes.
    results["q_nodes_completed"] = batch_q_nodes_completed
    return results
