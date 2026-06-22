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
    embeds_mean     : np.ndarray | None            (required if center=True)
    embeds_dim      : int                  (default: 2048; set 4096 for PRM)
    prm_embeds_layer: int                          (default: -1 = last)
    cov_update      : "exact" | "sm" (Sherman-Morrison) (default: "exact")
    revisit_policy  : "reuse" | "regenerate"       (default: "reuse")
    prm_batch_size  : int                          (default: 4)

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


# --------------------------------------------------------------------- #
# Diversity selection                                                   #
# --------------------------------------------------------------------- #

def _diverse_select(V_inv, q_embeds, q_scores, ds_alpha, ds_beta):
    """Pick the one arm maximizing `beta*score + alpha*diversity`.

    The diversity term for arm `x` is `sqrt(x^T V^-1 x)` — large when
    `x` points in a direction V has seen little of, small when V has
    accumulated many similar vectors. `V_inv` is the inverse covariance
    the caller maintains across selections (see MCTS.select_child); this
    function is a pure decision — it neither inverts nor mutates state.

    Ties (within `tol`) are broken by uniform random sampling, which
    avoids the systematic bias of picking the first argmax.

    (Single-arm selection: the caller always picks one child at a time.
    A K>1 batch variant would need a sequential within-call update of
    V_inv between picks — Sherman-Morrison, not a re-inversion — but no
    caller needs that today.)
    """
    q_embeds = np.asarray(q_embeds)
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

    Five gated steps, in this order:
      1. scope     pick which tokens contribute (full sequence or only
                   the assistant response).
      2. pool      reduce to (dim,): last token or mean over scope.
      3. project   optional sparse random projection to embeds_dim
                   (JL near-isometry; fixed matrix per run). Linear, so
                   it composes cleanly with centering below.
      4. center    optional mean subtraction (a held-out mean, in the
                   POST-projection space, to remove the embeddings'
                   shared anisotropic offset). Done before normalize:
                   the mean must be subtracted in the linear space.
      5. normalize optional L2 normalization (puts embeddings on the
                   unit sphere so the diversity term reads direction,
                   not magnitude). Non-linear, so it must come last.

    Steps 3-5 run on the numpy side; pooling/scope on the torch tensor.
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

    # 4. Center. The held-out mean lives in the post-projection space,
    # so its shape must match the current (projected) dim — guard so a
    # raw-space mean can't be silently subtracted from projected vecs.
    if sc.embeds_center:
        if sc.embeds_mean is None:
            raise ValueError("embeds_center=True requires search.embeds_mean")
        mean = np.asarray(sc.embeds_mean)
        if mean.shape[-1] != pooled.shape[-1]:
            raise ValueError(
                "embeds_mean dim "
                f"{mean.shape[-1]} != embedding dim {pooled.shape[-1]}; "
                "when embeds_proj='sparse' the mean must be computed in "
                "the post-projection space (same fixed projection)."
            )
        pooled = pooled - mean

    # 5. Normalize (L2), on the numpy side so it stays after centering.
    if sc.embeds_normalize:
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

    return pooled


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
        return embeds

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
        # One batched forward pass for all candidates of this question.
        # prm.embed returns [question][answer] -> (seq_len, dim) tensors;
        # one question here, so [0] is the per-candidate list. The PRM
        # builds the plain candidate chat with config.gen.system_prompt
        # so the embedded text matches the policy path's.
        raw_embeds = prm.embed(
            [question], [candidate_texts],
            system_prompt=config.gen.system_prompt,
            batch_size=sc.prm_batch_size,
            layer=getattr(sc, "prm_embeds_layer", -1),
        )[0]
        # response_start_idx is unused under "full" scope; pass it
        # through so _extract_embeds keeps one signature.
        return [
            _extract_embeds(raw, config, response_start_idx)
            for raw in raw_embeds
        ]

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
    V: NDArray[Shape["2048, 2048"], np.float32] = None
    V_inv: NDArray[Shape["2048, 2048"], np.float32] = None
    # Count of nodes that hit max_depth (mirrors mcts_cnt_search_v05's
    # cnt_node_max_depth); incremented in create_child.
    cnt_node_max_depth: int = 0

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
        embeds_dim = self.config.search.embeds_dim
        lam = self.config.search.lam
        if self.config.search.cov_update == "sm":
            self.V = None
            self.V_inv = (1.0 / lam) * np.eye(embeds_dim)
        else:
            self.V = lam * np.eye(embeds_dim)
            self.V_inv = (1.0 / lam) * np.eye(embeds_dim)

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
            self.V_inv, embeds, q_values,
            self.config.search.ds_alpha * log_nvisit_parent,
            self.config.search.ds_beta,
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
        # including the first-visit q-value path that never reads V_inv.
        # That path still commits to a child, so its direction must
        # enter the covariance or V_inv would go stale (no longer equal
        # inv(V)) and later diversity bonuses would be wrong. This is
        # the one place exact and sm (Sherman-Morrison) differ:
        u = selected_node.embeds.reshape(-1, 1)
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
            # via a less stable routine). This is the O(d^3)-per-
            # selection baseline; sm above is the fast path.
            self.V = self.V + u @ u.T
            self.V_inv = np.linalg.solve(
                self.V, np.eye(self.V.shape[0])
            )
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
    current_convs = [
        build_conv(question, current_text, config.gen.system_prompt)
    ]
    current_templated = tokenizer.apply_chat_template(
        current_convs,
        add_generation_prompt=current_node.depth == 0,
        continue_final_message=current_node.depth > 0,
        date_string=config.gen.date_string,
        tokenize=False,
    )
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
    # avoiding the cross-model confound (docs/decisions.md 2026-06-13).
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
