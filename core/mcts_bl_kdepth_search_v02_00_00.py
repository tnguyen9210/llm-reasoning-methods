"""
Budget-Limited MCTS with best-first leaf selection (depth-shaping
knapsack density, no embeddings) -- v02: delayed-eager terminal
backprop AND a selectable path-aware value term (score_mode).

Sibling: mcts_bl_kdepth_search_v01_00_00.py (unmodified). v02 applies
the SAME terminal-split + delayed-eager-backprop fix as
mcts_bl_cnt_search_v02_00_00.py / mcts_bl_kube_search_v02_00_00.py
(see mcts_bl_cnt_search_v02_00_00.py's docstring for the full
"why delayed, not immediate" rationale) -- a max-depth dead-end
always has cost <= 0, so it scores -inf and (in v01) sits in
leaf_nodes forever, permanently satisfying the kube_affordable
filter's "always eligible" clause and silently disabling that
filter's own empty-set fallback (see
docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.1,
written for the KUBE sibling but the affordability mechanics are
identical here -- both files share the same feasibility filter).

NOTE on delayed-vs-immediate here specifically: unlike the cnt/kube
siblings, the delay is BEHAVIORALLY INERT in this file -- the
frontier value term reads q_value(s) (own, parent, or path) that are
all already-frozen before the flush, and the delayed-flush queue is
emptied strictly AFTER the immediately-following selection resolves,
so no terminal's outcome can leak into the selection ranking among
the candidates its own generation produced. The queue pattern is
kept for structural consistency with the cnt/kube v02 siblings.

Selectable value term (score_mode), aligned with
mcts_bl_kube_search_v02_00_00.py's two modes (2026-07-21). The blend
touches ONLY the density's q-term; kdepth's DEPTH bonus and cost are
its exploration signal and stay untouched. This resolves the design
doc's §7.2 "no channel to blend" objection: that is correct for the
depth BONUS (no visit/parent-visit/clock term to blend), but the
density also carries a plain q-term, which IS blendable exactly as
bl_cnt/bl_kube blend theirs (see
docs/decisions-log.md 2026-07-21). The two scorers share no code and
are joined only by MCTS.frontier_score -- the loser of the planned
sweep is expected to be deleted (a pure-deletion diff).

  score_mode="parent_blend" (default) -- one-hop blend:
      blended_q = alpha*q(leaf) + (1-alpha)*q(parent)
      density = (blended_q + depth_beta*(1-depth_frac**depth_alpha))
                / cost
    alpha=1.0 recovers v01's depth_density EXACTLY (own-q only) --
    the ONLY exact-v01 control arm.

  score_mode="path_decay" -- full-path decayed value:
      q_path = sum_k gamma^k q(ancestor_k) / sum_k gamma^k
      density = (q_path + depth_beta*(1-depth_frac**depth_alpha))
                / cost
    Same gamma-decayed leaf->root value walk as the kube sibling's
    path_decay; only the bonus differs (kdepth keeps its DEPTH bonus,
    kube uses an AlphaZero-shaped VISIT bonus). No clock/schedule
    here -- kdepth's bonus is depth-based, not visit- or time-clocked.
    gamma=0.0 reads only the leaf's own q (NOT a v01 control arm).

What this v02 actually changes, concretely:
  - A terminal child is queued at creation (inside the expand step)
    and never enters leaf_nodes, instead of waiting to win a frontier
    selection; the queue is flushed (backpropped) right after the
    very next selection resolves -- see the NOTE above on why this
    delay is inert here despite being load-bearing in the cnt/kube
    siblings.
  - The terminal-split removes only terminal-selection phases (a
    phase that, in v01, does nothing but re-select an already-known
    dead-end and immediately backprop it) -- pure overhead removed.
    Under score_mode="parent_blend", alpha=1.0, the surviving
    non-terminal ranking is exactly v01's (own-q + depth bonus).
  - The kube_affordable feasibility fallback becomes reachable again
    (dead-ends can no longer keep the "affordable" set permanently
    non-empty), and the frontier stops accumulating permanently-
    stuck dead-end clutter.

Everything else -- expansion, backprop, node classes, candidate
generation, output shape, the kube_affordable filter, the cost<=0
-inf guard, the loop shape -- is unchanged from v01. Only the value
term of the frontier density is now score_mode-selectable (the depth
bonus and cost mapping are shared across modes).

Algorithm
    Initialize completion_list = [], leaf_nodes = [], gen_cnt = 0,
        current = root, pending = []  (queued terminal children, NOT
        yet backpropped -- see the delayed-flush NOTE above)
    While gen_cnt < gen_budget:
        If current.is_terminal:
            Backprop: update_recursive(current.q_value(), root)
        Else:
            Expand: generate n next-step continuations, dedupe,
                    score with PRM, attach as current's children
            gen_cnt += 1
            For each child:
                child.update(prm_score)
                if completed (EOS/length): mark terminal,
                    add to completion_list
                if not completed and depth >= max_depth:
                    mark terminal, score = negative_reward
                If child.is_terminal:
                    pending.append(child)   -- QUEUED, not backpropped
                Else:
                    Add child to leaf_nodes
        residual = gen_budget - gen_cnt
        candidates = {x in leaf_nodes : x terminal
                      or cost(x) <= residual}   (if kube_affordable;
                      falls back to all of leaf_nodes if empty -- the
                      "x terminal" clause is now dead in practice,
                      since terminals never reach leaf_nodes, but is
                      kept rather than removed -- see
                      mcts_bl_kube_search_v02_00_00.py's docstring
                      "Note on the now-dead is_terminal clause",
                      identical situation here)
        Select: current = argmax_{x in candidates} frontier_score(x);
                remove current from leaf_nodes
        For each node in pending:
            Backprop: update_recursive(node.q_value(), root)
        pending = []

    The loop body ordering, selection SCOPE, and behavior-preservation
    of the generate -> expand -> select rotation are unchanged from
    v01 -- see that module's docstring. The frontier score is now
    score_mode-selectable (MCTS.frontier_score dispatches to the two
    node methods); the flush's timing is inert given that the value
    term reads only already-frozen q_values -- see the NOTE above.

Selection criterion: (value_term + depth bonus) / cost, where the
value term is score_mode-selectable (parent_blend / path_decay, see
above and MCTS.frontier_score); the depth bonus f_a(z) and cost
mapping are shared with v01 -- see that module's docstring.

Variant lineage: docs/algorithms.md,
docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.2
(design -- the "no channel to blend" analysis, resolved for the
q-term 2026-07-21, see docs/decisions-log.md), docs/decisions/
bl-cnt-v02-eager-backprop-path-aware.md (this implementation).
"""

import random
import logging
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from pydantic import BaseModel

from vllm import SamplingParams

from sal.config import Config   # noqa: F401  re-exported for callers
from sal.models.reward_models import PRM   # noqa: F401  re-exported
from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv, generate_k_steps


logging.basicConfig(format='%(message)s', level=logging.FATAL + 1)


# --------------------------------------------------------------------- #
# Node classes                                                          #
# --------------------------------------------------------------------- #

@dataclass(slots=True)
class BaseNode:
    """Generic tree node. Carries the running text and bookkeeping for
    depth / terminal status. `MCTSNode` extends with q-value and
    visit counts.
    """
    state: Dict[str, str] = field(
        default_factory=lambda: {"text": "", "step": "", "extra_info": ""}
    )
    parent: Optional[Any] = None
    children: List[Any] = field(default_factory=list)

    tag: str = "0"              # dotted lineage, e.g. "0.1.2"
    depth: int = 0
    phase: int = 0              # which `num_phases` outer loop made this
    gen_cnt: int = 0            # gen_budget value at creation time
    is_terminal: bool = False   # EOS / max-depth / empty completion
    is_completed: bool = False  # specifically: ended via EOS / length

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
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value, start_node) -> None:
        """Backprop: update self, then ancestors, until `start_node`."""
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def _depth_bonus(
        self, max_depth: int, depth_beta: float, depth_alpha: float,
    ) -> float:
        """kdepth's exploration signal: depth_beta*(1-depth_frac**
        depth_alpha). Shared by both score_modes -- ONLY the value
        term (q vs. blended-q vs. path-decayed q) differs between
        modes; the depth bonus and cost mapping are identical (see
        module docstring / the two density methods below).

        depth_frac = depth / max_depth: 1 at the root (max bonus),
        0 at max_depth (no bonus) -- monotonically prefers shallower
        nodes. No visit-count term -- depth replaces visits as the
        exploration pressure (this is kdepth's identity).
        """
        depth_frac = self.depth / max_depth
        return depth_beta * (1.0 - depth_frac ** depth_alpha)

    def parent_blend_depth_density(
        self, max_depth: int, depth_beta: float, depth_alpha: float,
        alpha: float,
    ) -> float:
        """v02 score_mode="parent_blend": one-hop blend of own q with
        the parent's, plus the depth bonus, divided by cost.

        density = (alpha*q(self) + (1-alpha)*q(parent)
                   + depth_beta*(1-depth_frac**depth_alpha)) / cost

        The value blend mirrors
        mcts_bl_cnt_search_v02_00_00.py's / mcts_bl_kube_search_v02_
        00_00.py's parent blend, but hooks kdepth's DEPTH bonus (not
        a visit bonus) -- kdepth has no visit/clock term to blend
        into, only the plain q-term of its density (see module
        docstring).

        alpha=1.0 recovers mcts_bl_kdepth_search_v01_00_00.py's
        depth_density() EXACTLY (own-q only) -- the built-in v01
        control arm.

        cost = max_depth - depth; nodes at or past max_depth get
        density = -inf (guard -- should already be terminal), as v01.
        """
        cost = max_depth - self.depth
        if cost <= 0:
            return -float("inf")
        own_q = self.q_value() if self.visit_count() > 0 else 0.0
        if self.parent and self.parent.visit_count() > 0:
            parent_q = self.parent.q_value()
        else:
            parent_q = own_q
        q = alpha * own_q + (1 - alpha) * parent_q
        bonus = self._depth_bonus(max_depth, depth_beta, depth_alpha)
        return (q + bonus) / cost

    def path_decay_depth_density(
        self, max_depth: int, depth_beta: float, depth_alpha: float,
        gamma: float,
    ) -> float:
        """v02 score_mode="path_decay": gamma-decayed average of
        q_value along the leaf->root path, plus the depth bonus,
        divided by cost.

        q_path = sum_k gamma^k * q(ancestor_k) / sum_k gamma^k
                 (k = 0 at this leaf, walking to the root; the k=0
                 weight is 1 even at gamma=0, so norm > 0 always)
        density = (q_path + depth_beta*(1-depth_frac**depth_alpha))
                  / cost

        The q_path value walk is IDENTICAL to
        mcts_bl_kube_search_v02_00_00.py's path_decay_kube_density;
        the two differ only in the bonus -- kdepth keeps its DEPTH
        bonus, kube uses an AlphaZero-shaped VISIT bonus. No clock
        or schedule here: kdepth's bonus is depth-based, not visit-
        or time-clocked (see module docstring).

        gamma=0.0 reads only the leaf's own q -- NOT a v01 control
        arm (v01 == parent_blend alpha=1.0). cost<=0 -> -inf.

        Deliberately shares NO code with parent_blend_depth_density:
        the two score_modes are sweep arms and the loser is expected
        to be deleted afterward -- independence keeps that removal a
        pure deletion (see MCTS.frontier_score).
        """
        cost = max_depth - self.depth
        if cost <= 0:
            return -float("inf")
        node, acc, norm, k = self, 0.0, 0.0, 0
        while node is not None:
            q = node.q_value() if node.visit_count() > 0 else 0.0
            acc += (gamma ** k) * q
            norm += gamma ** k
            k += 1
            node = node.parent
        q_path = acc / norm
        bonus = self._depth_bonus(max_depth, depth_beta, depth_alpha)
        return (q_path + bonus) / cost

    def __repr__(self):
        return (
            f"MCTSNode(state={self.state}, "
            f"is_terminal={self.is_terminal}, "
            f"nvisits={self.__visit_count})"
        )


# --------------------------------------------------------------------- #
# Tree class                                                            #
# --------------------------------------------------------------------- #

class BaseTree(BaseModel):
    """Root holder. Actual search algorithm lives on `MCTS`."""
    config: Any
    question: Optional[str] = None
    root: Optional[Type[BaseNode]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = self.create_root()
        # Seed root with one update so visit_count >= 1 from the start,
        # matching the PUCT and KUBE siblings' convention (q_value()
        # well-defined immediately, even though this variant's density
        # doesn't use visit_count itself).
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with depth-shaping knapsack selection and eager terminal
    backprop.

    Holds `completed_nodes` (EOS/length-terminated leaves) and the
    algorithm methods: `expand_node`, `select_child_from_list`,
    `backprop`.
    """
    completed_nodes: List[Type[BaseNode]] = []
    cnt_node_max_depth: int = 0
    # Winning depth-discounted KUBE density of the most recent
    # select_child_from_list call; stashed rather than returned so the
    # selector signature is unchanged. Read each phase into
    # phase_selected_score. NaN until the first selection.
    last_selected_score: float = float("nan")

    def create_node(self, parent=None):
        return MCTSNode(parent=parent)

    # ----- Expansion ------------------------------------------------- #

    def create_child(
        self, current_node, candidate_info, candidate_score,
        phase, gen_cnt,
    ):
        """Append a single child to `current_node`. Marks terminal if
        vLLM hit EOS/length, or if depth would exceed `max_depths` —
        in the latter case overwrites the score with `negative_reward`.
        """
        new_node = self.create_node(parent=current_node)
        new_node.tag = f"{current_node.tag}.{len(current_node.children) + 1}"
        new_node.depth = current_node.depth + 1
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

        if not new_node.is_terminal and new_node.depth >= self.config.search.max_depth:
            new_node.is_terminal = True
            candidate_score = self.config.search.negative_reward
            self.cnt_node_max_depth += 1

        new_node.update(candidate_score)
        current_node.children.append(new_node)
        return new_node

    def expand_node(self, current_node, candidate_infos, candidate_scores,
                    phase, gen_cnt) -> List[Any]:
        """Append one child per (info, score) pair. Returns the list
        of newly created children (this call's batch only), so the
        caller can split it into terminal/non-terminal -- mirrors
        mcts_bl_cnt_search_v02_00_00.py's expand_node exactly.
        """
        new_children = []
        for info, score in zip(candidate_infos, candidate_scores):
            new_children.append(
                self.create_child(current_node, info, score, phase, gen_cnt)
            )
        return new_children

    # ----- Selection ------------------------------------------------- #

    def frontier_score(self, node) -> float:
        """Score a frontier leaf per config.search.score_mode -- the
        single point where the mode choice is read. Each mode's
        scorer is self-contained on MCTSNode; removing a mode is one
        branch here plus one node method (the modes are sweep arms,
        and the loser is expected to be deleted after the sweep).

        No `t`/clock argument (unlike the kube sibling): kdepth's
        bonus is depth-based, not visit- or time-clocked.
        """
        s = self.config.search
        if s.score_mode == "parent_blend":
            return node.parent_blend_depth_density(
                s.max_depth, s.depth_beta, s.depth_alpha, s.alpha)
        if s.score_mode == "path_decay":
            return node.path_decay_depth_density(
                s.max_depth, s.depth_beta, s.depth_alpha, s.gamma)
        raise ValueError(
            f"unknown score_mode: {s.score_mode!r} "
            "(expected 'parent_blend' or 'path_decay')"
        )

    def select_child_from_list(self, nodes: List[Any], residual: int):
        """Pick the node with the highest depth-shaping density (per
        score_mode) from an arbitrary list, uniform random tie-break.

        density(x) = (value_term(x) + depth_beta*(1-depth_frac(x)**
        depth_alpha)) / (max_depth - depth(x)) -- see frontier_score
        / the module docstring for the two score_modes; only the
        value term differs (the depth bonus and cost are shared).

        If config.search.kube_affordable, first restrict to nodes
        whose worst-case completion cost fits the residual generation
        budget — same feasibility step as v01 (see that module's
        docstring); the `node.is_terminal` disjunct is now dead in
        practice given the terminal-split, but kept — see
        mcts_bl_kube_search_v02_00_00.py's docstring "Note on the
        now-dead is_terminal clause", identical situation here.
        """
        max_depth = self.config.search.max_depth

        if self.config.search.kube_affordable:
            affordable = [
                node for node in nodes
                if node.is_terminal
                or max_depth - node.depth <= residual
            ]
            if affordable:
                nodes = affordable

        best_value = -float("inf")
        best_nodes: List[Any] = []

        for node in nodes:
            density = self.frontier_score(node)
            if density == best_value:
                best_nodes.append(node)
            elif density > best_value:
                best_value = density
                best_nodes = [node]

            logging.fatal(f"{node.tag}")
            logging.fatal(f"   q-value = {node.q_value():0.4f}")
            logging.fatal(f"   depth = {node.depth}")
            logging.fatal(f"   cost = {max_depth - node.depth}")
            logging.fatal(f"   density = {density:.4f}")
            logging.fatal(f"   nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {node.is_terminal}")

        selected_node = random.choice(best_nodes)
        # Stash the winning density for phase_selected_score.
        self.last_selected_score = float(best_value)
        logging.fatal(f"selected_leaf = {selected_node.tag}")
        return selected_node

    # ----- Backprop -------------------------------------------------- #

    def backprop(self, node):
        """Recursive q-value backprop up to and including the root."""
        node.update_recursive(node.q_value(), self.root)


# --------------------------------------------------------------------- #
# Candidate generation                                                  #
# --------------------------------------------------------------------- #

def _generate_candidates(
    question, current_node, d, config,
    tokenizer, llm_vllm, prm, sampling_params,
):
    """Generate, dedupe, and score next-step candidates branching off
    `current_node`. Returns (candidate_infos, candidate_scores).

    Two model calls per invocation:
      1. `generate_k_steps` produces `config.search.batch_size`
         continuations.
      2. `prm.score` scores each unique candidate text.
    """
    current_text = current_node.state["text"]
    logging.error(f"current_text = {current_text}")

    # Strip the step terminator before templating, then re-append it to
    # the templated string: some templates / transformers versions trim
    # or crash on trailing "\n\n" inside apply_chat_template, but the
    # model must see the separator to continue with a next step instead
    # of emitting EOS (docs/findings/coding-findings/
    # library-version-trajectory-completeness.md, 2026-06-11).
    current_text_clean = current_text.removesuffix("\n\n")
    current_convs = [build_conv(question, current_text_clean, config.gen.system_prompt)]
    current_templated = tokenizer.apply_chat_template(
        current_convs,
        add_generation_prompt=current_node.depth == 0,
        continue_final_message=current_node.depth > 0,
        date_string=config.gen.date_string,
        tokenize=False,
    )
    if current_text.endswith("\n\n"):
        current_templated = [t + "\n\n" for t in current_templated]
    current_templated = current_templated * config.search.batch_size

    lookahead = (
        0 if d == config.search.max_depth - 1 else config.search.lookahead
    )
    llm_outputs = generate_k_steps(
        current_templated, lookahead, llm_vllm, sampling_params, 1
    )
    logging.error("llm_outputs")
    logging.error(llm_outputs)

    # Dedupe by next-step text, keeping first occurrence.
    seen: Dict[str, int] = {}
    for idx, output in enumerate(llm_outputs):
        seen.setdefault(output.next_texts[0], idx)
    candidate_infos = [llm_outputs[idx] for idx in seen.values()]

    # All candidates branch off the same question, so pass one
    # question with its full list of candidate answers:
    # questions = [question], answers = [[cand_1, cand_2, ...]].
    cand_texts = [
        current_text + output.next_texts[0]
        for output in candidate_infos
    ]
    candidate_scores = prm.score(
        [question], [cand_texts],
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

    return candidate_infos, candidate_scores


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


def mcts_search(question, agent, config, llm_vllm, prm):
    """Run budget-limited best-first MCTS on a single `question`.

    Outer loop: `config.search.num_phases` iterations (safety cap).
    Each iteration expands the current node (or backprops it, if
    terminal), then selects the next node by depth-shaping density
    globally across the leaf frontier — generate -> expand -> select,
    mirroring mcts_bl_kdepth_search_v01_00_00.py's walk step. Newly
    created terminal children are queued (delayed eager backprop,
    inert here given what depth_density reads -- see module
    docstring) and never enter the frontier; only non-terminal
    children are added to leaf_nodes. Only expansions charge gen_cnt.
    """
    tokenizer = llm_vllm.get_tokenizer()
    # Template selection (mirrors generate_bon / bon_search): default
    # is the model's NATIVE chat template — each model's own
    # in-distribution format, avoiding the cross-model confound (see
    # docs/decisions/chat-template-per-family.md).
    # llm.use_custom_template defaults True (custom) for Llama; Qwen
    # YAML groups set it False (native) — see
    # LLMConfig.use_custom_template. Either way the trailing "\n\n"
    # step separator is preserved by the strip-and-reappend in
    # _generate_candidates.
    if config.llm.use_custom_template:
        tokenizer.chat_template = config.gen.custom_chat_template

    sampling_params = SamplingParams(
        temperature=config.gen.temperature,
        max_tokens=config.gen.max_tokens,
        top_p=config.gen.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    gen_cnt = 0
    p = 0
    phase_depths: List[int] = []
    # Per-phase exploration diagnostics (one entry per selection). See
    # the results-dict assembly in _search for scope/meaning.
    phase_selected_depth: List[int] = []
    phase_selected_q: List[float] = []
    phase_selected_score: List[float] = []
    leaf_nodes: List[Any] = []
    current_node = agent.root
    # Delayed-eager backprop queue: see
    # mcts_bl_cnt_search_v02_00_00.py's identical comment for the full
    # rationale. depth_density reads neither parent.q_value() nor
    # parent.visit_count() (see module docstring), so delayed vs.
    # immediate flush timing is BEHAVIORALLY INERT here -- kept for
    # structural consistency with the cnt/kube v02 siblings (same
    # queue pattern, same invariant to reason about), not because it
    # changes anything this file's selection formula reads.
    pending_terminal_backprops: List[Any] = []

    for p in range(config.search.num_phases):
        logging.fatal(f"\n-> p = {p}")

        # Defensive: same boundary-case guard as v01 (root COULD in
        # principle be terminal at max_depth == 0). In the common
        # case this branch is dead: freshly created terminals are
        # queued below and never reach the frontier.
        if current_node.is_terminal:
            logging.fatal(f"current_node.is_terminal = True")
            agent.backprop(current_node)
            phase_depths.append(current_node.depth)
        else:
            gen_cnt += 1
            infos, scores = _generate_candidates(
                question, current_node, current_node.depth, config,
                tokenizer, llm_vllm, prm, sampling_params,
            )
            new_children = agent.expand_node(
                current_node, infos, scores, p, gen_cnt,
            )
            # Terminal split + DELAYED eager backprop: a terminal
            # child is queued, not backpropped yet, and never enters
            # leaf_nodes; only non-terminal children compete for the
            # selection immediately below. Flushed right after that
            # selection resolves (or right before any loop exit).
            for child in new_children:
                if child.is_terminal:
                    pending_terminal_backprops.append(child)
                else:
                    leaf_nodes.append(child)

        logging.fatal(f"gen_cnt = {gen_cnt}")
        if gen_cnt >= config.search.gen_budget:
            logging.fatal("run out of budget!")
            for child in pending_terminal_backprops:
                agent.backprop(child)
            pending_terminal_backprops = []
            break

        if not leaf_nodes:
            logging.fatal("leaf_nodes is empty — stopping.")
            for child in pending_terminal_backprops:
                agent.backprop(child)
            pending_terminal_backprops = []
            break

        # Select the next node with highest depth-shaping density
        # across the entire frontier — the children expanded above
        # compete against every older leaf here. residual =
        # generations left, for the affordability filter (see module
        # docstring).
        residual = config.search.gen_budget - gen_cnt
        current_node = agent.select_child_from_list(
            leaf_nodes, residual
        )
        leaf_nodes.remove(current_node)

        for child in pending_terminal_backprops:
            agent.backprop(child)
        pending_terminal_backprops = []

        # Record which node this phase chose to expand: depth (the
        # shallow-vs-deep signal), q-value, and the winning density
        # stashed inside the selector.
        phase_selected_depth.append(current_node.depth)
        phase_selected_q.append(current_node.q_value())
        phase_selected_score.append(agent.last_selected_score)

        logging.fatal(
            f"selected = {current_node.tag}  "
            f"density={agent.last_selected_score:.4f}  "
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


def _search(batch_of_questions, config, trial_idx, llm_vllm, prm):
    """Run `mcts_search` on each question sequentially.

    Per-question deterministic seed: 100_000 + trial_idx.
    Returns a defaultdict of per-question lists aligned to `q_idx`.
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
        ) = mcts_search(question, agent, config, llm_vllm, prm)

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
    # over phases.
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
    # q_last_phase: per question, final MCTS phase index reached.
    results["q_last_phase"] = batch_q_last_phase
    # phase_depths: per question, depth of the selected leaf on each
    # phase that backprops (not appended on an expand phase — this
    # frontier-based search has no fixed root-to-leaf walk per phase).
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
