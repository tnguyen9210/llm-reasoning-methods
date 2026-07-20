"""
Budget-Limited MCTS with best-first leaf selection (count-based, no
embeddings) -- v02: delayed-eager terminal backprop + a selectable
path-aware frontier score (score_mode: one-hop parent blend, or
full-path decayed subtree value).

Sibling: mcts_bl_cnt_search_v01_00_00.py (PUCT, unmodified). v02
changes two things relative to v01:

  1. Terminal split + DELAYED eager backprop. v01 pushes every
     freshly created child -- terminal or not -- onto `leaf_nodes`; a
     terminal only backprops once it later WINS a frontier selection
     (`current_node.is_terminal` branch at the top of the next loop
     iteration) -- possibly never. v02 instead QUEUES a terminal
     child (never puts it on the frontier at all) and flushes the
     queue (backprops every queued node) right after the VERY NEXT
     selection resolves -- not immediately at creation. This is
     deliberate, not an implementation shortcut (Tuan raised the
     distinction, 2026-07-18): backpropping a same-batch terminal
     sibling immediately, before the selection that's choosing among
     its non-terminal siblings, would let information causally
     CONCURRENT with those candidates (produced by the exact same
     _generate_candidates call) leak into the selection ranking them
     -- arguably not genuinely "prior" evidence at all. Delaying the
     flush by exactly one step means a terminal's outcome influences
     every selection AFTER the one immediately following its own
     creation, never that one -- still bounded, still-fast
     propagation (one step, not the unbounded "maybe never" of v01's
     lazy scheme), but never same-step. Non-terminal children still
     go to `leaf_nodes` as before.
  2. Path-aware frontier score, selectable via config
     `score_mode` (added 2026-07-19). Motivation: backprop timing
     alone doesn't change v01's ranking -- `puct()` reads a leaf's
     own frozen q plus its PARENT's visit count, never the parent's
     q, so a backpropagated value is write-only regardless of when
     it lands. Both modes below make ancestor values actually read
     by later selections; they are arms of one planned sweep, after
     which the losing mode is expected to be DELETED. The two
     scorers deliberately share no code and are joined only by the
     MCTS.frontier_score dispatcher, so that removal is a pure
     deletion touching zero lines of the survivor.

Frontier score, score_mode="parent_blend" (default) -- one-hop blend:
    blended_q(x) = alpha * q_value(x) + (1 - alpha) * q_value(parent(x))
    path_aware_puct(x) = blended_q(x) + u(x)
    u(x) = cpuct * sqrt(log(parent_visits(x)) / visits(x))   (v01's u,
           unchanged)

    alpha in [0, 1]; alpha = 1.0 recovers v01's puct() exactly (the
    parent term drops out) -- the built-in control arm for a sweep,
    and the ONLY exact-v01 arm in this file (path_decay has none).
    If parent has visit_count() == 0 (shouldn't happen once root is
    seeded with one update, but guarded anyway), falls back to the
    leaf's own q_value so blended_q reduces to plain q_value rather
    than dividing by zero or reading an undefined mean.

Frontier score, score_mode="path_decay" -- full-path decayed value:
    q_path(x) = sum_k gamma^k * q_value(ancestor_k(x)) / sum_k gamma^k
                (k = 0 at the leaf x itself, walking to the root)
    path_decay_score(x) = q_path(x) + u(x)
    u(x) = cpuct * sqrt(parent_visits(x)) / (1 + visits(x))
           (AlphaZero shape -- NOT v01's UCB1 form, so cpuct is NOT
           comparable across the two modes; sweep it per mode)

    gamma in [0, 1]; gamma = 1.0 is a plain path average, gamma = 0.0
    reads only the leaf's own q (still with the AZ-shaped u -- not a
    v01 control arm). A backpropagated ancestor value anywhere on
    the lineage is read directly, discounted by distance.

The cross-mode knob is idle by design: alpha is unused under
"path_decay", gamma under "parent_blend".

Everything else -- expansion, backprop, node classes, candidate
generation, output shape -- is unchanged from v01. The loop shape
(generate -> expand -> select) is also unchanged; only WHAT gets
added to leaf_nodes at expand time (non-terminals only, not all
children), WHEN a terminal's backprop is applied (delayed one step,
not immediate), and WHAT select_child_from_list scores (the
score_mode-selected frontier score, not puct) differ.

Algorithm
    Initialize completion_list = [], leaf_nodes = [], gen_cnt = 0,
        current = root, pending = []  (queued terminal children,
        NOT yet backpropped -- see "Delayed-eager flush timing" below)
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
        Select: current = argmax_{x in leaf_nodes}
                    frontier_score(x)   (per score_mode);
                remove current from leaf_nodes
        For each node in pending:
            Backprop: update_recursive(node.q_value(), root)
        pending = []

    The `current.is_terminal` check at the top of the loop is now a
    defensive no-op in the common case (freshly created terminals
    never reach the frontier, so they're never selected as `current`
    via that path) -- kept only because `current` can start as
    `agent.root` on iteration 0, and root COULD in principle be
    terminal (max_depth == 0), a boundary case v01 also guards the
    same way.

    Delayed-eager flush timing: the flush ("For each node in
    pending...") happens AFTER Select, so the selection that picks
    `current` for THIS iteration uses only state that predates THIS
    iteration's own Expand call -- a same-batch terminal sibling
    cannot influence the selection choosing among its own siblings.
    The flush also runs before EITHER of the loop's early-exit points
    (gen_budget exhausted, leaf_nodes empty) -- a phase whose entire
    batch of new children is terminal (leaf_nodes stays empty, the
    loop is about to break) still gets its pending backprops applied
    before that break, so no queued node is ever silently dropped.

Selection criterion: the score_mode-selected frontier score -- see
the two "Frontier score" blocks above and MCTS.frontier_score.

Variant lineage: docs/algorithms.md,
docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md (original v02
record; delayed-vs-immediate flush timing decided 2026-07-18), and
docs/decisions-log.md 2026-07-19 (score_mode: the two selectable
scores, and the design-for-deletion rationale).
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

    def path_aware_puct(self, cpuct=2, alpha=1.0) -> float:
        """v02 selection index: blend own q_value with the parent's,
        then add v01's unchanged UCB1 exploration term.

        blended_q = alpha*q_value(self) + (1-alpha)*q_value(parent)
        alpha=1.0 recovers v01's puct() exactly. If parent has never
        been updated (visit_count()==0 -- shouldn't happen once root
        is seeded, guarded anyway), fall back to the leaf's own
        q_value rather than reading an undefined mean.
        """
        if not self.parent:
            return 0.0
        own_q = self.q_value() if self.visit_count() > 0 else 0.0
        if self.parent.visit_count() > 0:
            parent_q = self.parent.q_value()
        else:
            parent_q = own_q
        blended_q = alpha * own_q + (1 - alpha) * parent_q

        parent_visits = self.parent.visit_count()
        visits = self.visit_count()
        if parent_visits == 0 or visits == 0:
            u = 0.0
        else:
            u = cpuct * np.sqrt(np.log(parent_visits) / visits)
        return blended_q + u

    def path_decay_score(self, cpuct=2, gamma=0.8) -> float:
        """score_mode="path_decay" index: gamma-decayed average of
        q_value along the leaf->root path, plus an AlphaZero-shaped
        exploration term.

        q_path = sum_k gamma^k * q(ancestor_k) / sum_k gamma^k
                 (k = 0 at this leaf, increasing toward the root;
                 the k=0 weight is 1 even at gamma=0, so norm > 0
                 always)
        u = cpuct * sqrt(N_parent) / (1 + N_leaf)

        The u term's shape is NOT v01's UCB1 (no log, N_leaf in a
        plain denominator), so cpuct here is not comparable to
        parent_blend's cpuct. gamma=1 is a plain path average;
        gamma=0 reads only the leaf's own q (still with this u, so
        NOT a v01 control arm).

        Deliberately shares NO code with path_aware_puct: the two
        score_modes are sweep arms and the loser is expected to be
        deleted afterward -- independence keeps that removal a pure
        deletion (see MCTS.frontier_score).
        """
        if not self.parent:
            return 0.0
        node, acc, norm, k = self, 0.0, 0.0, 0
        while node is not None:
            q = node.q_value() if node.visit_count() > 0 else 0.0
            acc += (gamma ** k) * q
            norm += gamma ** k
            k += 1
            node = node.parent
        q_path = acc / norm

        u = (cpuct * np.sqrt(self.parent.visit_count())
             / (1 + self.visit_count()))
        return q_path + u

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
        # Seed root with one update so visit_count >= 1 from the start;
        # prevents the visit_count==1 PUCT special case from triggering
        # on the wrong iteration.
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with a score_mode-selectable path-aware frontier score
    and delayed-eager terminal backprop.

    Holds `completed_nodes` (EOS/length-terminated leaves) and the
    algorithm methods: `expand_node`, `frontier_score`,
    `select_child`, `backprop`.
    """
    completed_nodes: List[Type[BaseNode]] = []
    cnt_node_max_depth: int = 0

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
        caller can split it into terminal/non-terminal without
        relying on `current_node.children` holding nothing else --
        each node is expanded exactly once, but returning the batch
        directly keeps that invariant local to this method.
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
        """
        s = self.config.search
        if s.score_mode == "parent_blend":
            return node.path_aware_puct(s.cpuct, s.alpha)
        if s.score_mode == "path_decay":
            return node.path_decay_score(s.cpuct, s.gamma)
        raise ValueError(
            f"unknown score_mode: {s.score_mode!r} "
            "(expected 'parent_blend' or 'path_decay')"
        )

    def select_child(self, node, from_root: bool = False):
        """Pick the child with the highest frontier score (per
        score_mode), uniform random tie-break. Returns None if no
        children.
        """
        best_value = -float("inf")
        best_childs: List[Any] = []

        for child_node in node.children:
            score = self.frontier_score(child_node)
            if score == best_value:
                best_childs.append(child_node)
            elif score > best_value:
                best_value = score
                best_childs = [child_node]

            logging.fatal(f"{child_node.tag}")
            q = child_node.q_value()
            logging.fatal(f"   q-value = {q:0.4f}")
            logging.fatal(f"   score - q = {score - q:0.4f}")
            logging.fatal(f"   score = {score:0.4f}")
            logging.fatal(f"   nvisit = {child_node.visit_count():0.2f}")
            logging.fatal(f"   parent.nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {child_node.is_terminal}")

        if not best_childs:
            return None
        selected_node = random.choice(best_childs)
        logging.fatal(f"selected_child = {selected_node.tag}")
        return selected_node

    def select_child_from_list(self, nodes: List[Any]):
        """Pick the node with the highest frontier score (per
        score_mode) from an arbitrary list, uniform random
        tie-break. Same logic as `select_child` but operates on the
        BL-MCTS global leaf frontier.
        """
        best_value = -float("inf")
        best_nodes: List[Any] = []

        for node in nodes:
            score = self.frontier_score(node)
            if score == best_value:
                best_nodes.append(node)
            elif score > best_value:
                best_value = score
                best_nodes = [node]

            logging.fatal(f"{node.tag}")
            logging.fatal(f"   q-value = {node.q_value():0.4f}")
            logging.fatal(f"   score - q = {score - node.q_value():0.4f}")
            logging.fatal(f"   score = {score:0.4f}")
            logging.fatal(f"   nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {node.is_terminal}")

        selected_node = random.choice(best_nodes)
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

def mcts_search(question, agent, config, llm_vllm, prm):
    """Run budget-limited best-first MCTS on a single `question`.

    Outer loop: `config.search.num_phases` iterations (safety cap).
    Each iteration expands the current node (or backprops it, if
    terminal), then selects the next node by the score_mode-selected
    frontier score globally across the leaf frontier — generate ->
    expand -> select, mirroring
    mcts_bl_cnt_search_v01_00_00.py's walk step. Newly created
    terminal children are queued (delayed eager backprop -- flushed
    right after the selection immediately following their own
    creation, see pending_terminal_backprops below) and never enter
    the frontier; only non-terminal children are added to leaf_nodes
    and compete against every older leaf at the next selection. Only
    expansions charge gen_cnt.
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
    leaf_nodes: List[Any] = []
    current_node = agent.root
    # Delayed-eager backprop queue: this step's newly created terminal
    # children are held here (not backpropped yet) so THIS step's
    # selection reads only state that existed before this step's own
    # generation call -- backpropping a same-batch terminal sibling
    # immediately would let information causally concurrent with the
    # candidates being ranked ("peek" at a sibling from the exact
    # generation call that produced the candidates themselves) leak
    # into the selection choosing among them. Flushed unconditionally
    # right after selection resolves (or right before any loop exit,
    # so a phase whose children are ALL terminal still gets its
    # backprops applied even though leaf_nodes ends up empty).
    pending_terminal_backprops: List[Any] = []

    for p in range(config.search.num_phases):
        logging.fatal(f"\n-> p = {p}")

        # Defensive: `current_node` starts as `agent.root` on
        # iteration 0, and root COULD in principle be terminal
        # (max_depth == 0) -- same boundary case v01 guards. In the
        # common case this branch is dead: freshly created terminals
        # are queued below and never reach the frontier, so they are
        # never selected as `current_node` via this path.
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
            # selection immediately below. The queue is flushed after
            # that selection resolves (see below and the loop-exit
            # points), so this step's own selection is unaffected by
            # this step's own terminal outcomes, but the VERY NEXT
            # selection already sees them -- one step of latency, not
            # the unbounded latency of the original lazy scheme (a
            # terminal that's never SELECTED under lazy backprop
            # contributes nothing, ever).
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

        # Select the next node with the highest frontier score (per
        # score_mode) across the entire frontier — the children
        # expanded above compete against every older leaf here,
        # using only state that existed before THIS step's own
        # generation call (see the pending_terminal_backprops
        # comment above).
        current_node = agent.select_child_from_list(leaf_nodes)
        leaf_nodes.remove(current_node)

        # Flush now: this step's terminal outcomes are no longer
        # "concurrent" with a selection in progress -- the next
        # iteration's selection (and, if current_node is itself
        # terminal, that very node's own backprop) is free to read
        # them.
        for child in pending_terminal_backprops:
            agent.backprop(child)
        pending_terminal_backprops = []

        selected_score = agent.frontier_score(current_node)
        logging.fatal(
            f"selected = {current_node.tag}  "
            f"score={selected_score:.4f}  "
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

    return (
        completions, comp_depth, comp_phase, comp_gen,
        gen_cnt, p, phase_depths, agent.cnt_node_max_depth,
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
        ) = mcts_search(question, agent, config, llm_vllm, prm)

        batch_completions[q_idx] = completions
        batch_comp_depth[q_idx] = comp_depth
        batch_comp_phase[q_idx] = comp_phase
        batch_comp_gen[q_idx] = comp_gen
        batch_q_total_gens[q_idx] = q_total_gens
        batch_q_last_phase[q_idx] = q_last_phase
        batch_phase_depths[q_idx] = phase_depths
        batch_q_nodes_max_depth[q_idx] = q_nodes_max_depth

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
    return results
