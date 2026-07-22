"""
Budget-Limited MCTS with best-first leaf selection (fractional-KUBE,
no embeddings) -- v02: delayed-eager terminal backprop + a selectable
path-aware frontier score (score_mode: one-hop parent blend, or
full-path decayed subtree value; the schedules differ only in the
bonus's clock, the score_modes only in the value term + bonus shape).

Sibling: mcts_bl_kube_search_v01_00_00.py (unmodified). v02 applies
the same terminal-split + delayed-eager-backprop change as
mcts_bl_cnt_search_v02_00_00.py (see that module's docstring for the
full "why delayed, not immediate" rationale), plus the same Option 1
parent-blend on the value term -- under BOTH of KUBE's bonus
schedules, as of 2026-07-18. (As first shipped, the blend was
"parent"-schedule-only; see the history note in item 2 below and
docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md for why that
scoping was revisited the same day.)

  1. Terminal split + DELAYED eager backprop (both schedules). v01
     has a structural defect the design doc's §7.1 flags directly: a
     max-depth dead-end always has cost <= 0, so kube_density()
     returns -inf and the dead-end is NEVER selected while any
     finite-density node remains -- it sits in leaf_nodes forever,
     scanned every selection round, and (worse) its permanent
     is_terminal==True membership permanently satisfies the
     kube_affordable feasibility filter's "always eligible" clause,
     silently disabling that filter's own empty-set fallback for the
     rest of the run. v02 fixes this directly: a terminal child is
     QUEUED (not backpropped yet) and never enters leaf_nodes, so it
     can no longer prop up kube_affordable's non-emptiness or
     accumulate as frontier clutter. The queue is flushed
     (backpropped) right after the VERY NEXT selection resolves --
     not immediately at creation -- so this step's own selection
     never reads a value produced by this same step's own generation
     call (see mcts_bl_cnt_search_v02_00_00.py's docstring for why
     that distinction matters, now under both schedules since the
     blend below reads parent q under both).

  2. Path-aware frontier score -- BOTH schedules (the blend is
     schedule-independent; only the bonus's clock differs). Per the
     design doc §7.1: kube_schedule="parent"'s bonus term is
     "exactly bl_cnt v01's PUCT bonus" (module docstring line ~76 in
     v01), so the identical Option 1 parent-blend from
     mcts_bl_cnt_search_v02_00_00.py drops in directly -- blend the
     leaf's own q_value with its immediate parent's q_value by
     `alpha`, apply the schedule's KUBE bonus term unchanged, then
     divide by cost (the blend happens on the numerator's q term,
     same position the plain q term occupies in the unmodified
     formula).

     History note (2026-07-18, same-day scoping reversal): as first
     shipped, the blend was "parent"-schedule-only, on the reasoning
     that "global"'s bonus (`kube_c*sqrt(log(1+t)/visits)`, a
     frontier-wide function of the shared clock t) has no per-node
     channel -- true for the BONUS term, but Option 1 never touches
     the bonus term; it blends the VALUE term, which exists
     identically under both schedules. On analysis, global+blend is
     actually the CLEANER test of the blend idea: under "parent", a
     backprop through parent P moves two entangled channels at once
     (the bonus clock N(P) -- the count-attraction burst §7.1
     flags -- AND the blended value q(P)); under "global", the bonus
     cannot burst (it reads only t), so the blend is the ONLY
     ancestor channel -- pure value-based discouragement of failed
     neighborhoods with no counterproductive count-attraction side
     channel. The scoping was reversed in place (no v03 bump)
     because zero kube-v02 runs had been launched or scored, and
     alpha=1.0 reproduces the pre-reversal "global" behavior
     exactly. Recorded in
     docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md and
     docs/decisions-log.md (2026-07-18).

Frontier density, score_mode="parent_blend" (default):
    blended_q(x) = alpha * q_value(x) + (1 - alpha) * q_value(parent(x))
    bonus(x) = kube_c * sqrt(log(clock(x)) / visits(x))
        clock(x) = parent_visits(x)  under kube_schedule="parent"
        clock(x) = 1 + t             under kube_schedule="global"
        (each schedule's v01 bonus, unchanged)
    path_aware_kube_density(x) = (blended_q(x) + bonus(x)) / cost(x)

    alpha in [0, 1]; alpha = 1.0 recovers v01's exact kube_density()
    under EITHER schedule (the parent term drops out) -- built-in
    control arm, and the ONLY exact-v01 arm in this file. Sweeps
    under either schedule need the alpha=1.0 arm as their no-blend
    baseline.

Frontier density, score_mode="path_decay" (aligned 2026-07-19 with
mcts_bl_cnt_search_v02_00_00.py's two score_modes):
    q_path(x) = sum_k gamma^k * q_value(ancestor_k(x)) / sum_k gamma^k
                (k = 0 at the leaf x itself, walking to the root)
    bonus(x) = kube_c * sqrt(clock(x)) / (1 + visits(x))
        (AlphaZero shape -- NOT the log form above, so kube_c is
        NOT comparable across the two score_modes; sweep per mode.
        Same clock substitution per schedule as parent_blend.)
    path_decay_kube_density(x) = (q_path(x) + bonus(x)) / cost(x)

    Under kube_schedule="parent" this is exactly bl_cnt v02's
    path_decay_score divided by cost; "global" swaps its shared
    clock into the same shape. gamma in [0, 1]; gamma = 1.0 is a
    plain path average, gamma = 0.0 reads only the leaf's own q
    (still with the AZ-shaped bonus -- not a v01 control arm).

The two score_modes are arms of one planned sweep; the loser is
expected to be DELETED afterward. The scorers share no code and
are joined only by MCTS.frontier_score, so that removal is a pure
deletion. The cross-mode knob is idle by design (alpha unused
under "path_decay", gamma under "parent_blend").

Everything else -- expansion, backprop, node classes, candidate
generation, output shape, the kube_affordable feasibility filter, the
cost<=0 -inf guard -- is unchanged from v01. The loop shape
(generate -> expand -> select) is also unchanged; only WHAT gets
added to leaf_nodes at expand time (non-terminals only) and WHAT
select_child_from_list scores (the score_mode-selected frontier
density) differ.

Algorithm
    Initialize completion_list = [], leaf_nodes = [], gen_cnt = 0,
        t = 0 (frontier clock: one tick per selection; only feeds the
        bonus when kube_schedule="global"), current = root,
        pending = []  (queued terminal children, NOT yet
        backpropped -- see "Delayed-eager flush timing" below)
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
        t += 1
        residual = gen_budget - gen_cnt
        candidates = {x in leaf_nodes : x terminal
                      or cost(x) <= residual}   (if kube_affordable;
                      falls back to all of leaf_nodes if empty -- the
                      "x terminal" clause is now dead in practice,
                      since terminals never reach leaf_nodes, but is
                      kept rather than removed -- see "Note on the
                      now-dead is_terminal clause" below)
        Select: current = argmax_{x in candidates}
                    frontier_score(x, t)   (per score_mode);
                remove current from leaf_nodes
        For each node in pending:
            Backprop: update_recursive(node.q_value(), root)
        pending = []

    The loop body ordering, selection SCOPE, and behavior-preservation
    of the generate -> expand -> select rotation are unchanged from
    v01 -- see that module's docstring. Only the terminal-split +
    (schedule-conditional) blend described above differ.

    Delayed-eager flush timing: identical mechanics to
    mcts_bl_cnt_search_v02_00_00.py -- the flush happens AFTER Select,
    so a same-batch terminal sibling cannot influence the selection
    choosing among its own non-terminal siblings; the flush also runs
    before either early-exit point (gen_budget exhausted, leaf_nodes
    empty), so an all-terminal batch still gets its pending backprops
    applied before that break.

Note on the now-dead is_terminal clause: `select_child_from_list`'s
kube_affordable filter still checks `node.is_terminal or cost(x) <=
residual`. With the terminal-split, `nodes` passed in from
mcts_search never contains a terminal (they backprop and exit at
expand time), so the `node.is_terminal` disjunct never fires here --
but the method is kept as a general-purpose helper (mirroring v01's
signature exactly) rather than special-cased to its one current
caller, so the clause stays for correctness if ever called on a list
that does contain terminals.

Selection criterion: the score_mode-selected frontier density --
see the two "Frontier density" blocks above and MCTS.frontier_score.

Variant lineage: docs/algorithms.md,
docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.1
(design), docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md
(this implementation).
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

    def path_aware_kube_density(
        self, max_depth: int, kube_c: float, t: int, schedule: str,
        alpha: float,
    ) -> float:
        """v02 selection index: blend own q_value with the parent's
        (both schedules), add the schedule's bonus, divide by cost.
        The value blend mirrors mcts_bl_cnt_search_v02_00_00.py's
        path_aware_puct(); the schedules differ ONLY in the bonus's
        clock (parent visit count vs. the shared frontier counter t)
        -- the blend is schedule-independent, per the 2026-07-18
        decision to extend it to "global" (see module docstring;
        originally "parent"-only).

        alpha=1.0 recovers the unblended value term exactly under
        EITHER schedule -- i.e. v01's kube_density() -- the built-in
        control arm.

        cost = max_depth - depth; nodes at or past max_depth get
        density = -inf (guard — should already be terminal), same as
        v01.
        """
        cost = max_depth - self.depth
        if cost <= 0:
            return -float("inf")
        own_q = self.q_value() if self.visit_count() > 0 else 0.0
        visits = self.visit_count()
        if self.parent and self.parent.visit_count() > 0:
            parent_q = self.parent.q_value()
        else:
            parent_q = own_q
        q = alpha * own_q + (1 - alpha) * parent_q
        if schedule == "parent":
            clock = self.parent.visit_count() if self.parent else 1
        elif schedule == "global":
            clock = 1 + t
        else:
            raise ValueError(
                f"unknown kube_schedule: {schedule!r} "
                "(expected 'parent' or 'global')"
            )
        if clock == 0 or visits == 0:
            bonus = 0.0
        else:
            bonus = kube_c * np.sqrt(np.log(clock) / visits)
        return (q + bonus) / cost

    def path_decay_kube_density(
        self, max_depth: int, kube_c: float, t: int, schedule: str,
        gamma: float,
    ) -> float:
        """score_mode="path_decay" index: gamma-decayed average of
        q_value along the leaf->root path, plus an AlphaZero-shaped
        bonus on the schedule's clock, divided by remaining cost.

        q_path = sum_k gamma^k * q(ancestor_k) / sum_k gamma^k
                 (k = 0 at this leaf, walking to the root; the k=0
                 weight is 1 even at gamma=0, so norm > 0 always)
        bonus  = kube_c * sqrt(clock) / (1 + N_leaf)
            clock = N_parent  under schedule="parent"
            clock = 1 + t     under schedule="global"
        density = (q_path + bonus) / cost;  cost <= 0 -> -inf

        Under schedule="parent" this mirrors
        mcts_bl_cnt_search_v02_00_00.py's path_decay_score exactly
        (that file's u IS this bonus with clock=N_parent) before
        the /cost division; "global" swaps its shared clock into
        the same AZ shape -- the identical clock substitution the
        two schedules have always differed by. kube_c is NOT
        comparable across score_modes (different bonus shapes).

        Deliberately shares NO code with path_aware_kube_density:
        the two score_modes are sweep arms and the loser is
        expected to be deleted afterward -- independence keeps that
        removal a pure deletion (see MCTS.frontier_score).
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
        if schedule == "parent":
            clock = self.parent.visit_count() if self.parent else 1
        elif schedule == "global":
            clock = 1 + t
        else:
            raise ValueError(
                f"unknown kube_schedule: {schedule!r} "
                "(expected 'parent' or 'global')"
            )
        bonus = kube_c * np.sqrt(clock) / (1 + self.visit_count())
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
        # Seed root with one update so visit_count >= 1 from the start;
        # prevents the visit_count==0 bonus special case from
        # triggering on the wrong iteration.
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with a score_mode-selectable fractional-KUBE frontier
    density and delayed-eager terminal backprop (both score_modes
    and the terminal split apply under both schedules; only the
    bonus's clock is schedule-specific).

    Holds `completed_nodes` (EOS/length-terminated leaves) and the
    algorithm methods: `expand_node`, `select_child_from_list`,
    `backprop`.
    """
    completed_nodes: List[Type[BaseNode]] = []
    cnt_node_max_depth: int = 0
    # Winning KUBE density of the most recent select_child_from_list
    # call; stashed rather than returned so the selector signature is
    # unchanged. Read each phase into phase_selected_score. NaN until
    # the first selection.
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

    def frontier_score(self, node, t) -> float:
        """Score a frontier leaf per config.search.score_mode -- the
        single point where the mode choice is read (t feeds only
        the "global" schedule's bonus clock). Each mode's scorer is
        self-contained on MCTSNode; removing a mode is one branch
        here plus one node method (the modes are sweep arms, and
        the loser is expected to be deleted after the sweep).
        """
        s = self.config.search
        if s.score_mode == "parent_blend":
            return node.path_aware_kube_density(
                s.max_depth, s.kube_c, t, s.kube_schedule, s.alpha)
        if s.score_mode == "path_decay":
            return node.path_decay_kube_density(
                s.max_depth, s.kube_c, t, s.kube_schedule, s.gamma)
        raise ValueError(
            f"unknown score_mode: {s.score_mode!r} "
            "(expected 'parent_blend' or 'path_decay')"
        )

    def select_child_from_list(
        self, nodes: List[Any], t: int, residual: int,
    ):
        """Pick the node with the highest fractional-KUBE frontier
        density (per score_mode) from an arbitrary list, uniform
        random tie-break.

        density(x) = (value_term(x) + bonus(x)) / (max_depth -
        depth(x)) -- see frontier_score / the module docstring for
        the two score_modes; only the bonus clock is
        schedule-specific.

        If config.search.kube_affordable, first restrict to nodes
        whose worst-case completion cost fits the residual
        generation budget — identical to v01 (see that module's
        docstring); the `node.is_terminal` disjunct is now dead in
        practice given the terminal-split, but kept — see module
        docstring's "Note on the now-dead is_terminal clause".
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
            density = self.frontier_score(node, t)
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
    terminal), then selects the next node by path-aware fractional-
    KUBE density globally across the leaf frontier — generate ->
    expand -> select, mirroring mcts_bl_kube_search_v01_00_00.py's
    walk step. Newly created terminal children are queued (DELAYED
    eager backprop -- flushed right after the selection immediately
    following their own creation, see pending_terminal_backprops
    below) and never enter the frontier; only non-terminal children
    are added to leaf_nodes. Only expansions charge gen_cnt.
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
    t = 0
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
    # rationale. The blend reads parent.q_value() under BOTH
    # schedules (and, under "parent", the bonus clock also reads
    # parent.visit_count()) -- a same-batch terminal sibling's
    # outcome must not leak into the selection choosing among its
    # non-terminal siblings from the SAME expand call. t's timing
    # (only feeds "global"'s bonus clock) is unaffected by when the
    # queue flushes -- t is incremented independently and never
    # touched by backprop.
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

        # Select the next node with highest path-aware fractional-KUBE
        # density across the entire frontier — the children expanded
        # above compete against every older leaf here, using only
        # state that existed before THIS step's own generation call
        # (pending_terminal_backprops above). t only feeds the bonus
        # when kube_schedule="global"; the default "parent" schedule
        # uses each node's parent visit count. residual = generations
        # left, for the affordability filter.
        t += 1
        residual = config.search.gen_budget - gen_cnt
        current_node = agent.select_child_from_list(
            leaf_nodes, t, residual
        )
        leaf_nodes.remove(current_node)

        # Flush now: this step's terminal outcomes are no longer
        # "concurrent" with a selection in progress.
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
