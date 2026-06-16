"""
Budget-Limited MCTS with best-first leaf selection (count-based, no embeddings).

Key difference from mcts_cnt_search_v05_00_00: instead of phase-based
root-to-leaf walks, maintains an explicit `leaf_nodes` frontier and
selects globally across all current leaves each iteration.

Sibling variant: mcts_bl_cnt_search_v02_00_00.py replaces PUCT with
KUBE density-based leaf selection. Both are active.

Algorithm
    Initialize completion_list = [], leaf_nodes = [root], gen_cnt = 0
    While gen_cnt < gen_budget:
        selected_node = argmax_{x in leaf_nodes} puct(x)
        Remove selected_node from leaf_nodes
        If selected_node.is_terminal:
            Backprop: update_recursive(selected_node.q_value(), root)
        Else:
            Expand: generate n next-step continuations, dedupe,
                    score with PRM, attach as children
            gen_cnt += 1
            For each child:
                child.update(prm_score)
                if completed (EOS/length): mark terminal,
                    add to completion_list
                if not completed and depth >= max_depth:
                    mark terminal, score = negative_reward
            Add non-terminal children to leaf_nodes

Selection criterion:
    PUCT: q_value(x) + cpuct * sqrt(log(parent.visit_count) / visit_count)
    q_value = value_sum / visit_count  (running mean, same as v05)

Variant lineage: docs/algorithms.md.
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

    def puct(self, cpuct=2) -> float:
        if not self.parent:
            return 0.0
        q = self.q_value() if self.visit_count() > 0 else 0.0
        parent_visits = self.parent.visit_count()
        visits = self.visit_count()
        if parent_visits == 0 or visits == 0:
            u = 0.0
        else:
            u = cpuct * np.sqrt(np.log(parent_visits) / visits)
        return q + u

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
        self.root.update(0)

    def create_root(self):
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent=None):
        pass


class MCTS(BaseTree):
    """MCTS with PUCT selection.

    Holds `completed_nodes` (EOS/length-terminated leaves) and the
    algorithm methods: `expand_node`, `select_child`, `backprop`.
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

        if not new_node.is_terminal and new_node.depth >= self.config.max_depth:
            new_node.is_terminal = True
            candidate_score = self.config.negative_reward
            self.cnt_node_max_depth += 1

        new_node.update(candidate_score)
        current_node.children.append(new_node)

    def expand_node(self, current_node, candidate_infos, candidate_scores,
                    phase, gen_cnt):
        """Append one child per (info, score) pair."""
        for info, score in zip(candidate_infos, candidate_scores):
            self.create_child(current_node, info, score, phase, gen_cnt)

    # ----- Selection ------------------------------------------------- #

    def select_child(self, node, from_root: bool = False):
        """Pick the child with the highest PUCT value, uniform
        random tie-break. Returns None if no children.
        """
        best_value = -float("inf")
        best_childs: List[Any] = []

        for child_node in node.children:
            puct_value = child_node.puct(cpuct=self.config.cpuct)
            if puct_value == best_value:
                best_childs.append(child_node)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child_node]

            logging.fatal(f"{child_node.tag}")
            logging.fatal(f"   q-value = {child_node.q_value():0.4f}")
            logging.fatal(f"   u-value = {puct_value - child_node.q_value():0.4f}")
            logging.fatal(f"   puct = {puct_value:0.4f}")
            logging.fatal(f"   nvisit = {child_node.visit_count():0.2f}")
            logging.fatal(f"   parent.nvisit = {node.visit_count():0.2f}")
            logging.fatal(f"   is_terminal = {child_node.is_terminal}")

        if not best_childs:
            return None
        selected_node = random.choice(best_childs)
        logging.fatal(f"selected_child = {selected_node.tag}")
        return selected_node

    def select_child_from_list(self, nodes: List[Any]):
        """Pick the node with highest PUCT from an arbitrary list,
        uniform random tie-break. Same logic as `select_child` but
        operates on the BL-MCTS global leaf frontier.
        """
        best_value = -float("inf")
        best_nodes: List[Any] = []

        for node in nodes:
            puct_value = node.puct(cpuct=self.config.cpuct)
            if puct_value == best_value:
                best_nodes.append(node)
            elif puct_value > best_value:
                best_value = puct_value
                best_nodes = [node]

            logging.fatal(f"{node.tag}")
            logging.fatal(f"   q-value = {node.q_value():0.4f}")
            logging.fatal(f"   u-value = {puct_value - node.q_value():0.4f}")
            logging.fatal(f"   puct = {puct_value:0.4f}")
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
      1. `generate_k_steps` produces `config.batch_size` continuations.
      2. `prm.score` scores each unique candidate text.
    """
    current_text = current_node.state["text"]
    logging.error(f"current_text = {current_text}")

    # Strip the step terminator before templating, then re-append it to
    # the templated string: some templates / transformers versions trim
    # or crash on trailing "\n\n" inside apply_chat_template, but the
    # model must see the separator to continue with a next step instead
    # of emitting EOS (docs/findings.md, 2026-06-11).
    current_text_clean = current_text.removesuffix("\n\n")
    current_convs = [build_conv(question, current_text_clean, config.system_prompt)]
    current_templated = tokenizer.apply_chat_template(
        current_convs,
        add_generation_prompt=current_node.depth == 0,
        continue_final_message=current_node.depth > 0,
        date_string=config.date_string,
        tokenize=False,
    )
    if current_text.endswith("\n\n"):
        current_templated = [t + "\n\n" for t in current_templated]
    current_templated = current_templated * config.batch_size

    lookahead = 0 if d == config.max_depth - 1 else config.lookahead
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
        [question], [cand_texts], batch_size=4
    )
    # score returns [question][answer][step]; one question here, so
    # candidate_scores[0] is a list of candidates, each a per-step
    # score list. Aggregate each candidate's step list to a scalar.
    candidate_scores = [
        aggregate_scores(cand_scores, config.agg_strategy)
        for cand_scores in candidate_scores[0]
    ]
    logging.error(f"candidate_scores = {candidate_scores}")

    return candidate_infos, candidate_scores


# --------------------------------------------------------------------- #
# Main search loop                                                      #
# --------------------------------------------------------------------- #

def mcts_search(question, agent, config, llm_vllm, prm):
    """Run budget-limited best-first MCTS on a single `question`.

    Outer loop: `config.num_phases` iterations (safety cap).
    Each iteration selects one leaf globally by PUCT and either
    backprops (if terminal) or expands (otherwise).
    Only expansions charge gen_cnt.
    """
    tokenizer = llm_vllm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    gen_cnt = 0
    p = 0
    ndepths_arr: List[int] = []
    leaf_nodes: List[Any] = [agent.root]

    for p in range(config.num_phases):
        logging.fatal(f"\n-> p = {p}")

        if not leaf_nodes:
            logging.fatal("leaf_nodes is empty — stopping.")
            break

        # Select leaf with highest PUCT across the entire frontier.
        selected = agent.select_child_from_list(leaf_nodes)
        leaf_nodes.remove(selected)

        logging.fatal(
            f"selected = {selected.tag}  "
            f"puct={selected.puct(config.cpuct):.4f}  "
            f"q={selected.q_value():.4f}  "
            f"nvisit={selected.visit_count()}"
        )

        if selected.is_terminal:
            logging.fatal(f"selected.is_terminal = True")
            agent.backprop(selected)
            ndepths_arr.append(selected.depth)
        else:
            gen_cnt += 1
            infos, scores = _generate_candidates(
                question, selected, selected.depth, config,
                tokenizer, llm_vllm, prm, sampling_params,
            )
            agent.expand_node(selected, infos, scores, p, gen_cnt)
            for child in selected.children:
                leaf_nodes.append(child)

        logging.fatal(f"gen_cnt = {gen_cnt}")
        if gen_cnt >= config.gen_budget:
            logging.fatal("run out of budget!")
            break

    # Collect unique completions from completed nodes.
    seen: Dict[str, int] = {}
    for idx, node in enumerate(agent.completed_nodes):
        seen.setdefault(node.state["text"], idx)

    completions: List[str] = []
    c_depths: List[int] = []
    c_phases: List[int] = []
    c_gen_cnts: List[int] = []
    for i in seen.values():
        node = agent.completed_nodes[i]
        completions.append(node.state["text"])
        c_depths.append(node.depth)
        c_phases.append(node.phase)
        c_gen_cnts.append(node.gen_cnt)

    return (
        completions, c_depths, c_phases, c_gen_cnts,
        gen_cnt, p, ndepths_arr, agent.cnt_node_max_depth,
    )


def _search(batch_of_questions, config, trial_idx, llm_vllm, prm):
    """Run `mcts_search` on each question sequentially.

    Per-question deterministic seed: 100_000 + trial_idx.
    Returns a defaultdict of per-question lists aligned to `q_idx`.
    """
    n = len(batch_of_questions)
    all_completions = [[] for _ in range(n)]
    all_c_depths = [[] for _ in range(n)]
    all_c_phases = [[] for _ in range(n)]
    all_c_gen_cnts = [[] for _ in range(n)]
    all_gen_cnts = [[] for _ in range(n)]
    all_last_phases = [[] for _ in range(n)]
    all_ndepths_arr = [[] for _ in range(n)]
    all_cnt_max_depth = [[] for _ in range(n)]

    for q_idx, question in enumerate(batch_of_questions):
        seed = 100_000 + trial_idx
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        agent = MCTS(config=config, question=question)
        (
            completions, c_depths, c_phases, c_gen_cnts,
            gen_cnt, last_phase, ndepths_arr, cnt_max_depth,
        ) = mcts_search(question, agent, config, llm_vllm, prm)

        all_completions[q_idx] = completions
        all_c_depths[q_idx] = c_depths
        all_c_phases[q_idx] = c_phases
        all_c_gen_cnts[q_idx] = c_gen_cnts
        all_gen_cnts[q_idx] = gen_cnt
        all_last_phases[q_idx] = last_phase
        all_ndepths_arr[q_idx] = ndepths_arr
        all_cnt_max_depth[q_idx] = cnt_max_depth

    results: Dict[str, Any] = defaultdict(list)
    results["completions"] = all_completions
    results["c_depths"] = all_c_depths
    results["c_phases"] = all_c_phases
    results["c_gen_cnts"] = all_c_gen_cnts
    results["gen_cnts"] = all_gen_cnts
    results["last_phases"] = all_last_phases
    results["ndepths_arr"] = all_ndepths_arr
    results["cnt_max_depth"] = all_cnt_max_depth
    return results
