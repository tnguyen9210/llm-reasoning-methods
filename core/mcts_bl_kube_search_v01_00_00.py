"""
Budget-Limited MCTS with best-first leaf selection (fractional-KUBE,
no embeddings).

Its own mcts_bl_kube algorithm family (v01 of that family) — a
distinct selection criterion from bl_cnt's PUCT, not a same-family
sibling version.

Frontier bookkeeping, expansion, backprop, and output shape are
shared with mcts_bl_cnt_search_v01_00_00.py's PUCT variant, including
the loop's generate -> expand -> select ordering (aligned 2026-07-17
— see docs/decisions-log.md — this file previously used an older
select-first loop shape; the algorithm and every selection is
unchanged, only the reading order of the loop body moved to match
v01's). See that module's docstring for the shared skeleton. Only the
leaf selection criterion differs: this file uses a fractional-KUBE
density index instead of PUCT.

Sibling comparison: mcts_bl_cnt_search_v01_00_00.py uses PUCT. Both
are active (a PUCT-vs-KUBE comparison at matched gen_budget), even
though KUBE now lives in its own algorithm family rather than as a
same-family sibling version.

Algorithm
    Initialize completion_list = [], leaf_nodes = [], gen_cnt = 0,
        t = 0 (frontier clock: one tick per selection; only feeds the
        bonus when kube_schedule="global"), current = root
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
            Add children to leaf_nodes
        t += 1
        residual = gen_budget - gen_cnt
        candidates = {x in leaf_nodes : x terminal
                      or cost(x) <= residual}   (if kube_affordable;
                      falls back to all of leaf_nodes if empty)
        Select: current = argmax_{x in candidates} kube_density(x, t);
                remove current from leaf_nodes

    The loop body is ordered generate -> expand -> select to mirror
    mcts_bl_cnt_search_v01_00_00.py's walk step; the selection SCOPE
    stays global (the whole leaf frontier — the children just
    expanded compete against every older leaf), not per-parent. This
    is a rotation of the older select-first loop: the same nodes are
    selected in the same order by the same density index (verified
    behavior-preserving — the reorder only changes when in the loop
    body t/residual are computed and where current_node is read from,
    not what gets selected or when gen_cnt/phase caps trigger).

Selection criterion (Fractional KUBE, Tran-Thanh et al. arXiv:1204.1909
sec. 3.3 — see the reference implementation in the sibling `budget-mab`
repo, src/algorithms.py::FractionalKUBE):
    density(x) = (q_value(x) + bonus(x)) / cost(x)
    cost(x) = max_depth - depth(x)   (remaining generations to reach
              the depth limit — the MCTS analogue of an arm's fixed
              pull price in the bandit abstraction; nodes closer to
              max_depth are "cheaper" to finish)
    bonus(x) — two schedules, selected by the kube_schedule knob:
      "parent" (default):
          kube_c * sqrt(log(parent_visits(x)) / visits(x))
          UCT-style local clock: each parent's children form their
          own bandit instance, so the parent's visit count is that
          bandit's elapsed time. This is exactly bl_cnt v01's PUCT
          bonus, so this file differs from bl_cnt v01 only by the
          cost division — the PUCT-vs-KUBE ablation is a
          single-factor comparison.
          Frontier nodes keep visits == 1 for life (born with one
          update, removed from the frontier on first selection),
          so per-node discrimination comes entirely from
          parent_visits, which grows as terminal backprops flow
          through a branch.
      "global":
          kube_c * sqrt(log(1 + t) / visits(x))
          t = number of frontier selections so far, shared by
          every node — faithful to KUBE's flat-bandit clock, but
          with visits == 1 across the frontier the bonus is a
          frontier-wide constant: no per-node discrimination, it
          only tilts the depth/cost tradeoff as t grows. Kept as
          an ablatable schedule — see docs/decisions-log.md
          (2026-07-09, kube_schedule entry) and
          docs/decisions/global-vs-local-exploration-schedule.md.
    q_value = value_sum / visit_count  (running mean, same as bl_cnt v01)

    Affordability (kube_affordable, default true): the argmax is
    restricted to nodes whose cost fits the remaining generation
    budget, mirroring Fractional KUBE's feasibility step (the paper
    argmaxes over affordable arms only; budget-mab restricts before
    the argmax for the same reason). Terminal nodes consume no
    generations and are always eligible; if nothing is affordable
    the filter relaxes to the full frontier, since cost is a
    worst-case bound (EOS can finish a path early). See
    docs/decisions/kube-affordability-restriction.md.

    Constant note: the paper's bonus is sqrt(2*ln(t)/n); the 2 is
    folded into the tunable kube_c here, the same convention bl_cnt
    v01's cpuct uses for UCT's constant. kube_c = 2.0 is a starting
    point, not the literal sqrt(2).

    Nodes at cost(x) <= 0 (already at max_depth) get density = -inf so
    they are never selected over a node that can still make progress
    (they should already be terminal by construction, but this guards
    against a boundary case at exactly max_depth).

Variant lineage: docs/algorithms.md, docs/decisions-log.md (2026-07-09
correction from an earlier depth-decay-only, non-UCB density formula).
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

    def kube_density(
        self, max_depth: int, kube_c: float, t: int, schedule: str,
    ) -> float:
        """Fractional-KUBE index: (q_value + UCB bonus) / cost.

        cost = max_depth - depth; nodes at or past max_depth get
        density = -inf (guard — should already be terminal).
        Bonus clock per `schedule`: "parent" uses the parent's
        visit count (UCT-style local bandit; root, having no
        parent, gets clock 1 → bonus 0, matching v01's puct());
        "global" uses the frontier clock t. A zero clock or zero
        visits yields bonus 0.0 (not inf), same as v01's puct() —
        an unvisited node scores on q_value alone rather than
        forcing uniform exploration of every new node, which the
        small gen_budget can't afford. Module docstring has the
        full rationale.
        """
        cost = max_depth - self.depth
        if cost <= 0:
            return -float("inf")
        q = self.q_value() if self.visit_count() > 0 else 0.0
        visits = self.visit_count()
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
    """MCTS with fractional-KUBE selection.

    Holds `completed_nodes` (EOS/length-terminated leaves) and the
    algorithm methods: `expand_node`, `select_child_from_list`,
    `backprop`.
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

    def expand_node(self, current_node, candidate_infos, candidate_scores,
                    phase, gen_cnt):
        """Append one child per (info, score) pair."""
        for info, score in zip(candidate_infos, candidate_scores):
            self.create_child(current_node, info, score, phase, gen_cnt)

    # ----- Selection ------------------------------------------------- #

    def select_child_from_list(
        self, nodes: List[Any], t: int, residual: int,
    ):
        """Pick the node with highest fractional-KUBE density from an
        arbitrary list, uniform random tie-break.

        density(x) = (q_value(x) + bonus(x)) / (max_depth - depth(x))
        bonus clock follows config.search.kube_schedule: "parent"
        (UCT-style, parent visit count) or "global" (frontier clock
        t) — see the module docstring.

        If config.search.kube_affordable, first restrict to nodes
        whose worst-case completion cost fits the residual
        generation budget — Fractional KUBE's feasibility step,
        applied BEFORE the argmax so the ranking among affordable
        nodes is preserved (restricting after discards it; see
        budget-mab docs/issues.md). Terminal nodes consume no
        generations and are always eligible. If no node is
        affordable, relax to the full set rather than stopping:
        cost is a worst-case bound (EOS can finish a path early),
        so the remaining budget can still buy completions.
        """
        max_depth = self.config.search.max_depth
        kube_c = self.config.search.kube_c
        schedule = self.config.search.kube_schedule

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
            density = node.kube_density(max_depth, kube_c, t, schedule)
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
    terminal), then selects the next node by fractional-KUBE density
    globally across the leaf frontier — generate -> expand -> select,
    mirroring mcts_bl_cnt_search_v01_00_00.py's walk step, with the
    freshly expanded children competing against every older leaf at
    that selection. Only expansions charge gen_cnt.
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
            infos, scores = _generate_candidates(
                question, current_node, current_node.depth, config,
                tokenizer, llm_vllm, prm, sampling_params,
            )
            agent.expand_node(current_node, infos, scores, p, gen_cnt)
            for child in current_node.children:
                leaf_nodes.append(child)

        logging.fatal(f"gen_cnt = {gen_cnt}")
        if gen_cnt >= config.search.gen_budget:
            logging.fatal("run out of budget!")
            break

        if not leaf_nodes:
            logging.fatal("leaf_nodes is empty — stopping.")
            break

        # Select the next node with highest fractional-KUBE density
        # across the entire frontier — the children expanded above
        # compete against every older leaf here. t only feeds the
        # bonus when kube_schedule="global"; the default "parent"
        # schedule uses each node's parent visit count (see module
        # docstring). residual = generations left, for the
        # affordability filter.
        t += 1
        residual = config.search.gen_budget - gen_cnt
        current_node = agent.select_child_from_list(
            leaf_nodes, t, residual
        )
        leaf_nodes.remove(current_node)

        selected_density = current_node.kube_density(
            config.search.max_depth, config.search.kube_c, t,
            config.search.kube_schedule,
        )
        logging.fatal(
            f"selected = {current_node.tag}  "
            f"density={selected_density:.4f}  "
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
