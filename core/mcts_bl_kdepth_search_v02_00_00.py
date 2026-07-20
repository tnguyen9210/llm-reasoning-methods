"""
Budget-Limited MCTS with best-first leaf selection (depth-shaping
knapsack density, no embeddings) -- v02: delayed-eager terminal
backprop, hygiene only, no formula change.

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
siblings, the delay is BEHAVIORALLY INERT in this file --
depth_density() (see below) reads neither a parent's q_value() nor
its visit_count(), so there is no channel through which "was this
terminal's backprop flushed before or after the immediately-following
selection" could ever change a ranking. The delayed-flush queue
pattern is kept anyway, for structural consistency with the cnt/kube
v02 siblings (same invariant to reason about across all three files),
not because it changes anything this file's selection formula reads.

Unlike the other two v02 siblings, `depth_density()` is UNCHANGED --
byte-identical to v01's. Per the design doc §7.2: `depth_density`
reads only a leaf's own frozen q_value, its own depth, and two
constants (depth_beta, depth_alpha) -- "No visit_count/parent_visit/
global-clock term anywhere," by explicit design. There is no channel
in this formula that WHEN a node backprops, or what a parent's
q_value is, could ever reach -- so "no backprop timing -- eager,
lazy, or never -- can change which non-terminal node gets expanded
next" (§7.2, verbatim). Consequently this v02 does NOT add a
parent-blend or any alpha knob: there is nothing designed for it to
blend, and inventing one is explicitly out of scope of the existing
design doc (a decision confirmed 2026-07-17 when this file was
written -- see docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md
"kdepth scope").

What this v02 actually changes, concretely:
  - A terminal child is queued at creation (inside the expand step)
    and never enters leaf_nodes, instead of waiting to win a frontier
    selection; the queue is flushed (backpropped) right after the
    very next selection resolves -- see the NOTE above on why this
    delay is inert here despite being load-bearing in the cnt/kube
    siblings.
  - Ranking among NON-terminal nodes is provably unchanged: every
    non-terminal node that would have been in leaf_nodes under v01
    is in leaf_nodes under v02 too, at the same relative order
    relevant to depth_density's inputs (depth_density reads only
    depth/q_value/constants -- none of which the terminal-split
    touches for a non-terminal node). The only removed events are
    terminal-selection phases (a phase that, in v01, does nothing
    but re-select an already-known dead-end and immediately
    backprop it) -- pure overhead removed, not a ranking change.
  - The kube_affordable feasibility fallback becomes reachable again
    (dead-ends can no longer keep the "affordable" set permanently
    non-empty), and the frontier stops accumulating permanently-
    stuck dead-end clutter.

Everything else -- expansion, backprop, node classes, candidate
generation, output shape, depth_density, the kube_affordable filter,
the cost<=0 -inf guard, the loop shape -- is unchanged from v01.

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
        Select: current = argmax_{x in candidates} depth_density(x);
                remove current from leaf_nodes
        For each node in pending:
            Backprop: update_recursive(node.q_value(), root)
        pending = []

    The loop body ordering, selection SCOPE, and behavior-preservation
    of the generate -> expand -> select rotation are unchanged from
    v01 -- see that module's docstring. Only the terminal-split
    described above differs; depth_density itself is byte-identical
    to v01's, and the flush's timing (immediately after Select, or
    before either early-exit point) is inert given what
    depth_density reads -- see the NOTE above.

Selection criterion: identical to v01 -- see that module's docstring
(depth_density, cost mapping, f_a(z) depth-decay shape). This v02
does not change the formula.

Variant lineage: docs/algorithms.md,
docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.2
(design -- establishes the goal is unreachable via backprop timing
alone in this variant), docs/decisions/
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

    def depth_density(
        self, max_depth: int, depth_beta: float, depth_alpha: float,
    ) -> float:
        """Depth-shaping knapsack index: (q_value + depth bonus) / cost.

        Byte-identical to mcts_bl_kdepth_search_v01_00_00.py's --
        this v02 does not change the formula (see module docstring).

        cost = max_depth - depth; nodes at or past max_depth get
        density = -inf (guard — should already be terminal).
        bonus = depth_beta * (1 - depth_frac**depth_alpha),
        depth_frac = depth / max_depth: 1 at the root (max bonus),
        0 at max_depth (no bonus) — monotonically prefers shallower
        nodes. No visit-count/confidence term (see module docstring).
        """
        cost = max_depth - self.depth
        if cost <= 0:
            return -float("inf")
        q = self.q_value() if self.visit_count() > 0 else 0.0
        depth_frac = self.depth / max_depth
        bonus = depth_beta * (1.0 - depth_frac ** depth_alpha)
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

    def select_child_from_list(self, nodes: List[Any], residual: int):
        """Pick the node with highest depth-shaping density from an
        arbitrary list, uniform random tie-break.

        Byte-identical selection logic to v01's -- see that module's
        docstring. density(x) = (q_value(x) + depth_beta*(1-
        depth_frac(x)**depth_alpha)) / (max_depth - depth(x))

        If config.search.kube_affordable, first restrict to nodes
        whose worst-case completion cost fits the residual generation
        budget — same feasibility step as v01 (see that module's
        docstring); the `node.is_terminal` disjunct is now dead in
        practice given the terminal-split, but kept — see
        mcts_bl_kube_search_v02_00_00.py's docstring "Note on the
        now-dead is_terminal clause", identical situation here.
        """
        max_depth = self.config.search.max_depth
        depth_beta = self.config.search.depth_beta
        depth_alpha = self.config.search.depth_alpha

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
            density = node.depth_density(max_depth, depth_beta, depth_alpha)
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

        selected_density = current_node.depth_density(
            config.search.max_depth, config.search.depth_beta,
            config.search.depth_alpha,
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
