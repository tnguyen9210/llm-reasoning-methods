'''
MCTS implementation from the rStar-Math's repo
'''

# from __future__ import annotations

import os
import pprint
from collections import defaultdict
from abc import abstractmethod

import math
import random
import numpy as np

from typing import Optional, Dict, Any, List, Type, Union
from pydantic import BaseModel, PrivateAttr, field_validator

from omegaconf import DictConfig, OmegaConf

from vllm import SamplingParams

# from sal.search.utils import Beam, build_conv, generate_k_steps, last
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv, generate_k_steps, last


import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL+1)
# logging.basicConfig(level=logging.error)


class BaseNode(BaseModel):
    state: Dict[str, str] = {"text": "", "extra_info": ""}
    additional_state_keys: List[str] = []
    parent: Optional[Any] = None
    children: List[Any] = []
    depth: int = 0
    is_terminal: bool = False
    is_completed: bool = False 
    # reward: Optional[float] = None
    # value: Optional[float] = 0

    tag: str = "0"
    consecutive_errors: int = 0 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        for key in self.additional_state_keys:
            self.state[key] = ""

    def has_children(self) -> bool:
        return self.children != []


class MCTSNode(BaseNode):
    c_puct: float = 2
    inited: bool = False
    
    __visit_count: int = PrivateAttr(default=0)
    __value_sum: float = PrivateAttr(default=0)

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0

        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update(self, value: float) -> None:
        # if self.inited is False:
        #     self.inited = True
        #     self.value = value 

        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: Type[BaseNode]) -> None:
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        
        self.parent.update_recursive(value, start_node)
    
    def puct(self) -> float:
        if not self.parent: return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = self.c_puct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
        return q_value + u_value

    def __repr__(self):
        return f"MCTSNode(state={self.state}, is_terminal={self.is_terminal})"


class BaseTree(BaseModel):
    config: Any
    question: str = None
    ground_truth: Optional[Union[str, List[str]]] = None
    llm: Any = None
    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 
    stop: Optional[List[str]] = None
    # node_max_retry: int = 5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.root = self.create_root()
        self.current_node = self.root


    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root


    @abstractmethod
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
        pass

    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))


class BS(BaseTree):
    current_nodes: List[Type[BaseNode]] = []
    # final_answer_nodes: List[Type[BaseNode]] = [] 
    completed_nodes: List[Type[BaseNode]] = [] 
    candidate_nodes: List[Type[BaseNode]] = [] 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.candidate_nodes.append(self.current_node)
        # self.current_top_num = self.config.step_beam_width


    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent
        )

class MCTS(BS):

    search_node: Type[BaseNode] = None
    
    def create_node(self, parent = None):
        return MCTSNode(
            parent=parent
        )


    def expand_node(self, llm_outputs, current_node):
        for idx, output in enumerate(llm_outputs):
            self.create_child(output, current_node)


    def create_child(self, output, current_node):
        new_node = self.create_node(parent=current_node)
        parent_child_count = len(current_node.children)
        new_node.tag = f"{current_node.tag}.{parent_child_count + 1}"
        new_node.depth = current_node.depth + 1

        new_node.state["text"] = current_node.state["text"] + output.next_texts[0]
        if (output.stop_reasons[0] == "EOS"
            or output.stop_reasons[0] == "length"
                or output.next_texts[0] == ""
        ):
            new_node.is_completed = True
            new_node.is_terminal = True
            self.completed_nodes.append(new_node)
            
        if not new_node.is_terminal and new_node.depth > self.config.max_depths:
            new_node.is_terminal = True

        current_node.children.append(new_node)

    def select_child(self, node):
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            # logging.debug(f"child = {child}")
            if child.is_terminal:
                continue
            puct_value = child.puct()
            # logging.debug(puct_value)
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        #return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    
    def selection(self, from_root=False):
        logging.error(f"\n-> selection")
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node 

        node = start_node 
        # logging.info(f"start_node = {start_node}")
        if node is None:
            return None 

        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)      # To encourage exploration, select from non-terminal children 
            if next_node is None:
                node.is_terminal = True 
            node  = next_node
            
        # logging.info(f"selected_node = {node}")
        return None if (node is None or node.is_terminal) else node

        
    def select_next_step(self, prm_scores=None, from_root=False):
        logging.error(f"\n-> select_next_step")
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        logging.info(f"search_node")
        if self.search_node is None:
            logging.info("None")
        else:
            logging.info(self.search_node.state['text'])
            logging.info(self.search_node.is_terminal)
            logging.info(self.search_node.is_completed)
        logging.info(f"prm_scores")
        logging.info(prm_scores)
        
        if prm_scores:
            for candidate_node, score in zip(self.candidate_nodes, prm_scores):
                # backup 
                if candidate_node.is_terminal: # for terminal node: update recursively upto the root 
                    logging.info(f"candidate: update: terminal")
                    candidate_node.update_recursive(score[0], self.root)
                else:  # for intermediate node: only update this intermediate node 
                    logging.info(f"candidate: update: intermediate")
                    candidate_node.update(score[0])

        selected_node = self.selection(from_root=from_root)
        logging.warn(f"selected_node")
        if selected_node is None:
            logging.warn("None")
        else:
            logging.warn(selected_node.state['text'])
            logging.warn(selected_node.is_terminal)
            logging.warn(selected_node.is_completed)
            
        if selected_node is not None:
            self.current_nodes.append(selected_node)

    def generate_next_step(self, llm_outputs):
        logging.error(f"\n-> generate_next_step")
        self.candidate_nodes = []

        self.expand_node(llm_outputs, self.current_nodes[0])
    
        for child_node in self.current_nodes[0].children:
            if child_node not in self.candidate_nodes and child_node.visit_count() < 1:
                self.candidate_nodes.append(child_node)

        logging.warn(f"candidate_nodes")
        for node in self.candidate_nodes:
            logging.warn("")
            logging.warn(node.state['text'])
            logging.warn(node.is_terminal)
            logging.warn(node.is_completed)


def mcts_search(question, agent, config, llm_vllm, prm):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    batch_cnt = 0
    for p in range(config.num_phases):
        logging.error(f"\n-> p = {p}")
        agent.select_next_step(from_root=True)

        for d in range(config.max_depths):
            logging.error(f"\n-> d = {d}")
            # promt and get llm_outputs

            # if current branch reaches a terminal node, continue 
            if len(agent.current_nodes) == 0:
                break
            
            # partial_solution = agent.collect_partial_solution(agent.current_nodes[0])
            partial_solution = agent.current_nodes[0].state["text"]
            logging.error(f"partial_solution = {partial_solution}")
            convs = [build_conv(question, partial_solution, config.system_prompt)]*config.n

            add_generation_prompt = agent.current_nodes[0].depth == 0
            continue_final_message = agent.current_nodes[0].depth > 0
        
            tokenizer = llm_vllm.get_tokenizer()
        
            if config.custom_chat_template is not None:
                tokenizer.chat_template = config.custom_chat_template
                
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )
            # logging.error(templated_convs[0])

            lookahead = 0 if d == config.max_depths - 1 else config.lookahead
            llm_outputs = generate_k_steps(
                templated_convs, lookahead, llm_vllm, sampling_params, 1
            )

            # logging.error(f"llm_outputs")
            # logging.error(llm_outputs)
            agent.generate_next_step(llm_outputs)
    
            # apply prm and assign candidate steps with prm scores -> prm_outputs
            all_prompts = []
            all_completions = []
            for _, node in enumerate(agent.candidate_nodes):
                all_prompts.append(question)
                all_completions.append([node.state["text"]])
            
            prm_all_scores = prm.score(all_prompts, all_completions, batch_size=4)
            prm_agg_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in prm_all_scores
            ]
            logging.fatal(f"prm_agg_scores = {prm_agg_scores}")
            
            agent.select_next_step(prm_agg_scores)

            batch_cnt += 1
            if batch_cnt >= config.batch_budget:
                break 
                
        if batch_cnt >= config.batch_budget:
                break 

    unique_completion_dict = {}
    for idx, node in enumerate(agent.completed_nodes):
        if node.state["text"] not in unique_completion_dict:
            unique_completion_dict[node.state["text"]] = (idx)
            
    completions = [agent.completed_nodes[i].state["text"] for i in unique_completion_dict.values()]
        
    return completions

def _search(batch_of_questions, config, llm_vllm, prm):

    all_completions = [[] for _ in range(len(batch_of_questions))]
    for q_idx, question in enumerate(batch_of_questions):
        agent = MCTS(config=config, question=question)
        completions = mcts_search(question, agent, config, llm_vllm, prm)
        all_completions[q_idx] = completions
                                          
    results = defaultdict(list)
    results["completions"] = all_completions

    return results