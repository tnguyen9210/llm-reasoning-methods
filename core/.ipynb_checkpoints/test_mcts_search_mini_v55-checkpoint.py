'''
MCTS Vglobal2.
Add llm_vllm_embeds to extract the last hidden embeds.
When selection from the leaf use uct. when selection from the root use combination of uct and diversity.
Update V both when selection from the root and from the leaf.
'''

import os
import pprint
import copy
from collections import defaultdict
from abc import abstractmethod

import math
import random
import numpy as np
np.set_printoptions(precision=4)

from typing import Optional, Dict, Any, List, Type, Union
from pydantic import BaseModel, PrivateAttr, field_validator

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import SamplingParams

# from sal.search.utils import Beam, build_conv, generate_k_steps, last
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv, generate_k_steps, last

# from cores.llms import rm_generate


import logging
logging.basicConfig(format='%(message)s', level=logging.FATAL)
# logging.disable(logging.fatal)
# logging.basicConfig(level=logging.error)

# global match_cnt = 0
# global total_cnt = 0 

def _diverse_select(K, V, q_embeds, q_scores, ds_alpha, ds_beta, q_nll=None, q_ppl=None):
    q_scores = np.array(q_scores)
    num_arms = len(q_embeds)
    _V = copy.deepcopy(V)
    A_idxes = []
    A_embeds = []
    tol = 0.0001
    
    for it in range(K):
        _V_inv = np.linalg.inv(_V)   # 

        #
        q_diversity = np.sqrt(np.einsum('ij,jk,ik->i', q_embeds, _V_inv, q_embeds))
        q_vals = ds_beta*q_scores + ds_alpha*q_diversity
        
        max_val = np.max([val for idx, val in enumerate(q_vals) if idx not in A_idxes])
        
        # candidate_idxes = np.where(np.abs(q_vals-max_val) < tol)[0]
        candidate_idxes = [
            arm_idx for arm_idx, arm_val in enumerate(q_vals)
            if (np.abs(max_val - arm_val) <= tol) and (arm_idx not in A_idxes)
        ]

        # best_idx = min(candidate_idxes, key=lambda i: q_nll[i])
        # best_idx = min(candidate_idxes, key=lambda i: q_ppl[i])
        best_idx = random.sample(candidate_idxes, 1)[0]
        logging.fatal(f"vals = {q_vals}")
        logging.fatal(f"beta*q_values = {ds_beta*q_scores}")
        logging.fatal(f"alpha*q_diversity = {ds_alpha*q_diversity}")
        logging.fatal(f"candidate_idxes = {[idx + 1 for idx in candidate_idxes]}")
        logging.fatal(f"best_idx = {best_idx + 1}") 
        
        best_embeds = q_embeds[best_idx]
        best_embeds = best_embeds.reshape(-1, 1)
        # print(best_embeds.shape)

        # update V
        _V = _V + np.matmul(best_embeds, best_embeds.T)

        # update A
        A_idxes.append(best_idx)

        # print(_V.shape)
        # print(max_val)
        # print(max_idx)
        # print(A_idxes)

    return A_idxes, _V


class BaseNode(BaseModel):
    state: Dict[str, str] = {"text": "", "extra_info": ""}
    additional_state_keys: List[str] = []
    parent: Optional[Any] = None
    children: List[Any] = []
    depth: int = 0
    phase: int = 0
    step_cnt: int = 0
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
    c_puct: float = 0
    inited: bool = False
    embeds: Optional[Any] = None
    
    __visit_count: int = PrivateAttr(default=0)
    __value_sum: float = PrivateAttr(default=0)

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0

        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update(self, value: float) -> None:

        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: Type[BaseNode]) -> None:
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        
        self.parent.update_recursive(value, start_node)
    
    def puct(self, cpuct=2) -> float:
        if not self.parent: return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0

            
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = cpuct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
            
        return q_value + u_value

    def __repr__(self):
        return f"MCTSNode(state={self.state}, is_terminal={self.is_terminal}, nvisits={self.__visit_count})"


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
    V: Optional[Any] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.candidate_nodes.append(self.current_node)
        self.V = self.config.lam*np.eye(2048)
        # self.current_top_num = self.config.step_beam_width


    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent
        )


class MCTS(BS):

    search_node: Type[BaseNode] = None
    match_cnt: int = 0
    total_cnt: int = 0
    
    def create_node(self, parent = None):
        return MCTSNode(
            parent=parent
        )


    def expand_node(self, current_node, llm_outputs, phase, step_cnt):
        for output in llm_outputs:
            self.create_child(current_node, output, phase, step_cnt)


    def create_child(self, current_node, output, phase, step_cnt):
        new_node = self.create_node(parent=current_node)
        parent_child_count = len(current_node.children)
        # logging.fatal(f"num_children = {parent_child_count}")
        new_node.tag = f"{current_node.tag}.{parent_child_count}"
        new_node.depth = current_node.depth + 1
        new_node.embeds = output["embeds"]
        new_node.phase = phase
        new_node.step_cnt = step_cnt

        new_node.state["text"] = output["text"]

        if output["is_completed"] == True:
            new_node.is_completed = True 
            new_node.is_terminal = True 
            self.completed_nodes.append(new_node)
            
        if not new_node.is_terminal and new_node.depth >= self.config.max_depths:
            new_node.is_terminal = True

        current_node.children.append(new_node)

    def select_child(self, node, from_root=False):
        # logging.fatal(f"select_child")
        # logging.fatal(f"{node.children}")
        
        if not from_root:
            
            best_value = -float("inf")
            best_childs = []
            for child_node in node.children:
                if child_node.is_terminal:
                    continue

                puct_value = child_node.puct(cpuct=self.config.cpuct)
                if puct_value == best_value:
                    best_childs.append(child_node)
                elif puct_value > best_value:
                    best_value = puct_value
                    best_childs = [child_node]

                logging.fatal(f"{child_node.tag}")
                logging.fatal(f"   q-value = {child_node.q_value():0.4f}")
                logging.fatal(f"   u-value = {puct_value-child_node.q_value():0.4f}")
                logging.fatal(f"   puct = {puct_value:0.4f}")
                logging.fatal(f"   nvisit = {child_node.visit_count():0.2f}")
                logging.fatal(f"   parent.nvisit = {node.visit_count():0.2f}")
                logging.fatal(f"   is_terminal = {child_node.is_terminal}")

            if len(best_childs) == 0:
                return None
                
            selected_node = random.sample(best_childs, 1)[0]
            selected_embeds = selected_node.embeds
            selected_embeds = selected_embeds.reshape(-1, 1)
            
            self.V = self.V + np.matmul(selected_embeds, selected_embeds.T)

        else:
            children_q_values = [] 
            children_embeds = []
            _children = []
            for child_node in node.children:
                if child_node.is_terminal:
                    continue
                    
                q_value = child_node.q_value()
                children_q_values.append(q_value)
                children_embeds.append(child_node.embeds)
                _children.append(child_node)
    
            if len(children_q_values) == 0:
                return None
                
            # logging.fatal(f"nchildren = {len(children_puct_values)}")
            # logging.fatal(f"V = {self.V[:2,:2]}")

            A_idxes, new_V = _diverse_select(
                1, self.V, children_embeds, children_q_values, 
                self.config.ds_alpha, self.config.ds_beta, q_nll=None, q_ppl=np.zeros(len(node.children))) 

            self.V = copy.deepcopy(new_V)
            
            selected_node = _children[A_idxes[0]] if _children else None
        
        # logging.fatal(f"A_idxes = {A_idxes}")
        # logging.fatal(f"V = {self.V[:2,:2]}") 
        # logging.fatal(f"selected_node")
        logging.fatal(f"selected_child = {selected_node.tag}")
        # return _children[A_idxes[0]] if _children else None 
        return selected_node

    
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

        while node.has_children() and not node.is_terminal:
            next_node = self.select_child(node, from_root)      # To encourage exploration, select from non-terminal children 
            if next_node is None:
                node.is_terminal = True 
                break
            node = next_node
            
        # logging.info(f"selected_node = {node}")
        return None if (node is None or node.is_terminal) else node


    def select_next_step(self, candidate_scores=None, from_root=False):
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
        logging.info(f"candidate_scores")
        logging.info(candidate_scores)
        
        if candidate_scores:
            for candidate_node, score in zip(self.candidate_nodes, candidate_scores):
                # backup 
                if candidate_node.is_terminal: # for terminal node: update recursively upto the root 
                    logging.info(f"candidate: update: terminal")
                    candidate_node.update_recursive(score[0], self.root)
                else:  # for intermediate node: only update this intermediate node 
                    logging.info(f"candidate: update: intermediate")
                    candidate_node.update(score[0])

        selected_node = self.selection(from_root=from_root)
        # logging.fatal(f"selected_node")
        if selected_node is None:
            logging.fatal(f"selected_node = None")
        else:
            logging.fatal(f"selected_node = {selected_node.tag}")
            logging.warn(selected_node.state['text'])
            logging.warn(selected_node.is_terminal)
            logging.warn(selected_node.is_completed)
            
        if selected_node is not None:
            self.current_nodes.append(selected_node)

    def generate_next_step(self, llm_outputs, phase, step_cnt):
        logging.error(f"\n-> generate_next_step")
        self.candidate_nodes = []

        self.expand_node(self.current_nodes[0], llm_outputs, phase, step_cnt)
        # logging.fatal(f"current_node")
        # logging.fatal(self.current_nodes[0])
    
        for child_node in self.current_nodes[0].children:
            if child_node not in self.candidate_nodes and child_node.visit_count() < 1:
                self.candidate_nodes.append(child_node)

        logging.warn(f"candidate_nodes")
        for node in self.candidate_nodes:
            logging.warn("")
            logging.warn(node.state['text'])
            logging.warn(node.is_terminal)
            logging.warn(node.is_completed)
            
def mcts_search(question, agent, config, tree_dict):
    # print(config.max_depths)
    # tokenizer = llm_vllm.get_tokenizer()
    # if config.custom_chat_template is not None:
    #     tokenizer.chat_template = config.custom_chat_template
        
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        # prompt_logprobs=2,
    )

    step_cnt = 0
    should_terminate = False
    do_terminate = False

    ndepths_arr = []
    for p in range(config.num_phases):
        logging.fatal(f"\n-> p = {p}")

        cur_depth = 0
        if do_terminate:
            break
            
        agent.select_next_step(from_root=True)

        for d in range(config.max_depths):
            # logging.fatal(f"\n-> d = {d}")
            # promt and get llm_outputs

            # if current branch reaches a terminal node, continue 
            if len(agent.current_nodes) == 0:
                if cur_depth > 0:
                    ndepths_arr.append(cur_depth)
                if should_terminate:
                    do_terminate = True
                else:
                    should_terminate = True
                break

            should_terminate = False
            
            # partial_solution = agent.collect_partial_solution(agent.current_nodes[0])
            current_node = agent.current_nodes[0]
            current_tag = current_node.tag

            cur_depth = current_node.depth + 1
            step_cnt += 1
            logging.fatal(f"\n-> d = {current_node.depth}")
            
            llm_outputs = []
            llm_embeds = []
            for i in range(config.n):
                i_tag = f"{current_tag}.{i}"
                if i_tag in tree_dict:
                    llm_outputs.append(tree_dict[i_tag])
                    # llm_embeds.append(tree_dict[i_tag]['embeds'])

            agent.generate_next_step(llm_outputs, p, step_cnt)
    
            # # apply prm and assign candidate steps with prm scores -> prm_outputs    
            candidate_scores = []
            for _, node in enumerate(agent.candidate_nodes):
                # node_tag = node.tag
                candidate_scores.append([tree_dict[node.tag]['q_value']])
            
            # logging.error(prm_convs)
            # candidate_scores = rm_generate(prm_model, prm_v_head, prm_convs, prm_tokenizer)
            logging.error(f"candidate_scores = {candidate_scores}")
            
            agent.select_next_step(candidate_scores)

            logging.fatal(f"step_cnt = {step_cnt}")
            if step_cnt >= config.step_budget:
                break 

        
        if step_cnt >= config.step_budget:
            break

    # logging.fatal(f"matching cnt = {agent.match_cnt}/{agent.total_cnt} = {agent.match_cnt/agent.total_cnt:0.4f}")
    
    tree_dict = defaultdict()
    tag_list = list()
    root_node = agent.root
    save_tree(root_node, tree_dict, tag_list)
    
    unique_completion_dict = {}
    for idx, node in enumerate(agent.completed_nodes):
        if node.state["text"] not in unique_completion_dict:
            unique_completion_dict[node.state["text"]] = (idx)
            
    # completions = [agent.completed_nodes[i].state["text"] for i in unique_completion_dict.values()]
    completions = []
    c_depths = []
    c_phases = []
    c_step_cnts = []
    for i in unique_completion_dict.values():
        node = agent.completed_nodes[i]
        completions.append(node.state["text"])
        c_depths.append(node.depth)
        c_phases.append(node.phase)
        c_step_cnts.append(node.step_cnt)
        
    
    return tree_dict, tag_list, completions, c_depths, c_phases, c_step_cnts, p, ndepths_arr

def save_tree(node, tree_dict, tag_list):
    if node is None:
        return 

    tree_dict[node.tag] = node.state['text']
    tag_list.append((node.tag, node.q_value(), node.is_terminal))
    # for child in node.children:
    #     tree_dict[child.tag] = child.text

    for child in node.children:
        save_tree(child, tree_dict, tag_list)

    

def _search(batch_of_questions, config, llm_vllm, llm_vllm_embeds, prm):

    all_completions = [[] for _ in range(len(batch_of_questions))]
    for q_idx, question in enumerate(batch_of_questions):
        agent = MCTS(config=config, question=question)
        completions = mcts_search(question, agent, config, llm_vllm, llm_vllm_embeds, prm)
        all_completions[q_idx] = completions
                                          
    results = defaultdict(list)
    results["completions"] = all_completions

    return results