'''
MCTS with diversity scores, update V globally
Add llm_vllm_embeds to extract the last hidden embeds
When selection from the leaf use uct. when selection from the root use combination of uct and diversity
Update V both when selection from the root and from the leaf
'''

import os
import pprint
import copy
from collections import defaultdict
from abc import abstractmethod

import math
import random
import numpy as np

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
logging.basicConfig(format='%(message)s', level=logging.CRITICAL+1)
# logging.basicConfig(format='%(message)s', level=logging.FATAL)
# logging.disable(logging.CRITICAL)
# logging.basicConfig(level=logging.error)

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
        best_idx = min(candidate_idxes, key=lambda i: q_ppl[i])
        # print(q_vals)
        # print(q_nll)
        # print(candidate_idxes)
        # print(best_idx)
        
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
    
    def puct(self, cpuct=2) -> float:
        if not self.parent: return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0

        logging.fatal(cpuct)
        
        if cpuct == 0:
            return q_value
            
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = cpuct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
            
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
    
    def create_node(self, parent = None):
        return MCTSNode(
            parent=parent
        )


    def expand_node(self, current_node, llm_outputs, llm_embeds):
        for output, embeds in zip(llm_outputs, llm_embeds):
            self.create_child(current_node, output, embeds)


    def create_child(self, current_node, output, embeds):
        new_node = self.create_node(parent=current_node)
        parent_child_count = len(current_node.children)
        # logging.fatal(f"num_children = {parent_child_count}")
        new_node.tag = f"{current_node.tag}.{parent_child_count + 1}"
        new_node.depth = current_node.depth + 1
        new_node.embeds = embeds

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

    def select_child(self, node, from_root=False):
        
        if not from_root:

            best_value = -float("inf")
            best_childs = []
            for child_node in node.children:
                if child_node.is_terminal:
                    continue

                puct_value = child_node.puct(cpuct=self.config.cpuct_leaf)
                if puct_value == best_value:
                    best_childs.append(child_node)
                elif puct_value > best_value:
                    best_value = puct_value
                    best_childs = [child_node]

            if len(best_childs) == 0:
                return None
                
            selected_node = random.sample(best_childs, 1)[0]
            selected_embeds = selected_node.embeds
            self.V = self.V + np.matmul(selected_embeds, selected_embeds.T)

        else:
            children_puct_values = [] 
            children_embeds = []
            _children = []
            for child_node in node.children:
                if child_node.is_terminal:
                    continue
                    
                puct_value = child_node.puct(cpuct=self.config.cpuct_root)
                children_puct_values.append(puct_value)
                children_embeds.append(child_node.embeds)
                _children.append(child_node)
    
            if len(children_puct_values) == 0:
                return None
                
            # logging.fatal(f"nchildren = {len(children_puct_values)}")
            logging.fatal(f"V = {self.V[:2,:2]}")

            A_idxes, new_V = _diverse_select(
                1, self.V, children_embeds, children_puct_values, 
                self.config.ds_alpha, self.config.ds_beta, q_nll=None, q_ppl=np.zeros(len(node.children))) 

            self.V = copy.deepcopy(new_V)

            selected_node = _children[A_idxes[0]] if _children else None
        
        # logging.fatal(f"A_idxes = {A_idxes}")
        # logging.fatal(f"V = {self.V[:2,:2]}") 
        # logging.fatal(f"selected_node")
        # logging.fatal(_children[A_idxes[0]])
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

        if node.has_children() and not node.is_terminal:
            next_node = self.select_child(node, from_root)      # To encourage exploration, select from non-terminal children 
            if next_node is None:
                node.is_terminal = True 
            node = next_node
            
        # logging.info(f"selected_node = {node}")
        return None if (node is None or node.is_terminal) else node

    def generate_next_step(self, llm_outputs, llm_embeds):
        logging.error(f"\n-> generate_next_step")
        self.candidate_nodes = []

        self.expand_node(self.current_nodes[0], llm_outputs, llm_embeds)
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
        logging.warn(f"selected_node")
        if selected_node is None:
            logging.warn("None")
        else:
            logging.warn(selected_node.state['text'])
            logging.warn(selected_node.is_terminal)
            logging.warn(selected_node.is_completed)
            
        if selected_node is not None:
            self.current_nodes.append(selected_node)

def mcts_search(question, agent, config, llm_vllm, llm_vllm_embeds, prm):

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
        # prompt_logprobs=2,
    )

    batch_cnt = 0
    for p in range(config.num_phases):
        # print(f"phase = {p}")
        logging.fatal(f"\n-> p = {p}")
        agent.select_next_step(from_root=True)

        for d in range(config.max_depths):
            logging.fatal(f"\n-> d = {d}")
            # promt and get llm_outputs

            # if current branch reaches a terminal node, continue 
            if len(agent.current_nodes) == 0:
                break
            
            # partial_solution = agent.collect_partial_solution(agent.current_nodes[0])
            current_node = agent.current_nodes[0]
            current_partial_solution = current_node.state["text"]
            logging.error(f"current_partial_solution = {current_partial_solution}")
            current_convs = [build_conv(question, current_partial_solution, config.system_prompt)]

            # add_generation_prompt = agent.current_nodes[0].depth == 0
            # continue_final_message = agent.current_nodes[0].depth > 0
        
            current_templated_convs = tokenizer.apply_chat_template(
                current_convs,
                add_generation_prompt=current_node.depth == 0,
                continue_final_message=current_node.depth > 0,
                # add_generation_prompt=True,
                # continue_final_message=False,
                tokenize=False,
            )
            current_templated_convs = current_templated_convs*config.n
            # logging.error(current_templated_convs[0])

            # llm_outputs = llm_vllm.generate(current_templated_convs, sampling_params, use_tqdm=False)
            # logging.fatal("llm_outputs")
            # logging.fatal(llm_outputs)
            # # stop
            
            lookahead = 0 if d == config.max_depths - 1 else config.lookahead
            llm_outputs = generate_k_steps(
                current_templated_convs, lookahead, llm_vllm, sampling_params, 1
            )

            # logging.error(f"llm_outputs")
            # logging.error(llm_outputs)
            
            llm_embeds = []
            for cdix, output in enumerate(llm_outputs):
                # logging.error(f"cand_idx = {cdix}")
                cand_text = output.next_texts[0]
                
                # logging.fatal(f"cand_text = {cand_text}")
                cand_convs = [build_conv(question, cand_text, config.system_prompt)]
                cand_templated_convs = tokenizer.apply_chat_template(
                    cand_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                )
                # logging.fatal(cand_templated_convs[0]) 

                outputs =  llm_vllm_embeds.encode(cand_templated_convs, use_tqdm=False)
                outputs_embeds = outputs[0].outputs.data

                if config.normalize_embeds:
                    outputs_embeds = F.normalize(outputs_embeds, p=2, dim=-1)
                    
                llm_embeds.append(outputs_embeds.detach().cpu().numpy())
                # logging.fatal(outputs)
                # logging.fatal(outputs_embeds) 

            agent.generate_next_step(llm_outputs, llm_embeds)
            
    
            # apply prm and assign candidate steps with prm scores -> prm_outputs
            candidate_prompts = []
            candidate_completions = []
            for _, node in enumerate(agent.candidate_nodes):
                candidate_prompts.append(question)
                candidate_completions.append([node.state["text"]])
            
            candidate_scores = prm.score(candidate_prompts, candidate_completions, batch_size=4)
            candidate_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in candidate_scores
            ]
            # logging.error(prm_convs)
            # candidate_scores = rm_generate(prm_model, prm_v_head, prm_convs, prm_tokenizer)
            logging.error(f"candidate_scores = {candidate_scores}")
            
            agent.select_next_step(candidate_scores)

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

def _search(batch_of_questions, config, llm_vllm, llm_vllm_embeds, prm):

    all_completions = [[] for _ in range(len(batch_of_questions))]
    for q_idx, question in enumerate(batch_of_questions):
        agent = MCTS(config=config, question=question)
        completions = mcts_search(question, agent, config, llm_vllm, llm_vllm_embeds, prm)
        all_completions[q_idx] = completions
                                          
    results = defaultdict(list)
    results["completions"] = all_completions

    return results