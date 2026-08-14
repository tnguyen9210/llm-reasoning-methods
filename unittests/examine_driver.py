"""Headless examine-trace driver for the mcts_bl_* comparison.

Reuses the EXACT core code path of
unittests/examine_search_trace_bl_v1.ipynb (compose -> build MCTS
agent -> core_mod.mcts_search -> dump per-node tree JSON + trace
log), but loops over several (method, question) pairs so it can run
headless on one GPU. Offline, no W&B, no ledger, no scoring.

Runnable from any cwd (repo root + conf/ are resolved from this
file's location):
    python unittests/examine_driver.py --methods mcts_bl_sem_v02,... \
        --questions 0,5 --llm qwen_3b --level 5 --budget 80

Outputs default to unittests/results/ (override with --out), per
(method, question):
    examine_search_<METHOD>_q<IDX>.log        (trace)
    examine_search_tree_<METHOD>_q<IDX>.json  (full tree)
    examine_summary_<METHOD>_q<IDX>.json      (per-question fields)
"""
import os

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "offline")

import sys
import time
import json
import random
import logging
import argparse
import importlib

# Resolve the repo root from THIS file's location so the driver
# runs from anywhere (it lives in unittests/, but core/, utils/,
# and conf/ are one level up). Put the repo root on sys.path for
# the `from core...`/`from utils...` imports below, and derive the
# absolute conf/ path for Hydra's initialize (a relative
# config_path is resolved against the caller's file, which breaks
# once this is imported/launched from a different cwd).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_CONF_DIR = os.path.join(_REPO_ROOT, "conf")

import numpy as np
import torch
from vllm import LLM
from hydra import initialize_config_dir, compose
from hydra.core.config_store import ConfigStore

from core.reward_models import build_prm
from utils.configs import (
    ExpConfig,
    BLMCTSCntConfig, BLMCTSCntV02Config,
    BLMCTSKubeV01Config, BLMCTSKubeV02Config,
    BLMCTSKdepthV01Config, BLMCTSKdepthV02Config,
    BLMCTSSemConfig, BLMCTSSemV02Config,
    MCTSCntConfig, MCTSSemV01Config, MCTSSemV02Config,
)
from utils.load_data import load_data_hf

# method -> (search core module, hydra root config, takes embeds?)
METHODS = {
    "mcts_bl_cnt_v01": ("core.mcts_bl_cnt_search_v01_00_00",
                        "mcts_bl_cnt_v01_prm800k", False),
    "mcts_bl_cnt_v02": ("core.mcts_bl_cnt_search_v02_00_00",
                        "mcts_bl_cnt_v02_prm800k", False),
    "mcts_bl_kube_v01": ("core.mcts_bl_kube_search_v01_00_00",
                         "mcts_bl_kube_v01_prm800k", False),
    "mcts_bl_kube_v02": ("core.mcts_bl_kube_search_v02_00_00",
                         "mcts_bl_kube_v02_prm800k", False),
    "mcts_bl_kdepth_v01": ("core.mcts_bl_kdepth_search_v01_00_00",
                           "mcts_bl_kdepth_v01_prm800k", False),
    "mcts_bl_kdepth_v02": ("core.mcts_bl_kdepth_search_v02_00_00",
                           "mcts_bl_kdepth_v02_prm800k", False),
    "mcts_bl_sem_v01": ("core.mcts_bl_sem_search_v01_00_00",
                        "mcts_bl_sem_v01_prm800k", True),
    "mcts_bl_sem_v02": ("core.mcts_bl_sem_search_v02_00_00",
                        "mcts_bl_sem_v02_prm800k", True),
    # Non-bl production methods (added 2026-08-13 for the
    # sem-vs-cnt artifact). Their cores return the short 8/9-item
    # tuple — run_one recovers the node counters from the tree.
    "mcts_cnt_v01": ("core.mcts_cnt_search_v01_00_00",
                     "mcts_cnt_prm800k", False),
    "mcts_sem_v02": ("core.mcts_sem_search_v02_00_00",
                     "mcts_sem_v02_prm800k", True),
}


def register_schemas():
    cs = ConfigStore.instance()
    cs.store(name="exp_schema", node=ExpConfig)
    pairs = [
        ("mcts_bl_cnt_v01_schema", BLMCTSCntConfig),
        ("mcts_bl_cnt_v02_schema", BLMCTSCntV02Config),
        ("mcts_bl_kube_v01_schema", BLMCTSKubeV01Config),
        ("mcts_bl_kube_v02_schema", BLMCTSKubeV02Config),
        ("mcts_bl_kdepth_v01_schema", BLMCTSKdepthV01Config),
        ("mcts_bl_kdepth_v02_schema", BLMCTSKdepthV02Config),
        ("mcts_bl_sem_v01_schema", BLMCTSSemConfig),
        ("mcts_bl_sem_v02_schema", BLMCTSSemV02Config),
        ("mcts_cnt_schema", MCTSCntConfig),
        ("mcts_sem_v01_schema", MCTSSemV01Config),
        ("mcts_sem_v02_schema", MCTSSemV02Config),
    ]
    for name, node in pairs:
        cs.store(group="search", name=name, node=node)


def node_to_dict(node):
    """Recursive node -> plain-dict (matches the notebook's dump)."""
    step_text = node.state["text"]
    if node.parent is not None:
        step_text = step_text.removeprefix(node.parent.state["text"])
    return {
        "tag": node.tag,
        "depth": node.depth,
        "phase": node.phase,
        "gen_cnt": node.gen_cnt,
        "n": node.visit_count(),
        "q": node.q_value(),
        "is_terminal": node.is_terminal,
        "is_completed": node.is_completed,
        "step_text": step_text,
        "children": [node_to_dict(c) for c in node.children],
    }


def load_models(cfg, takes_embeds):
    """Mirror the launcher/notebook model load exactly."""
    llm_vllm = LLM(
        model=cfg.llm.llm_dir,
        tensor_parallel_size=cfg.llm.tensor_parallel_size,
        max_model_len=cfg.llm.max_model_len,
        gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
        enforce_eager=cfg.llm.enforce_eager,
        distributed_executor_backend=None,
        dtype=cfg.llm.dtype,
        quantization=cfg.llm.quantization,
        load_format=cfg.llm.load_format,
        seed=cfg.gen.seed,
    )
    llm_vllm_embeds = None
    if takes_embeds and cfg.search.embeds_source == "policy":
        llm_vllm_embeds = LLM(
            model=cfg.llm.llm_dir,
            runner="pooling",
            tensor_parallel_size=cfg.llm.tensor_parallel_size,
            max_model_len=cfg.llm.max_model_len,
            gpu_memory_utilization=(
                cfg.search.embeds_gpu_memory_utilization
            ),
            enforce_eager=cfg.llm.enforce_eager,
            distributed_executor_backend=None,
            dtype=cfg.llm.dtype,
            seed=cfg.gen.seed,
        )
    prm = build_prm(
        cfg.prm.kind, cfg.prm.prm_dir, device=cfg.prm.device_map,
    )
    return llm_vllm, llm_vllm_embeds, prm


def run_one(method, q_idx, question, cfg, core_mod, takes_embeds,
            llm_vllm, llm_vllm_embeds, prm, out_dir, trial_idx=0,
            tag_suffix=""):
    # tag_suffix distinguishes multiple config arms of the SAME method
    # (e.g. a depth_alpha sweep of mcts_bl_kdepth_v01) so their output
    # files do not collide.
    tag = f"{method}{tag_suffix}_q{q_idx}"
    log_path = os.path.join(out_dir, f"examine_search_{tag}.log")
    tree_path = os.path.join(out_dir, f"examine_search_tree_{tag}.json")
    summ_path = os.path.join(out_dir, f"examine_summary_{tag}.json")

    # Per-question seed mirrors _search's (100_000 + trial_idx).
    seed = 100_000 + trial_idx
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    agent = core_mod.MCTS(config=cfg, question=question)

    search_args = [question, agent, cfg, llm_vllm]
    if takes_embeds:
        search_args.append(llm_vllm_embeds)
    search_args.append(prm)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(format="%(message)s")
    root_logger.setLevel(logging.FATAL)   # "selection" verbosity
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(fh)

    start = time.time()
    try:
        ret = core_mod.mcts_search(*search_args)
    finally:
        root_logger.removeHandler(fh)
        fh.close()
    elapsed = time.time() - start

    (completions, comp_depth, comp_phase, comp_gen,
     q_total_gens, q_last_phase, phase_depths,
     q_nodes_max_depth) = ret[:8]
    if len(ret) == 14:
        # bl cores: six extra diagnostics recorded in-search.
        (phase_selected_depth, phase_selected_q,
         phase_selected_score, q_nodes_total,
         q_nodes_terminal, q_nodes_completed) = ret[8:]
    else:
        # mcts_cnt (8-tuple) / mcts_sem_v02 (9-tuple ending in
        # cnt_cov_nodes): no per-phase selection trace; the node
        # counters are recovered from the finished tree instead
        # (identical to the bl counters, which include the root).
        phase_selected_depth = None
        phase_selected_q = None
        phase_selected_score = None

        def _count(node):
            tot, term, comp = 1, int(node.is_terminal), \
                int(node.is_completed)
            for c in node.children:
                t, e, m = _count(c)
                tot, term, comp = tot + t, term + e, comp + m
            return tot, term, comp

        (q_nodes_total, q_nodes_terminal,
         q_nodes_completed) = _count(agent.root)

    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(node_to_dict(agent.root), f, indent=1,
                  ensure_ascii=False)
        f.write("\n")

    summary = {
        "method": method,
        "tag_suffix": tag_suffix,
        "question_idx_level5": q_idx,
        "seconds": elapsed,
        "gen_budget": cfg.search.gen_budget,
        "q_total_gens": q_total_gens,
        "q_last_phase": q_last_phase,
        "phase_depths": phase_depths,
        "q_nodes_max_depth": q_nodes_max_depth,
        "n_completions": len(completions),
        "n_completed_nodes": len(agent.completed_nodes),
        "comp_depth": comp_depth,
        "comp_phase": comp_phase,
        "comp_gen": comp_gen,
        # New per-phase exploration trace + tree scalars (the six keys
        # now recorded by every bl core; see docs/decisions/
        # bl-search-tree-diagnostics.md).
        "phase_selected_depth": phase_selected_depth,
        "phase_selected_q": phase_selected_q,
        "phase_selected_score": phase_selected_score,
        "q_nodes_total": q_nodes_total,
        "q_nodes_terminal": q_nodes_terminal,
        "q_nodes_completed": q_nodes_completed,
    }
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[{tag}] {elapsed:0.1f}s  gens={q_total_gens} "
          f"comps={len(completions)} last_phase={q_last_phase} "
          f"nmaxdepth={q_nodes_max_depth}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True,
                    help="comma-separated method names")
    ap.add_argument("--questions", required=True,
                    help="comma-separated level-5 question indices")
    ap.add_argument("--llm", default="qwen_3b")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results"),
        help="output dir (default: unittests/results next to this file)",
    )
    ap.add_argument(
        "--overrides", default="",
        help="comma-separated extra Hydra overrides appended to the "
             "compose list, e.g. search.depth_alpha=0.5",
    )
    ap.add_argument(
        "--tag-suffix", default="",
        help="suffix appended to the method name in output filenames, "
             "to distinguish config arms of the same method "
             "(e.g. _a0.5 for a depth_alpha sweep)",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    q_idxs = [int(x) for x in args.questions.split(",")]
    os.makedirs(args.out, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    print(f"device = {torch.cuda.get_device_name(0)}", flush=True)
    register_schemas()

    overrides = [
        f"llm={args.llm}",
        f"data.level={args.level}",
        f"search.gen_budget={args.budget}",
    ]
    overrides += [o.strip() for o in args.overrides.split(",") if o.strip()]

    # Group methods by whether they need the embeds engine so we
    # only rebuild models when the (llm, embeds-need) actually
    # changes. All four v02 targets here: sem needs PRM embeds
    # (embeds_source=prm -> no 2nd engine), the rest need none.
    for method in methods:
        mod_name, cfg_name, takes_embeds = METHODS[method]
        core_mod = importlib.import_module(mod_name)
        with initialize_config_dir(version_base=None,
                                    config_dir=_CONF_DIR):
            cfg = compose(config_name=cfg_name, overrides=overrides)

        llm_vllm, llm_vllm_embeds, prm = load_models(cfg, takes_embeds)

        # Load the dataset once per method (cheap; keeps cfg local).
        load_kwargs = {"ds_split": cfg.data.ds_split}
        if cfg.data.level is not None:
            load_kwargs["level"] = cfg.data.level
        dataset = load_data_hf(cfg.data.ds_dir, **load_kwargs)

        for q_idx in q_idxs:
            question = dataset[q_idx][cfg.data.question_field]
            run_one(method, q_idx, question, cfg, core_mod,
                    takes_embeds, llm_vllm, llm_vllm_embeds, prm,
                    args.out, tag_suffix=args.tag_suffix)

        # Free GPU before the next method's engines load.
        del llm_vllm, llm_vllm_embeds, prm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
