"""Compute a fixed embeds_center mean, per model and pooled.

Reads candidate completions from existing mcts_cnt_v01--level-5 runs
(one per LLM family, all scored by the same PRM), pools/projects each
candidate through the SAME _extract_embeds pipeline sem-mcts uses at
search time, and averages. Output feeds search.embeds_mean_dir for
BOTH mcts_sem_search_v02_00_00 and mcts_bl_sem_search_v01_00_00 —
they share _extract_embeds/_center_and_normalize, so one set of means
serves both cores; no per-core variant needed.

GPU + the real PRM checkpoint required (this computes the actual
embeddings, not a placeholder) — this is NOT a Hydra-launched search,
just a config composed offline for its embeds_*/prm settings, mirroring
status.py's compose() usage. Run from the repo root:

    python compute_embeds_mean.py

Everything is currently hardcoded to the level-5 / qwen-prm / the 5
model-family result dirs this was built for; promote to CLI overrides
if a second use case shows up.
"""

import os
import json
import glob
import time

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from core.mcts_sem_search_v02_00_00 import _extract_embeds
from core.reward_models import build_prm
from utils.configs import (
    ExpConfig, MCTSCntConfig, MCTSSemV02Config,
    config_hash, find_run_dir,
)
from utils.load_data import load_data_hf

ROOT = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = f"{ROOT}/conf"
OUT_DIR = f"{ROOT}/results/embeds_mean/level-5"
PRM_TAG = "qwen-prm"  # matches the prm=qwen_prm group used below

# One entry per LLM family scored under mcts_cnt_v01--level-5. The
# tag is used only for output filenames; llm_group must be a real
# conf/llm/<name>.yaml.
MODELS = [
    {"tag": "llama_1b", "llm_group": "llama_1b"},
    {"tag": "llama_3b", "llm_group": "llama_3b"},
    {"tag": "qwen_3b", "llm_group": "qwen_3b"},
    {"tag": "qwen_7b_gptq", "llm_group": "qwen_7b_gptq_int4"},
    {"tag": "qwen_math_1_5b", "llm_group": "qwen_math_1_5b"},
]

# Overrides shared by every cnt-mcts source config, matching how the
# 5 mcts_cnt_v01--level-5 result dirs were actually launched (see each
# dir's manifest.json config_identity).
CNT_OVERRIDES = [
    "prm=qwen_prm",
    "data.level=5",
    "search.batch_size=4",
    "search.max_depth=20",
    "search.gen_budget=80",
]


def _register_schemas():
    cs = ConfigStore.instance()
    cs.store(name="exp_schema", node=ExpConfig)
    cs.store(group="search", name="mcts_cnt_schema", node=MCTSCntConfig)
    cs.store(group="search", name="mcts_sem_v02_schema", node=MCTSSemV02Config)


def _compose(config_name, overrides):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _embeds_pipeline_cfg():
    """Compose a real mcts_sem_v02 config purely to read the
    embeds_strategy/scope/proj/dim/prm_embeds_layer settings the mean
    must match — never runs search. This IS the schema+YAML that
    search launches read at runtime, so there is no second place these
    settings could drift out of sync."""
    cfg = _compose("mcts_sem_v02_prm800k", ["prm=qwen_prm"])
    sc = cfg.search
    print(
        f"embeds pipeline: strategy={sc.embeds_strategy} "
        f"scope={sc.embeds_scope} proj={sc.embeds_proj} "
        f"dim={sc.embeds_dim} prm_layer={sc.prm_embeds_layer}"
    )
    return cfg


def _find_source_dir(llm_group):
    cfg = _compose(
        "mcts_cnt_prm800k", [f"llm={llm_group}"] + CNT_OVERRIDES
    )
    result_dir = find_run_dir(ROOT, cfg)
    if result_dir is None:
        raise FileNotFoundError(
            f"no mcts_cnt_v01--level-5 result dir matches the composed "
            f"config for llm={llm_group} (hash {config_hash(cfg)}); "
            "check CNT_OVERRIDES against that dir's manifest.json"
        )
    return result_dir


def _load_candidates(result_dir):
    """Flatten every trial's completions into one list of (question,
    candidate_text) pairs, matching PRM.embed's questions/answers
    shape (per-question list of candidate strings)."""
    questions, answers = [], []
    trial_paths = sorted(glob.glob(f"{result_dir}/generate_*--trial-*.jsonl"))
    if not trial_paths:
        raise FileNotFoundError(f"no generate_*.jsonl trials in {result_dir}")
    for path in trial_paths:
        with open(path, encoding="utf-8") as fin:
            record = json.loads(fin.readline())
        questions.append(record["completions"])
    # Each trial covers the SAME question set; concat across trials so
    # every trial's candidates contribute, per question.
    num_questions = len(questions[0])
    flat_questions, flat_answers = [], []
    for q_idx in range(num_questions):
        cands = []
        for trial in questions:
            cands.extend(trial[q_idx])
        flat_questions.append(f"q{q_idx}")  # placeholder id, unused below
        flat_answers.append(cands)
    return flat_questions, flat_answers


def _embed_model(prm, pipeline_cfg, llm_group, tag):
    result_dir = _find_source_dir(llm_group)
    print(f"[{tag}] source: {result_dir}")

    # Real question text is needed by prm.embed (it builds the chat
    # per (question, answer) pair) — reload the dataset rather than
    # threading question strings through the jsonl (they aren't saved
    # there; only "problem" from the HF split is authoritative).
    dataset = load_data_hf(
        f"{pipeline_cfg.base_dir}/prm800k/math_splits",
        ds_split="test", level=5,
    )
    real_questions = [q["problem"] for q in dataset]

    _, flat_answers = _load_candidates(result_dir)
    if len(flat_answers) != len(real_questions):
        raise ValueError(
            f"[{tag}] {len(flat_answers)} questions in {result_dir} vs "
            f"{len(real_questions)} in the level-5 test split — "
            "dataset/result dir mismatch, aborting rather than "
            "silently mis-pairing questions and candidates"
        )

    raw_embeds = prm.embed(
        real_questions, flat_answers,
        system_prompt=pipeline_cfg.gen.system_prompt,
        batch_size=8,
        layer=pipeline_cfg.search.prm_embeds_layer,
    )
    pooled = [
        _extract_embeds(raw, pipeline_cfg, response_start_idx=0)
        for per_question in raw_embeds
        for raw in per_question
    ]
    pooled = np.stack(pooled)  # (num_candidates, embeds_dim)
    print(f"[{tag}] {pooled.shape[0]} candidates, dim={pooled.shape[1]}")
    return pooled


def main():
    _register_schemas()
    pipeline_cfg = _embeds_pipeline_cfg()
    prm = build_prm(
        pipeline_cfg.prm.kind, pipeline_cfg.prm.prm_dir,
        device=pipeline_cfg.prm.device_map,
    )

    per_model_pooled = {}
    for entry in MODELS:
        per_model_pooled[entry["tag"]] = _embed_model(
            prm, pipeline_cfg, entry["llm_group"], entry["tag"]
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
        "prm": {"kind": pipeline_cfg.prm.kind, "name": pipeline_cfg.prm.name},
        "embeds_pipeline": {
            "embeds_strategy": pipeline_cfg.search.embeds_strategy,
            "embeds_scope": pipeline_cfg.search.embeds_scope,
            "embeds_proj": pipeline_cfg.search.embeds_proj,
            "embeds_dim": pipeline_cfg.search.embeds_dim,
            "prm_embeds_layer": pipeline_cfg.search.prm_embeds_layer,
        },
        "models": {},
    }

    raw_dir = f"{OUT_DIR}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    all_pooled = []
    for tag, pooled in per_model_pooled.items():
        out_path = f"{OUT_DIR}/embeds_mean--{tag}--{PRM_TAG}.npy"
        mean = pooled.mean(axis=0, keepdims=True)
        np.save(out_path, mean)

        # Full per-candidate array, not just the mean -- lets the
        # examination notebook compute within-model variance without
        # re-running the PRM. Same data already in memory; one extra
        # np.save, no recomputation.
        raw_path = f"{raw_dir}/{tag}--{PRM_TAG}.npy"
        np.save(raw_path, pooled)

        manifest["models"][tag] = {
            "num_candidates": int(pooled.shape[0]),
            "npy": os.path.relpath(out_path, ROOT),
            "raw_npy": os.path.relpath(raw_path, ROOT),
        }
        print(f"saved {out_path}  (n={pooled.shape[0]})")
        all_pooled.append(pooled)

    # Pooled mean = candidate-weighted average over ALL models'
    # candidates concatenated (NOT the mean of the 5 per-model means)
    # — a model with more candidates contributes proportionally more,
    # matching how a single combined dataset would be averaged.
    combined = np.concatenate(all_pooled, axis=0)
    pooled_mean = combined.mean(axis=0, keepdims=True)
    pooled_path = f"{OUT_DIR}/embeds_mean--pooled--{PRM_TAG}.npy"
    np.save(pooled_path, pooled_mean)
    manifest["pooled"] = {
        "num_candidates": int(combined.shape[0]),
        "npy": os.path.relpath(pooled_path, ROOT),
    }
    print(f"saved {pooled_path}  (n={combined.shape[0]}, candidate-weighted)")

    manifest_path = f"{OUT_DIR}/embeds_mean--manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
