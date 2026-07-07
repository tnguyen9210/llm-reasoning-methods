#!/usr/bin/env bash
# Offline determinism smoke test for generate_mcts_cnt.py.
#
# Runs the real launcher twice, back to back, with an identical
# config (num_trials=1, num_questions=2, run.results_subdir=smoketest
# so output lands under its own results/smoketest/ tree, isolated
# from real runs, with the SAME config_hash as the real run since
# run.results_subdir isn't a hash group). Both launches hash to the
# SAME result_dir, so run 1's output is moved aside before run 2
# starts. Then diffs the two trial-000 raw .jsonl dumps byte-for-byte.
#
# Usage: ./smoke_test_determinism.sh [hydra overrides...]
# Example: ./smoke_test_determinism.sh llm=llama_1b prm=llama_prm

set -euo pipefail

OVERRIDES=(
    "run.results_subdir=smoketest"
    "run.num_trials=1"
    "run.num_questions=2"
    "$@"
)

export WANDB_MODE=disabled

# Resolve result_dir up front so a stale results/smoketest/ tree from
# a prior interrupted/aborted run can be wiped before run 1 even
# starts (otherwise run 1 may "resume" onto an already-.done trial
# and produce no fresh output at all). omegaconf/antlr4 print a
# version-mismatch warning to stdout on every invocation, so only the
# LAST stdout line (the path itself) is taken, not the whole capture.
resolve_result_dir() {
    python3 - "${OVERRIDES[@]}" <<'PYEOF' | tail -n1
import sys
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils.configs import (
    ExpConfig, MCTSCntConfig, config_name, level_dir, results_root,
)

# generate_mcts_cnt.py registers these structured schemas as a
# module-level side effect of being run as __main__; hydra.compose
# needs the same registration done here since this runs standalone.
cs = ConfigStore.instance()
cs.store(name="exp_schema", node=ExpConfig)
cs.store(group="search", name="mcts_cnt_schema", node=MCTSCntConfig)

with hydra.initialize(config_path="conf", version_base=None):
    cfg = hydra.compose(
        config_name="mcts_cnt_prm800k", overrides=sys.argv[1:],
    )
print(f"results/{results_root(cfg)}/{level_dir(cfg)}/{config_name(cfg)}")
PYEOF
}

RESULT_DIR=$(resolve_result_dir)
echo "result_dir = ${RESULT_DIR}"

RUN1_DIR="${RESULT_DIR}--smoketest-run1"
rm -rf "${RESULT_DIR}" "${RUN1_DIR}"

echo "=== run 1 ==="
python3 generate_mcts_cnt.py "${OVERRIDES[@]}"

mv "${RESULT_DIR}" "${RUN1_DIR}"

echo "=== run 2 ==="
python3 generate_mcts_cnt.py "${OVERRIDES[@]}"

echo "=== diff trial-000 raw generations ==="
RUN_NAME=$(basename "${RESULT_DIR}")
F1="${RUN1_DIR}/generate_${RUN_NAME}--trial-000.jsonl"
F2="${RESULT_DIR}/generate_${RUN_NAME}--trial-000.jsonl"

if cmp -s "${F1}" "${F2}"; then
    echo "IDENTICAL: ${F1} == ${F2}"
    echo "RESULT: deterministic"
else
    echo "DIFFER: ${F1} != ${F2}"
    echo "sizes:"
    wc -c "${F1}" "${F2}"
    echo "RESULT: NOT deterministic"
fi
