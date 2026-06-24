# exp-smoke-test

Loaded when: Tuan wants to sanity-check that a generation
launcher runs to completion and produces sensible trajectory
output — e.g. "smoke-test generate_mcts_cnt with the new
feature", "run a small offline test on prm800k level 4 and let
me eyeball the trajectories", "does my new code actually run".
Sibling of `exp-new-comparison-table` / `exp-record-results`.

This is a **does-my-code-work** check, not an experiment. Its
entire job: run the launcher small + offline + into an isolated
dir, confirm it exits cleanly and wrote trajectory files, then
surface a trajectory or two for Tuan to inspect. It does NOT
score, does NOT analyze metrics, does NOT touch W&B, and does
NOT go into the tracking system.

---

## 0. What this skill is and is NOT

**IS:** a fast feasibility/visual check after a code change —
"runs to completion + the generated trajectories look right."

**IS NOT:**
- a scoring/eval run — **no `compute_stats.py`, no
  `prepare_scored_dataset`**, no pass@gb numbers. Success is
  "the launcher exited 0 and produced
  `generate_...--trial-000.jsonl`", full stop.
- a tracked experiment — **no `experiments.yaml` entry**, no
  `feeds`, no `recorded`. Smoke runs are throwaway.
- a determinism test — that's the separate
  `smoke_test_determinism.sh` (double-run + byte-diff); out of
  scope here.

The boundary is enforced by isolation (below) + the fact that
`status.py` already ignores the smoketest dir
(`_is_smoketest`), so a smoke run can never pollute the
ledger, the reconciler, or a real experiment dir.

---

## 1. The two isolation mechanisms (both required)

1. **Output isolation → `run.results_subdir=smoketest`.** The
   result dir is `results/{results_root}/{level_dir}/
   {config_name}`, where `results_root` = `run.results_subdir`
   if set, else `data.name` (see `results_root()` in
   `utils/configs.py`). Setting `run.results_subdir=smoketest`
   reroutes the whole output tree under `results/smoketest/...`.
   - **This is hash-neutral and dataset-neutral.**
     `run.results_subdir` lives in `RunConfig`, which is NOT a
     hash group — so a smoke run hashes IDENTICALLY to the real
     run of the same config (it's the same experiment, just a
     throwaway location). The dataset (`data.name`,
     `data.ds_dir`, `data.level`) is completely untouched: real
     data in, smoketest folder out.
   - `status.py._is_smoketest()` excludes any
     `results/smoketest/` path from reconciliation/backfill —
     so even though the smoke run shares a hash with the real
     run, `find_dir_by_hash` (which searches only the real
     `results/{data.name}/...` tree) never sees it, and backfill
     skips it. That's why this is safe.
   - (Historical note: this used to be done by overriding
     `data.name=smoketest`, which confusingly overloaded the
     dataset name. Replaced by `run.results_subdir` 2026-06-24.)

2. **W&B off → `WANDB_MODE=disabled`** (env var, not a config
   key). Makes `wandb.init()` a no-op; nothing hits the
   server. Set it on the command, e.g.
   `WANDB_MODE=disabled python generate_...`.

Both together = the run is invisible to W&B and isolated from
real result dirs.

---

## 2. Keeping it small (fast feedback)

Override down to a tiny workload so it finishes in minutes:
- `run.num_trials=1` — one trial is enough to see if it runs.
- `run.num_questions=2` (or a small N) — a couple of questions,
  not the full split.
- Leave the search budget at default unless the *feature being
  tested* is budget-related — a smoke test wants the real code
  path, just less of it.

---

## 3. Dataset axis (generalizes beyond prm800k)

The dataset is a **separate axis** from the smoketest routing.
Today only `conf/data/prm800k.yaml` exists (level 4 default).
When another dataset is added later (e.g. `aime2025`):
- select it with `data=aime2025` (a new `conf/data/*.yaml`),
- AND still override `data.name=smoketest` for output routing.

The two don't interfere: `data=<group>` picks what to load,
`data.name=smoketest` picks where to write. To smoke-test a
specific level: `data.level=4`.

---

## 4. Launcher / config_root map

| method | launcher | config_root |
|---|---|---|
| cnt-mcts | `generate_mcts_cnt.py` | `mcts_cnt_prm800k` |
| sem-mcts v01 | `generate_mcts_sem.py` | `mcts_sem_v01_prm800k` |
| sem-mcts v02 | `generate_mcts_sem.py` | `mcts_sem_v02_prm800k` |
| bl-cnt v01 | `generate_mcts_bl_cnt_v01.py` | (its root config) |

Per-model template is baked into `conf/llm/*.yaml` (qwen
native, llama custom) — don't pass `use_custom_template` when
selecting a whole `llm=` group.

---

## 5. Procedure

1. **Gather**: which launcher (what code/feature is being
   tested), which `llm=` (default a small one like `llama_1b`
   for speed unless the feature is model-specific), which
   dataset/level (default prm800k L4), and any
   feature-specific override Tuan names.
2. **Build the command** — offline, isolated, small:
   ```
   WANDB_MODE=disabled python <launcher> \
       --config-name <config_root> \
       llm=<model> [data=<group>] [data.level=<n>] \
       <feature-override(s)> \
       run.results_subdir=smoketest \
       run.num_trials=1 run.num_questions=2
   ```
   Show Tuan the command before running.
3. **Run it** (via Bash). Watch for a clean exit. If it
   crashes, surface the traceback — that's a *useful* smoke
   result ("the new code doesn't run yet"), report it as such,
   don't paper over it.
4. **Confirm artifacts**: the run dir under
   `results/smoketest/.../` exists and contains
   `generate_<config_name>--trial-000.jsonl` plus its `.done`
   marker. No scored `.txt` files are expected (no scoring).
5. **Surface a trajectory for inspection** — the point of the
   skill. From `generate_...--trial-000.jsonl`, show 1-2
   records. The keys to look at:
   - `completions` — the generated reasoning text (does it
     look coherent / on-task / not degenerate?).
   - `comp_depth`, `comp_phase`, `phase_depths`,
     `q_nodes_max_depth` — the search-structure fields (does
     the tree shape look sane for the budget?).
   - `q_total_gens` — generations spent on the question.
   Pretty-print a record (truncate long `completions` for
   readability) and point out anything that looks off.
6. **Report**: exit status, where the output landed, and the
   trajectory excerpt. Explicitly confirm it's in
   `results/smoketest/` (ignored by `status.py`) so Tuan knows
   nothing real was touched. Offer to clean up the smoketest
   dir if he's done (`rm -rf results/smoketest/<...>`), but
   don't delete without asking.

---

## 6. What "looks correct" means (the inspection)

Tuan only wants: does it run, and do trajectories look right.
So flag, in plain terms:
- ✅ healthy: coherent `completions`, depth/phase fields
  populated and plausible, the run hit `.done`.
- ⚠️ suspicious: empty/truncated/repetitive `completions`,
  zero or degenerate depths, a trajectory that stopped at
  depth 0, NaNs, or a question that produced no completions.
Do NOT compute or comment on scores/accuracy — that's a
different task (`compute_stats.py`), explicitly out of scope.

---

## 7. Failure modes / cautions

- **Forgetting `run.results_subdir=smoketest`** → output lands
  in the REAL `results/{data.name}/...` tree. Worse than the
  old `data.name` approach: the smoke run now shares a hash
  with the real run, so an omitted reroute drops smoke output
  into the EXACT real run dir and `status.py` treats it as the
  real run. This is the cardinal mistake this skill prevents —
  never omit it.
- **Forgetting `WANDB_MODE=disabled`** → creates a stray W&B
  run for a throwaway test. Always set it.
- **Running the scoring step** → out of scope; don't.
- **Treating a crash as a skill failure** → a crash IS a valid
  smoke-test outcome (the code doesn't run). Report the
  traceback plainly.
- **Stale smoketest dir** → a prior smoke run at the same
  config hash may already have a `.done` trial, so the
  launcher "resumes" and produces no fresh output. If output
  looks stale/missing, `rm -rf` the specific smoketest run dir
  and re-run (the determinism script does this up front for
  the same reason).
