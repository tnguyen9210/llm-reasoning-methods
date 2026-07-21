
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


# W&B run id: recorded into manifest.json (run_id field) so later
# post-processing (prepare_scored_dataset / compute_stats, separate
# processes after the run is closed) can reattach via
# wandb.init(id=..., resume="must") and log onto the SAME run. Not
# known until wandb.init() returns, so generation writes the manifest
# twice: once before wandb.init (run_id=None, so the dir is still
# identity-locatable if wandb.init itself never returns), once after
# (run_id=wandb_run.id). Older dirs carry the id in a standalone
# wandb_run_id.txt sidecar instead — load_wandb_run_id falls back to
# that file when manifest.json has no run_id.
WANDB_RUN_ID_FILE = "wandb_run_id.txt"


def load_wandb_run_id(result_dir: str) -> Optional[str]:
    """Return the saved W&B run id for this result dir, or None if
    neither manifest.json nor the legacy sidecar has one."""
    path = f"{result_dir}/{MANIFEST_FILE}"
    try:
        with open(path, encoding="utf-8") as fin:
            run_id = json.load(fin).get("run_id")
        if run_id:
            return run_id
    except (OSError, json.JSONDecodeError):
        pass
    legacy_path = f"{result_dir}/{WANDB_RUN_ID_FILE}"
    if not os.path.exists(legacy_path):
        return None
    with open(legacy_path) as fin:
        return fin.read().strip() or None


# Timing sidecar: the launcher logs a running average of per-trial
# timing to W&B (time_per_question_s, time_per_trial_hr), averaged
# over all trials completed so far. A resume restarts this process
# with an empty in-memory average, so the running average up through
# the last completed trial is persisted here and folded into the
# next trial's average on resume — same rationale as the run-id
# sidecar (separate process, needs the prior state on disk).
TIMING_STATE_FILE = "timing_state.json"


def save_timing_state(
    result_dir: str,
    n_done: int,
    avg_time_per_question_s: float,
    avg_time_per_trial_hr: float,
) -> None:
    with open(f"{result_dir}/{TIMING_STATE_FILE}", "w") as fout:
        json.dump({
            "n_done": n_done,
            "avg_time_per_question_s": avg_time_per_question_s,
            "avg_time_per_trial_hr": avg_time_per_trial_hr,
        }, fout)


def load_timing_state(result_dir: str) -> tuple[int, float, float]:
    """Return (n_done, avg_time_per_question_s, avg_time_per_trial_hr)
    saved by a prior run in this result dir, or (0, 0.0, 0.0) if no
    sidecar exists (fresh launch, or a run from before this was
    added)."""
    path = f"{result_dir}/{TIMING_STATE_FILE}"
    if not os.path.exists(path):
        return 0, 0.0, 0.0
    with open(path) as fin:
        d = json.load(fin)
    return (
        d["n_done"], d["avg_time_per_question_s"],
        d["avg_time_per_trial_hr"],
    )


# Prompt assets — vendored from sal so this project has no sal
# dependency. system_prompt and custom_chat_template are the
# values the search pipeline uses; keep in sync if sal changes.
system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

custom_chat_template = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'


def build_conv(prompt, response, system_prompt):
    """Build a [system, user, (assistant)] conversation.

    Vendored from sal.search.utils. The assistant turn is added
    only when response is non-empty, so an empty response leaves
    the prompt open for generation.
    """
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if response != "":
        conversation.append(
            {"role": "assistant", "content": response}
        )
    return conversation


@dataclass
class GenConfig:
    """Generation + scoring params, shared by notebooks and
    launchers. Defaults match sal.config.Config.

    Was named LLMConfig; renamed so LLMConfig can denote the LLM
    *model* group (paths/dtype) in the structured schema below.
    """
    # Generation
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int = 42
    date_string: str = "Aug 1 2025"

    # Scoring / aggregation
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # Best-of-N
    filter_duplicates: bool = False
    sort_completed: bool = False

    # Chat template related options
    system_prompt: str = system_prompt
    custom_chat_template: str = custom_chat_template



# ---------------------------------------------------------------
# Structured experiment schema (Hydra structured configs).
#
# One typed source of truth for launchers (Hydra binds YAML config
# groups onto it) and notebooks (instantiate directly, no Hydra).
# Grouped by concern: gen / llm / prm / data / search / run.
# ExpConfig composes them.
#
# mcts_cnt is migrated: generate_mcts_cnt + mcts_cnt_search_v01_00_00
# read this nested config directly (config.search.cpuct,
# config.gen.temperature, ...). Other search files/launchers are
# still on the flat sal.Config and migrate in later sessions.
# ---------------------------------------------------------------


@dataclass
class LLMConfig:
    """Generative model group: which checkpoint and how to load it
    under vLLM. Selected via the conf/llm/ Hydra group."""
    name: str = "Llama3.2-1B-Instruct"
    llm_dir: str = "???"          # required; set in the YAML group
    dtype: str = "float16"        # V100 (sm_70): no bf16
    tensor_parallel_size: int = 1
    max_model_len: int = 5000
    gpu_memory_utilization: float = 0.3
    # False = capture CUDA graphs (faster throughput, slower
    # startup, more memory). True for quick/debug runs.
    enforce_eager: bool = False
    # vLLM quantization method, e.g. "gptq"; None = unquantized.
    # llm_dir must point at the pre-quantized checkpoint when set.
    quantization: Optional[str] = None
    load_format: str = "auto"
    # When True, override this model's native chat template with
    # GenConfig.custom_chat_template (vendored from Llama 3.1). When
    # False, use whatever template the model ships with. Default True
    # (custom) for Llama; Qwen YAML groups (conf/llm/qwen_*.yaml) set
    # this False — the vendored template is Llama-3.1-specific and
    # isn't trained-on for Qwen (docs/decisions/
    # chat-template-per-family.md; see the template-bug note in
    # llm-reasoning-mcts-comparison-main for what happens otherwise).
    # CLI override (llm.use_custom_template=...) always wins.
    use_custom_template: bool = True


@dataclass
class PRMConfig:
    """Process reward model group. Selected via conf/prm/."""
    # Which PRM wrapper class to instantiate in launchers.
    # Supported values: "rlhflow", "qwen".
    kind: str = "rlhflow"
    name: str = "Llama3.1-8B-PRM-Deepseek-Data"
    prm_dir: str = "???"          # required; set in the YAML group
    device_map: str = "cuda:0"
    # PRM forward-pass micro-batch for scoring. Distinct from
    # search.batch_size (candidates per expansion) — that's a search
    # param, this is how many (question, answer) pairs the PRM scores
    # per forward pass. Raise to cut total passes; lower if it OOMs
    # or completions are long (padding waste).
    score_batch_size: int = 8


@dataclass
class DataConfig:
    """Dataset group: where it lives and how to read it. Selected
    via conf/data/. level is optional (None = whole split)."""
    name: str = "prm800k"
    ds_dir: str = "???"           # required; set in the YAML group
    ds_split: str = "test"
    question_field: str = "problem"
    level: Optional[int] = None
    # Which utils/parser.py `data_name` vocabulary ("math", "gsm8k",
    # ...) the ground-truth grader uses for this dataset. Derived
    # from `name`, not an independent experimental axis -- see
    # _HASH_EXCLUDE.
    grader_name: str = "math"


@dataclass
class SearchConfig:
    """Base search params shared by all methods. One subclass per
    method adds its own knobs (selected via conf/search/)."""
    method: str = "base"
    batch_size: int = 4           # candidates generated per expansion
    lookahead: int = 0
    max_depth: int = 20
    negative_reward: float = 0


@dataclass
class MCTSCntConfig(SearchConfig):
    """Count-based MCTS search params."""
    method: str = "mcts_cnt_v01"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    cpuct: float = 2.0

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # MCTSSemV01Config.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSCntConfig(SearchConfig):
    """Count-based MCTS search params (baseline/v01 variant)."""
    method: str = "mcts_bl_cnt_v01"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    cpuct: float = 2.0

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # MCTSCntConfig.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSCntV02Config(SearchConfig):
    """Count-based MCTS with delayed-eager terminal backprop and a
    selectable path-aware frontier score (`score_mode`).

    v02 of BLMCTSCntConfig's family, not a same-family bugfix -- see
    docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md for the
    original v02 record. Motivation for value-aware scoring: v01's
    puct() never reads a parent's q_value, only its own frozen q
    plus the parent's visit count -- a backpropagated value is
    otherwise write-only to selection.

    Changes relative to BLMCTSCntConfig, all in
    core/mcts_bl_cnt_search_v02_00_00.py:

    1. Terminal split + DELAYED eager backprop (unconditional): a
       terminal child never enters the leaf frontier; it is queued
       at creation and backpropped right after the immediately
       following selection resolves (one step of latency -- never
       same-step, never the unbounded "maybe never" of v01's lazy
       scheme). See the core module's docstring for the full
       delayed-vs-immediate rationale.

    2. A selectable frontier score, `score_mode` (added 2026-07-19;
       both modes are arms of one planned sweep, after which the
       losing mode is expected to be DELETED -- the two scorers are
       deliberately independent, joined only by the
       MCTS.frontier_score dispatcher, so removal is a pure
       deletion):

       score_mode="parent_blend" (default) -- one-hop blend:
           blended_q = alpha*q(leaf) + (1-alpha)*q(parent)
           score = blended_q + cpuct*sqrt(log(N_parent)/N_leaf)
         Exploration term is v01's UCB1 form, unchanged. alpha in
         [0, 1]; alpha=1.0 recovers BLMCTSCntConfig's exact puct()
         -- the ONLY exact-v01 control arm in this config; include
         it in any sweep.

       score_mode="path_decay" -- full-path decayed subtree value:
           q_path = sum_k gamma^k * q(ancestor_k) / sum_k gamma^k
                    (k = 0 at the leaf, walking to the root)
           score = q_path + cpuct*sqrt(N_parent) / (1 + N_leaf)
         Exploration term is the AlphaZero shape, NOT v01's UCB1 --
         so cpuct values are NOT comparable across modes; sweep
         cpuct per mode. gamma in [0, 1]: gamma=1 is a plain path
         average; gamma=0 reads only the leaf's own q (still with
         the AZ-shaped u, so NOT a v01 control arm).

       The cross-mode knob is idle by design (alpha unused under
       "path_decay", gamma unused under "parent_blend").
    """
    method: str = "mcts_bl_cnt_v02"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    cpuct: float = 2.0             # both modes; NOT comparable across
                                   # modes (different u-term shapes)
    score_mode: str = "parent_blend"   # "parent_blend" | "path_decay"
    alpha: float = 0.8            # parent_blend only: own-q vs.
                                   # parent-q blend; 1.0 = v01's puct
    gamma: float = 0.8             # path_decay only: per-hop decay

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # BLMCTSCntConfig.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSKubeV01Config(SearchConfig):
    """Count-based MCTS with fractional-KUBE frontier selection.

    Its own mcts_bl_kube algorithm family (v01 of that family) — a
    distinct selection criterion rather than a same-family variant of
    BLMCTSCntConfig's PUCT.

    Frontier counterpart of MCTSCntConfig, exactly as BLMCTSCntConfig
    is but with the leaf-selection index replaced: instead of PUCT
    (q_value + cpuct*sqrt(log(parent_visits)/visits)), selects by
    fractional-KUBE density — a UCB index divided by remaining cost —
    following Tran-Thanh et al.'s Fractional KUBE
    (arXiv:1204.1909 sec. 3.3), as implemented as the reference in the
    sibling `budget-mab` repo (`src/algorithms.py::FractionalKUBE`):

        density_i = (q_value_i + bonus_i) / cost_i
        cost_i = max_depth - depth_i   (remaining generations to reach
                                         the depth limit; the MCTS
                                         analogue of an arm's fixed
                                         pull price in budget-mab)
        bonus_i — selected by kube_schedule:
          "parent" (default):
              kube_c*sqrt(log(parent_visits_i)/visits_i)
              UCT-style local clock — exactly bl_cnt v01's PUCT
              bonus, so this differs from bl_cnt v01 only by the
              cost division (single-factor PUCT-vs-KUBE ablation).
              Frontier nodes keep visits == 1 for life, so
              discrimination comes from parent_visits, grown by
              terminal backprops.
          "global":
              kube_c*sqrt(log(1+t)/visits_i), t = frontier
              selections so far — faithful to KUBE's flat-bandit
              clock, but a frontier-wide constant when visits == 1
              (no per-node discrimination; only tilts the
              depth/cost tradeoff as t grows). Kept as an ablation
              arm — docs/decisions-log.md (2026-07-09) and
              docs/decisions/global-vs-local-exploration-schedule.md.

    kube_affordable (default True) mirrors Fractional KUBE's
    feasibility step: the argmax is restricted, before ranking, to
    nodes whose cost_i fits the remaining generation budget.
    Terminal nodes consume no generations and are always eligible;
    an empty affordable set relaxes to the full frontier (cost_i is
    a worst-case bound — EOS can finish a path early). With it on,
    bl_cnt-v01-vs-bl_kube-v01 compares the full KUBE package;
    kube_affordable=false is the middle arm isolating cost
    normalization alone. See
    docs/decisions/kube-affordability-restriction.md.

    An earlier version of this file used a static depth-decay bonus
    (beta * (1 - ((max_depth-depth)/max_depth)**alpha), no UCB term at
    all) that didn't match budget-mab's actual FractionalKUBE — see
    docs/decisions-log.md (2026-07-09) for the correction.
    """
    method: str = "mcts_bl_kube_v01"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    kube_c: float = 2.0            # UCB exploration coefficient
    kube_schedule: str = "parent"  # "parent" | "global" (see above)
    kube_affordable: bool = True   # restrict argmax to affordable
                                   # nodes (KUBE feasibility step)

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # BLMCTSCntConfig.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSKubeV02Config(SearchConfig):
    """Fractional-KUBE with delayed-eager terminal backprop and a
    path-aware frontier score (both schedules; the schedules differ
    only in the bonus's clock).

    v02 of BLMCTSKubeV01Config's family -- see
    docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md §
    "kube mechanism" for the full design and
    docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.1
    for the analysis this implements.

    Two changes relative to BLMCTSKubeV01Config, both in
    core/mcts_bl_kube_search_v02_00_00.py:
      1. Terminal split + DELAYED eager backprop (BOTH schedules).
         v01 has a structural defect: a max-depth dead-end always has
         cost <= 0, so kube_density() returns -inf and it is NEVER
         selected while any finite-density node remains -- it sits in
         leaf_nodes forever, and its permanent is_terminal==True
         membership permanently satisfies kube_affordable's "always
         eligible" clause, silently disabling that filter's own
         empty-set fallback. v02 queues a terminal at creation
         (never on the frontier) and backprops it right after the
         immediately following selection resolves, fixing this
         directly.
      2. A selectable path-aware frontier score, `score_mode`
         (aligned 2026-07-19 with BLMCTSCntV02Config's two modes;
         the loser of the planned sweep is expected to be DELETED
         -- the two scorers share no code, joined only by the
         MCTS.frontier_score dispatcher):

         score_mode="parent_blend" (default) -- one-hop blend:
             blended_q = alpha*q(leaf) + (1-alpha)*q(parent)
             density = (blended_q + kube_c*sqrt(log(clock)/N_leaf))
                       / cost
           The "parent" bonus is exactly BLMCTSCntV02Config's PUCT
           bonus (this file differs from bl_cnt only by the /cost
           division). alpha=1.0 recovers BLMCTSKubeV01Config's
           exact kube_density under either schedule -- the ONLY
           exact-v01 control arm. (As first shipped 2026-07-18 the
           blend was "parent"-only; extended to "global" the same
           day -- see the core module docstring's history note.)

         score_mode="path_decay" -- full-path decayed value +
         AlphaZero-shaped bonus:
             q_path = sum_k gamma^k q(ancestor_k) / sum_k gamma^k
             density = (q_path + kube_c*sqrt(clock)/(1+N_leaf))
                       / cost
           Mirrors BLMCTSCntV02Config's path_decay exactly under
           kube_schedule="parent" (same value walk, same AZ bonus
           with clock=N_parent) before the /cost division;
           "global" swaps clock=1+t into the same shape -- the
           identical clock substitution the schedules have always
           differed by. kube_c is NOT comparable across modes
           (different bonus shapes); sweep it per mode. gamma=0.0
           reads only the leaf's own q (NOT a v01 control arm).

         clock = N_parent under kube_schedule="parent",
         1 + t under "global", for both modes. The cross-mode knob
         is idle by design (alpha unused under "path_decay", gamma
         unused under "parent_blend").
    """
    method: str = "mcts_bl_kube_v02"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    kube_c: float = 2.0            # bonus coefficient, both modes;
                                   # NOT comparable across modes
                                   # (different bonus shapes)
    kube_schedule: str = "parent"  # "parent" | "global" (see above)
    kube_affordable: bool = True   # restrict argmax to affordable
                                   # nodes (KUBE feasibility step)
    score_mode: str = "parent_blend"   # "parent_blend" | "path_decay"
    alpha: float = 0.8            # parent_blend only: own-q vs.
                                   # parent-q blend; 1.0 = the exact
                                   # BLMCTSKubeV01Config kube_density
                                   # under either schedule (no-blend
                                   # control arm -- include in sweeps)
    gamma: float = 0.8             # path_decay only: per-hop decay

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # BLMCTSKubeV01Config.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSKdepthV01Config(SearchConfig):
    """Knapsack-cost-normalized MCTS with a depth-shaping frontier
    selection bonus (no visit counts).

    Its own mcts_bl_kdepth algorithm family (v01 of that family) —
    "kdepth" = knapsack cost normalization + deterministic
    depth-shaping. A deliberately different theoretical basis from
    anything in bl_cnt/bl_kube, not a refinement: "cnt" specifically
    denotes count-based (visit-count) exploration, which this variant
    has none of.

    Sibling of BLMCTSKubeV01Config (Fractional KUBE): same
    knapsack-style objective and cost mapping, but the UCB confidence
    bonus is replaced with a fixed depth-preference function — there
    is no visit-count/exploration term of any kind, and no
    bandit/regret guarantee — see
    docs/decisions/bl-kdepth-knapsack-bonus.md.

        density_i = (q_value_i + depth_beta*f_a(depth_frac_i)) / cost_i
        cost_i = max_depth - depth_i    (same mapping as bl_cnt v01 /
                                          bl_kube v01)
        depth_frac_i = depth_i / max_depth  (0 at root, 1 at max_depth)
        f_a(z) = 1 - z**depth_alpha
            f_a(0)=1 (root, max bonus), f_a(1)=0 (max_depth, no
            bonus) — monotonically prefers shallower nodes. (Indexed
            on depth_frac, NOT on the cost fraction
            (max_depth-depth)/max_depth — the latter inverts the
            direction and rewards deep nodes instead; see the
            decision doc for the sign check.)

    kube_affordable (default True): identical feasibility-restriction
    semantics to BLMCTSKubeV01Config — same knapsack constraint, only
    the per-arm value term differs. See
    docs/decisions/kube-affordability-restriction.md.
    """
    method: str = "mcts_bl_kdepth_v01"
    num_phases: int = 1000
    gen_budget: int = 80            # total generations across the run
    depth_beta: float = 2.0         # depth-bonus coefficient
    depth_alpha: float = 1.0        # depth-bonus exponent
    kube_affordable: bool = True    # restrict argmax to affordable
                                    # nodes (knapsack feasibility step)

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # BLMCTSKubeV01Config.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class BLMCTSKdepthV02Config(SearchConfig):
    """Depth-shaping knapsack MCTS with eager terminal backprop only
    -- no formula change.

    v02 of BLMCTSKdepthV01Config's family -- see
    docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md §
    "kdepth scope" for why this v02 is hygiene-only, and
    docs/decisions/bl-cnt-path-aware-frontier-score-design.md §7.2
    for the analysis establishing the goal is unreachable via
    backprop timing alone here: depth_density() reads only a leaf's
    own frozen q_value, its own depth, and two constants -- no
    visit-count or parent-q channel exists at all for a blend to hook
    into, so "no backprop timing -- eager, lazy, or never -- can
    change which non-terminal node gets expanded next" (§7.2,
    verbatim).

    One change relative to BLMCTSKdepthV01Config, in
    core/mcts_bl_kdepth_search_v02_00_00.py: terminal split +
    delayed-eager backprop. A terminal child is queued at creation
    (backpropped right after the immediately following selection;
    timing provably inert here, applied for cross-file consistency)
    and never enters the leaf frontier -- fixes the same permanently-
    stuck-dead-end / kube_affordable-fallback-suppression defect
    BLMCTSKubeV02Config's docstring describes (this file shares the
    identical feasibility filter). depth_density() itself is BYTE-
    IDENTICAL to BLMCTSKdepthV01Config's -- no alpha knob, since
    there is nothing designed for it to blend.
    """
    method: str = "mcts_bl_kdepth_v02"
    num_phases: int = 1000
    gen_budget: int = 80            # total generations across the run
    depth_beta: float = 2.0         # depth-bonus coefficient
    depth_alpha: float = 1.0        # depth-bonus exponent
    kube_affordable: bool = True    # restrict argmax to affordable
                                    # nodes (knapsack feasibility step)

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set. Mirrors
    # BLMCTSKdepthV01Config.prm_batch_size.
    prm_batch_size: int = 1


@dataclass
class MCTSSemV01Config(SearchConfig):
    """Semantic (embedding-diversity) MCTS — v01 baseline.

    "Semantic" is the method label; the mechanism is hidden-state
    embeddings, so the per-knob fields keep their embeds_* names.

    Selection mixes the PRM q-value with a diversity bonus
    sqrt(x^T V^-1 x) over the candidates' pooled hidden-state
    embeddings; V is a ridge-regularized covariance accumulator
    (V_0 = lam * I). Reads off cfg.search.* in
    mcts_sem_search_v01_00_00, same as MCTSCntConfig does for cnt.

    v01 sources embeds from the POLICY: a second vLLM engine in
    pooling mode (runner="pooling") alongside the generative one.
    v02 (MCTSSemV02Config) sources them from the PRM instead and
    drops that engine. embeds_source selects which; the launcher
    builds the pooling engine only when embeds_source == "policy".

    The embeds-engine load knob (embeds_gpu_memory_utilization)
    lives here, not on LLMConfig: it's a method-specific concern
    (a second engine), not a generic model property. It is unused
    when embeds_source != "policy".
    """
    method: str = "mcts_sem_v01"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run

    # Where the diversity embeddings come from:
    #   "policy" — second vLLM pooling engine on the generator
    #              (v01 baseline; needs embeds_gpu_memory_utilization).
    #   "prm"    — the PRM's last-layer hidden states, folded into the
    #              in-loop prm.score forward pass (v02; no 2nd engine).
    embeds_source: str = "policy"

    # Diversity selection: q_val = ds_beta*score + ds_alpha*diversity.
    lam: float = 0.01             # ridge: V_0 = lam * I
    ds_alpha: float = 100.0       # diversity weight
    ds_beta: float = 1.0          # q-value weight

    # Embedding extraction (see core._extract_embeds). Pipeline order
    # is pool -> project -> center -> normalize.
    embeds_strategy: str = "last"     # "last" | "avg"  (pooling)
    embeds_scope: str = "full"        # "full" | "response"
    embeds_normalize: bool = True     # L2-normalize pooled vector
    embeds_center: bool = False       # subtract a mean (see mode)
    # Which mean embeds_center subtracts when it's on:
    #   "fixed" — held-out precomputed mean (embeds_mean_dir),
    #             loaded once by the launcher. The original mode.
    #   "local" — mean of the current expansion's own sibling
    #             candidates, recomputed fresh at every expansion,
    #             never carried forward (rep_exp-style local
    #             centering — docs/decisions/
    #             rep-exp-elliptical-bonus-review.md). Implemented
    #             in the v02 core only; v01 ignores it.
    # ("online" reserved for the planned Welford mode — see
    # docs/decisions/embeds-centering-design.md.) Hash: excluded
    # iff equal to the pinned neutral value "fixed", so adding
    # this field left every pre-existing config_hash unchanged
    # (see _HASH_EXCLUDE_IF_DEFAULT).
    embeds_center_mode: str = "fixed"  # "fixed" | "local"
    embeds_mean_dir: str = ""         # results/-relative .npy prefix
    # Size of the covariance V AND the final embedding dim fed to it.
    # With embeds_proj="sparse" this is the POST-projection dim (the
    # raw source dim — e.g. 4096 for the PRM — is read off the pooled
    # tensor at runtime, so it isn't configured separately).
    embeds_dim: int = 2048

    # Optional sparse random projection of the pooled embedding down to
    # embeds_dim, applied between pool and center (see core).
    #   "none"   — no projection (pooled dim must equal embeds_dim).
    #   "sparse" — Johnson-Lindenstrauss SparseRandomProjection from the
    #              raw pooled dim to embeds_dim. The matrix is fixed for
    #              the whole run (a drifting map would make V incoherent),
    #              built once from a fixed internal seed (see core) and
    #              cached. The seed isn't exposed: JL holds w.h.p. for
    #              any seed, so seed choice doesn't matter empirically;
    #              it's pinned internally only so resumes rebuild the
    #              same matrix.
    embeds_proj: str = "none"         # "none" | "sparse"

    # Covariance bookkeeping + expansion policy.
    cov_update: str = "exact"         # "exact" | "sm" (Sherman-Morrison)
    revisit_policy: str = "reuse"     # "reuse" | "regenerate"
    # Precision for V / V_inv and the embeddings multiplied against
    # them. "fp64" matches the long-standing de facto behavior:
    # np.eye()/np.linalg.solve() with no dtype= already default to
    # float64, so V/V_inv have always been float64 while the pooled
    # embeddings (torch .float() -> float32) get silently upcast at
    # every V_inv @ u / einsum. "fp32" makes that explicit and
    # uniform: V/V_inv seeded as float32, embeds cast to float32
    # before any covariance op, so fp32-vs-fp64 can be A/B'd instead
    # of relying on NumPy's implicit promotion. See docs/decisions/
    # covariance-precision.md. Hash: excluded iff equal to the pinned
    # neutral value "fp64", so adding this field left every
    # pre-existing config_hash unchanged (see _HASH_EXCLUDE_IF_DEFAULT).
    cov_dtype: str = "fp64"           # "fp32" | "fp64"

    # Second (pooling) vLLM engine's share of GPU memory. Only used
    # when embeds_source == "policy". The generative engine uses
    # llm.gpu_memory_utilization; the two must sum to leave headroom
    # for the PRM.
    embeds_gpu_memory_utilization: float = 0.1

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset). Kept
    # small because in-loop scoring is per-candidate-set.
    prm_batch_size: int = 1

    # Populated at runtime by the launcher when embeds_center=True
    # (np.load of the mean .npy). Not set from YAML.
    embeds_mean: Optional[Any] = None


@dataclass
class MCTSSemV02Config(MCTSSemV01Config):
    """Semantic MCTS — v02: embeddings from the PRM, not the policy.

    Same diversity-selection mechanism and the same embeds_* knobs as
    v01, but the pooled embeddings are read from the PRM's last-layer
    hidden states (folded into the in-loop prm.score forward pass)
    instead of a second vLLM pooling engine. So the embedding space
    shifts policy -> reward-model; everything downstream (pooling,
    normalize, center, the covariance bonus) is identical, which makes
    v01-vs-v02 a clean ablation of the embedding *source* alone.

    Inherits every field from v01; overrides the method label and the
    embeds_source default, and adds prm_embeds_layer (a knob v01 can't
    have). embeds_gpu_memory_utilization is inherited but unused here
    (no second engine) — left in place so the two variants share one
    schema; v02's YAML simply never sets it.

    NOTE embeds_dim: the PRM's hidden size differs from the generator's
    (e.g. Llama3.1-8B-PRM = 4096 vs Llama3.2-1B = 2048). embeds_dim
    sizes the covariance V, so v02's YAML MUST set it to the PRM's
    hidden size; the inherited 2048 default is wrong for the PRM.
    """
    method: str = "mcts_sem_v02"
    embeds_source: str = "prm"

    # Which PRM hidden-state layer to pool (-1 = last, closest to the
    # reward head). The v02-specific knob; v01 has no analogue.
    prm_embeds_layer: int = -1


@dataclass
class BLMCTSSemConfig(SearchConfig):
    """Semantic MCTS with best-first frontier selection.

    Frontier counterpart of MCTSSemV02Config, exactly as
    BLMCTSCntConfig is to MCTSCntConfig: selection is global across
    the leaf frontier (mcts_bl_sem_search_v01_00_00), with the sem
    family's diversity-adjusted value replacing bl_cnt's PUCT.

    A fresh SearchConfig subclass (not a MCTSSemV02Config child) so it
    carries only knobs that mean something here: no cpuct (no PUCT)
    and no revisit_policy (a frontier node is expanded at most once by
    construction). The embeds_*/ds_*/lam/cov_update knobs are the sem
    family's, with the same semantics; defaults match sem_v02's YAML
    operating point (PRM embeds, 512-dim sparse projection, sm).

    Adds ds_alpha_schedule: how the effective diversity weight evolves
    over the run. sem_v02 hardcodes the per-parent form; on a global
    frontier the choice is a real design axis, so it's exposed:
      "global" — ds_alpha * sqrt(log(1+t)), t = frontier selections
                 so far. The frontier is a flat arm set and
                 sqrt(x^T V^-1 x) is the LinUCB confidence width, so
                 the global-clock growth is the OFUL-standard
                 schedule (default).
      "parent" — ds_alpha * sqrt(log(1+parent_visits)) per node: the
                 literal sem_v02 transplant. Nodes get tree-position-
                 dependent exploration scales.
      "none"   — constant ds_alpha (no schedule).
    """
    method: str = "mcts_bl_sem_v01"
    num_phases: int = 1000        # iteration cap (safety), not budget
    gen_budget: int = 80          # total generations across the run

    # Where the diversity embeddings come from (see MCTSSemV01Config):
    # "prm" needs no second engine; "policy" builds the pooling engine.
    embeds_source: str = "prm"

    # Diversity selection: q_val = ds_beta*score + ds_alpha*sched*div.
    lam: float = 0.01             # ridge: V_0 = lam * I
    ds_alpha: float = 100.0       # diversity weight
    ds_beta: float = 1.0          # q-value weight
    ds_alpha_schedule: str = "global"  # "global" | "parent" | "none"

    # Embedding extraction (see core._extract_embeds). Pipeline order
    # is pool -> project -> center -> normalize.
    embeds_strategy: str = "last"     # "last" | "avg"  (pooling)
    embeds_scope: str = "full"        # "full" | "response"
    embeds_normalize: bool = True     # L2-normalize pooled vector
    embeds_center: bool = False       # subtract a mean (see mode)
    # Which mean embeds_center subtracts when it's on:
    #   "fixed" — held-out precomputed mean (embeds_mean_dir),
    #             loaded once by the launcher. The original mode.
    #   "local" — mean of the current expansion's own sibling
    #             candidates, recomputed fresh at every expansion,
    #             never carried forward (rep_exp-style local
    #             centering — docs/decisions/
    #             rep-exp-elliptical-bonus-review.md).
    # Hash: excluded iff equal to the pinned neutral value "fixed",
    # so adding this field left every pre-existing config_hash
    # unchanged (see _HASH_EXCLUDE_IF_DEFAULT).
    embeds_center_mode: str = "fixed"  # "fixed" | "local"
    embeds_mean_dir: str = ""         # results/-relative .npy prefix
    # Post-projection dim = size of the covariance V (the raw source
    # dim is read off the pooled tensor at runtime).
    embeds_dim: int = 512
    embeds_proj: str = "sparse"       # "none" | "sparse"

    # Which PRM hidden-state layer to pool (-1 = last). Only used
    # when embeds_source == "prm".
    prm_embeds_layer: int = -1

    # Covariance bookkeeping. Default sm (Sherman-Morrison): O(d^2)
    # persistent inverse update, validated machine-precision-identical
    # to exact in sem_v02 (see conf/search/mcts_sem_v02.yaml).
    cov_update: str = "sm"            # "exact" | "sm"
    # Precision for V / V_inv and the embeddings multiplied against
    # them. "fp64" matches the long-standing de facto behavior:
    # np.eye()/np.linalg.solve() with no dtype= already default to
    # float64, so V/V_inv have always been float64 while the pooled
    # embeddings (torch .float() -> float32) get silently upcast at
    # every V_inv @ u / einsum. "fp32" makes that explicit and
    # uniform. See docs/decisions/covariance-precision.md. Hash:
    # excluded iff equal to the pinned neutral value "fp64", so
    # adding this field left every pre-existing config_hash
    # unchanged (see _HASH_EXCLUDE_IF_DEFAULT).
    cov_dtype: str = "fp64"           # "fp32" | "fp64"

    # Second (pooling) vLLM engine's share of GPU memory. Only used
    # when embeds_source == "policy".
    embeds_gpu_memory_utilization: float = 0.1

    # PRM forward-pass micro-batch *inside* the search loop (distinct
    # from prm.score_batch_size, which scores the final dataset).
    prm_batch_size: int = 1

    # Populated at runtime by the launcher when embeds_center=True
    # (np.load of the mean .npy). Not set from YAML.
    embeds_mean: Optional[Any] = None


@dataclass
class BLMCTSSemV02Config(BLMCTSSemConfig):
    """Semantic MCTS with best-first frontier selection, delayed-eager
    terminal handling, and a selectable path-aware value term
    (mcts_bl_sem_search_v02_00_00).

    Subclasses BLMCTSSemConfig (v01); inherits every diversity /
    embedding / covariance knob unchanged. Two v02 additions:

    1. Terminal split + DELAYED eager backprop (unconditional): a
       terminal child never enters the leaf frontier; it is queued at
       creation and, one step after the following selection resolves,
       both backpropped AND folded into the diversity covariance V.
       No flag -- v01 vs v02 is a version comparison. See the v02
       search module's docstring and docs/decisions-log.md 2026-07-20.

    2. A selectable frontier VALUE term, `score_mode` (added
       2026-07-20). The frontier score is always

           ds_beta * q_term(leaf) + ds_alpha * sched * sqrt(x^T V^-1 x)

       score_mode swaps ONLY q_term -- the diversity/exploration term
       (sqrt(x^T V^-1 x)) is untouched, unlike BLMCTSCntV02Config where
       score_mode also swaps the exploration term (UCB1 -> AZ). So
       ds_alpha/ds_beta stay comparable across modes here.

         score_mode="own" (default) -- q_term = q(leaf). Byte-identical
             to BLMCTSSemConfig (v01) selection; the exact-v01 control
             arm, and the default so runs already on disk keep their
             behavior. (bl_cnt_v02 defaults to parent_blend instead --
             it was born with score_mode; bl_sem_v02 was not.)

         score_mode="parent_blend" -- one-hop blend (Option 1):
             q_term = alpha*q(leaf) + (1-alpha)*q(parent)
           Makes a backpropped parent value readable to selection (it
           is otherwise write-only to the q channel). alpha in [0, 1];
           alpha=1.0 recovers "own" exactly.

         score_mode="path_decay" -- full-path decayed subtree value
             (Option 2, value term only -- the AlphaZero u-term has no
             analog here, the diversity term stays):
             q_term = sum_k gamma^k * q(ancestor_k) / sum_k gamma^k
                      (k = 0 at the leaf, walking to the root)
           gamma in [0, 1]: gamma=1 is a plain path average; gamma=0
           reads only the leaf's own q, i.e. equals "own".

       The cross-mode knobs are idle by design (alpha unused unless
       "parent_blend"; gamma unused unless "path_decay").
    """
    method: str = "mcts_bl_sem_v02"
    score_mode: str = "own"       # "own" | "parent_blend" | "path_decay"
    alpha: float = 0.8            # parent_blend only: own-q vs.
                                   # parent-q blend; 1.0 = "own"
    gamma: float = 0.8             # path_decay only: per-hop decay;
                                   # 0.0 = "own"


@dataclass
class BoNConfig(SearchConfig):
    """Best-of-N search params. n completions sampled per question
    in a single expansion (no tree), so depth/lookahead are unused."""
    method: str = "bon"
    n: int = 256                  # completions sampled per question
    filter_duplicates: bool = True


@dataclass
class RunConfig:
    """Run-level / hardware params not tied to a model or method."""
    num_trials: int = 2
    num_questions: int = -1       # -1 = use full dataset
    # Worker processes for the CPU scoring maps (answer parsing +
    # sympy canonicalization) in build_scored_dataset. 1 = single
    # process. Raise to match the cores your session actually has
    # (check `nproc`); exceeding them just thrashes.
    num_proc: int = 1
    # Top-level results subdir override. Empty -> use data.name (the
    # normal case: results/{data.name}/...). Set to e.g. "smoketest"
    # to reroute output to results/smoketest/{data.name}/... WITHOUT
    # changing the dataset or the config hash (run is not a hash
    # group). This is how throwaway smoke-test runs stay isolated
    # from real result dirs (and from each other, across datasets)
    # while still loading the real dataset. See results_root().
    results_subdir: str = ""


@dataclass
class ExpConfig:
    """Top-level experiment config: composes the groups.

    Launchers receive this (typed) from Hydra; notebooks may build
    it directly. Search code reads it nested (config.search.cpuct,
    config.gen.temperature, ...).
    """
    gen: GenConfig = field(default_factory=GenConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # Base type so any method's subclass can bind here; the concrete
    # schema is selected per-run via the conf/search/ group (each
    # launcher registers its subclass under the "search" group).
    search: SearchConfig = field(default_factory=SearchConfig)
    run: RunConfig = field(default_factory=RunConfig)
    algo: str = "mcts_cnt"
    # Run-wide base path; group files interpolate ${base_dir}/...
    base_dir: str = "/groups/chichengz/tnn/datasets"


# config_name is a module-level function, not a method/property on
# ExpConfig. Hydra hands main() a struct-mode DictConfig that exposes
# only the declared *data* fields — dataclass methods/properties are
# not carried over, so `cfg.config_name` would raise. As a plain
# function taking the config as an argument, it works identically on
# a DictConfig (launcher) and a real ExpConfig (notebook).


def results_root(cfg) -> str:
    """The top-level results subdir for this run: `{results_subdir}/
    {data.name}` if run.results_subdir is set, else just data.name.
    One definition so every path-builder (launchers + find_run_dir +
    status.py) agrees on where a run's dir lives. Overriding
    run.results_subdir (e.g. 'smoketest') reroutes output to
    results/smoketest/{data.name}/... WITHOUT changing the dataset or
    the config hash -- the dataset subfolder keeps smoke tests across
    different datasets from mixing in one flat dir."""
    sub = getattr(cfg.run, "results_subdir", "") or ""
    return f"{sub}/{cfg.data.name}" if sub else cfg.data.name


def level_dir(cfg) -> str:
    """Per-level grouping dir under results/{results_root}/, e.g.
    'bon--level-3'. Drops the level suffix to just 'bon' when level
    is None (whole split). The run dir is then nested inside it:
    results/{results_root}/{level_dir}/{config_name}."""
    level_str = (
        f"--level-{cfg.data.level}"
        if cfg.data.level is not None else ""
    )
    return f"{cfg.search.method}{level_str}"


# --------------------------------------------------------------- #
# Run identity: readable prefix + a hash of the full config.       #
#                                                                  #
# The dir name is `{readable prefix}--cfg-{hash8}`. The prefix is a #
# curated, *cosmetic* subset of fields for eyeball-skimming; the    #
# hash is the collision-safe identity, computed over the full       #
# run-affecting config. This splits the two jobs the old all-knobs- #
# in-the-name scheme conflated: identity (the hash, complete) vs.   #
# human display (the prefix, partial) — so adding a knob no longer  #
# changes the prefix and orphans old dirs, and the hash still       #
# guarantees distinct configs get distinct dirs for resume/.done.   #
#                                                                   #
# IDENTITY RULE: the hash is RECORDED into manifest.json at run     #
# creation; readers locate a run by matching that recorded hash     #
# (find_run_dir), NOT by trusting a re-derived name. Only the        #
# launcher recomputes (to decide resume-vs-fresh). Full rationale:   #
# vault note `question-config-name-experiment-naming` /              #
# docs/decisions-log.md 2026-06-21.                                  #
# --------------------------------------------------------------- #

MANIFEST_FILE = "manifest.json"

# Config groups whose fields define a run's *results*. Everything in
# these (resolved) goes into the identity hash + manifest. Cosmetic /
# environment-only fields are stripped below.
_HASH_GROUPS = ("search", "gen", "llm", "prm", "data")

# Fields excluded from the identity hash: they don't change the
# produced results, only how/where the job runs or reports. Keyed by
# group. (num_trials/num_questions live in `run`, which isn't a hash
# group at all, so they're excluded by omission.)
_HASH_EXCLUDE = {
    "llm": {"gpu_memory_utilization"},
    "prm": {"device_map", "score_batch_size"},
    "search": {"embeds_gpu_memory_utilization", "embeds_mean"},
    "data": {"ds_dir", "question_field", "grader_name"},
}

# Fields added after many configs already had hashes: excluded from
# the identity iff they equal this pinned "neutral" value (the value
# that reproduces the old, pre-field behavior), so every existing
# config_hash stays unchanged. The pinned value is frozen forever,
# independent of the dataclass default above.
_HASH_EXCLUDE_IF_DEFAULT = {
    "search": {"cov_dtype": "fp64", "embeds_center_mode": "fixed"},
}


def _resolved(cfg):
    """cfg as a plain nested dict, interpolations resolved. Imported
    lazily so notebooks that build dataclasses directly (no OmegaConf)
    can still import this module."""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    # Already a plain object (dataclass instance from a notebook).
    from dataclasses import asdict, is_dataclass
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def config_identity(cfg) -> dict:
    """The run-affecting config, flat-by-group, with cosmetic/env
    fields stripped. This is exactly what config_hash hashes and what
    write_manifest records — one definition so the two never drift."""
    full = _resolved(cfg)
    out: dict = {}
    for group in _HASH_GROUPS:
        gvals = full.get(group)
        if not isinstance(gvals, dict):
            continue
        exclude = _HASH_EXCLUDE.get(group, set())
        neutral = _HASH_EXCLUDE_IF_DEFAULT.get(group, {})
        out[group] = {
            k: v for k, v in gvals.items()
            if k not in exclude
            and not (k in neutral and v == neutral[k])
        }
    return out


def config_hash(cfg, n: int = 8) -> str:
    """First `n` hex chars of a sha1 over config_identity. Canonical
    JSON (sorted keys, no whitespace) so the hash is stable across
    field-ordering and platform. RECORD this once; do not rely on
    re-deriving it to find old runs (see find_run_dir)."""
    blob = json.dumps(
        config_identity(cfg), sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def config_name(cfg) -> str:
    """Run dir name: readable prefix + `--cfg-{hash}`.

    Prefix order (curated for skimming): algo, level (only when set —
    a None level, e.g. a full split or a level-less dataset like AIME,
    is omitted), llm, prm, depth, batch_size, budget. Everything else
    (cpuct, lam, proj, cov, tmpl, prm_batch_size, ...) is NOT in the
    name — it's in the hash + manifest. Works on a DictConfig
    (launcher) or a real ExpConfig (notebook).
    """
    algo = cfg.search.method
    # Drop the redundant "-Instruct" marker for shorter dirs.
    llm_name = cfg.llm.name.replace("-Instruct", "")
    prm_name = getattr(cfg.prm, "kind", None)
    # level: optional, dataset-specific. Shown only when set; the hash
    # carries it unconditionally so a level-N vs full-split run never
    # collide regardless of whether the prefix shows it.
    level_str = (
        f"--level-{cfg.data.level}"
        if cfg.data.level is not None else ""
    )
    prm_str = f"--{prm_name}" if prm_name is not None else ""
    return (
        f"{algo}{level_str}--{llm_name}{prm_str}"
        f"--d-{cfg.search.max_depth}--bs-{cfg.search.batch_size}"
        f"--b-{cfg.search.gen_budget:03d}"
        f"--cfg-{config_hash(cfg)}"
    )


def write_manifest(
    result_dir: str, cfg, varied=None, run_id: Optional[str] = None,
) -> None:
    """Record the run's identity into {result_dir}/manifest.json so it
    can be located later by recorded fact (find_run_dir), not by
    re-deriving the name. Atomic write (temp + rename). `varied` is an
    optional list of the knob names this run sweeps — a display hint
    for tables/readers, not part of identity. `run_id` is the W&B run
    id; called once before wandb.init (run_id=None) and once after
    (run_id=wandb_run.id) — see load_wandb_run_id."""
    payload = {
        "config_name": config_name(cfg),
        "config_hash": config_hash(cfg),
        "config_identity": config_identity(cfg),
        "varied": list(varied) if varied else [],
        "run_id": run_id,
    }
    path = f"{result_dir}/{MANIFEST_FILE}"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, default=str)
        fout.write("\n")
    os.replace(tmp, path)


def find_run_dir(root_dir: str, cfg) -> Optional[str]:
    """Locate an existing run dir for `cfg` by matching the recorded
    identity hash in each candidate's manifest.json — NOT by trusting
    a re-derived name. Searches
    results/{results_root}/{level_dir}/*/. Returns the dir path, or
    None if no manifest matches (a fresh run, or an old dir not yet
    backfilled with a manifest)."""
    target = config_hash(cfg)
    parent = f"{root_dir}/results/{results_root(cfg)}/{level_dir(cfg)}"
    for manifest_path in glob.glob(f"{parent}/*/{MANIFEST_FILE}"):
        try:
            with open(manifest_path, encoding="utf-8") as fin:
                rec = json.load(fin)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("config_hash") == target:
            return os.path.dirname(manifest_path)
    return None


def manifest_run_name(result_dir: str) -> Optional[str]:
    """The `config_name` recorded in a dir's manifest, or None. This is
    the authoritative basename for the dir's trial files
    ({run_name}--trial-NNN.jsonl) — readers should use it rather than
    recompute config_name, so files resolve even if config_name's
    format later changes."""
    path = f"{result_dir}/{MANIFEST_FILE}"
    try:
        with open(path, encoding="utf-8") as fin:
            return json.load(fin).get("config_name")
    except (OSError, json.JSONDecodeError):
        return None


def resolve_result_dir(root_dir: str, cfg, override=None):
    """Locate a run's (result_dir, run_name) for a reader
    (compute_stats / prepare_scored_dataset). Resolution order:

      1. explicit `override` path (e.g. CLI +result_dir=...) — for old
         dirs with no manifest, or any direct addressing;
      2. find_run_dir — match cfg's recorded identity hash in a
         manifest (the robust, record-once path);
      3. fall back to the freshly-computed config_name path — covers a
         brand-new run whose dir doesn't exist yet, and reproduces the
         legacy behavior when no manifest is present.

    run_name is the dir's recorded config_name when a manifest exists
    (authoritative for the trial filenames), else the dir basename."""
    if override is not None:
        result_dir = (
            override if os.path.isabs(override)
            else f"{root_dir}/{override}"
        )
    else:
        result_dir = find_run_dir(root_dir, cfg)
        if result_dir is None:
            # No manifest matched: fresh run, or an un-backfilled old
            # dir. Recompute the name (legacy path).
            result_dir = (
                f"{root_dir}/results/{results_root(cfg)}"
                f"/{level_dir(cfg)}/{config_name(cfg)}"
            )
    run_name = manifest_run_name(result_dir) or os.path.basename(
        result_dir.rstrip("/")
    )
    return result_dir, run_name