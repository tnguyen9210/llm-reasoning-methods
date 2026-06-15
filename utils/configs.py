
from dataclasses import dataclass, field
from typing import Optional


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
# mcts_cnt is migrated: generate_mcts_cnt + mcts_cnt_search_v05_00_00
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


@dataclass
class PRMConfig:
    """Process reward model group. Selected via conf/prm/."""
    name: str = "Llama3.1-8B-PRM-Deepseek-Data"
    prm_dir: str = "???"          # required; set in the YAML group
    device_map: str = "cuda:0"


@dataclass
class DataConfig:
    """Dataset group: where it lives and how to read it. Selected
    via conf/data/. level is optional (None = whole split)."""
    name: str = "prm800k"
    ds_dir: str = "???"           # required; set in the YAML group
    ds_split: str = "test"
    question_field: str = "problem"
    level: Optional[int] = None


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
    method: str = "mcts_cnt"
    num_phases: int = 1000
    gen_budget: int = 80          # total generations across the run
    cpuct: float = 2.0


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
    num_trials: int = 4
    num_questions: int = -1       # -1 = use full dataset


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


def config_name(cfg) -> str:
    """Self-describing run name. Dispatches on the search method so
    each algo emits its own knobs. Works on ExpConfig or its
    DictConfig (struct mode exposes only declared data fields)."""
    level_str = (
        f"--level-{cfg.data.level}"
        if cfg.data.level is not None else ""
    )
    method = cfg.search.method
    if method == "mcts_cnt":
        return (
            f"mcts_cnt{level_str}"
            f"--bs-{cfg.search.batch_size}--d-{cfg.search.max_depth}"
            f"--b-{cfg.search.gen_budget:03d}"
            f"--cpuct-{cfg.search.cpuct}"
        )
    if method == "bon":
        return (
            f"bon{level_str}"
            f"--n-{cfg.search.n}--temp-{cfg.gen.temperature}"
            f"--mtoks-{cfg.gen.max_tokens}"
        )
    raise ValueError(f"Unknown search method: {method!r}")