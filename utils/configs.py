
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    # General
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    data_string: str = "Aug 1 2025"
    seed: int = 0 

    # PRM 
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # LLM 
    llm_mem: float = 0.1
    vllm_mem: float = 0.1
    
    # Chat template related options
    system_prompt: str = system_prompt
    custom_chat_template: str = custom_chat_template

    
    
@dataclass
class SearchConfig:
    #
    algo_name: str = "mcts"
    algo_version: str = "v01"

    # General  
    bs: int = 4 # number of nodes to be generated per LLM call
    beam_width = 4 # number of nodes left after selection 
    lookahead = 0  # ...
    max_depth = 20   # max tree depth. note: after reaching max_depth then terminate search 
    
    # MCTS parameters
    num_batches: int = 4
    step_budget = num_batches*max_depths
    max_nphases = 1000

    lam = 0.01 
    use_ppl = True 
    embeds_normalizing = True 
    embeds_strategy = 'avg'
    embeds_centering = False 
    embeds_mean_file_name = ""

    # DivOp's parameters
    ds_beta = 1.0 
    ds_alpha = 100.0 
    negative_reward = 0
    
    sort_completed: bool = False 
    filter_duplicates: bool = True # whether to remove duplicates in the final list of completions 

    

@dataclass:
class ExperimentConfig:
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)

    # base_dir
    base_dir: str = "/groups/chichengz/tnn/datasets/"

    # llm_dir and prm_dir
    llm_name: str = "Llama-3.2-1B-Instruct"
    llm_dir: str = base_dir + f"{llm_name}"
    prm_name: str = "Llama3.1-8B-PRM-Deepseek-Data"
    prm_dir: str = base_dir + f"{prm_name}"
    
    # dataset dir 
    ds_name: str = "aime"
    ds_split: str = "test" 
    ds_dir: str = base_dir + f"{ds_name}"
    
    @property
    def config_name(self) -> str: 
        return f""

system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

custom_chat_template = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'