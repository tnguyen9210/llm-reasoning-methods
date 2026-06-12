# Findings

Append-only log of empirical observations about repo behavior:
environment sensitivity, library quirks, format gotchas — anything
that affects how experiments are run or interpreted. Not a home for
scientific results (algorithm comparisons belong in W&B and paper
notes). Newest first. One `##` per finding. Decisions motivated by a
finding go in [decisions.md](decisions.md) and reference it.

## 2026-06-11 — Library versions change generated output content

**Observation:** the same code, config, and seeds produce materially
different trajectories under different library stacks. Root cause
(established with tokenizer-level tests): trajectories complete (end
with "The final answer is") iff the model's continuation prompt ends
with the `\n\n` step separator — and whether it does depended on the
interaction of an explicit `removesuffix("\n\n")` in the search code
with the transformers version. With `mcts_cnt_search_v05_00_00` on
prm800k level 4 (old env: vLLM 0.6.4 / transformers 4.45.2 /
torch 2.5.1; py311 env: newer stack):

| | old env | py311 env |
|---|---|---|
| with strip | ~12% complete | ~12% complete (byte-identical) |
| without strip | ~12% complete (byte-identical) | **81–97% complete** |

Mechanism, piece by piece:

- SAL's custom chat template (`sal/config.py`, applied on every run)
  renders message content without `| trim` — its one difference from
  the stock Llama template, which trims content (and, in
  transformers 4.45, crashes with `ValueError: substring not found`
  on content ending `\n\n` under `continue_final_message=True`).
- transformers 4.45's `continue_final_message` *truncation* step
  drops the trailing separator from the rendered prompt even when the
  template preserved it; newer transformers keeps it.
- The explicit strip in the search code deleted it in any version.
  Without the separator the model treats the message as finished and
  emits EOS immediately; the empty-step path then records the
  abandoned text as a "completed" trajectory (~80% of completions).
  The `\r\n\r\n` anomalies (which evade the `\n\n` vLLM stop string)
  co-occur with the broken prompts.
- Side observation: given identical prompts and seeds, generations
  are byte-identical across vLLM 0.6.4 and the newer stack.

**Fix (same day):** search files now strip for templating, then
re-append the separator to the templated string
(`mcts_cnt_search_v05_00_00`, `mcts_bl_cnt_search_v01/v02_00_00`) —
prompt correctness no longer depends on the transformers version.

Related, same env: transformers 4.45 returns a raw tensor from
`apply_chat_template(..., return_tensors="pt")` unless
`return_dict=True` is passed; `reward_models.py` was fixed
accordingly so PRM scoring works in both stacks.

**Implication:** validate output *format* after any environment
change, not just exit codes — a run can succeed while silently
producing garbage. Tools: `unittests/check_trajectory_completeness.py`
(results-level) and
`unittests/test_chat_template_continuation_v1.ipynb` (tokenizer-only
env gate, runs in seconds).
