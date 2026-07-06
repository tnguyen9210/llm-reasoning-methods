# Library versions change generated output content

*2026-06-11*

**Observation:** the same code, config, and seeds produce materially
different trajectories under different library stacks. With
`mcts_cnt_search_v05_00_00` on prm800k level 4 (2 questions,
2 trials), the old env (vLLM 0.6.4 / transformers 4.45.2 /
torch 2.5.1) yielded 0–12.5% complete trajectories; the newer py311
stack yielded 81–97%.

## The one fact that drives everything

The search builds solutions step by step, with `\n\n` between steps.
To generate the next step, the partial solution is templated into a
prompt. Everything hinges on the last two characters of that prompt:

```
Prompt A: "...The dot product is 13.\n\n"   <- ends with separator
Prompt B: "...The dot product is 13."       <- separator missing
```

- Prompt A: the model sees "start of a new step" → writes the next
  step → eventually writes "The final answer is" → complete.
- Prompt B: the model sees a finished-looking message → emits EOS
  immediately, generating nothing. The empty-step path in
  `create_child` then records the half-finished text as a "completed"
  trajectory. This produced the ~80% abandoned trajectories.

So completeness reduces entirely to: does the trailing `\n\n` survive
into the prompt the model actually sees?

## Two different things delete the separator

1. **Our own code:** an explicit `removesuffix("\n\n")` before
   templating — deletes it in every environment.
2. **The old library:** transformers 4.45's `continue_final_message`
   truncation trims the trailing separator during templating even
   when we don't strip it; newer transformers preserves it. (SAL's
   custom chat template itself is whitespace-preserving — no `| trim`
   on message content, its one difference from the stock Llama
   template.)

## The 2×2 that explains the confusion

| | old env (tf 4.45) | py311 (newer) |
|---|---|---|
| with strip | ~12% complete (we deleted it) | ~12% (we deleted it) |
| without strip | ~12% (library deleted it) | **81–97% complete** |

The three broken cells produce **byte-identical** generations —
identical final prompts plus identical seeds, reproducible even
across the two vLLM versions.

## Why the diagnosis took three rounds

1. Removed the strip, tested in the old env → no change (the library
   deletes the separator anyway) → the strip looked cosmetic.
2. Tested in py311 without the strip → complete trajectories → looked
   like a pure environment effect (the original framing of this
   finding).
3. Restored the strip, re-ran py311 → incomplete again, byte-identical
   to the old env → the strip itself is harmful; the environment only
   matters when the strip is absent.

## Why the strip existed, and the fix

The strip was added (commit 34b7d11) to avoid a real crash: with the
*stock* Llama template, `apply_chat_template` raises
`ValueError: substring not found` when assistant content ends with
`\n\n` under `continue_final_message=True`. The same commit also
switched back to SAL's custom template, which doesn't have that
problem — the guard became unnecessary but stayed behind, silently
sabotaging the prompts.

Fix (same day), in `mcts_cnt_search_v05_00_00` and
`mcts_bl_cnt_search_v01/v02_00_00` — strip-and-reappend:

```python
clean = text.removesuffix("\n\n")    # templating can never crash/trim
prompt = apply_chat_template(clean)
prompt = prompt + "\n\n"             # nothing can remove it anymore
```

`apply_chat_template` is the only place the separator can be lost;
re-appending after that call makes the prompt correct by
construction, independent of the transformers version.

## Related observations, same env

- transformers 4.45 returns a raw tensor from
  `apply_chat_template(..., return_tensors="pt")` unless
  `return_dict=True` is passed; `reward_models.py` was fixed
  accordingly so PRM scoring works in both stacks.
- `\r\n\r\n` separator anomalies (which evade the `\n\n` vLLM stop
  string) co-occur with the broken prompts and vanish with them.

**Implication:** validate output *format* after any environment
change, not just exit codes — a run can succeed while silently
producing garbage. Tools: `unittests/check_trajectory_completeness.py`
(results-level) and
`unittests/test_step_separator_affect_generation.ipynb`
(tokenizer-only env gate, runs in seconds; optional GPU check for the
model behavior itself).

## Addendum 2026-07-06 — the template-family dimension, and sem never got the fix

Two updates from smoke-testing `mcts_sem_v02` on the native
Llama template (per-template analysis:
`unittests/examine_llm_chat_templates_v1.ipynb` and the vault
note `llm-prm-deep-dive/findings/llm-chat-templates.md`).

**1. The library isn't the only thing that trims — templates
differ by family.** Llama's *native* template trims the trailing
`\n\n` (`| trim` on message content); Qwen's native template
preserves it. So on the current py311 stack the failure mode
above reappears whenever a Llama model runs on its native
template without the strip-and-reappend guard.

**2. The strip-and-reappend fix never reached the sem family.**
It lives in `mcts_cnt_search_v01_00_00` (:263-273) and
`mcts_bl_cnt_search_v01/v02_00_00`, but `mcts_sem_search_v01/
v02_00_00` template `current_text` directly (their only
`removesuffix("\n\n")` is on the embed/score copy of
candidates, not the generation prompt). Nothing has broken in
practice because the 2026-06-19 per-family default keeps Llama
on the custom (whitespace-preserving) template — configuration
masking the missing guard.

Measured (sem-v02, Llama3.2-1B, prm=qwen, level 4):

| | Llama native (no guard) | Llama custom (control) | Qwen-Math-1.5B native (full trial) |
|---|---|---|---|
| nodes reaching a final answer | **0/26** | 8/8 | **99.7%** (2809/2817) |
| 1-step stubs (<600 chars) | 77% | 0% | 1.8% |
| tree depth | 1–2 | up to 16 | median 12 |

**Implications:** recorded qwen sem-v02 results are unaffected
(full-trial census healthy). Llama sem runs are safe only by
configuration. Fix: add the same strip-and-reappend block to
`mcts_sem_search_v01/v02` (finish the migration this finding's
fix started).

**Fixed 2026-07-06.** Ported the identical strip-and-reappend
block (`mcts_cnt_search_v01_00_00:263-273`) into both
`mcts_sem_search_v01_00_00` and `mcts_sem_search_v02_00_00`'s
`_generate_candidates`. No version bump — every currently
recorded sem run uses a template that already preserves the
separator (Llama+custom or Qwen+native), so the fix reproduces
byte-identical prompts at every existing hash; zero Llama+native
sem runs existed prior to this fix. Re-ran the same smoke test
after the change:

| | Llama native, before fix | Llama native, after fix |
|---|---|---|
| nodes reaching a final answer | 0/26 | **32/39 (82%)** |
| 1-step stubs (<600 chars) | 77% | **2.6%** |
| tree depth | 1–2 | back to normal multi-step trees |

Matches the healthy controls (Llama+custom 8/8, Qwen+native
99.7%). Migration is now complete across cnt/bl_cnt/sem.
