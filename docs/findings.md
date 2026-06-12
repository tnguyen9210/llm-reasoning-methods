# Findings

Append-only log of empirical observations about repo behavior:
environment sensitivity, library quirks, format gotchas — anything
that affects how experiments are run or interpreted. Not a home for
scientific results (algorithm comparisons belong in W&B and paper
notes). Newest first. One `##` per finding. Decisions motivated by a
finding go in [decisions.md](decisions.md) and reference it.

## 2026-06-11 — Library versions change generated output content

**Observation:** the same code, config, and seeds produce materially
different trajectories under different library stacks. With
`mcts_cnt_search_v05_00_00` on prm800k level 4 (2 questions,
2 trials), the old env (vLLM 0.6.4 / transformers 4.45.2 /
torch 2.5.1) yielded 0–12.5% complete trajectories; the newer py311
stack yielded 81–97%.

### The one fact that drives everything

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

### Two different things delete the separator

1. **Our own code:** an explicit `removesuffix("\n\n")` before
   templating — deletes it in every environment.
2. **The old library:** transformers 4.45's `continue_final_message`
   truncation trims the trailing separator during templating even
   when we don't strip it; newer transformers preserves it. (SAL's
   custom chat template itself is whitespace-preserving — no `| trim`
   on message content, its one difference from the stock Llama
   template.)

### The 2×2 that explains the confusion

| | old env (tf 4.45) | py311 (newer) |
|---|---|---|
| with strip | ~12% complete (we deleted it) | ~12% (we deleted it) |
| without strip | ~12% (library deleted it) | **81–97% complete** |

The three broken cells produce **byte-identical** generations —
identical final prompts plus identical seeds, reproducible even
across the two vLLM versions.

### Why the diagnosis took three rounds

1. Removed the strip, tested in the old env → no change (the library
   deletes the separator anyway) → the strip looked cosmetic.
2. Tested in py311 without the strip → complete trajectories → looked
   like a pure environment effect (the original framing of this
   finding).
3. Restored the strip, re-ran py311 → incomplete again, byte-identical
   to the old env → the strip itself is harmful; the environment only
   matters when the strip is absent.

### Why the strip existed, and the fix

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

### Related observations, same env

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
`unittests/test_chat_template_continuation_v1.ipynb` (tokenizer-only
env gate, runs in seconds).
