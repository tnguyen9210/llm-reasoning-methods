# Context-length overflow in BL search: root cause of the llama-3b bl_sem crashes, and the terminal-on-context-exhaustion guard

*2026-07-10 — decided; implementation pending*

Records the root-cause investigation of the two failed
`mcts_bl_sem_v01` llama-3b fp16 cells (2026-07-09), the
token-length evidence gathered across models, and the decision on
how the search should handle prompts that outgrow the model's
context window.

## The failure

Both llama-3b bl_sem cells crashed with the identical unhandled
exception:

```
ValueError: The decoder prompt (length 5000) plus the number of
requested output tokens (at least 1) is longer than the maximum
model length of 5000.
```

- `cfg-0f06296f` (run `2goolnzd`, w_eff=100): died during trial 0
  → 0/2 trials, empty result dir.
- `cfg-3ca318f6` (run `yf562ig8`, w_eff=10): trial 0 completed
  (6.19 hr), died during trial 1 → 1/2 trials.

The two were launched ~80 s apart as part of the same sweep and
hit the same bug independently (an earlier note wrongly recorded
them as two attempts at one cell).

Mechanism: `conf/llm/llama_3b.yaml` caps `max_model_len: 5000`.
On a deep best-first path, system prompt + question + accumulated
steps reached 5000 tokens; the next expansion called
`llm.generate` on it and vLLM's V1 input validator raised. The
raise happens in `generate_k_steps`
(`sal/search/utils.py`, `llm.generate` on
`initial_prompt + lookahead_text`, no length check), reached from
`_generate_candidates` in the search core. Nothing above it
catches: no guard in any search variant, no per-question
try/except in the launcher — so one over-long path kills the
whole trial process and every remaining question with it.

## Evidence: who actually grazes the cap

Completion token lengths (model's own tokenizer,
`add_special_tokens=False`), per question, bl_sem w_eff=10.
"Near-cap" = any completion within 700 tokens of the model's
`max_model_len`. Full per-prompt tables:
`zlogs/comp_token_lens_bl_sem_weff10_4models.txt` (temporary,
uncommitted; regenerate by tokenizing `completions` in each run's
`mcts_...--trial-00N.jsonl`).

| model (cap) | trial | mean of per-q avgs | near-cap questions |
|---|---|---|---|
| llama-1b fp16 (5000) | 0 | 788 | 7 |
| llama-1b fp16 | 1 | 735 | 5 |
| llama-3b fp16 (5000) | 0 | 681 | 6 (trial 1 crashed) |
| qwen-3b fp16 (5000) | 0 / 1 | 608 / 665 | 1 / 1 |
| qwen-7b gptq-int4 (5000) | 0 / 1 | 486 / 490 | 1 / 1 |
| qwen-math-1.5b (4096) | 0 / 1 | 687 / 685 | 1 / 0 |

Three reads:

1. **Cap-grazing is a Llama-family behavior.** Both llama models
   flirt with the cap on 5–7 questions per trial; every qwen
   model touches it on at most one. qwen-math-1.5b never comes
   within 500 of its (smaller) 4096 cap. That llama-3b crashed
   while llama-1b survived twice is luck — with 5–7 grazers per
   trial, both llama cells carry substantial per-trial crash
   probability.
2. **The frontier protocol amplifies it.** The phase-based
   counterpart (`mcts_sem_v02`, llama-3b, same lam/ds_alpha,
   `cfg-0c3fa88a`) completed 2/2 trials with only ONE near-cap
   question (q62) and mean completion length ~500 vs bl_sem's
   681. On q53, phase-based produced 0 completions where bl_sem
   produced 29 averaging ~3,974 tokens: same hard question, two
   failure styles — phase-based gives up empty, frontier "wins"
   by filling the context window.
3. **Scores are already contaminated where the run survives.**
   On cap-saturating questions (llama-3b q62/q98: *average*
   completion ~4,400–4,600 tokens), the search spent its budget
   on runaway paths clipped at the window, so those cells partly
   measure context exhaustion, not search quality. This applies
   to the already-scored llama-1b bl_sem cells (~5% of questions
   per trial).

## Decision

Treat **context exhaustion exactly like `max_depth`: a
terminality condition**, plus containment at the launcher:

1. **Length guard in the search (the fix).** Before expanding a
   node, tokenize its prompt; if
   `prompt_tokens + step_headroom >= max_model_len`, mark the
   node terminal (backprop its value, drop it from the frontier)
   instead of calling `llm.generate`. This mirrors the existing
   max-depth terminal handling, so no new failure vocabulary is
   introduced. The guard belongs in every variant (cnt, sem, and
   all bl siblings) — the bug is latent everywhere; the bl
   variants just concentrate probability on it.
2. **Per-question try/except in the launcher (containment).** A
   question whose search still escapes with an exception loses
   its completions (scores 0 — consistent with how
   budget-exhausted questions already score) instead of killing
   the trial and all remaining questions.

**Rejected: raising `max_model_len`** (e.g. 5000 → 6144+ for
llama_3b). It only moves the cliff — the grazing questions run
away until *something* stops them — and costs KV-cache memory on
the V100, where the PRM shares the GPU. It also silently changes
the experiment (longer completions on runaway questions), whereas
the guard is behavior-neutral for every run that never hits the
cap.

## Consequences / follow-ups

- Both llama-3b bl_sem cells (`0f06296f`, `3ca318f6`) need reruns
  after the guard lands; both are marked FAILED in
  `experiments.yaml` with pointers here.
- The scored llama-1b bl_sem cells carry the interpretation
  caveat from read 3 above.
- The guard adds a terminal condition without a config knob, so
  config hashes are unchanged; runs that never hit the cap are
  byte-identical. Runs that would have crashed become defined
  behavior instead — strictly more data, no comparability break.
- Open question for later: whether "terminal because out of
  context" should score the node's partial path at all, or count
  as a failed path (current lean: treat like any other terminal —
  the PRM scores what exists — and revisit if it distorts
  completion-rate comparisons).
