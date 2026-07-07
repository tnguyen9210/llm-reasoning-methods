# `embeds_scope`: which tokens get pooled, and why one combination is blocked

Originating log entry:
[decisions-log.md #2026-07-07](../decisions-log.md#2026-07-07--search-embeds_scoperesponse-stays-unimplemented-for-embeds_sourceprm).

## What it does

`embeds_scope` (`MCTSSemV01Config`, inherited by v02) picks which
tokens of a candidate's tokenized sequence contribute to the pooled
diversity embedding, applied in `_extract_embeds` before pooling
(scope → pool → project → center → normalize):

- **`"full"`** (default) — the entire tokenized sequence (system +
  user + assistant) contributes.
- **`"response"`** — only the assistant-response tokens contribute;
  everything before `response_start_idx` (the system/user prefix) is
  sliced off first (`raw[response_start_idx:, :]`).

`response_start_idx` is computed once per question by
`_compute_response_start_idx`: render the chat with an empty assistant
message and the tokenizer's `add_generation_prompt=True`, then count
tokens up to where the assistant turn begins.

## Support matrix (verified 2026-07-07)

| `embeds_scope` | `embeds_source="policy"` (v01) | `embeds_source="prm"` (v02) |
|---|---|---|
| `"full"` | works | works |
| `"response"` | works | **blocked** (`NotImplementedError`) |

Both v01 and v02 have their own copy of `_extract_embeds` and
`_compute_response_start_idx` (structurally identical logic, not
literally shared code — consistent with the project's framing that
v01/v02 differ only in embedding *source*, sharing everything else).
For `"full"`, both scopes work identically in either file — no
tokenizer dependency, since the whole sequence is used regardless of
which model produced it.

For `"response"` with `embeds_source="policy"` (v01): `_embed_candidates`
computes `response_start_idx` using `llm_vllm.get_tokenizer()` — the
**generator's** tokenizer — and the embedding being sliced was *also*
produced by the generator (via the pooling engine). Same tokenizer
produced both the index and the sequence being sliced, so the slice is
correct by construction.

For `"response"` with `embeds_source="prm"` (v02): this is where it
breaks. `response_start_idx` is still computed with the **generator's**
tokenizer (via `mcts_search` → `_compute_response_start_idx`), but the
sequence being sliced under this source comes from a **PRM** forward
pass, over the PRM's own tokenization and chat template. Slicing the
PRM's hidden-state tensor at an index computed from a different
tokenizer's token count is not an approximation — it's a different
tokenizer, over a different chat template, so the index has no defined
meaning in the PRM's token stream. This would silently produce a
valid-shaped but wrong slice (pooling over whatever tokens happen to
land at that position, not the actual assistant response) rather than
crash — a worse failure mode to leave unguarded than to block outright.

## The decision

`_embed_candidates`'s `source == "prm"` branch explicitly checks
`if sc.embeds_scope != "full": raise NotImplementedError(...)` rather
than attempting to reuse or approximately adjust the generator's
`response_start_idx`. The guard is scoped to the `prm` source
specifically — `policy`-source `"response"` scope is unaffected and
untouched, since it's correct as described above.

## What a correct implementation would need, if ever prioritized

1. A parallel `_compute_prm_response_start_idx(question, config,
   prm.tokenizer)` that renders the **PRM's own** prefix-only chat
   (via the PRM's `apply_chat_template`) and counts **its** tokens —
   the generator's index cannot simply be reused or adjusted, since
   the two tokenizers segment text differently.
2. Threading the PRM's tokenizer to wherever this gets computed —
   currently only the generator's tokenizer is passed around for this
   purpose (`mcts_search` holds `tokenizer = llm_vllm.get_tokenizer()`
   and nothing else).
3. Likely a **per-row** start index rather than a single scalar, if
   `PRM._embed_batch` ever batches candidates across more than one
   question in a single forward pass. Today it's one question at a
   time via `_embed_candidates`, so a scalar suffices only
   incidentally — this would need revisiting the moment cross-question
   batching is introduced.
4. Verification via decoded token spans — confirm the computed index
   actually lands at the assistant turn for the PRM's specific
   template — since a wrong-but-plausible index wouldn't crash, it
   would just quietly pool the wrong tokens. A shape-only unit test
   would not catch this class of bug.

## Why deferred rather than fixed now

The real config (`conf/search/mcts_sem_v02.yaml`) already runs
`embeds_scope=full`, so no current experiment needs the blocked path;
the guard's job is to keep a future misconfiguration loud (`raise`)
rather than let it silently mispool tokens. Revisit if a future
ablation specifically wants to isolate the response-only embedding
under the PRM source — at that point, implement per the four points
above rather than attempting a quick reuse of the existing generator
index.
