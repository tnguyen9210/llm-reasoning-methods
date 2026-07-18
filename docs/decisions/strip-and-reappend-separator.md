# Strip-and-reappend: the `\n\n` step separator survives across every MCTS variant

*Originating entries:
[decisions-log.md #2026-06-13](../decisions-log.md#2026-06-13--prompting-use-native-chat-templates-not-one-custom-template)
(introduces the mechanism),
[#2026-07-06](../decisions-log.md#2026-07-06--search-sem-mcts-gets-the-strip-and-reappend-separator-guard-applied-in-place)
(ports it to sem-mcts). This doc covers the mechanism generally across
all current MCTS search methods, verified against code 2026-07-07.*

## The mechanism

Every MCTS variant's `_generate_candidates` builds the next-step
generation prompt from an accumulated `current_text` that may already
end in the `"\n\n"` step separator (the vLLM stop string, kept via
`stop=["\n\n"], include_stop_str_in_output=True`). The guard:

```python
current_text_clean = current_text.removesuffix("\n\n")
current_convs = [build_conv(question, current_text_clean, system_prompt)]
current_templated = tokenizer.apply_chat_template(
    current_convs,
    add_generation_prompt=(depth == 0),
    continue_final_message=(depth > 0),
    ...,
    tokenize=False,
)
if current_text.endswith("\n\n"):
    current_templated = [t + "\n\n" for t in current_templated]
```

Strip the separator **before** `apply_chat_template`, then re-append it
**after**, on the templated string. This makes the separator's survival
independent of which chat template or transformers version is in play.

## Why the guard exists at all

Some chat templates — and some transformers versions, independent of
template — trim or otherwise mishandle a trailing `"\n\n"` inside
`apply_chat_template`. Concretely, Llama's *native* chat template
silently trims it. Without the guard, the model's prompt ends up
looking like a finished message rather than one awaiting a continued
step, so the model emits EOS immediately instead of continuing —
collapsing the search tree to 1-step stubs. This is the same failure
class as
[findings/coding-findings/library-version-trajectory-completeness.md](../findings/coding-findings/library-version-trajectory-completeness.md).
`apply_chat_template` is the one place the separator is lost;
re-appending it afterward on the templated string is correct by
construction, since nothing downstream of that point touches the
separator again.

## Current coverage (verified 2026-07-07)

`removesuffix("\n\n")` on the templating path is present in all five
current MCTS search cores:

| file | line |
|---|---|
| `core/mcts_cnt_search_v01_00_00.py` | 263 |
| `core/mcts_bl_cnt_search_v01_00_00.py` | 320 |
| `core/mcts_bl_kube_search_v01_00_00.py` (renamed 2026-07-16 from `mcts_bl_cnt_search_v02_00_00.py` — see [bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md)) | 403 |
| `core/mcts_sem_search_v01_00_00.py` | 526 |
| `core/mcts_sem_search_v02_00_00.py` | 725 |

cnt-mcts and both BL-MCTS variants had the guard from early on; it
originated alongside the 2026-06-13 native-chat-template migration
(`mcts_cnt_search_v05_00_00` first, per that entry). Sem-mcts (v01 and
v02) did **not** get it until 2026-07-06 — their `_generate_candidates`
had been templating `current_text` directly, with `removesuffix`
applied only to the separate embed/score copy of a candidate, never to
the actual generation prompt. This went unnoticed in practice because
the 2026-06-19 per-family template default keeps Llama on the custom,
whitespace-preserving template — so the code path that would trigger
the bug (Llama + native template) had zero recorded runs before the fix
landed. Verified via smoke test (Llama3.2-1B + native template +
sem-v02): before the fix, 0/26 nodes reached a final answer (77% were
1-step stubs); after, 32/39 (82%) reached a final answer, 2.6% stubs —
in line with the healthy controls.

**Applied in place for sem-mcts, no method-string bump** — normally a
core-file behavior change needs a new `search.method` label so old and
new code don't collide on the same result-dir hash, but since every
recorded sem run used a separator-preserving template combination
(Llama+custom or Qwen+native), the fix reproduces byte-identical
prompts at every existing hash. There was no prior data at the one
combination (Llama+native) this fix actually changes behavior for.

Sem-mcts's `_generate_candidates` additionally strips `"\n\n"` from
candidate texts a second time before PRM scoring/embedding
(`candidate_texts.append(cand_text.removesuffix("\n\n"))`,
`core/mcts_sem_search_v02_00_00.py:774`) — this is a separate,
redundant protection against the *scoring-side* trailing-separator bug
(see
[findings/coding-findings/prm-step-split-trailing-separator.md](../findings/coding-findings/prm-step-split-trailing-separator.md)),
belt-and-suspenders alongside `PRM._split_steps`, not a substitute for
the generation-prompt guard described here.

## Revisit if

A new MCTS search core is added by cloning an existing
`_generate_candidates` — check for this exact strip-and-reappend block
by name; the 2026-07-06 gap (sem-mcts missing it for weeks before
anyone noticed, only surfacing because no Llama+native runs existed
yet) shows this is easy to silently omit when copying the surrounding
structure without carrying every line.
