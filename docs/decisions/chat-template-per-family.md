# Chat template is per-model-family: Llama uses custom, Qwen uses native

*Decided 2026-06-13, refined 2026-06-19 —
[decisions-log.md #2026-06-13](../decisions-log.md#2026-06-13--prompting-use-native-chat-templates-not-one-custom-template),
[#2026-06-19](../decisions-log.md#2026-06-19--models-configs-chat-template-default-lives-on-llmconfig-set-per-model-family)*

## Current state

`LLMConfig.use_custom_template` (`utils/configs.py`) defaults `True`
(custom template). No `conf/llm/llama_*.yaml` group overrides it, so
**Llama models use the vendored custom template**. Every
`conf/llm/qwen_*.yaml` group (`qwen_3b`, `qwen_3b_gptq_int4`,
`qwen_7b_gptq_int4`, `qwen_math_1_5b`, `qwen_math_7b`) sets it `False`,
so **Qwen models use their own native chat template**. A CLI override
(`llm.use_custom_template=...`) wins over the YAML default. All
template-selection read sites (`mcts_cnt`, `mcts_sem` v01/v02, `bon`,
`mcts_bl_cnt`) and `config_name`'s `--tmpl-` dir tag read
`cfg.llm.use_custom_template`.

The field lives on `LLMConfig` (the dataclass that already describes
the model) rather than as a resolved/computed value, because it's
static per-model-family state, set once per YAML group — not something
that needs runtime logic to determine.

## Why this split, not "native for everyone" or "custom for everyone"

Both single-template extremes create the same kind of problem: forcing
one model family's chat format onto a different family it wasn't
trained on.

- **A single hardcoded custom template for every model** (the original
  state) was added because Llama's *native* template silently trims
  the trailing `\n\n` step separator between reasoning steps — the
  custom template's real job was preserving that separator. But
  applying it to Qwen forced Llama-3.1 format onto a model trained on
  its own `<|im_start|>`-style format, producing malformed,
  non-terminating output (stray foreign special tokens leaking into
  the completion).
- **Switching everyone to native** (a considered alternative) removes
  the Qwen confound but reintroduces the original Llama problem: native
  Llama templates drop the step separator, and text generation
  collapses into a single stub instead of a multi-step trajectory
  (same failure class as
  [findings/coding-findings/library-version-trajectory-completeness.md](../findings/coding-findings/library-version-trajectory-completeness.md)).

The actual fix for the separator-loss problem is a strip-and-reappend
guard (`removesuffix("\n\n")` before `apply_chat_template`, re-append
after) — once that guard exists, Llama no longer *needs* the custom
template to keep the separator, so each family can use its own native,
in-distribution format without reintroducing either confound. Llama
happens to still default to `custom` (a config default, not a
requirement) rather than being flipped to `native` — the strip-and-
reappend guard makes native technically viable for Llama too, but the
per-family default wasn't changed opportunistically once the
regression (Qwen-on-custom) was fixed.

`custom_chat_template` stays in the config as a vendored asset even
though it's no longer applied universally.

## Why a plain per-family config field, not a resolver function

A first fix attempt added a `resolve_use_custom_template(cfg)` helper
to compute the right default per family at call time. Rejected: only a
handful of Qwen configs exist, and the value only needs setting once
per YAML group — a resolver function computes something that's really
just static data, adding an indirection layer for no benefit over
setting the field directly in each `conf/llm/*.yaml`.

## Revisit if

The per-family default needs to depend on more than just "which YAML
group is loaded" (e.g. varying by dataset or task) — at that point a
real resolver would earn its complexity. Also revisit if the
backlogged M2 template A/B (`llm-prm-deep-dive`) shows native is worse
than custom for some model even with the separator guard in place, or
if a model's native template can't be made to preserve the separator
even with strip-and-reappend.
