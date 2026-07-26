# qwen-math caps at `max_model_len=4096`: a global mml override raises for it while every other model accepts it

*2026-07-24 — finding*

`llm.max_model_len` is not uniformly overridable across the model
grid. The two Qwen2.5-Math configs are pinned at 4096 and *cannot*
go higher — vLLM refuses at startup — while the other six models
accept anything up to at least 6000. So a single global override
applied across a sweep succeeds for most cells and hard-fails for
exactly the qwen-math ones.

## The asymmetry

| config | pinned `max_model_len` | can it go higher? |
|---|---|---|
| `llama_1b`, `llama_3b`, `llama_3b_gptq` | 5000 | yes |
| `qwen_3b`, `qwen_3b_gptq_int4`, `qwen_7b_gptq_int4` | 5000 | yes |
| **`qwen_math_1_5b`, `qwen_math_7b`** | **4096** | **no** |

The 4096 pin is not a project convention — it is the model's
architectural ceiling. Qwen2.5-Math ships with
`max_position_embeddings=4096`, and vLLM derives its maximum from
that. The 5000 used elsewhere *is* a project convention, chosen
well below those models' real ceilings (all six accepted
`max_model_len=6000` without complaint on 2026-07-24, so their
derived maximum is at least 6000).

## What happens when you over-request

A too-large value is a **hard startup failure, not a downgrade**.
Asking qwen-math for 6000:

```
ValueError: User-specified max_model_len (6000) is greater than
the derived max_model_len (max_position_embeddings=4096.0 or
model_max_length=None in model's config.json). To allow
overriding this maximum, set the env var
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
```

It is raised inside `LLM(...)` construction — before any
generation — so the process dies during model init. The same
applies to `5000`: any value above 4096 raises. From
`vllm/config/model.py` (v0.18.x, lines ~2059-2083):

```python
elif max_model_len > derived_max_model_len:
    if model_max_length is None or max_model_len > model_max_length:
        msg = (...)
        if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
            logger.warning_once("%s %s", msg, warning)
        else:
            raise ValueError(...)   # <- default path
```

Note there is **no clamping branch**: vLLM never quietly reduces
your request to 4096. The only fall-back-to-derived path is when
`max_model_len` is left unset (`None`), where line ~2054 does
`max_model_len = int(derived_max_model_len)`. Our Hydra configs
always set it explicitly, so that path never applies here.

**Why the other models don't error:** their pinned 5000 is far
below their own derived maximum, so the `>` comparison never
trips — and raising them to 6000 still doesn't trip it.

## The trap this creates

Applying one override across a model sweep, e.g.

```
llm.max_model_len=6000
```

silently does the right thing for six cells and kills the two
qwen-math cells at init. The failure is easy to misread as a
scheduling/GPU problem because it looks like an immediately dead
process with no trial output — this is exactly how it surfaced on
2026-07-24, where three relaunches died instantly and all three
happened to be qwen-math.

If a sweep needs a larger window, qwen-math cannot join it at all:
there is no value that is both >5000 and ≤4096. The cell has to be
dropped and labelled, not retried.

## The escape hatch is unusable for research numbers

`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` downgrades the raise to a
warning, but vLLM's own message states that for RoPE models (all
Qwen and Llama here) positions past the trained range "lead to
nan", and for absolute-position models cause a CUDA array
out-of-bounds. That trades a loud crash for silent numerical
corruption, so it must not be used to force qwen-math above 4096.

## Consequence: qwen-math is never context-matched to its row-mates

Because the pin has always been 4096, qwen-math has run with a
smaller window than every other model in every table — verified
across `mcts_bl_cnt_v01` (L4, L5), `mcts_bl_kdepth_v01` (L4, L5),
and the b=320 sem cells, all recording `max_model_len=4096`.

So in any "model family / size / quantization comparison" that
includes qwen-math, the context window is a **second uncontrolled
variable** alongside model identity: 4096 vs 5000 of room. This is
longstanding and affects already-scored tables. It only bites on
problems whose prompts approach the cap, so for most rows it is
harmless — but a weak qwen-math row on long-chain problems has
this confound available and should not be read as a pure
model-quality result without checking prompt lengths.

## Open experiment: can qwen-math run level-5 b=320 at its own 4096?

**Worth trying; currently unresolved.** The 2026-07-24 level-5
b=320 batch excluded qwen-math because it cannot accept the
`max_model_len=6000` those cells use. But "qwen-math cannot do
level-5 b=320 *at all*" is a stronger claim, and it is **not
established**:

- It *was* launched at its own 4096 window (`cfg-d87ee48f`, W&B
  `08m3c7r9`, 2026-07-24 05:42). The model loaded, entered trial
  0, and ran **~1 h 25 min with no context error** before the
  process was killed by an unrelated external sweep at 07:09.
  W&B recorded it `crashed` (heartbeat lost), not `failed` (no
  non-zero exit) — i.e. terminated, not overflowed. That result
  dir was deleted 2026-07-24 as an empty stub; the evidence lives
  in the W&B run.
- The only *observed* level-5 b=320 overflow is a different cell:
  cnt llama-3b at mml=5000 (W&B `qyve9h2t`, `failed` after ~23
  min). Generalizing from it to qwen-math is inference — and
  qwen-math's completions are not the longest in the grid (per
  [../../decisions/context-length-overflow-guard.md](../../decisions/context-length-overflow-guard.md),
  cap-grazing is a Llama-family behavior; qwen-math never came
  within 500 tokens of its 4096 cap at b=80).
- Prior in favor: qwen-math **completed** level-4 b=320 at
  mml=4096, 2/2 trials, twice (`cfg-799bfbc6`, `cfg-c67e46ee`).

**The test:** one run of the level-5 b=320 cell with
`llm=qwen_math_1_5b llm.max_model_len=4096`. If trial 0 completes,
qwen-math belongs in the level-5 b=320 tables at its own window
rather than being marked `n/a (4096 ctx)` — with the window
difference noted, exactly as the level-4 b=320 tables already do.
If it overflows, the exclusion becomes measured rather than
assumed, and the `n/a` label is justified.

Either outcome is worth the ~14 h: it decides whether three
level-5 b=320 tables read 4/5 models or 5/5.

## How to handle it

- **Don't** set `llm.max_model_len` globally in a multi-model
  sweep without checking the qwen-math cells separately.
- **Do** leave the per-model pins alone as the default: 4096 for
  qwen-math is correct and load-bearing.
- If a study genuinely needs >4096 of context, qwen-math is out of
  scope for it; mark the cell `n/a (4096 ctx)` rather than
  `planned` or `failed`, so it doesn't read as unfinished work.
- When comparing qwen-math against 5000-window models on
  long-prompt problems, note the window difference in the table's
  limitations.
