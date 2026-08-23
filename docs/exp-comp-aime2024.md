# Experiment comparisons — AIME 2024

Tracking doc for search-method experiments on **AIME 2024**
(`data=aime2024`: 30 problems, `grader_name=aime`, no level
split). Sibling of `exp-comp-aime2025.md`; same structure,
conventions, and ledger workflow (workflow v2 —
`orchestration/ledgers/aime2024.yaml`).

## Contents

- [Tuning tables [gen_budget=80]](#tuning-tables-gen_budget80)
  - [cnt-mcts](#cnt-mcts)
    - [cpuct sweep (llama-1b)](#cpuct-sweep-llama-1b) · `tbl-6c1b54`
    - [cpuct sweep (llama-3b)](#cpuct-sweep-llama-3b) · `tbl-3544e1`
    - [cpuct sweep (qwen-3b)](#cpuct-sweep-qwen-3b) · `tbl-e35b67`
    - [cpuct sweep (qwen-7b gptq-int4)](#cpuct-sweep-qwen-7b-gptq-int4) · `tbl-323b0e`
    - [cpuct sweep (qwen-math-1.5b)](#cpuct-sweep-qwen-math-15b) · `tbl-7f1b6a`

## Purpose

Replicate the two informative endpoints of the prm800k-level-5
cpuct bracket — the interior reference `cpuct=2.0` and the
pure-exploitation limit `∞` (constructed as `q_beta=0`) — on a
second competition-style dataset, per model. The level-5 sweep
(`tbl-e0b779`/`tbl-ed4e96`/`tbl-c93854`/`tbl-0f6c1a`/
`tbl-57b084`) found a flat, non-monotone interior with the
`q_beta=0` arm at or near the top on most models; these tables
test whether that read transfers to AIME 2024.

## Structure and use

Same conventions as `exp-comp-aime2025.md`: `####` tables under
budget/method headings, scored cells `.NNNN<br>±.NNNN` (4 dp),
status cells derived from the ledger via
`orchestration/status.py --sync-doc aime2024`. AIME docs fix
**`run.num_trials=4`** (30 questions per trial → per-cell
quantization is 1/30 ≈ .0333; SEMs are large — read tables at
the ~2-SEM level only).

## Tuning tables [gen_budget=80]

### cnt-mcts

#### cpuct sweep (llama-1b)
<!-- table-id: tbl-6c1b54 -->
> **Compares:** the {`2.0`, `∞`} endpoints of the level-5 cpuct
> bracket on AIME 2024. `cpuct=2.0` is the file-default interior
> reference (`q_beta=1.0`); `∞` is the pure-exploitation limit,
> constructed as `cpuct=2.0, q_beta=0` exactly as in the level-5
> sweep (see `tbl-e0b779`'s preamble for the construction and
> the sweep-wide `tie_tol=1e-4` tie band, kept here for
> comparability). The interior `{0, 1.0, 10}` arms are not
> replicated — level 5 found the interior flat and the `0` arm
> budget-confounded.
>
> ⚠️ **n=30**: one AIME trial is 30 questions, so cell values
> quantize at ≈ .0333 and 4-trial SEMs stay wide; treat gaps
> under ~2 SEM as noise.
>
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, bs-4, d-20, b=80, prm_batch_size=1, tie_tol=1e-4,
> tmpl=model-family default, llm=llama-1b, data=aime2024,
> **run.num_trials=4**.

| llm | prm | cpuct | q_beta | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 2.0 | 1.0 | — | running | — | — | — | — | — |
| llama-1b | qwen | 2.0 | 0 | — | running | — | — | — | — | — |

> **Analysis.** No AIME2024 data yet — nothing to take away.
> **Limitations / follow-up:** entire table queued 2026-08-22
> (priority 2); launch is
> the level-5 sweep counterpart's command with `data=aime2024`
> and `run.num_trials=4`. Feeds key: `aime2024-cnt-cpuct-llama1b`.

#### cpuct sweep (llama-3b)
<!-- table-id: tbl-3544e1 -->
> **Compares:** same {`2.0`, `∞`} endpoints as the llama-1b
> table above — see its preamble for the construction, the
> tie band, and the n=30 ⚠.
>
> **Fixed:** as llama-1b table, llm=llama-3b.

| llm | prm | cpuct | q_beta | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 2.0 | 1.0 | — | inqueue | — | — | — | — | — |
| llama-3b | qwen | 2.0 | 0 | — | inqueue | — | — | — | — | — |

> **Analysis.** No AIME2024 data yet — nothing to take away.
> **Limitations / follow-up:** entire table queued 2026-08-22
> (priority 2); launch is
> the level-5 sweep counterpart's command with `data=aime2024`
> and `run.num_trials=4`. Feeds key: `aime2024-cnt-cpuct-llama3b`.

#### cpuct sweep (qwen-3b)
<!-- table-id: tbl-e35b67 -->
> **Compares:** same {`2.0`, `∞`} endpoints as the llama-1b
> table above — see its preamble for the construction, the
> tie band, and the n=30 ⚠. On level 5 this model's `q_beta=0`
> arm was its best pass@gb cell (`.7276`, `tbl-c93854`).
>
> **Fixed:** as llama-1b table, llm=qwen-3b.

| llm | prm | cpuct | q_beta | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | 2.0 | 1.0 | — | inqueue | — | — | — | — | — |
| qwen-3b | qwen | 2.0 | 0 | — | inqueue | — | — | — | — | — |

> **Analysis.** No AIME2024 data yet — nothing to take away.
> **Limitations / follow-up:** entire table queued 2026-08-22
> (priority 2); launch is
> the level-5 sweep counterpart's command with `data=aime2024`
> and `run.num_trials=4`. Feeds key: `aime2024-cnt-cpuct-qwen3b`.

#### cpuct sweep (qwen-7b gptq-int4)
<!-- table-id: tbl-323b0e -->
> **Compares:** same {`2.0`, `∞`} endpoints as the llama-1b
> table above — see its preamble for the construction, the
> tie band, and the n=30 ⚠.
>
> **Fixed:** as llama-1b table, llm=qwen-7b gptq-int4.

| llm | prm | cpuct | q_beta | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 2.0 | 1.0 | — | inqueue | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 2.0 | 0 | — | inqueue | — | — | — | — | — |

> **Analysis.** No AIME2024 data yet — nothing to take away.
> **Limitations / follow-up:** entire table queued 2026-08-22
> (priority 2); launch is
> the level-5 sweep counterpart's command with `data=aime2024`
> and `run.num_trials=4`. Feeds key: `aime2024-cnt-cpuct-qwen7b`.

#### cpuct sweep (qwen-math-1.5b)
<!-- table-id: tbl-7f1b6a -->
> **Compares:** same {`2.0`, `∞`} endpoints as the llama-1b
> table above — see its preamble for the construction, the
> tie band, and the n=30 ⚠.
>
> **Fixed:** as llama-1b table, llm=qwen-math-1.5b.

| llm | prm | cpuct | q_beta | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 2.0 | 1.0 | — | inqueue | — | — | — | — | — |
| qwen-math-1.5b | qwen | 2.0 | 0 | — | inqueue | — | — | — | — | — |

> **Analysis.** No AIME2024 data yet — nothing to take away.
> **Limitations / follow-up:** entire table queued 2026-08-22
> (priority 2); launch is
> the level-5 sweep counterpart's command with `data=aime2024`
> and `run.num_trials=4`. Feeds key: `aime2024-cnt-cpuct-qwenmath`.
