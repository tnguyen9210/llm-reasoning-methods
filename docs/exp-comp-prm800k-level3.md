# LLM Reasoning — MCTS Experiment Comparison — PRM800K Level 3

> **Provenance:** created 2026-07-28 by mirroring the single
> `cnt-mcts` model-family table from
> [exp-comp-prm800k-level5.md](exp-comp-prm800k-level5.md)
> (`tbl-afdda0`) at `data.level=3`. All five cells were queued the
> same day at priority 2 (`orchestration/ledgers/prm800k-level3.yaml`)
> — **no level-3 results exist yet**; the rows read `inqueue`
> until a GPU picks them up.
>
> **Grid:** the 5-model level-5 grid (llama-1b, llama-3b fp16,
> qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b), not level-4's
> 7 — llama-3b gptq and qwen-3b gptq-int4 are dropped, same as at
> level 5.
>
> **Sample size:** level 3 is **105 problems** (vs 128 at level 4
> and 134 at level 5), so standard errors here are ~13 % wider
> than level 5's at the same accuracy. Read cross-level
> differences with that in mind.

Tracker for MCTS search experiments on PRM800K **level 3**.
Scope is deliberately narrow: one table, the cnt-mcts model
family comparison under the Qwen PRM, so the level-3 point can
be read against its level-4 and level-5 counterparts
(`tbl-afdda0` there) without importing the rest of those docs'
grids.




<!-- toc:begin -- generated, do not hand-edit -->
## Contents

- [**Purpose**](#purpose)
- [**Structure and use**](#structure-and-use)
- [**Tuning tables \[gen_budget=80\]**](#tuning-tables-gen_budget80)
  - [cnt-mcts](#cnt-mcts)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm) · `tbl-01884a`
  - [sem-mcts-v02](#sem-mcts-v02)
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha1) · `tbl-8d2a3d`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha10) · `tbl-49eedf`

*3 tables. Regenerate with `python scripts/gen_toc.py`.*
<!-- toc:end -->

## Purpose

One question: **how does the model-family ranking at
`gen_budget=80` change when the problems get easier?** The same
five models, the same PRM, the same search — only `data.level`
moves. Levels 4 and 5 are already measured; level 3 is the
missing low-difficulty anchor.

## Structure and use

The `status` column is DERIVED from the ledger — never hand-edit
it; `python orchestration/status.py --sync-doc prm800k-level3
--apply` settles it. Numbers come from `compute_stats.py` only
(format `.NNNN<br>±.NNNN`, 4 dp); `hr/trial` comes from the
result dir's `timing_state.json`, never from `compute_stats`.
Empty cells are `—`. Full workflow:
[decisions/experiment-workflow-v2.md](decisions/experiment-workflow-v2.md).

## Tuning tables [gen_budget=80]

### cnt-mcts

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-01884a -->
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama),
> **data.level=3**, run.num_trials=2. Default context window (no
> `llm.max_model_len` override): the overflow that forced
> mml6000 elsewhere was observed at level 5 and b=320
> ([decisions/context-length-overflow-guard.md](decisions/context-length-overflow-guard.md)),
> and neither condition applies here.
>
> **Launch:** the level-5 counterpart's command plus
> `data.level=3` — config hashes and `--level-3--` run names
> follow automatically.
>
> ⚠️ `expected_hr` was set at 8–12 (2× the level-5 twin's
> hr/trial, rounded up). Measured: **1.7–2.9 hr/trial**, so the
> estimates were 3–6× too generous. Level 3's 105 questions
> and shorter searches make it much cheaper than level 5.
>
> **W&B:** 8dzw7eyr (llama-1b), s8z756dr (llama-3b), 7hsq6d3w
> (qwen-3b), sgtyrdii (qwen-7b), oy0iw2ph (qwen-math-1.5b).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .8000<br>±.0277 | .7048<br>±.0316 | .6810<br>±.0322 | .6381<br>±.0332 | 1.66 |
| llama-3b fp16 | 2 | scored | .9286<br>±.0178 | .8571<br>±.0242 | .8286<br>±.0261 | .7952<br>±.0279 | 2.89 |
| qwen-3b fp16 | 2 | scored | .9619<br>±.0132 | .9381<br>±.0167 | .9381<br>±.0167 | .9286<br>±.0178 | 2.54 |
| qwen-7b gptq-int4 | 2 | scored | .9857<br>±.0082 | .9429<br>±.0161 | .9190<br>±.0189 | .9048<br>±.0203 | 1.94 |
| qwen-math-1.5b fp16 | 2 | scored | .9857<br>±.0082 | .9476<br>±.0154 | .9381<br>±.0167 | .9381<br>±.0167 | 1.99 |

> **Analysis.** No level-3 data yet — nothing to take away.
> **Limitations / follow-up:** the comparison this table exists
> for is cross-level, so it only becomes readable once all five
> cells are scored; a partially filled column invites reading a
> model-family difference that is really a
> which-cells-finished difference. For reference, the level-5
> counterpart runs 2.98–5.13 hr/trial across these five models,
> and level 3 should be faster per problem (shorter solutions)
> on 105 rather than 134 problems.

### sem-mcts-v02

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)
<!-- table-id: tbl-8d2a3d -->
> **Compares:** model family, size, and quantization jointly,
> scored with `prm=qwen` (Qwen-Math-7B-PRM), at `lam=0.01,
> ds_alpha=1.0` (`w_eff = ds_alpha/sqrt(lam) = 10`). The level-3
> counterpart of level 5's `tbl-73533c`, so the two read directly
> against each other.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=80, tmpl=model-family default (native for Qwen, custom
> for Llama), `embeds_proj=sparse512`, `cov_update=sm`, lam=0.01,
> ds_alpha=1.0 (w_eff=10), ds_beta=1.0, prm_batch_size=1,
> **data.level=3**, run.num_trials=2.
>
> **Launch:** the level-5 counterpart's command plus
> `data.level=3`.
>
> ⚠️ `expected_hr` was set at 10–14 (2× the level-5 twin's
> hr/trial). Measured: **2.6–3.7 hr/trial**, so the estimates
> were 3–5× too generous.
>
> **W&B:** aockwewn (llama-1b), ro7g3y5n (llama-3b), ye7ukzo8
> (qwen-3b), sqg3c8s5 (qwen-7b), bxfl2j6f (qwen-math-1.5b).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .7810<br>±.0286 | .7143<br>±.0312 | .6619<br>±.0327 | .6143<br>±.0337 | 2.85 |
| llama-3b fp16 | 2 | scored | .9048<br>±.0203 | .8714<br>±.0232 | .8286<br>±.0261 | .8143<br>±.0269 | 3.66 |
| qwen-3b fp16 | 2 | scored | .9619<br>±.0132 | .9286<br>±.0178 | .9286<br>±.0178 | .8952<br>±.0212 | 3.42 |
| qwen-7b gptq-int4 | 2 | scored | .9762<br>±.0105 | .9238<br>±.0184 | .9190<br>±.0189 | .9095<br>±.0198 | 2.64 |
| qwen-math-1.5b fp16 | 2 | scored | .9810<br>±.0095 | .9381<br>±.0167 | .9333<br>±.0173 | .9286<br>±.0178 | 2.83 |

> **Analysis.** Level 3 compresses the model-family spread that
> level 5 shows clearly. pass@gb runs .7810 → .9810 here against
> level 5's .3209 → .7687, so the weakest model (llama-1b) gains
> far more from the easier split than the strongest. qwen-math-
> 1.5b edges out qwen-7b gptq-int4 on every metric despite being
> ~4× smaller, matching the level-5 ordering on wei/maj.
> **Limitations / follow-up:** level 5 has this table complete
> (llama-1b .3209, llama-3b .5784, qwen-3b .6903, qwen-7b
> gptq-int4 .7687, qwen-math-1.5b .7500 pass@gb), so the level-3
> read is a clean difficulty contrast once filled.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-49eedf -->
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`). The
> level-3 counterpart of level 5's `tbl-cf8fea`.
>
> **Fixed:** identical to the `ds_alpha=1` table above except
> **ds_alpha=10** (w_eff=100).
>
> **Launch:** the level-5 counterpart's command plus
> `data.level=3`.
>
> ⚠️ `expected_hr` was set at 10–14 (2× the level-5 twin's
> hr/trial). Measured: **2.7–3.7 hr/trial**, so the estimates
> were 3–5× too generous.
>
> ⚠️ **qwen-math-1.5b is `failed`, not missing.** Its allocation
> (23446851, gpu_windfall) was preempted after trial 1 of 2 and
> re-queued; the run is resumable — the result dir is
> hash-addressed, so a requeue skips the finished trial.
>
> **W&B:** it83yx2s (llama-1b), kvlybe2r (llama-3b), 81eni7sd
> (qwen-3b), evzz4tch (qwen-7b). qwen-math-1.5b: none (failed).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .8000<br>±.0277 | .7143<br>±.0312 | .6714<br>±.0325 | .5857<br>±.0341 | 2.84 |
| llama-3b fp16 | 2 | scored | .9095<br>±.0198 | .8524<br>±.0245 | .8190<br>±.0266 | .8048<br>±.0274 | 3.74 |
| qwen-3b fp16 | 2 | scored | .9619<br>±.0132 | .9048<br>±.0203 | .9143<br>±.0194 | .9000<br>±.0208 | 3.35 |
| qwen-7b gptq-int4 | 2 | scored | .9810<br>±.0095 | .9286<br>±.0178 | .9000<br>±.0208 | .8905<br>±.0216 | 2.68 |
| qwen-math-1.5b fp16 | — | failed | — | — | — | — | — |

> **Analysis.** Four of five cells scored. Against the
> `ds_alpha=1` table (`tbl-8d2a3d`), raising `w_eff` from 10 to
> 100 moves pass@gb by at most .0190 on any model and the
> direction is not consistent (llama-1b +.0190, qwen-7b +.0048,
> qwen-3b .0000, llama-3b +.0047) — every gap is inside one
> standard error, so at level 3 this knob does nothing
> measurable. maj@gb on llama-1b is the one visible drop
> (.6143 → .5857), also within error.
> **Limitations / follow-up:** the level-5 counterpart carries a
> ⚠️ mismatch flag — recomputing its cited llama-1b, qwen-7b
> gptq-int4 and qwen-math-1.5b baselines produced different
> numbers than the table shows, and it was not overwritten. That
> flag is about level-5 bookkeeping, not this config; level-3
> cells here are computed fresh and are unaffected.
