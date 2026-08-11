# LLM Reasoning — MCTS Experiment Comparison

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on PRM800K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.

<!-- toc:begin -- generated, do not hand-edit -->
## Contents

- [**Purpose**](#purpose)
- [**Structure (why it's shaped this way)**](#structure-why-its-shaped-this-way)
- [**How to use**](#how-to-use)
- [**Cross-algorithm summary (qwen PRM)**](#cross-algorithm-summary-qwen-prm)
- [**Algorithm name ↔ code mapping**](#algorithm-name-code-mapping)
- [**Summary — results per (algorithm, model, budget)**](#summary-results-per-algorithm-model-budget)
- [**Tuning tables \[gen_budget=80\]**](#tuning-tables-gen_budget80)
  - [cnt-mcts](#cnt-mcts)
    - [custom vs native template comparison](#custom-vs-native-template-comparison) · `tbl-c1962a`
    - [prm_batch_size sweep](#prm_batch_size-sweep) · `tbl-0642eb`
    - [rlhflow vs qwen PRM comparison](#rlhflow-vs-qwen-prm-comparison) · `tbl-ab03de`
    - [enforce_eager comparison](#enforce_eager-comparison) · `tbl-bfab79`
    - [model family, size, quantization comparison](#model-family-size-quantization-comparison) · `tbl-8ca223`
  - [cnt-mcts (updated)](#cnt-mcts-updated)
    - [custom vs native template comparison](#custom-vs-native-template-comparison-1) · `tbl-55d130`
    - [prm_batch_size sweep](#prm_batch_size-sweep-1) · `tbl-b5bc59`
    - [rlhflow vs qwen PRM comparison](#rlhflow-vs-qwen-prm-comparison-1) · `tbl-ef6f98`
    - [enforce_eager comparison](#enforce_eager-comparison-1) · `tbl-adf2f8`
    - [model family, size, quantization comparison](#model-family-size-quantization-comparison-1) · `tbl-702925`
    - [model family, size, quantization comparison (qwen PRM)](#model-family-size-quantization-comparison-qwen-prm) · `tbl-6fe5a2`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b)](#agg_strategy-comparison-qwen-3b-qwen-math-15b) · `tbl-3ea294`
  - [sem-mcts-v02](#sem-mcts-v02)
    - [embeds_proj × cov_update sweep (v02)](#embeds_proj-cov_update-sweep-v02) · `tbl-860167`
    - [embeds_strategy × scope sweep (v02, qwen PRM)](#embeds_strategy-scope-sweep-v02-qwen-prm) · `tbl-82c90f`
    - [ds_alpha sweep (v02)](#ds_alpha-sweep-v02) · `tbl-1c9c2c`
    - [ds_alpha sweep (v02, qwen PRM)](#ds_alpha-sweep-v02-qwen-prm) · `tbl-93c239`
    - [lam / ds_alpha joint sweep (v02, llama-1b, step 1 done)](#lam-ds_alpha-joint-sweep-v02-llama-1b-step-1-done) · `tbl-0efc55`
    - [lam / ds_alpha joint sweep (v02, llama-3b, step 1 done)](#lam-ds_alpha-joint-sweep-v02-llama-3b-step-1-done) · `tbl-f50e22`
    - [lam / ds_alpha joint sweep (v02, qwen-math-1.5b)](#lam-ds_alpha-joint-sweep-v02-qwen-math-15b) · `tbl-7491b1`
    - [model family, size, quantization comparison](#model-family-size-quantization-comparison-2) · `tbl-0c4ffd`
    - [model family, size, quantization comparison (qwen PRM)](#model-family-size-quantization-comparison-qwen-prm-1) · `tbl-352d94`
    - [rlhflow vs qwen PRM comparison](#rlhflow-vs-qwen-prm-comparison-2) · `tbl-b4c266`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-1) · `tbl-baf795`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=10)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam01-w_eff10) · `tbl-b1e565`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=100)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam01-w_eff100) · `tbl-db5810`
    - [LLM vs PRM embeds comparison](#llm-vs-prm-embeds-comparison) · `tbl-1eed5c`
  - [sem-mcts-v02 \[cov_scope=local\]](#sem-mcts-v02-cov_scopelocal)
    - [lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-1b-embeds_refrelative) · `tbl-db0cf7`
    - [lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-3b-embeds_refrelative) · `tbl-43996a`
    - [lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-3b-embeds_refrelative) · `tbl-ecabc0`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4-embeds_refrelative) · `tbl-e6a2f9`
    - [lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-math-15b-embeds_refrelative) · `tbl-c76d49`
  - [cnt-mcts-bl-v01](#cnt-mcts-bl-v01)
    - [model family, size, quantization comparison (qwen PRM)](#model-family-size-quantization-comparison-qwen-prm-2) · `tbl-deb9f9`
  - [kube-mcts-bl-v01](#kube-mcts-bl-v01)
    - [model family, size, quantization comparison (qwen PRM)](#model-family-size-quantization-comparison-qwen-prm-3) · `tbl-fbd467`
  - [kdepth-mcts-bl-v01](#kdepth-mcts-bl-v01)
    - [model family, size, quantization comparison (qwen PRM)](#model-family-size-quantization-comparison-qwen-prm-4) · `tbl-7367f8`
  - [sem-mcts-bl](#sem-mcts-bl)
    - [model family, size, quantization comparison (qwen PRM, w_eff=100)](#model-family-size-quantization-comparison-qwen-prm-w_eff100) · `tbl-ed6194`
    - [model family, size, quantization comparison (qwen PRM, w_eff=10)](#model-family-size-quantization-comparison-qwen-prm-w_eff10) · `tbl-7fec69`
- [**Tuning tables \[gen_budget=160, 320, …\] *(future)***](#tuning-tables-gen_budget160-320-future)
  - [cnt-mcts](#cnt-mcts-1)
    - [model family comparison (b=320, qwen PRM)](#model-family-comparison-b320-qwen-prm) · `tbl-4e21d6`
  - [sem-mcts-v02](#sem-mcts-v02-1)
    - [model family comparison (b=320, qwen PRM, lam=0.1, w_eff=10)](#model-family-comparison-b320-qwen-prm-lam01-w_eff10) · `tbl-e144a5`
    - [model family comparison (b=320, qwen PRM, lam=0.1, w_eff=100)](#model-family-comparison-b320-qwen-prm-lam01-w_eff100) · `tbl-179d62`
- [**Run log (newest first)**](#run-log-newest-first)
  - [2026-06-18 — cnt-mcts / llama-1b / custom / cpuct=2.0 / b=80](#2026-06-18-cnt-mcts-llama-1b-custom-cpuct20-b80)
- [**Standing comparison questions**](#standing-comparison-questions)
- [**Links & connections**](#links-connections)

*39 tables. Regenerate with `python scripts/gen_toc.py`.*
<!-- toc:end -->

## Purpose
The four algorithm tracks (`llm-reasoning-mcts-exp`,
`llm-reasoning-mcts-bl-exp`, + the `sem` variants) own
*implementation* milestones. This doc owns *experiment tracking
and comparison*: per-algorithm tuning grids and the cross-algorithm
verdict — the views none of the per-algorithm tracks give.

**This is a living log, not a milestone doc.** No `progress: N/M`;
"done" isn't a state here.

## Structure (why it's shaped this way)
Two activities, two shapes:
- **Tuning *within* an algorithm, at a given budget and
  model** → small, algorithm-specific grid of its own knobs
  (template, cpuct, embeds-strategy…). One
  `## Tuning tables [gen_budget=N]` section per budget →
  nested `### algorithm` → `##### model` (or an `llm` column
  when several model tables share one comparison), so each
  table is just the config rows for one (budget, algorithm,
  model) cell. Different algorithms show different columns —
  no forced shared schema.
- **Comparing *across* algorithms at a fixed budget** →
  sparse, one row per algorithm × model. The **Summary**
  table (your fixed columns), one row per algorithm × model
  × budget.

`gen_budget` is a top-level tuning-tables section; algorithm
and model are *subsection* levels within it, not columns —
model gets its own level (or column, for grouped comparisons)
because its GPU constraints (cf. the trial-loop OOM) and
scaling behavior are what you tune around. The
within-algorithm scaling curve (80→160→320) is read by
scanning one algorithm/model across the `gen_budget=N` tuning
sections; the cross-algorithm-per-budget cut lives in the
Summary. (Larger budgets need less tuning, so those sections
will be sparser — the nesting keeps that asymmetry clean
instead of one wide sparse grid.)

## How to use
- **Plan/run** → add a config row under the matching
  `## Tuning tables [gen_budget=N]` → `### algorithm` →
  `##### model`; log hypothesis + follow-up in the Run log.
- **A config wins at a budget** → promote it to the Summary
  as that (algorithm, model, budget)'s best config, linking
  back to the tuning row.
- **Metrics**: tuning tables carry pass@gb only (terse);
  the Summary carries the full set (pass/naive/weighted/
  maj@gb). Everything else (depth, ncomps, timing) stays in
  W&B / the result dir, linked.
- **Best-config rule:** best-scoring row *at that budget*,
  same LLM/level/trials, picked across **all tuning knobs
  jointly** (template, cpuct, …) — not "best template" with
  other knobs held fixed. Don't promote a config from a
  different budget or LLM into a Summary row — that breaks
  the comparison.

---

## Cross-algorithm summary (qwen PRM)
> One table per model, one row per algorithm — pulled directly from
> each algorithm's own "model family, size, quantization comparison
> (qwen PRM)" table above/below (`cnt-mcts (updated)`, `sem-mcts`,
> `cnt-mcts-bl-v01`, `kube-mcts-bl-v01`, `kdepth-mcts-bl-v01`,
> `sem-mcts-bl-v01`). All rows fixed at b=80, bs-4, d-20,
> agg_strategy=`last`, tmpl=model-family default (native for Qwen,
> custom for Llama), prm=qwen. `cnt-mcts` row is method=`mcts_cnt_v01`
> (the post-`_split_steps`-fix table, not the older pre-fix
> `mcts_cnt` section). `sem-mcts` row is `mcts_sem_v02` (PRM embeds),
> `ds_alpha=100` (w_eff not applicable — that knob is bl_sem-specific).
> `sem-mcts-bl-v01` row uses the `w_eff=100` table (more complete than
> `w_eff=10` at time of writing: 5/7 vs. 4/7 cells scored); see that
> algorithm's own section for the `w_eff=10` comparison point.
> `kube-mcts-bl-v01` (Fractional KUBE) and `kdepth-mcts-bl-v01`
> (depth-shaping) are each filled at 5 of 7 models as of 2026-07-09
> (see `docs/decisions/bl-kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/bl-kdepth-knapsack-bonus.md` for the
> algorithms); llama-3b gptq and qwen-3b gptq-int4 still unqueued for
> both.

**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .6367<br>±.0301 | .5352<br>±.0312 | .4961<br>±.0313 | .4531<br>±.0312 | 2.38 |
| sem-mcts | 2 | scored | .6133<br>±.0305 | .4961<br>±.0313 | .4492<br>±.0311 | .3906<br>±.0306 | 3.90 |
| cnt-mcts-bl-v01 | 2 | scored | .4414<br>±.0311 | .4297<br>±.0310 | .3984<br>±.0307 | .3789<br>±.0304 | 2.12 |
| kube-mcts-bl-v01 | 2 | scored | .5586<br>±.0311 | .5117<br>±.0313 | .4688<br>±.0312 | .4531<br>±.0312 | 2.30 |
| kdepth-mcts-bl-v01 | 2 | scored | .5742<br>±.0310 | .5430<br>±.0312 | .5117<br>±.0313 | .4883<br>±.0313 | 2.21 |
| sem-mcts-bl-v01 | 2 | scored | .5195<br>±.0313 | .4219<br>±.0309 | .3242<br>±.0293 | .2422<br>±.0268 | 5.18 |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .7656<br>±.0265 | .6758<br>±.0293 | .6523<br>±.0298 | .6445<br>±.0300 | 4.02 |
| sem-mcts | 2 | scored | .7656<br>±.0265 | .6562<br>±.0297 | .6289<br>±.0303 | .6016<br>±.0307 | 5.43 |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | 2 | scored | .7305<br>±.0278 | .6602<br>±.0297 | .6367<br>±.0301 | .6211<br>±.0304 | 3.55 |
| kdepth-mcts-bl-v01 | 2 | scored | .7227<br>±.0280 | .6680<br>±.0295 | .6758<br>±.0293 | .6445<br>±.0300 | 3.25 |
| sem-mcts-bl-v01 | — | failed | — | — | — | — | — |

<!-- TEMPORARILY HIDDEN (2026-07-21): llama-3b gptq block. Restore
     by uncommenting. bl rows still unrun; cnt/sem-mcts scored.
**llama-3b gptq**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .7148<br>±.0283 | .6055<br>±.0306 | .5781<br>±.0309 | .5625<br>±.0311 | 2.85 |
| sem-mcts | 2 | scored | .7148<br>±.0283 | .6094<br>±.0306 | .5625<br>±.0311 | .5078<br>±.0313 | 4.45 |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |
-->

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7695<br>±.0264 | .7617<br>±.0267 | 3.76 |
| sem-mcts | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7734<br>±.0262 | .7461<br>±.0273 | .7227<br>±.0280 | 5.00 |
| cnt-mcts-bl-v01 | 2 | scored | .6445<br>±.0300 | .6328<br>±.0302 | .6172<br>±.0304 | .6094<br>±.0306 | 3.50 |
| kube-mcts-bl-v01 | 2 | scored | .8320<br>±.0234 | .7617<br>±.0267 | .7422<br>±.0274 | .7344<br>±.0277 | 3.31 |
| kdepth-mcts-bl-v01 | 2 | scored | .8164<br>±.0242 | .7539<br>±.0270 | .7461<br>±.0273 | .7344<br>±.0277 | 3.00 |
| sem-mcts-bl-v01 | 2 | scored | .8320<br>±.0234 | .6836<br>±.0291 | .6484<br>±.0299 | .6016<br>±.0307 | 5.19 |

<!-- TEMPORARILY HIDDEN (2026-07-21): qwen-3b gptq-int4 block. Restore
     by uncommenting. cnt/kube/kdepth-bl rows unrun; cnt/sem-mcts and
     sem-mcts-bl-v01 scored.
**qwen-3b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .8320<br>±.0234 | .7031<br>±.0286 | .7109<br>±.0284 | .6914<br>±.0289 | 2.68 |
| sem-mcts | 2 | scored | .7930<br>±.0254 | .6953<br>±.0288 | .6953<br>±.0288 | .6875<br>±.0290 | 3.87 |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | 2 | scored | .7422<br>±.0274 | .6133<br>±.0305 | .5625<br>±.0311 | .5273<br>±.0313 | 4.18 |
-->

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .9102<br>±.0179 | .8086<br>±.0246 | .8164<br>±.0242 | .8008<br>±.0250 | 3.11 |
| sem-mcts | 2 | scored | .9375<br>±.0152 | .8164<br>±.0242 | .8086<br>±.0246 | .8047<br>±.0248 | 4.20 |
| cnt-mcts-bl-v01 | 2 | scored | .8125<br>±.0244 | .7578<br>±.0268 | .7461<br>±.0273 | .7422<br>±.0274 | 2.78 |
| kube-mcts-bl-v01 | 2 | scored | .8750<br>±.0207 | .7930<br>±.0254 | .7852<br>±.0257 | .7656<br>±.0265 | 2.58 |
| kdepth-mcts-bl-v01 | 2 | scored | .9023<br>±.0186 | .8281<br>±.0236 | .8320<br>±.0234 | .8203<br>±.0240 | 2.43 |
| sem-mcts-bl-v01 | 2 | scored | .8906<br>±.0195 | .7500<br>±.0271 | .7109<br>±.0284 | .6953<br>±.0288 | 4.15 |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .8906<br>±.0195 | .8008<br>±.0250 | .8047<br>±.0248 | .7891<br>±.0255 | 2.84 |
| sem-mcts | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7969<br>±.0252 | .7734<br>±.0262 | .7578<br>±.0268 | 3.96 |
| cnt-mcts-bl-v01 | 2 | scored | .6836<br>±.0291 | .6562<br>±.0297 | .6602<br>±.0297 | .6562<br>±.0297 | 2.75 |
| kube-mcts-bl-v01 | 2 | scored | .8359<br>±.0232 | .7773<br>±.0261 | .7617<br>±.0267 | .7695<br>±.0264 | 2.71 |
| kdepth-mcts-bl-v01 | 2 | scored | .8164<br>±.0242 | .7422<br>±.0274 | .7461<br>±.0273 | .7461<br>±.0273 | 2.58 |
| sem-mcts-bl-v01 | 2 | scored | .8320<br>±.0234 | .6992<br>±.0287 | .6445<br>±.0300 | .6484<br>±.0299 | 4.19 |

> **Analysis.** Across the 5 models with cnt-mcts-bl-v01 data
> (llama-1b, qwen-3b, qwen-3b-gptq-int4 partial, qwen-7b-gptq-int4,
> qwen-math-1.5b), the ranking is consistently **cnt-mcts ≳ sem-mcts
> > sem-mcts-bl-v01 > cnt-mcts-bl-v01** on pass@gb — both frontier
> (best-first) variants trail their phase-based counterparts, and
> cnt-mcts-bl-v01 trails sem-mcts-bl-v01 at every model where both
> are scored (e.g. qwen-7b-gptq-int4: .8125 vs. .8906; qwen-math-1.5b:
> .6836 vs. .8320). This is consistent with bl_cnt_v01's documented
> ~18% zero-completion rate at this budget (see the `cnt-mcts-bl-v01`
> section above) — a frontier search that exhausts budget without
> completing a path loses all credit for that question, and bl_sem's
> diversity bonus may be mitigating that failure mode somewhat where
> bl_cnt's plain PUCT does not.
> `kube-mcts-bl-v01` (Fractional KUBE) is now filled at 5 of 7
> models (2026-07-09) and confirms this: cost-aware density
> selection substantially closes the frontier-vs-phase-based gap
> that cnt-mcts-bl-v01 showed. v02 beats v01 at every shared model
> (llama-1b .5586 vs .4414, qwen-3b .8320 vs .6445, qwen-7b-gptq-int4
> .8750 vs .8125, qwen-math-1.5b .8359 vs .6836) and lands close to
> (though still slightly behind) cnt-mcts/sem-mcts at the same
> models — e.g. qwen-7b-gptq-int4: cnt-mcts .9102, sem-mcts .9375,
> kube-mcts-bl-v01 .8750, a ~.05-.06 gap versus bl_cnt_v01's ~.10 gap
> at the same model. This supports the hypothesis above: KUBE's
> cost-normalized `(q+bonus)/cost` density discounts
> expensive-to-finish nodes rather than treating them as equally
> attractive regardless of remaining budget, mitigating (not fully
> closing) the budget-exhaustion failure mode without needing sem's
> diversity bonus.
> `kdepth-mcts-bl-v01` (fixed depth-shaping bonus, no exploration term)
> is also now filled at the same 5 models and tells a similar story:
> it beats v01 at every shared model (llama-1b .5742 vs .4414,
> qwen-3b .8164 vs .6445, qwen-7b-gptq-int4 .9023 vs .8125,
> qwen-math-1.5b .8164 vs .6836) and is competitive with v02 — edging
> it out at 3 of 4 directly-shared models (llama-1b .5742 vs .5586,
> qwen-7b-gptq-int4 .9023 vs .8750; qwen-math-1.5b .8164 vs .8359 is
> the one exception, -.0195) — despite having no confidence-bound or
> regret guarantee behind its bonus. That a fixed, visit-count-free
> depth preference matches or beats an evidence-based UCB bonus at
> this single budget point is a useful negative result on how much
> the exploration term actually buys here, though it says nothing
> about behavior off this budget or across a `depth_beta`/
> `depth_alpha` sweep, which doesn't exist yet.
> **Limitations / follow-up:** llama-3b gptq and qwen-3b-gptq-int4
> still have no bl_cnt-v01/bl_cnt-v02/bl_cnt-v03 data;
> qwen-3b-gptq-int4's sem-mcts-bl-v01 cell reuses the `w_eff=100`
> point (see the `sem-mcts-bl` section's own w_eff=10 table for a
> lower-diversity comparison point at .8086, now also scored).
> llama-3b fp16's sem-mcts-bl-v01 cell is marked failed: the
> w_eff=100 run this row reads from (`0f06296f`, 0/2 trials)
> crashed on a vLLM `max_model_len=5000` context overflow — the
> search has no prompt-length guard — and the sibling w_eff=10
> cell (`3ca318f6`, 1/2) died the same way; both need a length
> guard (or a larger max_model_len) plus a rerun before this cell
> can fill.

---

## Algorithm name ↔ code mapping
> Row labels are conceptual names; `method=` is what
> `config_name()` emits into
> `results/<dataset>/<method>--level-N--...--b-NNN--.../`.

| Concept | Code `method=` | Core module | Status |
|---|---|---|---|
| cnt-mcts | `mcts_cnt` | `mcts_cnt_search_v01_00_00` | runs logged |
| sem-mcts (PRM) | `mcts_sem_v02` | `mcts_sem_search_v02_00_00` | runs logged |
| sem-mcts (policy) | `mcts_sem_v01` | `mcts_sem_search_v01_00_00` | runnable, no runs yet |
| cnt-mcts-bl v01 | `mcts_bl_cnt_v01` | `mcts_bl_cnt_search_v01_00_00` | runs logged |
| kube-mcts-bl v01 (renamed 2026-07-16 from cnt-mcts-bl v02) | `mcts_bl_kube_v01` | `mcts_bl_kube_search_v01_00_00` | runs logged |
| kdepth-mcts-bl v01 (renamed 2026-07-17 from cnt-mcts-bl v03) | `mcts_bl_kdepth_v01` | `mcts_bl_kdepth_search_v01_00_00` | runs logged |
| sem-mcts-bl | `mcts_bl_sem_v01` | `mcts_bl_sem_search_v01_00_00` | runnable, no runs yet |

> Every sem-mcts row elsewhere in this doc is **v02** (PRM-sourced
> embeddings) — v01 (policy-sourced, via a 2nd vLLM pooling
> engine) is wired up on `ExpConfig` but has no runs yet. v01 vs.
> v02 is a clean embedding-*source* ablation on the same
> diversity algorithm; v02 additionally supports
> `embeds_proj=none|sparse` (sparse = JL projection to 512-dim,
> ~2.5x faster) and `cov_update=exact|sherman_morrison`.
> `mcts_bl_kube_v01` (renamed 2026-07-16 from `mcts_bl_cnt_v02`) has a
> launcher + config now (previously flat/unmigrated); see the
> `kube-mcts-bl-v01` rows above and section below for its now-scored
> runs. `mcts_bl_kdepth_v01` (renamed 2026-07-17 from `mcts_bl_cnt_v03`)
> (2026-07-09) is a sibling of the KUBE variant sharing the same knapsack
> skeleton/cost mapping/affordability step, but replaces its
> UCB confidence bonus with a fixed depth-preference function
> (`depth_beta`/`depth_alpha`, no visit-count term) — see
> `docs/decisions/bl-kdepth-knapsack-bonus.md`; see the
> `kdepth-mcts-bl-v01` rows above and section below for its now-scored
> runs. `sem-mcts-bl`
> (`mcts_bl_sem_v01`) is now implemented (2026-07-08) — best-
> first frontier selection with sem's diversity-adjusted value,
> run from `generate_mcts_sem.py`, `algo=mcts_bl_sem_v01` — but
> has no runs yet.

## Summary — results per (algorithm, model, budget)
> The cross-model / cross-algorithm comparison: one row per
> algorithm × model × budget, showing the metrics of its
> **best overall config by pass@gb** — template, cpuct, and
> any other tuning knob are all on equal footing here; this
> row is whichever config in the tuning tables below scored
> highest, not a template-specific pick (full per-knob detail
> lives in those tables). Read down a model to compare
> algorithms, or across models under one algorithm. Dataset =
> PRM800K level-4, bs-4 d-20. Trials = scored trials.

| algorithm | model | budget | trials | pass@gb | naive@gb | wei@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| cnt-mcts | llama-1b | 80 | 4 | .648<br>±.042 | .492<br>±.044 | .469<br>±.044 | .414<br>±.044 |
| cnt-mcts | llama-3b | 80 | 4 | .744<br>±.019 | .508<br>±.022 | .586<br>±.022 | .582<br>±.022 |
| cnt-mcts | qwen-3b | 80 | 4 | .873<br>±.015 | .689<br>±.021 | .727<br>±.020 | .715<br>±.020 |
| cnt-mcts | qwen-math-1.5b | 80 | 2 | .879<br>±.020 | .746<br>±.027 | .770<br>±.026 | .758<br>±.027 |
| sem-mcts (PRM) | llama-1b | 80 | 2 | .594<br>±.031 | .445<br>±.031 | .430<br>±.031 | .414<br>±.031 |
| sem-mcts (PRM) | llama-3b | 80 | 1 ⚠ | .719<br>±.040 | .539<br>±.044 | .578<br>±.044 | .555<br>±.044 |
| sem-mcts (PRM) | qwen-3b | 80 | 2 | .840<br>±.023 | .629<br>±.030 | .703<br>±.029 | .699<br>±.029 |
| sem-mcts (PRM) | qwen-math-1.5b | 80 | 2 | .879<br>±.020 | .746<br>±.027 | .766<br>±.027 | .746<br>±.027 |
| sem-mcts (policy) | — | 80 | — | *planned (no runs yet)* | — | — | — |
| cnt-mcts-bl-v01 | — | 80 | — | *planned (no qwen-PRM runs yet)* | — | — | — |
| sem-mcts-bl | — | 80 | — | *planned (no runs yet)* | — | — | — |

> Winning config per row (cpuct fixed at 2.0 throughout — no
> sweep yet, so template is the only knob currently in play;
> see tuning tables for the full grid): cnt-mcts — llama-1b
> **custom** (.648 > native .566, the only model with both
> scored); llama-3b **custom** (.744 > native .732, both
> now scored); qwen-3b **native** (only scored);
> qwen-math-1.5b **native** (custom is scored at .894 over 2
> trials but template-bug — see tuning table; not a valid
> winner yet).
> `sem-mcts (policy)`/`cnt-mcts-bl-v01`/`sem-mcts-bl` — no runs yet.
>
> **What the numbers say (budget 80):**
> - Within cnt, model size/family dominates template: both
>   Qwen models hit .879 pass@gb, well above Llama
>   (.648/.744).
> - **Custom now beats native on both Llama sizes** (1B:
>   .648 > .566; 3B: .744 > .732) — but the 3B Qwen custom
>   run (`mcts_cnt--level-4--Qwen2.5-Math-1.5B--tmpl-custom`)
>   is producing malformed/leaking completions from a
>   template bug (custom_chat_template is hardcoded to Llama
>   3.1's tokens and gets force-applied to Qwen's tokenizer
>   too — `llm-reasoning-mcts-exp-todo` Track 1), so the
>   "custom wins" trend may not hold once that's fixed and
>   Qwen's custom numbers are comparable.

---

## Tuning tables [gen_budget=80]
> Hierarchy: `### <algorithm>` → `##### <model family + size>`
> → a table whose rows are configs (template × cpuct …).
> Algorithm and model are *subsection* levels, not columns
> (model sometimes becomes an `llm` column instead, for
> grouped comparisons across models). Higher budgets get their
> own `## Tuning tables [gen_budget=N]` section as runs land
> (expected sparser — less tuning at high budget).

### cnt-mcts
> knobs: template, cpuct (bs-4, d-20 fixed). method=`mcts_cnt`.
> No cpuct sweep yet — every row is the default 2.0. All four
> @gb metrics shown for review; the Summary above promotes
> whichever row scores highest on **pass@gb** across all knobs
> jointly (template + cpuct, once swept) per model — not a
> template-specific pick. Metric `—` = scored but not
> backfilled; `generated` = raw output, not scored.

#### custom vs native template comparison
<!-- table-id: tbl-c1962a -->
> **Compares:** `tmpl` (custom vs. native chat template) — the
> only varying knob; all other knobs held at default and dropped
> as columns. Kept as separate per-model tables (rather than one
> merged table) so each model's caveats stay attached to its own
> rows.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, prm_batch_size=2.
> `hr/trial`: GPTQ rows read from `timing_state.json`; fp16 rows
> predate that file, so theirs is the mean of `time_per_trial_hr`
> over all logged trials in W&B.
>
> ⚠️ **custom = template-bug on every Qwen row below
> (qwen-math-1.5b, qwen-3b gptq-int4, qwen-7b gptq-int4) — NOT a
> clean custom-vs-native signal for Qwen.** These runs predate the
> 2026-06-19 fix and force-apply the hardcoded Llama-3.1-vendored
> `custom_chat_template` to Qwen's tokenizer regardless of `llm=`;
> Qwen was never trained on these tokens, so completions can leak
> raw `<|eot_id|>`-style markup after the boxed answer
> (`llm-reasoning-mcts-exp-todo` Track 1). Llama rows are
> unaffected (custom is Llama's native template). **Fixed
> 2026-06-19** — `gen.use_custom_template` now defaults per model
> family (Qwen → native, else → custom); each ⚠ row below would
> need a post-fix re-run for a clean number.
> ⚠️ qwen-math-1.5b custom's `9.31*` hr/trial is a single-trial
> value (only trial 0's timing logged) inflated by the
> template-bug's leaked-text slowdown — not a clean 2-trial
> runtime read; see analysis below.
> ⚠️ qwen-3b/qwen-7b gptq-int4 rows run at `prm_batch_size=2`, not
> the fp16 rows' default — not directly comparable to the fp16
> pairs on `hr/trial`.
>
> **W&B:** qwen-math-1.5b custom `kk32i2lp`.

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b | **custom** | 4 | scored | **.648<br>±.042** | .492<br>±.044 | .469<br>±.044 | .414<br>±.044 | 2.42 |
| llama-1b | native | 2 | scored | .566<br>±.031 | .371<br>±.030 | .348<br>±.030 | .313<br>±.029 | 2.56 |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-3b | **custom** | 4 | scored | **.744<br>±.019** | .508<br>±.022 | .586<br>±.022 | .582<br>±.022 | 3.99 |
| llama-3b | native | 4 | scored | .732<br>±.020 | .520<br>±.022 | .547<br>±.022 | .529<br>±.022 | 3.98 |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-3b | native | 4 | scored | .873<br>±.015 | .689<br>±.021 | .727<br>±.020 | .715<br>±.020 | 3.80 |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | custom | 2 | scored ⚠ | .894<br>±.019 | .742<br>±.027 | .770<br>±.026 | .758<br>±.027 | 9.31* |
| qwen-math-1.5b | native | 2 | scored | .879<br>±.020 | .746<br>±.027 | .770<br>±.026 | .758<br>±.027 | 3.08 |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-3b gptq-int4 | custom | 3 | scored ⚠ | .680<br>±.024 | .578<br>±.025 | .628<br>±.025 | .604<br>±.025 | 2.87 |
| qwen-3b gptq-int4 | native | 2 | scored | .797<br>±.025 | .652<br>±.030 | .676<br>±.029 | .688<br>±.029 | 2.74 |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | custom | 3 | scored ⚠ | .867<br>±.017 | .760<br>±.022 | .794<br>±.021 | .784<br>±.021 | 3.18 |
| qwen-7b gptq-int4 | native | 2 | scored | .902<br>±.019 | .672<br>±.029 | .750<br>±.027 | .754<br>±.027 | 3.21 |

> **Analysis.** For Llama, custom beats native at both sizes
> (1b: .648 vs .566; 3b: .744 vs .732, smaller gap) — custom is
> Llama's native template, so this is the expected direction.
> For Qwen, the custom rows are all template-bug-corrupted, so no
> clean custom-vs-native verdict exists yet — qwen-math-1.5b's
> corrupted custom (.894) and clean native (.879) land within
> noise of each other anyway, and qwen-7b gptq-int4 custom trails
> native (.867 vs .902) despite the leak, suggesting native is
> the safer Qwen default regardless of precision. qwen-math-1.5b
> custom's `9.31*` hr/trial reflects the bug's leaked-text
> slowdown (the running average over both trials, not a
> single-trial artifact), not a real native-vs-custom runtime
> delta — don't read it as such.
> **Limitations / follow-up:** every Qwen custom row needs a
> post-2026-06-19-fix re-run to get a real custom-vs-native
> signal; the qwen-3b/qwen-7b gptq-int4 rows additionally aren't
> prmbs-matched to the fp16 rows, so their `hr/trial` only
> compares within their own pair, not across precision.

#### prm_batch_size sweep
<!-- table-id: tbl-0642eb -->
> **Compares:** the in-loop PRM scoring micro-batch
> (`search.prm_batch_size`, [utils/configs.py](../utils/configs.py))
> — same search config otherwise, so pass@gb should be ~flat
> across rows (modulo sampling noise); the point is the
> **runtime/throughput and memory** delta, not a new accuracy
> result.
>
> **Fixed:** llama-1b, tmpl=custom, cpuct=2.0, bs-4, d-20, b=80.
>
> ⚠️ rlhflow/qwen prm_bs∈{1,4} rows are 2-trial runs scored
> 2026-06-21; prm_bs=2 not explicitly run yet (the existing
> baseline cell uses a different trial count/path, so it's left
> out rather than presented as a matched data point).
>
> **W&B:** rlhflow prm_bs=1 `1c9026yj`, prm_bs=4 `wb2un007`;
> qwen prm_bs=1 `8vvw5usb`, prm_bs=4 `u9itrf7k`.

| llm | prm | prm_bs | trials | status | pass@gb | hr/trial | peak GPU mem (GB) |
|---|---|---|---|---|---|---|---|
| llama-1b | rlhflow | 1 | 2 | scored | .617<br>±.030 | 2.51 | 30.23 |
| llama-1b | rlhflow | 2 | — | not run | — | — | — |
| llama-1b | rlhflow | 4 | 2 | scored | .641<br>±.030 | 2.38 | 31.68 |
| llama-1b | qwen | 1 | 2 | scored | .633<br>±.030 | 2.35 | 27.49 |
| llama-1b | qwen | 4 | 2 | scored | .676<br>±.029 | 2.31 | 28.68 |

> **Analysis.** Within the n=2 rows, pass@gb is flat within ~1
> SEM across prm_bs (rlhflow: .617/.641; qwen: .633/.676) — no
> accuracy regression from larger micro-batches, as expected;
> hr/trial also flat (~2.3-2.5 hr), so this sweep shows no
> throughput win at this model/budget scale. **Peak GPU mem is
> NOT flat, though** — `prm_bs=4` consistently costs ~1.2-1.5 GB
> more than `prm_bs=1` (rlhflow: 30.23→31.68 GB; qwen:
> 27.49→28.68 GB), pulled from W&B's auto-logged
> `system.gpu.0.memoryAllocatedBytes` (max over each run's
> history; no explicit code instrumentation — not in
> `timing_state.json` or `wandb.log()` calls in
> [generate_mcts_cnt.py](../generate_mcts_cnt.py)). So `prm_bs=1`
> is the safer default if memory headroom is the binding
> constraint (V100S 32GB): same accuracy, same speed, less
> memory pressure, at this model/budget scale.
> **Limitations / follow-up:** n=2 trials per cell and prm_bs=2
> untested — full writeup incl. why the pass@gb gap isn't real
> and the trial count needed to actually resolve it:
> [findings/exp-findings/prm-batch-size-throughput-memory.md](findings/exp-findings/prm-batch-size-throughput-memory.md).

#### rlhflow vs qwen PRM comparison
<!-- table-id: tbl-ab03de -->
> **Compares:** `prm.kind` (Llama-8B-PRM "rlhflow" vs
> Qwen-Math-7B-PRM "qwen") — the *scoring* model, not the policy
> LLM. Both PRMs support scoring via `PRM_REGISTRY`/`build_prm()`
> (decisions-log.md, 2026-06-19); this is the scoring-side
> counterpart to the embeds-source ablation (sem-mcts,
> PRM-as-embedder, 2026-06-20).
>
> **Fixed:** tmpl=custom (legacy rlhflow rows) / model-family
> default (new qwen-PRM `cfg-*` rows), cpuct=2.0, bs-4, d-20, b=80.
>
> ⚠️ llama-1b has both PRMs at matched `prm_bs=1`, 2 trials each.
> llama-3b/qwen-3b/qwen-math-1.5b's qwen-PRM rows are matched to
> each other at `prm_bs=1`, 2 trials (new `cfg-*` dirs, scored +
> logged 2026-06-22) but NOT matched to their own rlhflow row: the
> rlhflow rows' trial count varies (4/4/2) and prmbs isn't pinned
> to 1 — llama-3b/qwen-3b's legacy `tmpl-custom` dirs predate the
> `prm_batch_size` field entirely (`mcts_cnt`'s schema has no such
> knob — assumed `prmbs=4` here, not recorded). GPTQ-int4 rows
> still have no qwen-PRM run (rlhflow-only so far).
>
> **W&B:** llama-1b rlhflow `1c9026yj`, llama-1b qwen `8vvw5usb`;
> llama-3b qwen `5opc7rii`; qwen-3b qwen `9kxy56vs`; qwen-math-1.5b
> qwen `9skdu6r4`.

| llm | prm | prmbs | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | rlhflow | 1 | 2 | scored | .617<br>±.030 | .434<br>±.031 | .461<br>±.031 | .434<br>±.031 | 2.51 |
| llama-1b | qwen | 1 | 2 | scored | .633<br>±.030 | .531<br>±.031 | .492<br>±.031 | .449<br>±.031 | 2.35 |
| llama-3b | rlhflow | 4 (assumed, legacy) | 4 | scored | .744<br>±.019 | .508<br>±.022 | .586<br>±.022 | .582<br>±.022 | 3.99 |
| llama-3b | qwen | 1 | 2 | scored | .785<br>±.026 | .688<br>±.029 | .648<br>±.030 | .629<br>±.030 | 4.16 |
| qwen-3b | rlhflow | 4 (assumed, legacy) | 4 | scored | .873<br>±.015 | .689<br>±.021 | .727<br>±.020 | .715<br>±.020 | 3.80 |
| qwen-3b | qwen | 1 | 2 | scored | .867<br>±.021 | .758<br>±.027 | .777<br>±.026 | .766<br>±.027 | 3.82 |
| qwen-math-1.5b | rlhflow | 4 (assumed, legacy) | 2 | scored | .879<br>±.020 | .746<br>±.027 | .770<br>±.026 | .758<br>±.027 | 3.08 |
| qwen-math-1.5b | qwen | 1 | 2 | scored | .898<br>±.019 | .809<br>±.025 | .773<br>±.026 | .785<br>±.026 | 2.86 |

> **Analysis.** At llama-1b and llama-3b, qwen-PRM scoring edges
> out rlhflow on pass@gb (.633 vs .617; .785 vs .744); at
> qwen-math-1.5b it edges out too (.898 vs .879); at qwen-3b
> rlhflow is marginally ahead (.873 vs .867). All gaps are within
> ~1 SEM at n=2-4 trials per cell — read as "qwen-PRM is at least
> competitive everywhere, possibly slightly ahead at
> smaller/non-Qwen models," not a settled result. naive/wei/maj
> follow the same direction as pass@gb at every model except
> qwen-math-1.5b, where qwen-PRM's naive (.809) and maj (.785)
> lead but wei (.773) trails rlhflow's wei (.770) only marginally
> — no metric flips the overall pass@gb ranking. **Runtime:**
> qwen-PRM hr/trial is close to rlhflow's at every model (within
> ±0.2hr) except llama-3b, where qwen-PRM is slower (4.16 vs
> 3.99) — the opposite direction from the sem-mcts version of
> this table, where qwen-PRM ran faster throughout; likely just
> noise at n=2 trials since both PRMs score at the same
> `prm_batch_size=1` here.
> **Limitations / follow-up:** the llama-3b/qwen-3b/qwen-math-1.5b
> qwen-PRM rows are new `cfg-*` dirs, distinct from the older
> unscored `llama-3b/prm-qwen/prmbs-4` legacy dir (still exists
> ungenerated/unscored, unrelated to this table now). rlhflow's
> prmbs-4-assumed legacy rows would need a clean prmbs-1 re-run
> to fully isolate `prm.kind` from `prm_batch_size`.

#### enforce_eager comparison
<!-- table-id: tbl-bfab79 -->
> **Compares:** `llm.enforce_eager` (vLLM's CUDA-graph toggle —
> `True` disables CUDA graphs, `False`/default uses them) at
> fixed model.
>
> **Fixed:** llama-3b, rlhflow, tmpl=custom, cpuct=2.0, bs-4, d-20,
> b=80.
>
> ⚠️ Only llama-3b/rlhflow currently has both values run (the
> legacy `tmpl-custom` dir at `enforce_eager=False`, confirmed via
> W&B config, vs. `cfg-e829c53b` at `enforce_eager=True`) — a
> single-row, single-model comparison, not a sweep. **Not
> matched:** trial count (4 vs 2) and `prm_batch_size` (legacy
> dir's PRM scoring batch size predates the field and isn't
> recorded; `cfg-e829c53b` is prm_batch_size=1) — treat as a rough
> signal, not a controlled ablation. Per [[feedback_prefer_eager_false]],
> every other table in this doc uses the `enforce_eager=False`
> run for a given cell — this is the only place an
> `enforce_eager=True` row should appear.
>
> **W&B:** llama-3b eager=False `97w1z01n`, eager=True `e5ki98he`.

| llm | prm | enforce_eager | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-3b | rlhflow | False (default) | 4 | scored | .744<br>±.019 | .508<br>±.022 | .586<br>±.022 | .582<br>±.022 | 3.99 |
| llama-3b | rlhflow | True | 2 | scored | .746<br>±.027 | .504<br>±.031 | .602<br>±.031 | .594<br>±.031 | 4.65 |

> **Analysis.** pass@gb is essentially identical (.744 vs .746,
> well within 1 SEM) — `enforce_eager` looks accuracy-neutral
> here, as expected (it only changes vLLM's execution mode, not
> sampling). **hr/trial is notably higher with eager mode on**
> (4.65 vs 3.99, ~17% slower) — consistent with CUDA graphs
> normally speeding up decode; eager mode forgoes that.
> **Limitations / follow-up:** with only 2-4 trials and an
> unmatched prm_batch_size, treat the runtime gap as suggestive,
> not conclusive — a matched-trial, matched-prmbs re-run would
> confirm it. No other model/PRM combination has both eager
> values run yet, so this can't be checked for generality.

#### model family, size, quantization comparison
<!-- table-id: tbl-8ca223 -->
> **Compares:** model family, size, and quantization jointly —
> `llm` is a single combined string (model-precision) rather
> than split columns, since this varies model+precision per row
> (unlike the template comparison's single-knob isolation).
>
> **Fixed:** cpuct=2.0, bs-4, d-20, b=80, tmpl=model-family
> default (native for Qwen, custom for Llama, per the
> 2026-06-19 per-family default fix above).
>
> ⚠️ GPTQ rows use prm_batch_size=2 (vs. the fp16 rows' default)
> and read `hr/trial` from `timing_state.json`; fp16 rows predate
> that file, so theirs is the mean of `time_per_trial_hr` over
> all logged trials in W&B (4 trials for llama-1b/3b and qwen-3b
> fp16, 2 for qwen-math-1.5b fp16) — the fp16/GPTQ runtime
> comparison isn't perfectly apples-to-apples.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .648<br>±.042 | .492<br>±.044 | .469<br>±.044 | .414<br>±.044 | 2.42 |
| llama-3b fp16 | 4 | scored | .744<br>±.019 | .508<br>±.022 | .586<br>±.022 | .582<br>±.022 | 3.99 |
| llama-3b gptq | 3 | scored | .721<br>±.023 | .492<br>±.026 | .537<br>±.026 | .531<br>±.026 | 2.92 |
| qwen-3b fp16 | 4 | scored | .873<br>±.015 | .689<br>±.021 | .727<br>±.020 | .715<br>±.020 | 3.80 |
| qwen-3b gptq-int4 | 2 | scored | .797<br>±.025 | .652<br>±.030 | .676<br>±.029 | .688<br>±.029 | 2.74 |
| qwen-7b gptq-int4 | 2 | scored | .902<br>±.019 | .672<br>±.029 | .750<br>±.027 | .754<br>±.027 | 3.21 |
| qwen-math-1.5b fp16 | 2 | scored | .879<br>±.020 | .746<br>±.027 | .770<br>±.026 | .758<br>±.027 | 3.08 |

> **Analysis.** GPTQ trades a modest accuracy hit for faster
> trials at matched budget — llama-3b gptq is ~27% faster than
> its fp16 counterpart (2.92 vs 3.99 hr) but loses ~2.3 pts
> pass@gb (.721 vs .744); qwen-3b gptq-int4 is ~28% faster than
> fp16 (2.74 vs 3.80 hr) but loses ~7.6 pts (.797 vs .873) — a
> bigger accuracy cost than Llama at the same size. qwen-7b
> gptq-int4 is the standout: .902 pass@gb, the best score in
> this table, while still running faster than every fp16 row
> except llama-1b — int4 lets the 7B model run cheaper than the
> 3B fp16 models while beating them on accuracy.
> **Limitations / follow-up:** trial counts are small and uneven
> (2-4), and the GPTQ rows' `prm_batch_size` mismatch means the
> fp16/GPTQ runtime comparison should be read for direction, not
> exact percentages. A matched-prmbs re-run would sharpen the
> runtime deltas.

### cnt-mcts (updated)
> method=`mcts_cnt_v01`. Corrected reruns following the
> `PRM._split_steps` fix (2026-07-06 — see
> [findings/coding-findings/prm-step-split-trailing-separator.md](findings/coding-findings/prm-step-split-trailing-separator.md)
> and `docs/decisions-log.md`), which affected `agg_strategy="last"`
> scoring for non-terminal candidates in every table below. The
> `### cnt-mcts` section above is kept as-is (method=`mcts_cnt`,
> pre-fix) for comparison; do not edit it. Table shapes copied
> from there; rows to be filled in as reruns land.

#### custom vs native template comparison
<!-- table-id: tbl-55d130 -->

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b | custom | — | to rerun | — | — | — | — | — |
| llama-1b | native | — | to rerun | — | — | — | — | — |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-3b | custom | — | to rerun | — | — | — | — | — |
| llama-3b | native | — | to rerun | — | — | — | — | — |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-3b | native | — | to rerun | — | — | — | — | — |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | custom | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | native | — | to rerun | — | — | — | — | — |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-3b gptq-int4 | custom | — | to rerun | — | — | — | — | — |
| qwen-3b gptq-int4 | native | — | to rerun | — | — | — | — | — |

| llm | tmpl | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | custom | — | to rerun | — | — | — | — | — |
| qwen-7b gptq-int4 | native | — | to rerun | — | — | — | — | — |

#### prm_batch_size sweep
<!-- table-id: tbl-b5bc59 -->

| prm | prm_bs | trials | status | pass@gb | hr/trial | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| rlhflow | 1 | — | to rerun | — | — | — |
| rlhflow | 2 | — | to rerun | — | — | — |
| rlhflow | 4 | — | to rerun | — | — | — |
| qwen | 1 | — | to rerun | — | — | — |
| qwen | 4 | — | to rerun | — | — | — |

#### rlhflow vs qwen PRM comparison
<!-- table-id: tbl-ef6f98 -->

| llm | prm | prmbs | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | rlhflow | 1 | — | to rerun | — | — | — | — | — |
| llama-1b | qwen | 1 | — | to rerun | — | — | — | — | — |
| llama-3b | rlhflow | — | — | to rerun | — | — | — | — | — |
| llama-3b | qwen | 1 | — | to rerun | — | — | — | — | — |
| qwen-3b | rlhflow | — | — | to rerun | — | — | — | — | — |
| qwen-3b | qwen | 1 | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | — | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1 | — | to rerun | — | — | — | — | — |

#### enforce_eager comparison
<!-- table-id: tbl-adf2f8 -->

| llm | prm | enforce_eager | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-3b | rlhflow | False (default) | — | to rerun | — | — | — | — | — |
| llama-3b | rlhflow | True | 2 | scored | .7461<br>±.0273 | .5039<br>±.0313 | .6016<br>±.0307 | .5938<br>±.0308 | 4.65 |

> **Analysis.** True arm scored 2026-07-23 (the mcts_cnt rerun,
> W&B `e5ki98he`); the False arm still awaits its rerun.
> Informal cross-check against the rlhflow model-family table's
> llama-3b fp16 cell below (.7422, 4.44 hr, eager off): accuracy
> unchanged (< 1 SEM) and eager costs ~5% wall-clock —
> consistent with `enforce_eager=True` only disabling CUDA
> graphs. Informal because that cell is a separate v01-flavor
> run, not this table's False arm.
> **Limitations / follow-up:** ledger
> orchestration/ledgers/prm800k-level4.yaml (`cnt-mcts-e829c53b`), feeds
> `tbl-adf2f8`. The controlled read needs the False-arm rerun.

#### model family, size, quantization comparison
<!-- table-id: tbl-702925 -->
> **Fixed:** method=`mcts_cnt_v01`, prm=rlhflow, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (default,
> unlike the pre-fix `### cnt-mcts` table's GPTQ rows which used
> prm_batch_size=2 — here every row uses the same default, so
> fp16/GPTQ runtimes are directly comparable), tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> **W&B:** llama-1b `w0e8cidi`, llama-3b fp16 `87yk0nf9`, llama-3b
> gptq `c9qcq6dk`, qwen-7b gptq-int4 `qumxbcc8` (all generated
> 2026-07-07; scoring backfilled via `compute_stats.py` — the
> llama-3b rows had no prior generation-time W&B run, so
> `compute_stats.py` created a fresh run rather than resuming one).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5430<br>±.0312 | .4258<br>±.0310 | .3984<br>±.0307 | .3867<br>±.0305 | 2.68 |
| llama-3b fp16 | 2 | scored | .7422<br>±.0274 | .5547<br>±.0311 | .6055<br>±.0306 | .5898<br>±.0308 | 4.44 |
| llama-3b gptq | 2 | scored | .6953<br>±.0288 | .5195<br>±.0313 | .5195<br>±.0313 | .5078<br>±.0313 | 3.35 |
| qwen-3b fp16 | 2 | scored | .8398<br>±.0230 | .6875<br>±.0290 | .7148<br>±.0283 | .7070<br>±.0285 | 4.01 |
| qwen-3b gptq-int4 | 2 | scored | .7812<br>±.0259 | .6445<br>±.0300 | .6797<br>±.0292 | .6641<br>±.0296 | 2.97 |
| qwen-7b gptq-int4 | 2 | scored | .9180<br>±.0172 | .7148<br>±.0283 | .7852<br>±.0257 | .7812<br>±.0259 | 3.36 |
| qwen-math-1.5b fp16 | 2 | scored | .9102<br>±.0179 | .7695<br>±.0264 | .7891<br>±.0255 | .7812<br>±.0259 | 3.15 |

> **Fixed a real `compute_stats.py` hang while filling this row.**
> `qwen-3b gptq-int4` (rlhflow) reproducibly hung — bisected to one
> record (`test/precalculus/920.json`, a matrix-power question)
> whose model completion boxed a whole equation instead of a value;
> comparing it via `sympy` hung so hard that `signal.alarm`
> (`utils/metrics.py::run_with_timeout`/`_grade_pred`) couldn't
> interrupt it — signals only fire between Python bytecode
> instructions, and the stuck call was in `sympy`'s C-level code.
> Fixed by passing `timeout=True` to `grader2.math_equal` at both
> call sites, routing symbolic comparison through `grader2.py`'s
> already-existing (but previously unused by `metrics.py`) hard-kill
> subprocess path (`call_with_timeout`/`symbolic_equal_process`,
> `multiprocessing.Process.terminate()`) instead of in-process
> comparison. Verified: the poison record now resolves in ~1-10s
> instead of hanging forever; both trial files replayed clean
> end-to-end; the real `compute_stats.py` invocation for this cell
> now completes in ~1 minute. See
> [findings/coding-findings/compute-stats-sympy-hang.md](findings/coding-findings/compute-stats-sympy-hang.md)
> for the full write-up.

#### model family, size, quantization comparison (qwen PRM)
<!-- table-id: tbl-6fe5a2 -->
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (default,
> matched across every row — same rationale as the rlhflow
> table above), tmpl=model-family default (native for Qwen,
> custom for Llama). Companion to the rlhflow-PRM table above;
> same 7 model/quant configs, different scoring PRM.
>
> **W&B:** llama-1b `cqbxegfu`, llama-3b `sfy5oinp`, llama-3b gptq
> `34hihgfu`, qwen-3b gptq-int4 `pr4yz0v3`, qwen-7b gptq-int4
> `pk4vy32g` (all generated 2026-07-07; scoring backfilled via
> `compute_stats.py`).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .6367<br>±.0301 | .5352<br>±.0312 | .4961<br>±.0313 | .4531<br>±.0312 | 2.38 |
| llama-3b fp16 | 2 | scored | .7656<br>±.0265 | .6758<br>±.0293 | .6523<br>±.0298 | .6445<br>±.0300 | 4.02 |
| llama-3b gptq | 2 | scored | .7148<br>±.0283 | .6055<br>±.0306 | .5781<br>±.0309 | .5625<br>±.0311 | 2.85 |
| qwen-3b fp16 | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7695<br>±.0264 | .7617<br>±.0267 | 3.76 |
| qwen-3b gptq-int4 | 2 | scored | .8320<br>±.0234 | .7031<br>±.0286 | .7109<br>±.0284 | .6914<br>±.0289 | 2.68 |
| qwen-7b gptq-int4 | 2 | scored | .9102<br>±.0179 | .8086<br>±.0246 | .8164<br>±.0242 | .8008<br>±.0250 | 3.11 |
| qwen-math-1.5b fp16 | 2 | scored | .8906<br>±.0195 | .8008<br>±.0250 | .8047<br>±.0248 | .7891<br>±.0255 | 2.84 |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
<!-- table-id: tbl-3ea294 -->
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. `"last"` is every
> other table's fixed default; `"min"` and `"prod"` are
> implemented but not yet reported anywhere in this doc. Prompted
> by the `_split_steps` fix (`agg="last"`-specific bug, see
> `### cnt-mcts (updated)` header above) — `"min"` in particular
> is a useful cross-check since it's structurally less exposed to
> that bug (a holistic bogus score rarely wins a min() over a
> trajectory with a genuinely bad step).
>
> **Fixed:** method=`mcts_cnt_v01`, cpuct=2.0, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | rlhflow | min | 2 | scored | .8633<br>±.0215 | .6367<br>±.0301 | .7344<br>±.0277 | .7148<br>±.0283 | 4.03 |
| qwen-3b | rlhflow | prod | 2 | scored | .8477<br>±.0225 | .7031<br>±.0286 | .7344<br>±.0277 | .7031<br>±.0286 | 4.00 |
| qwen-3b | rlhflow | last | 2 | scored | .8398<br>±.0230 | .6875<br>±.0290 | .7148<br>±.0283 | .7070<br>±.0285 | 4.01 |
| qwen-3b | qwen | min | 2 | scored | .8867<br>±.0198 | .7969<br>±.0252 | .7852<br>±.0257 | .7500<br>±.0271 | 3.78 |
| qwen-3b | qwen | prod | 2 | scored | .8398<br>±.0230 | .7891<br>±.0255 | .7891<br>±.0255 | .7422<br>±.0274 | 3.70 |
| qwen-3b | qwen | last | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7695<br>±.0264 | .7617<br>±.0267 | 3.76 |
| qwen-math-1.5b | rlhflow | min | 2 | scored | .8789<br>±.0204 | .6836<br>±.0291 | .7188<br>±.0282 | .7031<br>±.0286 | 3.12 |
| qwen-math-1.5b | rlhflow | prod | 2 | scored | .8906<br>±.0195 | .7617<br>±.0267 | .7812<br>±.0259 | .7422<br>±.0274 | 3.09 |
| qwen-math-1.5b | rlhflow | last | 2 | scored | .9102<br>±.0179 | .7695<br>±.0264 | .7891<br>±.0255 | .7812<br>±.0259 | 3.15 |
| qwen-math-1.5b | qwen | min | 2 | scored | .8867<br>±.0198 | .8125<br>±.0244 | .7656<br>±.0265 | .7422<br>±.0274 | 2.84 |
| qwen-math-1.5b | qwen | prod | 2 | scored | .8711<br>±.0210 | .8086<br>±.0246 | .7891<br>±.0255 | .7617<br>±.0267 | 2.90 |
| qwen-math-1.5b | qwen | last | 2 | scored | .8906<br>±.0195 | .8008<br>±.0250 | .8047<br>±.0248 | .7891<br>±.0255 | 2.84 |

> **Analysis (both models).**
> pass@gb is flat within ~1-2 SEM across all three strategies for
> both PRMs and both models (qwen-math-1.5b rlhflow: .879/.891/
> .910; qwen: .887/.871/.891 — qwen-3b rlhflow: .863/.848/.840;
> qwen: .887/.840/.879) — at n=2 trials, no strategy wins
> outright on the headline metric. naive@gb is where the two PRMs
> diverge, but only **one direction clears noise**: under
> rlhflow, `min` is clearly worst and replicates at both sizes
> (qwen-math-1.5b: .684 vs `last`'s .770, an 8.6pt gap against a
> ~2.9pt SEM; qwen-3b: .637 vs `prod`'s .703, a 6.6pt gap against
> a ~3.0pt SEM — both outside 2 SEM). Under qwen, `min` numerically
> edges out `last` at both sizes (qwen-math-1.5b: .813 vs .801,
> +1.2pt; qwen-3b: .797 vs .746, +5.1pt) but neither gap clears
> even 1 SEM (~2.5-2.7pt) — **not distinguishable from noise at
> n=2 trials**, so "qwen favors min" is not yet a supportable
> claim, only "rlhflow clearly penalizes min" is. hr/trial is flat
> across strategies within each PRM (~3.7-4.0hr for qwen-3b vs
> ~2.8-2.9hr for qwen-math-1.5b — model-size gap dominates over
> any agg_strategy effect), as expected — aggregation is a
> scoring-time choice, not a generation-time one.
> **Limitations / follow-up:** n=2 trials only per cell; the
> rlhflow `min`-penalty is the one finding here that clears noise
> and replicates across model sizes. The qwen-side gap needs more
> trials before claiming a direction at all.

### sem-mcts-v02
> **Runnable as of 2026-06-18** (rename + migration landed).
> Every table in this section is `mcts_sem_v02` (PRM embeds, no
> 2nd engine) — hence the heading. The one exception is the
> `LLM vs PRM embeds comparison` table below (`tbl-1eed5c`),
> which is a v01↔v02 head-to-head and therefore also carries
> `mcts_sem_v01` (policy embeds, 2nd vLLM engine) rows; v01 has
> zero runs to date, so that table is entirely `planned`. knobs beyond template: ds_alpha, ds_beta,
> lam, embeds_strategy (last/avg), embeds_normalize, and for
> v02 embeds_proj (none/sparse, dim 512) + cov_update
> (exact/sherman_morrison). Defaults in conf/search/mcts_sem_v0*.
> Run v01 and v02 at matched model/level/trials vs. cnt-mcts —
> the comparison the project exists for. sem-mcts has no
> `cpuct` knob (selection is q-value-only on first visit, then
> a ds_alpha/ds_beta-weighted diversity bonus on later visits;
> see `core/mcts_sem_search_v02_00_00.py:select_child`).

#### embeds_proj × cov_update sweep (v02)
<!-- table-id: tbl-860167 -->
> **Compares:** a 2×2-per-model grid instead of two single-knob
> sweeps. `embeds_proj`: `none` feeds the PRM's raw 4096-dim
> hidden state into the covariance bonus; `sparse512` JL-projects
> it to 512 first (~2.5× speed win, accuracy cost untested).
> `cov_update`: `exact` recomputes V^-1 each step;
> `sherman_morrison` (sm) updates it incrementally
> (path-identical to exact, proven; the question here is whether
> that holds at scale). pass@gb should match within noise across
> cov_update (same path) but may differ across embeds_proj (lossy
> projection); hr/trial is the throughput axis.
>
> **Fixed:** method=`mcts_sem_v02` (proj/cov_update don't exist
> on v01), bs-4, d-20, b=80.
>
> ⚠️ both `none×sm` rows run at `prm_batch_size=2`, not the
> sparse512×sm default-prmbs (prmbs-1) row — not directly
> comparable on throughput. A separate n=1 `sparse512×sm` run at
> `prm_batch_size=2` (W&B `ttsp0a0g`) was dropped from this table
> — prm_bs doesn't affect accuracy per the prm_batch_size sweep
> above, and n=1 added no comparable signal over the n=2 row.
>
> **W&B:** llama-1b sparse512×sm `kqn1lj13`, none×sm `f6ojjyik`;
> qwen-math-1.5b sparse512×exact `lkltpzc1`, sparse512×sm
> `qn3b8lg0`, none×sm `ni9v75j9`.

| llm | proj | cov_update | trials | status | pass@gb | hr/trial |
|---|---|---|---|---|---|---|
| llama-1b | none | exact | — | planned | — | — |
| llama-1b | none | sm (prmbs-2) | 2 | scored ⚠ | .6328<br>±.0302 | 12.05 |
| llama-1b | sparse512 | exact | — | planned | — | — |
| llama-1b | sparse512 | sm | 2 | scored | .5938<br>±.0308 | 4.27 |
| qwen-math-1.5b | none | exact | — | planned | — | — |
| qwen-math-1.5b | none | sm (prmbs-2) | 2 | scored ⚠ | .8789<br>±.0204 | 9.89 |
| qwen-math-1.5b | sparse512 | exact | 2 | scored | .8711<br>±.0210 | 4.34 |
| qwen-math-1.5b | sparse512 | sm | 2 | scored | .8789<br>±.0204 | 4.81 |

> **Analysis.** Other @gb metrics: llama-1b sparse512×sm (2
> trials) — naive .4453±.0311, weighted .4297±.0310, maj
> .4141±.0308, ncomps 14.2±0.8, depth 8.7±0.2, nphases
> 44.5±11.0, ndepths 9.4±0.2. llama-1b none×sm: naive
> .4453±.0311, weighted .4336±.0310, maj .3984±.0307, ncomps
> 14.1±0.7, depth 8.8±0.2, nphases 44.8±11.0, ndepths 9.5±0.3.
> Qwen-1.5B sparse512 (exact vs sm, 2 trials each): naive
> .7383±.0275 vs .7461±.0273; weighted .7500±.0271 (both); maj
> .7539±.0270 vs .7461±.0273 — all within ~1 SEM, no systematic
> effect from cov_update. ncomps 24.6±1.1 vs 23.9±1.1, nphases
> 16.1±4.1 vs 14.8±3.9. qwen-math-1.5b none×sm: naive
> .7539±.0270, weighted .7500±.0271, maj .7422±.0274, ncomps
> 23.5±1.1, depth 9.5±0.2, nphases 11.0±0.6, ndepths 10.2±0.2.
> **proj effect (none vs sparse512, both sm):** llama-1b none
> (.6328) > sparse512 (.5938) — suggestive of the JL projection
> costing some accuracy at this model size, though prm_bs differs
> across the comparison so it isn't fully isolated. qwen-math-1.5b:
> none (.8789) is within ~1 SEM of sparse512×sm (.8789) and
> sparse512×exact (.8711) — no clear separation at this model
> size. **Runtime: none is markedly slower** — llama-1b 12.05
> hr/trial vs sparse512's 4.27; qwen-math-1.5b 9.89 vs 4.32-4.81 —
> roughly 1.2-2.3× slower, consistent with the ~2.5× speedup
> `embeds_proj=sparse512` is expected to give (the projection
> avoids working with the raw 4096-dim covariance). So at this
> budget, sparse512 looks like the better default: comparable-or-
> better accuracy at noticeably lower cost.
> **Limitations / follow-up:** n=2 trials per cell, SEMs wide on
> ncomps/nphases — treat as preliminary. The prmbs confound on
> the proj comparison means a clean prmbs-1 `none` run would
> sharpen the proj-effect read; the `exact` cells for llama-1b are
> still entirely `planned`.

#### embeds_strategy × scope sweep (v02, qwen PRM)
<!-- table-id: tbl-82c90f -->
> **Compares:** how the PRM hidden state is pooled into the
> covariance bonus — `embeds_strategy` (`last` = final-token
> hidden state vs. `avg` = mean over tokens) crossed with
> `embeds_scope` (`full` = the whole prompt+response sequence
> vs. `response` = only the assistant response tokens). The
> question is whether averaging or response-only scoping
> changes the diversity signal enough to move pass@gb.
>
> **Fixed:** method=`mcts_sem_v02`, llm=llama-3b, prm=qwen,
> tmpl=custom (llama default), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, ds_beta=1.0.
>
> ⚠️ `embeds_scope=response` is **not supported on v02** (PRM
> source): `_extract_embeds` computes `response_start_idx`
> with the generator tokenizer, which doesn't apply to PRM
> hidden states, so the core raises for that combination (see
> `core/mcts_sem_search_v02_00_00.py:227`). The two `response`
> rows are therefore **blocked** — shown for completeness but
> not runnable until that fix lands; they are NOT queued in
> `experiments.yaml`. `last`×`full` is the v02 default config
> and is already done (it's the qwen-PRM llama-3b default run,
> W&B `kbwjqw96`-family — same cfg as the ds_alpha=100 cell).
>
> **W&B:** last×full done (cfg-2b647a18); avg×full planned;
> response rows blocked (no runs).
>
> **`lam=0.1` addendum (2026-07-08):** two extra `w_eff` checkpoints
> per strategy, at `lam=0.1` (`w_eff = ds_alpha/sqrt(lam)`, so
> `w_eff=10 → ds_alpha=3.16`, `w_eff=100 → ds_alpha=31.6`). The
> `last` rows at both checkpoints reuse already-scored cells from
> the `lam / ds_alpha joint sweep (v02, llama-3b)` table (same
> `lam=0.1, ds_alpha` pairs, same fixed config) — no new run. The
> `avg` rows at both checkpoints are new; both now run and scored
> (2026-07-08/09).
>
> **`lam=0.01` addendum (2026-07-08):** same two `w_eff` checkpoints
> at the table's default `lam=0.01` (`w_eff=10 → ds_alpha=1.0`,
> `w_eff=100 → ds_alpha=10`). The `last` rows reuse already-scored
> cells from the `lam / ds_alpha joint sweep (v02, llama-3b)` table
> (`cfg-23f6c64a`, `cfg-baa5b18e`) — no new run. The `avg` rows are
> new; both now run and scored (2026-07-08/09).

| llm | strategy | scope | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | last | full | 0.01 | 100 | 1000 | 2 | done (see ds_alpha=100) | — | — | — | — | — |
| llama-3b | last | full | 0.01 | 1.0 | 10 | 2 | scored (see lam/ds_alpha joint sweep) | .7500<br>±.0271 | .6562<br>±.0297 | .6562<br>±.0297 | .6523<br>±.0298 | 5.23 |
| llama-3b | last | full | 0.01 | 10 | 100 | 2 | scored (see lam/ds_alpha joint sweep) | .7695<br>±.0264 | .6797<br>±.0292 | .6445<br>±.0300 | .6211<br>±.0304 | 5.44 |
| llama-3b | last | full | 0.1 | 3.16 | 10 | 2 | scored (see lam/ds_alpha joint sweep) | .7578<br>±.0268 | .6719<br>±.0294 | .6602<br>±.0297 | .6289<br>±.0303 | 5.32 |
| llama-3b | last | full | 0.1 | 31.6 | 100 | 2 | scored (see lam/ds_alpha joint sweep) | .7812<br>±.0259 | .6562<br>±.0297 | .6211<br>±.0304 | .5938<br>±.0308 | 5.51 |
| llama-3b | avg | full | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |
| llama-3b | avg | full | 0.01 | 1.0 | 10 | 2 | scored | .7617<br>±.0267 | .6523<br>±.0298 | .6445<br>±.0300 | .6484<br>±.0299 | 5.51 |
| llama-3b | avg | full | 0.01 | 10 | 100 | 2 | scored | .7773<br>±.0261 | .6719<br>±.0294 | .6484<br>±.0299 | .6328<br>±.0302 | 5.79 |
| llama-3b | avg | full | 0.1 | 3.16 | 10 | 2 | scored | .7695<br>±.0264 | .6641<br>±.0296 | .6641<br>±.0296 | .6211<br>±.0304 | 5.46 |
| llama-3b | avg | full | 0.1 | 31.6 | 100 | 2 | scored | .7539<br>±.0270 | .6641<br>±.0296 | .6328<br>±.0302 | .6172<br>±.0304 | 5.57 |
| llama-3b | last | response | — | — | — | — | blocked | — | — | — | — | — |
| llama-3b | avg | response | — | — | — | — | blocked | — | — | — | — | — |

> **Analysis (updated 2026-07-09).** 4 of 5 `avg`×`full` cells now
> scored (only the default point `lam=0.01,ds_alpha=100,w_eff=1000`
> remains unrun). At matched `lam`/`w_eff`, `avg` vs. `last`:
> `lam=0.01,w_eff=10`: .7617 vs .7500 (+.0117); `lam=0.01,w_eff=100`:
> .7773 vs .7695 (+.0078); `lam=0.1,w_eff=10`: .7695 vs .7578
> (+.0117) — three checkpoints show `avg` a hair above `last`, all
> well inside 1 SEM (~.027), i.e. not distinguishable from noise at
> n=2 trials. `lam=0.1,w_eff=100` is the outlier: `avg` .7539 vs.
> `last` .7812 (**-.0273**, right at 1 SEM) — the one checkpoint
> where the two pooling strategies visibly diverge; worth a repeat
> run before treating it as a real effect rather than variance.
> **Limitations / follow-up:** n=2 trials/cell throughout — the
> `lam=0.1,w_eff=100` divergence is the one cell worth re-running at
> higher n to confirm before drawing a conclusion. The default point
> (`lam=0.01,ds_alpha=100,w_eff=1000`) is still `planned`. The two
> `response` rows are blocked on PRM-source `response_start_idx`
> support; queue them once the v02 core handles
> `embeds_scope=response` for `embeds_source=prm`. A v01
> (policy-embeds) version of this table would unblock the `response`
> axis, since v01 supports it.

#### ds_alpha sweep (v02)
<!-- table-id: tbl-1c9c2c -->
> **Compares:** `ds_alpha`, the diversity-bonus weight in
> `q_val = ds_beta*score + ds_alpha*diversity` (scaled by
> `sqrt(log(1 + parent_visits))` on subsequent visits; see
> `core/mcts_sem_search_v01_00_00.py:_select_by_diversity`).
> `ds_alpha=0` collapses selection to pure q-value (no diversity
> bonus at any visit count) — a useful lower-bound check against
> cnt-mcts-style greedy selection. Default is `100.0`
> (`utils/configs.py:MCTSSemV01Config`).
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=rlhflow, ds_beta=1.0,
> prm_batch_size=1.
>
> ⚠️ All cells are 2 trials at prmbs-1 — treat as preliminary
> (SEMs are wide at n=2), but the grid is now complete and
> internally consistent (same proj=sparse512/cov=sm/prmbs-1
> across every row).
>
> **W&B:** llama-1b ds_alpha=0 `bjz0yxrg`, ds_alpha=10
> `wsvy5q72`, ds_alpha=100 `hdiysdi6`, ds_alpha=1000 `nlx82zbw`;
> llama-3b ds_alpha=10 `8882rt6u`, ds_alpha=100 `gv2b7ajq`,
> ds_alpha=1000 `fv18snbn`; qwen-math-1.5b ds_alpha=0 `j2ms4lvk`,
> ds_alpha=10 `ihxrzedi`, ds_alpha=100 `qn3b8lg0`, ds_alpha=1000
> `kbwjqw96`.

| llm | ds_alpha | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b | 0 | 2 | scored | .4336<br>±.0310 | .4023<br>±.0307 | .4023<br>±.0307 | .3359<br>±.0296 | 3.36 |
| llama-1b | 10 | 2 | scored | .6133<br>±.0305 | .4453<br>±.0311 | .4180<br>±.0309 | .3906<br>±.0306 | 4.93 |
| llama-1b | 100 (default) | 2 | scored | .5898<br>±.0308 | .4336<br>±.0310 | .4336<br>±.0310 | .4062<br>±.0308 | 4.99 |
| llama-1b | 1000 | 2 | scored | .5938<br>±.0308 | .4258<br>±.0310 | .4375<br>±.0311 | .3906<br>±.0306 | 4.96 |
| llama-3b | 10 | 2 | scored | .7422<br>±.0274 | .5430<br>±.0312 | .5781<br>±.0309 | .5703<br>±.0310 | 6.74 |
| llama-3b | 100 (default) | 2 | scored | .7383<br>±.0275 | .5469<br>±.0312 | .5703<br>±.0310 | .5703<br>±.0310 | 6.61 |
| llama-3b | 1000 | 2 | scored | .7344<br>±.0277 | .5586<br>±.0311 | .5977<br>±.0307 | .5938<br>±.0308 | 6.71 |
| qwen-math-1.5b | 0 | 2 | scored | .7812<br>±.0259 | .7266<br>±.0279 | .7266<br>±.0279 | .7227<br>±.0280 | 3.03 |
| qwen-math-1.5b | 10 | 2 | scored | .8945<br>±.0192 | .7617<br>±.0267 | .7812<br>±.0259 | .7578<br>±.0268 | 4.78 |
| qwen-math-1.5b | 100 (default) | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7656<br>±.0265 | .7461<br>±.0273 | 4.81 |
| qwen-math-1.5b | 1000 | 2 | scored | .8867<br>±.0198 | .7656<br>±.0265 | .7656<br>±.0265 | .7422<br>±.0274 | 4.86 |

> **Analysis.** The grid is now complete for llama-1b and
> qwen-math-1.5b (full 0/10/100/1000), and both tell the **same
> story**: turning the diversity bonus *on* helps, but its
> magnitude past ~10 doesn't. llama-1b: .434 at `ds_alpha=0` →
> .613 at 10, then flat (.590 at 100, .594 at 1000). qwen-math-1.5b:
> .781 at `ds_alpha=0` → .894/.879/.887 at 10/100/1000 — same shape,
> a ~10pt lift off the lower bound then a plateau within ~1 SEM.
> So the 0→10 jump is the one real move at *both* model sizes, and
> the "bonus helps, amount doesn't" read now generalizes rather
> than resting on llama-1b alone. **llama-3b** (10/100/1000, no
> `ds_alpha=0` yet) is flat across its filled cells, consistent
> with the plateau, but can't speak to the lower-bound jump
> without a `ds_alpha=0` run.
> **Limitations / follow-up:** llama-3b needs a `ds_alpha=0` run
> to complete its grid and confirm the 0→on jump at that size too.
> All cells are n=2, so the jumps are suggestive, not settled —
> more trials would tighten the wide SEMs before treating the
> plateau as final.

#### ds_alpha sweep (v02, qwen PRM)
<!-- table-id: tbl-93c239 -->
> **Compares:** the same `ds_alpha` diversity-bonus sweep as the
> table above, but with `prm=qwen` (Qwen-Math-7B-PRM) as the
> scoring model instead of `prm=rlhflow` (Llama-8B-PRM). Reading
> this table against the rlhflow one isolates whether the
> `ds_alpha` behavior (the "bonus helps, magnitude past ~10
> doesn't" shape) is robust to the choice of PRM, or specific to
> rlhflow scoring. `ds_alpha=0` is omitted here — the lower-bound
> check is already covered in the rlhflow table; this sweep
> focuses on the on-bonus range (10/100/1000).
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1.
>
> ⚠️ 9/9 cells scored (2 trials each) as of 2026-07-07 — llama-3b
> ds_alpha=1000 completed and is filled in below. `hr/trial` read
> from each run's `timing_state.json` (`avg_time_per_trial_hr`).
>
> **W&B:** llama-1b 10/100/1000 `02xrjfdb`/`7hjxksmx`/`fgem65eg`;
> llama-3b 10/100/1000 `qvp2vneb`/`ynia3d1p`/`7ccy14de` — the 1000
> run is one of the two runs recovered by the 2026-06-24 run_id
> resume-fragmentation fix (`docs/decisions-log.md`), not a fresh run;
> qwen-math-1.5b 10/100/1000 `6hbme316`/`q0d6yk4f`/`sczanhp2`.

| llm | ds_alpha | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b | 10 | 2 | scored | .6211<br>±.0304 | .5352<br>±.0312 | .4844<br>±.0313 | .4258<br>±.0310 | 3.75 |
| llama-1b | 100 (default) | 2 | scored | .6133<br>±.0305 | .4961<br>±.0313 | .4492<br>±.0311 | .3906<br>±.0306 | 3.90 |
| llama-1b | 1000 | 2 | scored | .6289<br>±.0303 | .5078<br>±.0313 | .4688<br>±.0312 | .3945<br>±.0306 | 3.92 |
| llama-3b | 10 | 2 | scored | .7695<br>±.0264 | .6797<br>±.0292 | .6445<br>±.0300 | .6211<br>±.0304 | 5.44 |
| llama-3b | 100 (default) | 2 | scored | .7656<br>±.0265 | .6562<br>±.0297 | .6289<br>±.0303 | .6016<br>±.0307 | 5.43 |
| llama-3b | 1000 | 2 | scored | .7617<br>±.0267 | .6562<br>±.0297 | .6172<br>±.0304 | .5898<br>±.0308 | 5.61 |
| qwen-math-1.5b | 10 | 2 | scored | .8789<br>±.0204 | .7969<br>±.0252 | .7891<br>±.0255 | .7695<br>±.0264 | 3.98 |
| qwen-math-1.5b | 100 (default) | 2 | scored | .8750<br>±.0207 | .7969<br>±.0252 | .7734<br>±.0262 | .7578<br>±.0268 | 3.96 |
| qwen-math-1.5b | 1000 | 2 | scored | .8750<br>±.0207 | .8008<br>±.0250 | .7500<br>±.0271 | .7539<br>±.0270 | 3.92 |

> **Analysis.** The "magnitude past ~10 doesn't matter" shape
> from the rlhflow table holds under qwen-PRM scoring too: within
> each model, pass@gb is flat across 10/100/1000 to within SEM
> (llama-1b .621/.613/.629; llama-3b .770/.766/.762;
> qwen-math-1.5b .879/.875/.875). Now that all 9 cells are in, the
> flatness holds cleanly at llama-3b too — .762 (ds_alpha=1000) is
> within 1 SEM of both .770 (10) and .766 (100), no trend in
> either direction. So the ds_alpha plateau is robust to PRM
> choice, not an rlhflow artifact. Absolute levels track the
> model, as expected (qwen-math-1.5b highest at ~.88, llama-3b
> ~.77, llama-1b ~.62) — consistent with the `rlhflow vs qwen PRM
> comparison` below.
> **Limitations / follow-up:** 2 trials/cell — SEMs ~.03, so the
> within-model flatness is "no detectable trend," not "proven
> equal." A `ds_alpha=0` qwen-PRM row per model would extend the
> lower-bound check to this PRM, but is deferred — the rlhflow
> table already establishes the 0→on jump.

#### lam / ds_alpha joint sweep (v02, llama-1b, step 1 done)
<!-- table-id: tbl-0efc55 -->
> **Compares:** whether `lam` (the ridge constant setting `V`'s
> initial scale, `V_0 = lam*I`) and `ds_alpha` are truly redundant
> along the derived invariant `w_eff = ds_alpha/sqrt(lam)`, or
> whether `lam`'s second role (controlling how fast `V_inv` adapts
> as embeddings accumulate) has an independent effect on pass@gb.
> The two `ds_alpha sweep (v02)` tables above only ever tested
> `lam=0.01`; this table holds `w_eff` fixed across rows and varies
> `lam`/`ds_alpha` jointly, filling in `w_eff` values below the
> confirmed plateau (`w_eff≥100`, i.e. `ds_alpha≥10` at `lam=0.01`)
> that have never been tested. `w_eff ∈ {0.1, 0.3, 3.0}` extend the
> grid below `w_eff=1` — the existing plateau finding only shows
> *that* a switch happens somewhere in `(0, 100)`, not the shape of
> the on-ramp; these log-spaced points (matching the existing
> `{0.316, 3.16, 31.6}` spacing at `lam=0.1`) probe whether the
> switch is sharp near `w_eff≈1` or a gradual ramp from `w_eff=0`.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b. Queued *after* the llama-3b table
> below, which is the one actually running first.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and the full 5-step tuning procedure
> this table is step 1/2 of. **Step 1** is the two bolded cells
> below (`w_eff=10`, matched across `lam=1.0` and `lam=0.01`) — if
> their pass@gb/naive@gb agree within SEM, `lam`'s independent role
> is negligible and the remaining cells collapse to a 1D sweep over
> `w_eff` at one fixed `lam` (skip the rest of the grid); if they
> disagree outside SEM, run the full grid.
>
> ✅ Step 1 done (2026-07-08); every `w_eff∈{1,3,10,100}` row across
> all three `lam` values scored via `compute_stats.py` as of
> 2026-07-08 — no `experiments.yaml` entries were added for these runs
> (launched and generated directly; see below). Under `prm=qwen`,
> llama-1b still has **no existing `ds_alpha=0` (w_eff=0) baseline**
> (the qwen-PRM `ds_alpha sweep (v02, qwen PRM)` table above only has
> 10/100/1000 for llama-1b; only the *rlhflow* table has a llama-1b
> `ds_alpha=0` row) — that row is still not started, and neither are
> the `w_eff∈{0.1,0.3}` on-ramp cells. The `w_eff=100, lam=0.01` row
> reuses the already-scored llama-1b `ds_alpha=10` cell from the
> qwen-PRM sweep table.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned (new; no existing qwen-PRM baseline) | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1 | 1 | 2 | scored | .5781<br>±.0309 | .5078<br>±.0313 | .4883<br>±.0313 | .4805<br>±.0313 | — |
| llama-1b | qwen | 0.1 | 0.316 | 1 | 2 | scored | .6016<br>±.0307 | .5391<br>±.0312 | .5352<br>±.0312 | .5000<br>±.0313 | — |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .6094<br>±.0306 | .5391<br>±.0312 | .5039<br>±.0313 | .4727<br>±.0313 | — |
| llama-1b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .6172<br>±.0304 | .5391<br>±.0312 | .5117<br>±.0313 | .4727<br>±.0313 | 3.74 |
| llama-1b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .6133<br>±.0305 | .5195<br>±.0313 | .5312<br>±.0312 | .4961<br>±.0313 | 3.82 |
| llama-1b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .5938<br>±.0308 | .5508<br>±.0311 | .5234<br>±.0313 | .4688<br>±.0312 | 3.77 |
| llama-1b | qwen | **1.0** | **10** | **10** | **2** | **scored (step 1)** | **.6172<br>±.0304** | **.5273<br>±.0313** | **.4766<br>±.0313** | **.4375<br>±.0311** | — |
| llama-1b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .6133<br>±.0305 | .5156<br>±.0313 | .4766<br>±.0313 | .4375<br>±.0311 | — |
| llama-1b | qwen | **0.01** | **1.0** | **10** | **2** | **scored (step 1)** | **.6250<br>±.0303** | **.5469<br>±.0312** | **.5039<br>±.0313** | **.4648<br>±.0312** | — |
| llama-1b | qwen | 1.0 | 100 | 100 | 2 | scored | .6289<br>±.0303 | .5312<br>±.0312 | .4375<br>±.0311 | .3438<br>±.0297 | 3.93 |
| llama-1b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .6094<br>±.0306 | .5117<br>±.0313 | .4531<br>±.0312 | .4219<br>±.0309 | 3.90 |
| llama-1b | qwen | 0.01 | 10 | 100 | 2 | scored (see qwen-PRM ds_alpha=10 above) | .6211<br>±.0304 | .5352<br>±.0312 | .4844<br>±.0313 | .4258<br>±.0310 | 3.75 |

> **Analysis.** Step 1 done: `w_eff=10` at `lam=1.0`
> (pass@gb=.6172±.0304) and `lam=0.01` (pass@gb=.6250±.0303) agree
> within SEM (Δ=.008, naive@gb Δ=.020, both ≪ SEM≈.031) — same
> conclusion as the llama-3b table below. Per the procedure's own
> rule, `lam`'s independent role is negligible here too, so the
> remaining `lam≠0.01` on-ramp cells were low priority, but got run
> anyway (2026-07-08) and confirm the same pattern: **`w_eff=3` is
> flat across all three `lam`** (1.0→.6172, 0.1→.6133, 0.01→.5938
> pass@gb — all within ~1.5 SEM), and **`w_eff=1` was already flat**
> (1.0→.578, 0.1→.602, 0.01→.609). **`w_eff=100` is flat on pass@gb**
> (1.0→.6289, 0.1→.6094, 0.01→.6211 — all within SEM) but shows
> real spread on `maj@gb`/`wei@gb`: `lam=1.0` drops to maj=.3438,
> noticeably below `lam=0.1`'s .4219 and `lam=0.01`'s .4258 — the
> only clear @gb-metric divergence across `lam` anywhere in this
> table, though pass@gb itself (the primary metric) is unaffected.
> `w_eff=1000, lam=0.01` (pass@gb=.6289±.0303) was also generated but
> isn't part of this table's own grid — it matches the already-scored
> llama-1b `ds_alpha=1000` row in the qwen-PRM sweep table above
> (same config, reused, not duplicated here; a numerical coincidence
> that this also equals the `w_eff=100, lam=1.0` cell's pass@gb —
> different configs, not the same run).
> **Limitations / follow-up:** only `w_eff=0` (`lam=0.01, ds_alpha=0`)
> and the `w_eff∈{0.1,0.3}` on-ramp cells at non-`lam=0.01` values
> remain unrun — every `w_eff∈{1,3,10,100}` row across all three `lam`
> values is now filled. The `maj@gb` dip at `lam=1.0, w_eff=100` is
> worth a second look if a `lam=1.0` sweep is ever prioritized, but
> pass@gb itself shows no `lam` effect anywhere in the completed grid.

#### lam / ds_alpha joint sweep (v02, llama-3b, step 1 done)
<!-- table-id: tbl-f50e22 -->
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b table above, on llama-3b. **This is the table queued
> to run first** — llama-3b has no existing `ds_alpha=0` baseline
> under either PRM (closing that gap and running step 1 happen in
> the same pass), and its result determines how much of the llama-1b
> table (and any later cross-check per procedure step 5) is worth
> running. `w_eff ∈ {0.1, 0.3, 3.0}` fill in the on-ramp below
> `w_eff=1`, same rationale as the llama-1b table above.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and 5-step procedure. **Step 1** is the
> two bolded cells below (`w_eff=10`); the `w_eff=0` cell closes the
> pre-existing llama-3b `ds_alpha=0` gap (neither PRM had this row
> before) and is independent of the step-1 outcome, so it's fine to
> launch alongside step 1 rather than wait.
>
> ✅ Step 1 done (2026-07-08) — both step-1 cells generated 2026-07-07
> and scored via `compute_stats.py` 2026-07-08 (no `experiments.yaml`
> entries added; launched and generated directly). The `w_eff=0` gap-
> closer (`lam=0.01, ds_alpha=0`) is still not generated. Launch
> command used: `generate_mcts_sem.py --config-name
> mcts_sem_v02_prm800k llm=llama_3b prm=qwen_prm
> search.lam=<lam> search.ds_alpha=<ds_alpha> search.ds_beta=1.0`.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | **0** | **0** | — | planned (step 1: new gap-closing cell) | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 1 | 1 | 2 | scored | .7109<br>±.0284 | .6406<br>±.0300 | .6328<br>±.0302 | .6328<br>±.0302 | — |
| llama-3b | qwen | 0.1 | 0.316 | 1 | 2 | scored | .7109<br>±.0284 | .6641<br>±.0296 | .6562<br>±.0297 | .6484<br>±.0299 | — |
| llama-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7266<br>±.0279 | .6680<br>±.0295 | .6602<br>±.0297 | .6484<br>±.0299 | — |
| llama-3b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .7227<br>±.0280 | .6523<br>±.0298 | .6484<br>±.0299 | .6367<br>±.0301 | 4.90 |
| llama-3b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .7422<br>±.0274 | .6758<br>±.0293 | .6680<br>±.0295 | .6602<br>±.0297 | 4.97 |
| llama-3b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .7305<br>±.0278 | .6641<br>±.0296 | .6836<br>±.0291 | .6602<br>±.0297 | 4.92 |
| llama-3b | qwen | **1.0** | **10** | **10** | **2** | **scored (step 1)** | **.7422<br>±.0274** | **.6562<br>±.0297** | **.6445<br>±.0300** | **.6211<br>±.0304** | — |
| llama-3b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .7578<br>±.0268 | .6719<br>±.0294 | .6602<br>±.0297 | .6289<br>±.0303 | — |
| llama-3b | qwen | **0.01** | **1.0** | **10** | **2** | **scored (step 1)** | **.7500<br>±.0271** | **.6562<br>±.0297** | **.6562<br>±.0297** | **.6523<br>±.0298** | — |
| llama-3b | qwen | 1.0 | 100 | 100 | 2 | scored | .7383<br>±.0275 | .6602<br>±.0297 | .6016<br>±.0307 | .5742<br>±.0310 | — |
| llama-3b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .7812<br>±.0259 | .6562<br>±.0297 | .6211<br>±.0304 | .5938<br>±.0308 | — |
| llama-3b | qwen | 0.01 | 10 | 100 | 2 | scored (see qwen-PRM ds_alpha=10 above) | .7695<br>±.0264 | .6797<br>±.0292 | .6445<br>±.0300 | .6211<br>±.0304 | 5.44 |

> **Analysis.** Two of step 1's three bolded cells are done —
> `w_eff=10` at `lam=1.0` (pass@gb=.7422±.0274) and `lam=0.01`
> (pass@gb=.7500±.0271) agree within SEM (Δ=.008, naive@gb Δ=0,
> both ≪ SEM≈.027-.030). **Decision: `lam`'s independent role is
> negligible** — per the procedure's own rule, the grid collapses to
> a 1D sweep over `w_eff` at `lam=0.01` rather than the full 3×5
> grid. This also confirms the llama-1b table's analysis above
> (same conclusion, smaller model). The third bolded cell
> (`w_eff=0`, `lam=0.01, ds_alpha=0`, the gap-closer) is still not
> generated. Beyond step 1, most of the on-ramp grid at `w_eff=1`,
> `w_eff=3`, and `w_eff=100` also got generated as a side effect and
> is consistent with the no-lam-effect finding: at `w_eff=1`, pass@gb
> is .711/.711/.727 across `lam=1.0/0.1/0.01` (flat within SEM); at
> **`w_eff=3`, .7227/.7422/.7305 (all within ~1 SEM, no trend)**; at
> `w_eff=100`, .738/.781/.770 (a bit more spread, but still within
> ~2×SEM of each other). Unlike llama-1b, the `maj@gb`/`wei@gb`
> metrics at `w_eff=100` don't show the same `lam=1.0` dip here —
> `lam=1.0`'s maj@gb (.574) is the lowest of the three but the spread
> (.574/.594/.621) is milder than llama-1b's (.344/.422/.426). The
> `w_eff=100, lam=0.01` row reuses the already-scored llama-3b
> `ds_alpha=10` cell from the qwen-PRM `ds_alpha sweep (v02, qwen
> PRM)` table above (unchanged).
> **Limitations / follow-up:** with `lam` ruled out as of step 1, the
> remaining `w_eff∈{0.1,0.3,3.0}` on-ramp cells at `lam≠0.01` are low
> priority — only the `w_eff=0` gap-closer (`lam=0.01, ds_alpha=0`)
> and completing the `lam=0.01` 1D sweep across all `w_eff` values
> are worth running next.

#### lam / ds_alpha joint sweep (v02, qwen-math-1.5b)
<!-- table-id: tbl-7491b1 -->
> **Compares:** two `w_eff` checkpoints at `lam=0.1` against the
> existing `lam=0.01` baseline in the `ds_alpha sweep (v02, qwen
> PRM)` table above — a spot-check on qwen-math-1.5b, not a full
> joint sweep like the llama-1b/llama-3b tables (no `lam=1.0` arm
> run here, and no `w_eff=3` or on-ramp cells).
>
> **Fixed:** tmpl=model-family default (native), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .8633<br>±.0215 | .7852<br>±.0257 | .7695<br>±.0264 | .7695<br>±.0264 | 3.94 |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .8867<br>±.0198 | .7930<br>±.0254 | .7578<br>±.0268 | .7500<br>±.0271 | 4.02 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored (see qwen-PRM ds_alpha=10 above) | .8789<br>±.0204 | .7969<br>±.0252 | .7891<br>±.0255 | .7695<br>±.0264 | 3.98 |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .8672<br>±.0213 | .7891<br>±.0255 | .7656<br>±.0265 | .7422<br>±.0274 | 3.92 |

> **Analysis.** Both `lam=0.1` cells (w_eff=10: .8867, w_eff=100:
> .8672) are within ~1 SEM of the `lam=0.01, w_eff=100` baseline
> (.8789) — consistent with the llama-1b/llama-3b tables' finding
> that `lam` has no independent effect once `w_eff` is matched, now
> extended to a third model. Unlike llama-1b (which showed
> maj@gb/wei@gb degrading noticeably at `w_eff=100`), qwen-math-1.5b
> stays flat across all four @gb metrics between `w_eff=10` and
> `w_eff=100` — no sign of the same weighted/majority-vote
> degradation at high diversity weight for this model.
> With the `lam=0.01, w_eff=10` cell now filled (2026-07-09,
> pass@gb .8633), the direct `lam=0.01` vs. `lam=0.1` comparison
> at matched `w_eff=10` is .8633 vs .8867 (+.0234, within ~1 SEM
> of ±.02) — no independent `lam` effect at this `w_eff`, matching
> the pattern above.
> **Limitations / follow-up:** no `lam=1.0` arm run for this model
> (unlike llama-1b/llama-3b), so this isn't a full step-1
> replication — only two `lam` values (0.01, 0.1) tested per
> `w_eff`, treat as a spot-check, not confirmation. Completing the
> `lam=1.0` arm at `w_eff=10` and `100` would make this a proper
> step-1 check like the other two tables.

#### model family, size, quantization comparison
<!-- table-id: tbl-0c4ffd -->
> **Compares:** model family, size, and quantization jointly —
> same shape as cnt-mcts's table above, for cross-method
> comparability.
>
> **Fixed:** bs-4, d-20, b=80, tmpl=model-family default,
> method=`mcts_sem_v02` (PRM embeds), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm) — the project's default path
> (path-identical to exact, proven, see decisions-log.md).
>
> ⚠️ `prm_batch_size` differs by row (1 for llama-3b/gptq/qwen-3b/
> qwen-3b-gptq-int4/qwen-7b-gptq-int4; 2 for llama-1b/
> qwen-math-1.5b — no prmbs-1+rlhflow run exists yet for those
> two), so hr/trial isn't perfectly apples-to-apples across every
> row.
>
> **W&B:** llama-1b `kqn1lj13`, llama-3b `gv2b7ajq`, llama-3b
> gptq `p035tdjs`, qwen-3b `hkrjgbwl`, qwen-3b gptq-int4
> `ekf9b680`, qwen-7b gptq-int4 `f2dhl1ja`, qwen-math-1.5b
> `qn3b8lg0`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5938<br>±.0308 | .4453<br>±.0311 | .4297<br>±.0310 | .4141<br>±.0308 | 4.27 |
| llama-3b fp16 | 2 | scored | .7383<br>±.0275 | .5469<br>±.0312 | .5703<br>±.0310 | .5703<br>±.0310 | 6.61 |
| llama-3b gptq | 2 | scored | .6992<br>±.0287 | .5391<br>±.0312 | .5273<br>±.0313 | .5039<br>±.0313 | 5.71 |
| qwen-3b fp16 | 2 | scored | .8398<br>±.0230 | .6289<br>±.0303 | .7031<br>±.0286 | .6992<br>±.0287 | 5.85 |
| qwen-3b gptq-int4 | 2 | scored | .8242<br>±.0238 | .6836<br>±.0291 | .6992<br>±.0287 | .6914<br>±.0289 | 4.93 |
| qwen-7b gptq-int4 | 2 | scored | .9062<br>±.0183 | .7109<br>±.0284 | .7500<br>±.0271 | .7461<br>±.0273 | 5.02 |
| qwen-math-1.5b fp16 | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7656<br>±.0265 | .7461<br>±.0273 | 4.81 |

> **Analysis.** Accuracy scales with model size/quality as
> expected (llama-1b .594 → llama-3b .738 → qwen-7b .906),
> and qwen-7b gptq-int4 again posts the best score in the table
> despite being quantized — consistent with the cnt-mcts version
> of this comparison. GPTQ's accuracy cost at matched size is
> smaller here than for cnt-mcts: llama-3b gptq trails its fp16
> counterpart by ~4 pts (.699 vs .738) and qwen-3b gptq-int4
> trails by ~1.6 pts (.824 vs .840) — both gaps comfortably
> within noise at n=2.
> **Limitations / follow-up:** the prm_batch_size mismatch (2 vs
> 1) on the llama-1b/qwen-math-1.5b rows still needs a prmbs-1
> rerun before this table supports a fully apples-to-apples
> fp16-vs-GPTQ runtime verdict for sem-mcts. All cells are now
> n=2, so accuracy gaps within ~1 SEM should be read as ties.

#### model family, size, quantization comparison (qwen PRM)
<!-- table-id: tbl-352d94 -->
> **Compares:** the same 7-model family/size/quantization sweep
> as the sem table above, but scored with `prm=qwen`
> (Qwen-Math-7B-PRM) instead of the default `prm=rlhflow`
> (Llama-8B-PRM). Read against that table, it isolates whether
> the model-family ranking (and the GPTQ accuracy/speed
> tradeoff) is robust to the PRM, or specific to rlhflow
> scoring.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), ds_alpha=100, ds_beta=1.0,
> prm_batch_size=1.
>
> ⚠️ All 7 cells scored as of 2026-07-07. **qwen-3b fp16**
> (`cfg-77cae091`) and **qwen-math-1.5b fp16** (`cfg-7a4be169`)
> are read from the pre-fix `--prefix-backup` copies (numbers
> below), not the in-progress precautionary regen described in
> `docs/decisions-log.md` (2026-07-07 entry) — that regen is expected
> to reproduce these exact numbers (verified no-op at existing
> hashes); re-check this row once it lands and is diffed.
>
> **W&B:** fp16 rows are the cfg-`f24283b8`/`2b647a18`/`7a4be169`
> runs (see ds_alpha-sweep-qwen), qwen-3b fp16 `jun56c12`;
> gptq/gptq-int4 rows: llama-3b gptq `u4w3ylt1`, qwen-3b gptq-int4
> `oe1lbvdy`, qwen-7b gptq-int4 `l38fiewz`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .6133<br>±.0305 | .4961<br>±.0313 | .4492<br>±.0311 | .3906<br>±.0306 | 3.90 |
| llama-3b fp16 | 2 | scored | .7656<br>±.0265 | .6562<br>±.0297 | .6289<br>±.0303 | .6016<br>±.0307 | 5.43 |
| llama-3b gptq | 2 | scored | .7148<br>±.0283 | .6094<br>±.0306 | .5625<br>±.0311 | .5078<br>±.0313 | 4.45 |
| qwen-3b fp16 | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7734<br>±.0262 | .7461<br>±.0273 | .7227<br>±.0280 | 5.00 |
| qwen-3b gptq-int4 | 2 | scored | .7930<br>±.0254 | .6953<br>±.0288 | .6953<br>±.0288 | .6875<br>±.0290 | 3.87 |
| qwen-7b gptq-int4 | 2 | scored | .9375<br>±.0152 | .8164<br>±.0242 | .8086<br>±.0246 | .8047<br>±.0248 | 4.20 |
| qwen-math-1.5b fp16 | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7969<br>±.0252 | .7734<br>±.0262 | .7578<br>±.0268 | 3.96 |

> **Analysis.** qwen-7b gptq-int4 is the standout — **.9375**
> pass@gb, comfortably the best in this table and ahead of every
> fp16 row too, echoing the cnt-mcts version of this comparison
> (where qwen-7b gptq-int4 also topped its table). llama-3b gptq
> trails its fp16 counterpart by ~5 pts (.7148 vs .7656); qwen-3b
> gptq-int4 trails qwen-3b fp16 by a larger ~8pt gap (.7930 vs
> .8750) — so quantization's accuracy cost is actually *larger*
> for Qwen-3b here than for Llama-3b, the opposite of what the
> single gptq-int4 vs fp16 comparison alone might have suggested.
> qwen-3b fp16 (.8750) and qwen-math-1.5b fp16 (.8750) tie exactly
> at n=2 despite very different model sizes — worth another look
> once the regen confirms these numbers are stable, not just a
> coincidence of a 2-trial sample.
> **Limitations / follow-up:** qwen-3b fp16 and qwen-math-1.5b
> fp16 are currently read from pre-fix backups pending the
> regen's completion — see `docs/decisions-log.md` 2026-07-07 for why
> and what to verify (should be byte-identical; if not, re-open
> every sem-mcts result for scrutiny). n=2 throughout, so read
> gaps within ~1 SEM as ties.
> Once qwen-3b fp16 lands, the key remaining read is whether
> qwen-3b gptq-int4's accuracy cost (vs fp16) is smaller than
> Llama's, matching or diverging from the rlhflow table's
> pattern.

#### rlhflow vs qwen PRM comparison
<!-- table-id: tbl-b4c266 -->
> **Compares:** `prm.kind` (Llama-8B-PRM "rlhflow" vs
> Qwen-Math-7B-PRM "qwen") — the *scoring* model, not the policy
> LLM. Scoring-side counterpart to the cnt-mcts table of the same
> name. Unlike that table, all three models here (llama-1b,
> llama-3b, qwen-math-1.5b) have a scored qwen-PRM run, since
> v02's `embeds_source=prm` sweep already produced qwen-PRM
> generations at every model.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov=sm, ds_alpha=100.0, ds_beta=1.0 (sem-mcts has
> no `cpuct` — selection is q-value-only on first visit, then a
> ds_alpha/ds_beta-weighted diversity bonus on later visits; see
> `core/mcts_sem_search_v02_00_00.py:select_child`).
>
> ⚠️ `prm_batch_size` differs by row (llama-1b/llama-3b rlhflow
> rows use whatever prmbs the original v02 sweep ran at; every
> other row, including qwen-math-1.5b rlhflow, is prmbs-1) —
> doesn't affect accuracy per the prm_batch_size sweep (cnt-mcts,
> above), so left as-is rather than re-run.
>
> **W&B:** llama-1b rlhflow `kqn1lj13`, llama-1b qwen `j34q0wjq`;
> llama-3b rlhflow `gv2b7ajq`, llama-3b qwen `q4fz58mg`;
> qwen-math-1.5b rlhflow `qn3b8lg0`, qwen-math-1.5b qwen `g1z9k6mk`.

| llm | prm | prmbs | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | rlhflow | 4 | 2 | scored | .5938<br>±.0308 | .4453<br>±.0311 | .4297<br>±.0310 | .4141<br>±.0308 | 4.27 |
| llama-1b | qwen | 1 | 2 | scored | .6133<br>±.0305 | .5156<br>±.0313 | .4414<br>±.0311 | .4180<br>±.0309 | 3.93 |
| llama-3b | rlhflow | 1 | 2 | scored | .7383<br>±.0275 | .5469<br>±.0312 | .5703<br>±.0310 | .5703<br>±.0310 | 6.61 |
| llama-3b | qwen | 1 | 2 | scored | .7500<br>±.0271 | .6797<br>±.0292 | .6133<br>±.0305 | .5977<br>±.0307 | 5.39 |
| qwen-math-1.5b | rlhflow | 1 | 2 | scored | .8789<br>±.0204 | .7461<br>±.0273 | .7656<br>±.0265 | .7461<br>±.0273 | 4.81 |
| qwen-math-1.5b | qwen | 1 | 2 | scored | .8672<br>±.0213 | .7812<br>±.0259 | .7656<br>±.0265 | .7617<br>±.0267 | 3.90 |

> **Analysis.** llama-1b: qwen-PRM scoring edges out rlhflow
> (.6133 vs .5938). llama-3b: qwen-PRM scores higher (.7500 vs
> .7383), now both at 2 trials — a real but modest gap, ~1 SEM.
> qwen-math-1.5b: the two PRMs are within ~1 SEM of each other
> (.8789 vs .8672) — no real separation at this model size. Net:
> qwen-PRM scoring is at least as good as rlhflow at every model
> checked so far, never worse by more than noise. **Runtime:**
> qwen-PRM is faster than rlhflow at llama-1b and llama-3b (3.93
> vs 4.27; 5.39 vs 6.61 hr/trial) — but those rlhflow rows run at
> prmbs-4/1 vs qwen's prmbs-1, and prmbs is a throughput knob
> (prm_batch_size sweep, cnt-mcts, above), so part of that gap is
> the batch-size mismatch, not the PRM itself. At qwen-math-1.5b,
> where both rows are matched at prmbs-1, rlhflow is actually
> *slower* (4.81 vs 3.90 hr/trial) — the opposite direction,
> suggesting the earlier "qwen-PRM is faster" read was largely the
> prmbs confound, not a real per-PRM cost difference.
> **Limitations / follow-up:** n=2 trials per cell throughout —
> this is a lead to firm up with more trials, not a settled
> result. llama-1b/llama-3b need a prmbs-1 rlhflow re-run to
> isolate the runtime comparison from the batch-size confound,
> the way qwen-math-1.5b already is.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
<!-- table-id: tbl-baf795 -->
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. Scoring-side
> counterpart to the cnt-mcts table of the same name. Note:
> sem-mcts's `_generate_candidates` already strips the trailing
> `"\n\n"` before calling `prm.score` (verified — the embed and
> score paths share the same cleaned `candidate_texts`), so unlike
> cnt-mcts this table isn't tied to the `_split_steps` fix; it's a
> fresh sweep, not a rerun.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, ds_alpha=100.0, ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | rlhflow | min | — | to rerun | — | — | — | — | — |
| qwen-3b | rlhflow | prod | — | to rerun | — | — | — | — | — |
| qwen-3b | rlhflow | last | — | to rerun | — | — | — | — | — |
| qwen-3b | qwen | min | — | to rerun | — | — | — | — | — |
| qwen-3b | qwen | prod | — | to rerun | — | — | — | — | — |
| qwen-3b | qwen | last | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7734<br>±.0262 | .7461<br>±.0273 | .7227<br>±.0280 | 5.00 |
| qwen-math-1.5b | rlhflow | min | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | prod | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | last | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | to rerun | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | 2 | scored (pre-fix backup) | .8750<br>±.0207 | .7969<br>±.0252 | .7734<br>±.0262 | .7578<br>±.0268 | 3.96 |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=10)
<!-- table-id: tbl-b1e565 -->
> **Compares:** same `gen.agg_strategy` knob as the table above, at
> `lam=0.1` instead of the default `lam=0.01` — matched `w_eff` (via
> `w_eff = ds_alpha/sqrt(lam)`) rather than matched `ds_alpha`, so
> this is a cross-check on the `agg_strategy` finding under a
> different `lam` operating point, not a new axis.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.1, ds_alpha=3.16 (w_eff=10),
> ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-3b | qwen | min | 2 | scored | .8633<br>±.0215 | .8047<br>±.0248 | .7969<br>±.0252 | .7500<br>±.0271 | 4.92 |
| qwen-3b | qwen | prod | 2 | scored | .8594<br>±.0218 | .7852<br>±.0257 | .8008<br>±.0250 | .7695<br>±.0264 | 4.85 |
| qwen-3b | qwen | last | 2 | scored | .8438<br>±.0227 | .7344<br>±.0277 | .7070<br>±.0285 | .7031<br>±.0286 | 4.90 |
| qwen-math-1.5b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | 2 | scored | .9023<br>±.0186 | .8320<br>±.0234 | .8086<br>±.0246 | .7695<br>±.0264 | 4.08 |
| qwen-math-1.5b | qwen | prod | 2 | scored | .8867<br>±.0198 | .8125<br>±.0244 | .7891<br>±.0255 | .7852<br>±.0257 | 3.99 |
| qwen-math-1.5b | qwen | last | 2 | scored (see lam/ds_alpha joint sweep, qwen-math-1.5b) | .8867<br>±.0198 | .7930<br>±.0254 | .7578<br>±.0268 | .7500<br>±.0271 | 4.02 |

> **Analysis.** 6 of 12 cells scored. qwen-math-1.5b/qwen-PRM: all
> three agg_strategy values filled — `min` (.9023) edges out
> `prod`/`last` (both .8867 exactly) on pass@gb, though all three
> are within ~1 SEM of each other. Naive/wei/maj tell a different
> story: `prod` (.8125/.7891/.7852) clearly beats both `min`
> (.8320/.8086/.7695 — comparable naive, but lower maj) and `last`
> (.7930/.7578/.7500 — lowest of the three on all three metrics) —
> so unlike pass@gb, which is flat across agg_strategy here, the
> non-pass metrics show a real spread with `prod` on top. qwen-3b/
> qwen-PRM now shows the same pattern: pass@gb is flat across all
> three (`min` .8633, `prod` .8594, `last` .8438 — all within ~1
> SEM), but naive/wei/maj favor `min`/`prod` (.80–.80/.79–.80)
> over `last` (.73/.71/.70) — confirming the pass@gb-flat-but-
> naive/wei/maj-spread pattern holds across both models at this
> `w_eff`, not just qwen-math-1.5b.
> **Limitations / follow-up:** 6 of 12 cells still new/unqueued in
> `experiments.yaml` (all rlhflow-PRM cells for both models).

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=100)
<!-- table-id: tbl-db5810 -->
> **Compares:** same as the `w_eff=10` table above, at the next
> `w_eff` checkpoint.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.1, ds_alpha=31.6 (w_eff=100),
> ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-3b | qwen | min | 2 | scored | .8711<br>±.0210 | .7734<br>±.0262 | .7617<br>±.0267 | .7383<br>±.0275 | 4.88 |
| qwen-3b | qwen | prod | 2 | scored | .8672<br>±.0213 | .7930<br>±.0254 | .7773<br>±.0261 | .7305<br>±.0278 | 4.92 |
| qwen-3b | qwen | last | 2 | scored | .8750<br>±.0207 | .8086<br>±.0246 | .7656<br>±.0265 | .7266<br>±.0279 | 4.84 |
| qwen-math-1.5b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | 2 | scored | .8789<br>±.0204 | .8125<br>±.0244 | .7852<br>±.0257 | .7422<br>±.0274 | 4.02 |
| qwen-math-1.5b | qwen | prod | 2 | scored | .8594<br>±.0218 | .7930<br>±.0254 | .7656<br>±.0265 | .7422<br>±.0274 | 3.96 |
| qwen-math-1.5b | qwen | last | 2 | scored (see lam/ds_alpha joint sweep, qwen-math-1.5b) | .8672<br>±.0213 | .7891<br>±.0255 | .7656<br>±.0265 | .7422<br>±.0274 | 3.92 |

> **Analysis.** 6 of 12 cells scored. All three qwen-3b/qwen-PRM
> agg_strategy values are close on pass@gb (.8711/.8672/.8750 —
> within ~1 SEM), matching the flat-pass@gb pattern from the
> `w_eff=10` table, though the naive/wei/maj spread is milder here
> (`last` actually leads maj@gb at .7266 vs. `min`/`prod`'s
> .7383/.7305 — the `w_eff=10` table's "min/prod beat last on
> non-pass metrics" finding does not clearly repeat at this
> `w_eff`). qwen-math-1.5b/qwen-PRM: `min` (.8789) again edges out
> `prod`/`last` (.8594/.8672) on pass@gb, all within ~1 SEM — same
> ranking as `w_eff=10`, but maj@gb is now tied across all three
> (.7422 for all) rather than showing `prod`'s clear lead there.
> qwen-3b's `last` here (.8750) is close to its `w_eff=10`
> counterpart in the table above (.8438, +.031, within ~1.5 SEM) —
> consistent with the general `w_eff` plateau once past ~10.
> **Limitations / follow-up:** 6 of 12 cells still new/unqueued in
> `experiments.yaml` (all rlhflow-PRM cells for both models). The
> `w_eff=10` table's "min/prod beat last on non-pass metrics"
> pattern doesn't clearly hold here — worth flagging as `w_eff`-
> dependent rather than a stable agg_strategy effect once more
> data lands.

#### LLM vs PRM embeds comparison
<!-- table-id: tbl-1eed5c -->
> **Compares:** the diversity-embedding *source* — v01 sources
> from the policy LLM (2nd vLLM engine); v02 sources from the
> PRM. One table per model, at matched template, for the
> head-to-head the project exists for.
>
> **Fixed:** bs-4, d-20, b=80, ds_alpha/ds_beta/lam at v0{1,2}
> defaults. llama-1b/llama-3b use tmpl=custom (match cnt-mcts at
> those models); qwen-math-1.5b uses tmpl=native (cnt-mcts
> Qwen-Math custom has the template bug — match the clean native
> cnt row, 2 trials, for comparability).
>
> ⚠️ **v01 has zero runs of any kind** (see
> [[Algorithm name ↔ code mapping]] above) — every row in all
> three tables below is `planned`. This comparison cannot be made
> yet.
>
> **W&B:** none yet (no runs exist).

##### llama-1b
| method | tmpl | trials | status | pass@gb |
|---|---|---|---|---|
| sem v01 (policy) | custom | — | *planned* | — |
| sem v02 (PRM) | custom | — | *planned* | — |

##### llama-3b
| method | tmpl | trials | status | pass@gb |
|---|---|---|---|---|
| sem v01 (policy) | custom | — | *planned* | — |
| sem v02 (PRM) | custom | — | *planned* | — |

##### qwen-math-1.5b
| method | tmpl | trials | status | pass@gb |
|---|---|---|---|---|
| sem v01 (policy) | native | — | *planned* | — |
| sem v02 (PRM) | native | — | *planned* | — |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** v02 already has scored runs at all
> three models elsewhere in this doc (the model family/
> quantization and rlhflow-vs-qwen-PRM tables above); the missing
> half is v01. Launching v01 at these three model/template combos
> (matched trial count to the v02 rows already on hand) would
> complete this comparison in one batch.

### sem-mcts-v02 [cov_scope=local]

> **Same implementation, one flag.** Everything in this section
> runs `core/mcts_sem_search_v02_00_00.py` — the identical file
> the `sem-mcts-v02` section above uses — with
> `search.cov_scope=local`. No separate method, no separate
> `config_root`, no separate launcher; the cells differ from
> their global twins only by that override (and therefore by
> `config_hash`).
>
> **The reasoning behind the shifted grid lives in the level-5
> doc** (`docs/exp-comp-prm800k-level5.md`, same section
> heading) and is inherited verbatim: under local scope the
> per-node fold count `k` stays small (1–5, tens at the root)
> instead of running to the hundreds, so the diversity bonus
> stays *near* its nominal `w_eff = ds_alpha/sqrt(lam)` for the
> whole run rather than decaying an order of magnitude below
> it. Local's optimum should therefore sit near **`w_eff ≈ 1`**,
> roughly 10x below global's measured optimum of `w_eff = 10`.
> The grid below is shifted down and denser at the low end for
> that reason.
>
> **`lam` is swept at one value only** (0.01), inheriting the
> global joint sweeps' finding that `lam` has no independent
> effect once `w_eff` is fixed. If the local results look
> `lam`-sensitive in a way the global ones did not, that
> assumption is the first thing to revisit.
>
> ⚠️ **This section is the level-4 twin of level-5's
> `cov_scope=local` block.** Every table below has a
> same-model, same-grid counterpart there
> (`tbl-ba6b11`, `tbl-cf849a`, `tbl-b1cb82`, `tbl-5d64b1`,
> `tbl-3a76ce`). The comparison the section exists to serve is
> **does the local operating point transfer across difficulty**
> — level 4 is the easier split, so if the optimum sits at a
> different `w_eff` here than at level 5, `w_eff` has to be
> tuned per difficulty and cannot be fixed once for the
> program.

#### lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)
<!-- table-id: tbl-db0cf7 -->

> **Compares:** level-5's llama-1b local+relative sweep
> (`tbl-ba6b11`) on the easier level-4 split, same seven-point
> grid. At level 5 this model is the one whose `embeds_ref`
> comparison **crossed sign** — `relative` ahead at `w_eff=1`,
> behind at `w_eff=10` — so its curve shape is the most
> informative in the family and the most worth re-measuring
> where the problems are easier.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b, **lam=0.01**, level=4
> (config default), run.num_trials=2.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction.** With `ds_alpha=0` the diversity bonus is
> multiplied by zero, so this cell must reproduce the global
> `w_eff=0` value exactly. Queued here as a plumbing check on
> the relative path.
>
> **W&B:** —

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | 2 | scored | .5469<br>±.0312 | .5078<br>±.0313 | .4883<br>±.0313 | .4375<br>±.0311 | 3.24 |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .5703<br>±.0310 | .5273<br>±.0313 | .4805<br>±.0313 | .4609<br>±.0312 | 3.56 |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .6250<br>±.0303 | .5391<br>±.0312 | .4922<br>±.0313 | .4648<br>±.0312 | 3.58 |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .6289<br>±.0303 | .5469<br>±.0312 | .5078<br>±.0313 | .4648<br>±.0312 | 3.82 |
| llama-1b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .6367<br>±.0301 | .5273<br>±.0313 | .4922<br>±.0313 | .4609<br>±.0312 | 3.82 |
| llama-1b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .6406<br>±.0300 | .5312<br>±.0312 | .4805<br>±.0313 | .4219<br>±.0309 | 3.95 |
| llama-1b | qwen | 0.01 | 10 | 100 | 2 | scored | .6055<br>±.0306 | .5156<br>±.0313 | .4805<br>±.0313 | .4375<br>±.0311 | 3.94 |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** the five tables in this section
> are one experiment, not five — the readable quantity is
> whether each model's peak sits at the same `w_eff` as its
> level-5 twin, so a partially-filled set invites reading a
> model effect that is really a which-cells-finished effect.
> At ~3.8 hr/trial × 2 trials this table is ~53 GPU-hours.
> Feeds key: `tbl-db0cf7`.

#### lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)
<!-- table-id: tbl-43996a -->

> **Compares:** level-5's llama-3b local+relative sweep
> (`tbl-cf849a`) on level 4. `tbl-cf849a` gives this model the
> **clearest interior optimum anywhere in the program** —
> .4440 → .5821 at `w_eff=3` → .5485 by 10, a +.138 span at
> ~4.5 SE. Whether that interior peak survives on an easier
> split is the single sharpest transfer question in the
> section: an interior optimum that moves with difficulty
> cannot be tuned once.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b, **lam=0.01**, level=4
> (config default), run.num_trials=2.
>
> ⚠️ `w_eff=0` is `embeds_ref`-independent by construction; see
> the llama-1b table above.
>
> **W&B:** —

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | 0 | 0 | 2 | scored | .7031<br>±.0286 | .6797<br>±.0292 | .6797<br>±.0292 | .6406<br>±.0300 | 2.98 |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .7266<br>±.0279 | .6602<br>±.0297 | .6367<br>±.0301 | .6406<br>±.0300 | 4.96 |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .7070<br>±.0285 | .6367<br>±.0301 | .6289<br>±.0303 | .6250<br>±.0303 | 4.97 |
| llama-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7617<br>±.0267 | .6797<br>±.0292 | .6602<br>±.0297 | .6406<br>±.0300 | 5.24 |
| llama-3b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7578<br>±.0268 | .6641<br>±.0296 | .6680<br>±.0295 | .6484<br>±.0299 | 5.38 |
| llama-3b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .7812<br>±.0259 | .6289<br>±.0303 | .6406<br>±.0300 | .6133<br>±.0305 | 5.61 |
| llama-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .7852<br>±.0257 | .6523<br>±.0298 | .6250<br>±.0303 | .6211<br>±.0304 | 5.64 |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** the level-5 twin's interior peak
> at `w_eff=3` is the prediction this table tests; a flat grid
> here would mean the peak is a level-5 artifact rather than a
> property of the model. At ~5.0 hr/trial × 2 trials this table
> is ~70 GPU-hours — the most expensive of the five.
> Feeds key: `tbl-43996a`.

#### lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)
<!-- table-id: tbl-ecabc0 -->

> **Compares:** level-5's qwen-3b local+relative sweep
> (`tbl-b1cb82`) on level 4. On AIME2025 at b=320 the qwen-3b
> grid was the one the analysis flagged as **most likely to
> report a null** (no monotone structure, whole spread ~2 SE);
> level 4 is the split where a real effect, if there is one,
> should be easiest to resolve, so this table doubles as the
> test of whether that null is a hardness artifact.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-3b, **lam=0.01**, level=4
> (config default), run.num_trials=2.
>
> ⚠️ `w_eff=0` is `embeds_ref`-independent by construction; see
> the llama-1b table above.
>
> **W&B:** —

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | 0.01 | 0 | 0 | 2 | scored | .7891<br>±.0255 | .7500<br>±.0271 | .7539<br>±.0270 | .7422<br>±.0274 | 3.37 |
| qwen-3b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .8750<br>±.0207 | .7812<br>±.0259 | .7773<br>±.0261 | .7656<br>±.0265 | 4.93 |
| qwen-3b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .8672<br>±.0213 | .7617<br>±.0267 | .7617<br>±.0267 | .7461<br>±.0273 | 5.05 |
| qwen-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .8672<br>±.0213 | .7891<br>±.0255 | .7461<br>±.0273 | .7305<br>±.0278 | 5.03 |
| qwen-3b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .8711<br>±.0210 | .7812<br>±.0259 | .7461<br>±.0273 | .7188<br>±.0282 | 4.97 |
| qwen-3b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .8828<br>±.0201 | .7695<br>±.0264 | .7734<br>±.0262 | .7539<br>±.0270 | 5.04 |
| qwen-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .8672<br>±.0213 | .7539<br>±.0270 | .7305<br>±.0278 | .7227<br>±.0280 | 5.09 |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** if this grid is flat at level 4
> too, qwen-3b is the model to drop from future local sweeps —
> two nulls at two difficulties is enough. At ~4.9 hr/trial ×
> 2 trials this table is ~69 GPU-hours.
> Feeds key: `tbl-ecabc0`.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)
<!-- table-id: tbl-e6a2f9 -->

> **Compares:** level-5's qwen-7b local+relative sweep
> (`tbl-5d64b1`) on level 4, on the strongest policy in the
> grid. At level 5 `relative` peaked at `w_eff=1` (.7836) and
> decayed monotonically to .7537 by 100 — a *left* optimum.
> The AIME2025 b=320 counterpart is still climbing at
> `w_eff=3` (.3667) with 100 in flight, i.e. pointing *right*.
> Level 4 sits between those two difficulties and should say
> which way the optimum actually moves with hardness.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4, **lam=0.01**,
> level=4 (config default), run.num_trials=2.
>
> ⚠️ `w_eff=0` is `embeds_ref`-independent by construction; see
> the llama-1b table above.
>
> **W&B:** —

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | 0 | 0 | 2 | scored | .8008<br>±.0250 | .7852<br>±.0257 | .7734<br>±.0262 | .7461<br>±.0273 | 1.76 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .8945<br>±.0192 | .8164<br>±.0242 | .8281<br>±.0236 | .8164<br>±.0242 | 3.64 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .8984<br>±.0189 | .7969<br>±.0252 | .8164<br>±.0242 | .8086<br>±.0246 | 3.86 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | 2 | scored | .9141<br>±.0176 | .7891<br>±.0255 | .8242<br>±.0238 | .8164<br>±.0242 | 4.12 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3 | 2 | scored | .9219<br>±.0168 | .8047<br>±.0248 | .8125<br>±.0244 | .7930<br>±.0254 | 4.26 |
| qwen-7b gptq-int4 | qwen | 0.01 | 1.0 | 10 | 2 | scored | .9062<br>±.0183 | .8242<br>±.0238 | .8008<br>±.0250 | .7969<br>±.0252 | 4.25 |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | 2 | scored | .9219<br>±.0168 | .7773<br>±.0261 | .7930<br>±.0254 | .7891<br>±.0255 | 4.32 |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** this is the one table in the
> section with a *contradiction* to resolve rather than a
> prediction to confirm — level 5 says the optimum is left,
> AIME2025 says right. At ~4.2 hr/trial × 2 trials it is
> ~59 GPU-hours. Feeds key: `tbl-e6a2f9`.

#### lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)
<!-- table-id: tbl-c76d49 -->

> **Compares:** level-5's qwen-math-1.5b local+relative sweep
> (`tbl-3a76ce`) on level 4. This model is the family outlier
> twice over: math-specialized embeddings, and the one level-5
> model whose measured `relative` points peak at `w_eff=10`
> rather than low. On AIME2025 b=320 it posted **.4000**, the
> best pass@gb in that doc, also at a high `w_eff`. If its peak
> stays right at level 4 while the others sit near 1, the
> per-model tuning story is about the embedding space, not the
> difficulty.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b, **lam=0.01**, level=4
> (config default), run.num_trials=2.
>
> ⚠️ `w_eff=0` is `embeds_ref`-independent by construction; see
> the llama-1b table above.
>
> **W&B:** —

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | 0 | 0 | 2 | scored | .8203<br>±.0240 | .7891<br>±.0255 | .7734<br>±.0262 | .7383<br>±.0275 | 2.83 |
| qwen-math-1.5b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .9141<br>±.0176 | .7930<br>±.0254 | .8008<br>±.0250 | .8047<br>±.0248 | 4.06 |
| qwen-math-1.5b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .9141<br>±.0176 | .7812<br>±.0259 | .8047<br>±.0248 | .7969<br>±.0252 | 4.06 |
| qwen-math-1.5b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .8984<br>±.0189 | .7812<br>±.0259 | .7891<br>±.0255 | .7812<br>±.0259 | 4.11 |
| qwen-math-1.5b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .8750<br>±.0207 | .7852<br>±.0257 | .7812<br>±.0259 | .7617<br>±.0267 | 4.11 |
| qwen-math-1.5b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .8828<br>±.0201 | .7500<br>±.0271 | .7695<br>±.0264 | .7461<br>±.0273 | 3.99 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored | .8594<br>±.0218 | .7852<br>±.0257 | .7656<br>±.0265 | .7500<br>±.0271 | 4.05 |

> **Analysis.** No data yet — nothing to take away.
> **Limitations / follow-up:** cheapest table in the section at
> ~4.0 hr/trial × 2 trials ≈ 56 GPU-hours, and the one whose
> result is most likely to differ from the other four — queue
> it even if the section gets cut for compute.
> Feeds key: `tbl-c76d49`.

### cnt-mcts-bl-v01
> knobs: template, cpuct (bs-4, d-20 fixed). method=`mcts_bl_cnt_v01`.
> No cpuct sweep yet — every row is the default 2.0. Same
> selection rule as cnt-mcts: the Summary above promotes
> whichever row scores highest on **pass@gb** across all
> knobs jointly. (`num_phases` cap exists but isn't a tuned
> knob yet — open backlog question in
> `llm-reasoning-mcts-bl-exp-todo` on whether to keep it,
> replace it, or remove it.)

#### model family, size, quantization comparison (qwen PRM)
<!-- table-id: tbl-deb9f9 -->
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts (updated)'s equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled. All 7 cells are new for bl_cnt_v01 with the
> qwen PRM.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (the
> new default — see `generate_mcts_bl_cnt.py`/
> `BLMCTSCntConfig` alignment fix), tmpl=model-family default
> (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .4414<br>±.0311 | .4297<br>±.0310 | .3984<br>±.0307 | .3789<br>±.0304 | 2.12 |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .6445<br>±.0300 | .6328<br>±.0302 | .6172<br>±.0304 | .6094<br>±.0306 | 3.50 |
| qwen-3b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | 2 | scored | .8125<br>±.0244 | .7578<br>±.0268 | .7461<br>±.0273 | .7422<br>±.0274 | 2.78 |
| qwen-math-1.5b fp16 | 2 | scored | .6836<br>±.0291 | .6562<br>±.0297 | .6602<br>±.0297 | .6562<br>±.0297 | 2.75 |

> **Analysis.** 4 of 7 cells now scored (2026-07-09): llama-1b
> .4414, qwen-3b fp16 .6445, qwen-7b gptq-int4 .8125,
> qwen-math-1.5b fp16 .6836. Every scored bl_cnt cell trails its
> cnt-mcts counterpart at the same model/qwen-PRM config (llama-1b
> .6367, qwen-3b .8789, qwen-7b gptq-int4 .9102, qwen-math-1.5b
> .8906 — see the `cnt-mcts` model family/size/quantization
> (qwen PRM) table above) — the gap ranges from -.0964
> (qwen-math-1.5b) to -.2344 (qwen-3b fp16), consistently large
> and one-directional across all 4 models tested so far, not
> just the llama-1b point noted previously. This is consistent
> with the earlier (now-removed) rlhflow-PRM finding and with
> the ~18% zero-completion rate documented for bl_cnt_v01 at this
> budget (see `docs/findings/` and the single-question trace in
> `docs/benchmarks.md` — the same question produced 0 completions
> for bl_cnt_v01 at b=80): a frontier-selection run that exhausts
> its budget without completing loses all credit for that
> question, which plausibly explains bl_cnt's across-the-board
> pass@gb deficit against cnt-mcts's depth-first-with-backup
> selection.
> **Limitations / follow-up:** llama-3b fp16, llama-3b gptq, and
> qwen-3b gptq-int4 still unqueued in `experiments.yaml`. Given
> the now-4-model consistent underperformance, worth deciding
> whether to (a) finish the remaining 3 cells anyway for a
> complete grid, or (b) prioritize investigating/fixing the
> zero-completion issue before spending more budget on bl_cnt_v01
> sweeps — check `llm-reasoning-mcts-bl-exp-todo`.

### kube-mcts-bl-v01
> knobs: template, kube_c, kube_schedule, kube_affordable (bs-4,
> d-20 fixed). method=`mcts_bl_kube_v01`. No kube_c sweep
> yet — every row is the default (kube_c=2.0, kube_schedule=parent,
> kube_affordable=true). Same best-first frontier as
> cnt-mcts-bl-v01, but selects by fractional-KUBE density (a UCB
> confidence bonus divided by remaining cost) instead of PUCT,
> following Tran-Thanh et al. arXiv:1204.1909 sec. 3.3. See
> `docs/algorithms.md` ("BL-KUBE-MCTS") and
> `docs/decisions/bl-kube-bonus-schedule.md` /
> `docs/decisions/kube-affordability-restriction.md` for the
> algorithm and its schedule/feasibility design.

#### model family, size, quantization comparison (qwen PRM)
<!-- table-id: tbl-fbd467 -->
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled. All 7 cells are new for bl_kube_v01.
>
> **Fixed:** method=`mcts_bl_kube_v01`, prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent (default — UCT-style
> local clock, matches v01's PUCT bonus so the v01-vs-v02
> comparison isolates cost normalization; see
> `docs/decisions/bl-kube-bonus-schedule.md`), kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5586<br>±.0311 | .5117<br>±.0313 | .4688<br>±.0312 | .4531<br>±.0312 | 2.30 |
| llama-3b fp16 | 2 | scored | .7305<br>±.0278 | .6602<br>±.0297 | .6367<br>±.0301 | .6211<br>±.0304 | 3.55 |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .8320<br>±.0234 | .7617<br>±.0267 | .7422<br>±.0274 | .7344<br>±.0277 | 3.31 |
| qwen-3b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | 2 | scored | .8750<br>±.0207 | .7930<br>±.0254 | .7852<br>±.0257 | .7656<br>±.0265 | 2.58 |
| qwen-math-1.5b fp16 | 2 | scored | .8359<br>±.0232 | .7773<br>±.0261 | .7617<br>±.0267 | .7695<br>±.0264 | 2.71 |

> **Analysis.** 5 of 7 cells now scored (2026-07-09): llama-1b
> .5586, llama-3b fp16 .7305, qwen-3b fp16 .8320, qwen-7b
> gptq-int4 .8750, qwen-math-1.5b .8359. Comparing to v01's
> equivalent table above at the 4 shared models (llama-1b, qwen-3b
> fp16, qwen-7b gptq-int4, qwen-math-1.5b), v02 (fractional-KUBE
> density) beats v01 (PUCT) on every one: +.1172 (llama-1b, .5586
> vs .4414), +.1875 (qwen-3b, .8320 vs .6445), +.0625 (qwen-7b
> gptq-int4, .8750 vs .8125), +.1523 (qwen-math-1.5b, .8359 vs
> .6836) — consistent, one-directional, and the opposite sign of
> the bl_cnt-vs-cnt gap noted above. Plausibly the cost-normalized
> `(q+bonus)/cost` density (v02) is less prone to the
> budget-exhaustion / zero-completion failure mode than PUCT's
> `q+bonus` (v01), since density naturally discounts deep/
> expensive-to-finish nodes rather than treating them as equally
> attractive regardless of remaining budget.
> **Limitations / follow-up:** llama-3b gptq and qwen-3b
> gptq-int4 still unqueued in `experiments.yaml`. No
> `kube_c`/`kube_schedule`/`kube_affordable` sweep yet — every row
> is the same fixed point until one exists.

### kdepth-mcts-bl-v01
> knobs: template, depth_beta, depth_alpha (bs-4, d-20 fixed).
> method=`mcts_bl_kdepth_v01`. No
> depth_beta/depth_alpha sweep yet — every row is the default
> (depth_beta=2.0, depth_alpha=1.0). Sibling of cnt-mcts-bl-v01
> (PUCT) and kube-mcts-bl-v01 (Fractional KUBE): same best-first
> frontier / knapsack-style selection and cost mapping as both,
> but the leaf-selection bonus is a fixed depth-preference
> function instead of a confidence bound — no
> visit-count/exploration term, no bandit/regret guarantee. See
> `docs/algorithms.md` ("BL-KDEPTH-MCTS") and
> `docs/decisions/bl-kdepth-knapsack-bonus.md` for the algorithm
> and the sign-correction note (`f_a` is indexed on depth
> fraction, not cost fraction, so it favors shallow nodes as
> intended).

#### model family, size, quantization comparison (qwen PRM)
<!-- table-id: tbl-7367f8 -->
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled. All 7 cells are new for bl_kdepth_v01.
>
> **Fixed:** method=`mcts_bl_kdepth_v01`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5742<br>±.0310 | .5430<br>±.0312 | .5117<br>±.0313 | .4883<br>±.0313 | 2.21 |
| llama-3b fp16 | 2 | scored | .7227<br>±.0280 | .6680<br>±.0295 | .6758<br>±.0293 | .6445<br>±.0300 | 3.25 |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .8164<br>±.0242 | .7539<br>±.0270 | .7461<br>±.0273 | .7344<br>±.0277 | 3.00 |
| qwen-3b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | 2 | scored | .9023<br>±.0186 | .8281<br>±.0236 | .8320<br>±.0234 | .8203<br>±.0240 | 2.43 |
| qwen-math-1.5b fp16 | 2 | scored | .8164<br>±.0242 | .7422<br>±.0274 | .7461<br>±.0273 | .7461<br>±.0273 | 2.58 |

> **Analysis.** 5 of 7 cells now scored (2026-07-09): llama-1b
> .5742, llama-3b fp16 .7227, qwen-3b fp16 .8164, qwen-7b
> gptq-int4 .9023, qwen-math-1.5b .8164. Comparing to v01's
> equivalent table above at the 4 shared models, v03
> (depth-shaping bonus, no exploration term) also beats v01
> (PUCT) on every one: +.1328 (llama-1b, .5742 vs .4414), +.1719
> (qwen-3b, .8164 vs .6445), +.0898 (qwen-7b gptq-int4, .9023 vs
> .8125), +.1328 (qwen-math-1.5b, .8164 vs .6836) — same
> direction and similar magnitude to v02's gap over v01 above.
> v03 also edges out v02 on 3 of 4 shared models (llama-1b .5742
> vs .5586, qwen-7b gptq-int4 .9023 vs .8750, qwen-math-1.5b
> .8164 vs .8359 is the one exception, a small -.0195 dip), so
> the smoke-test worry about depth-shaping concentrating budget
> exhaustion at `max_depth` doesn't show up as a pass@gb deficit
> here — if anything, a fixed depth preference does at least as
> well as an evidence-based UCB bonus at this budget, though with
> no regret guarantee to fall back on off-distribution.
> **Limitations / follow-up:** llama-3b gptq and qwen-3b
> gptq-int4 still unqueued in `experiments.yaml`. No
> depth_beta/depth_alpha sweep yet — every row is the same fixed
> point until one exists.

### sem-mcts-bl
> knobs: model family/size/quantization (this table); lam,
> ds_alpha, ds_alpha_schedule not yet swept — every row below
> is the same fixed point. method=`mcts_bl_sem_v01`
> (`core/mcts_bl_sem_search_v01_00_00.py`), best-first frontier
> selection with the sem family's diversity-adjusted value
> (frontier counterpart of sem-mcts-v02, as cnt-mcts-bl-v01 is to
> cnt-mcts). Run from `generate_mcts_sem.py`,
> `algo=mcts_bl_sem_v01`. See `docs/algorithms.md`
> ("BL-Sem-MCTS") and `docs/decisions-log.md` (2026-07-08) for
> the algorithm and its `ds_alpha_schedule` design.

#### model family, size, quantization comparison (qwen PRM, w_eff=100)
<!-- table-id: tbl-ed6194 -->
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled. All 7 cells are new.
>
> **Fixed:** method=`mcts_bl_sem_v01`, prm=qwen (both scoring
> AND diversity embeds — `embeds_source=prm` is the schema
> default, no second pooling engine), agg_strategy=`last`, bs-4,
> d-20, b=80, prm_batch_size=1, `ds_alpha_schedule=global`
> (default — see decisions-log), `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse` (defaults), tmpl=
> model-family default (native for Qwen, custom for Llama).
> **lam=0.1, ds_alpha=31.6** (`w_eff = ds_alpha/sqrt(lam) = 100`
> — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row; `ds_beta=1.0` fixed throughout, so only the
> ratio matters).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5195<br>±.0313 | .4219<br>±.0309 | .3242<br>±.0293 | .2422<br>±.0268 | 5.18 |
| llama-3b fp16 | 0/2 | failed | — | — | — | — | — |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .8320<br>±.0234 | .6836<br>±.0291 | .6484<br>±.0299 | .6016<br>±.0307 | 5.19 |
| qwen-3b gptq-int4 | 2 | scored | .7422<br>±.0274 | .6133<br>±.0305 | .5625<br>±.0311 | .5273<br>±.0313 | 4.18 |
| qwen-7b gptq-int4 | 2 | scored | .8906<br>±.0195 | .7500<br>±.0271 | .7109<br>±.0284 | .6953<br>±.0288 | 4.15 |
| qwen-math-1.5b fp16 | 2 | scored | .8320<br>±.0234 | .6992<br>±.0287 | .6445<br>±.0300 | .6484<br>±.0299 | 4.19 |

> **Analysis.** 5 of 7 cells now scored (2026-07-09): llama-1b .5195,
> qwen-3b .8320, qwen-3b-gptq-int4 .7422, qwen-7b-gptq-int4 .8906,
> qwen-math-1.5b .8320. Comparing to the `w_eff=10` table below at
> the same models, the higher diversity weight (`w_eff=100`) scores
> lower on every shared model so far — consistent with
> [ds-alpha-diversity-bonus-plateau.md](findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)'s
> finding that turning the diversity bonus on matters but pushing its
> magnitude past `ds_alpha≈10` (`w_eff≈32` at `lam=0.1`) does not help
> further and may hurt on the frontier-selection (bl_sem) variant
> specifically — this hadn't been checked for bl_sem before.
> **Limitations / follow-up:** llama-3b fp16 FAILED (config-hash
> `0f06296f`, run `2goolnzd`): crashed during trial 0 with vLLM's
> "decoder prompt (length 5000) ... longer than the maximum model
> length of 5000" — a deep frontier path filled llama_3b's
> `max_model_len=5000` and the search has no context-length guard,
> so the unhandled ValueError killed the trial (0/2 on disk, empty
> result dir). Same root cause as the w_eff=10 llama-3b failure
> below; needs a length guard (or larger max_model_len) before
> rerunning. llama-3b gptq still unqueued. `lam`/
> `ds_alpha_schedule` are fixed at one point (`global` schedule) —
> no sweep along those axes yet for this algorithm.

#### model family, size, quantization comparison (qwen PRM, w_eff=10)
<!-- table-id: tbl-7fec69 -->
> **Compares:** same 7-model/quant grid as the `w_eff=100` table
> above, at one order of magnitude lower effective diversity
> weight — the two tables together give a first (coarse) read on
> whether the model-family ranking is sensitive to `w_eff` for
> this algorithm, ahead of a proper `w_eff` sweep.
>
> **Fixed:** identical to the `w_eff=100` table above (method=
> `mcts_bl_sem_v01`, prm=qwen, agg_strategy=`last`, bs-4, d-20,
> b=80, prm_batch_size=1, `ds_alpha_schedule=global`,
> `cov_update=sm`, `embeds_dim=512`/`embeds_proj=sparse`, tmpl=
> model-family default) except the diversity weight.
> **lam=0.1, ds_alpha=3.16** (`w_eff = ds_alpha/sqrt(lam) = 10`
> — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .5859<br>±.0308 | .5078<br>±.0313 | .4766<br>±.0313 | .4336<br>±.0310 | 4.57 |
| llama-3b fp16 | 1/2 | failed | — | — | — | — | — |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .8477<br>±.0225 | .7734<br>±.0262 | .7461<br>±.0273 | .7422<br>±.0274 | 5.25 |
| qwen-3b gptq-int4 | 2 | scored | .8086<br>±.0246 | .7266<br>±.0279 | .7031<br>±.0286 | .6992<br>±.0287 | 4.05 |
| qwen-7b gptq-int4 | 2 | scored | .9102<br>±.0179 | .8008<br>±.0250 | .8008<br>±.0250 | .7930<br>±.0254 | 4.23 |
| qwen-math-1.5b fp16 | 2 | scored | .8555<br>±.0220 | .7734<br>±.0262 | .7461<br>±.0273 | .7383<br>±.0275 | 4.15 |

> **Analysis.** 5 of 7 cells now scored (2026-07-09): llama-1b .5859,
> qwen-3b .8477, qwen-3b-gptq-int4 .8086, qwen-7b-gptq-int4 .9102,
> qwen-math-1.5b .8555. Comparing to the `w_eff=100` table above at
> the 4 shared models, `w_eff=10` scores higher on all 4 (.5859 vs
> .5195 llama-1b, .8477 vs .8320 qwen-3b, .8086 vs .7422
> qwen-3b-gptq-int4, .8555 vs .8320 qwen-math-1.5b) — consistent with
> [ds-alpha-diversity-bonus-plateau.md](findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)'s
> plateau-onset framing: `w_eff=10` (right at the plateau's onset in
> sem_v02's rlhflow-PRM data) outperforms `w_eff=100` (well past it)
> here too, so bl_sem's plateau appears to sit in a similar place, at
> least directionally on these first data points.
> **Limitations / follow-up:** llama-3b fp16 FAILED (config-hash
> `3ca318f6`, run `yf562ig8`): trial 0 completed (6.19 hr) but
> trial 1 crashed mid-search with vLLM's "decoder prompt (length
> 5000) ... longer than the maximum model length of 5000" — a deep
> frontier path filled llama_3b's `max_model_len=5000` and the
> search has no context-length guard, so the unhandled ValueError
> killed the trial (1/2 on disk, `missing trials, skipped: [1]` in
> compute_stats). The w_eff=100 llama-3b cell above (`0f06296f`,
> run `2goolnzd`, 0/2) died the same way ~80s after launch of the
> same sweep — same root cause, one failure per cell, NOT two
> attempts at one cell (earlier note here said otherwise). Needs a
> length guard (or larger max_model_len) before rerunning. llama-3b
> gptq still unqueued. Not yet a full 7-cell grid or a real `w_eff`
> sweep — just two coarse points.

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

### cnt-mcts

#### model family comparison (b=320, qwen PRM)
<!-- table-id: tbl-4e21d6 -->
> **Compares:** a 5-model family/size sweep (llama-1b, llama-3b
> fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b — GPTQ
> variants llama-3b gptq and qwen-3b gptq-int4 excluded, out of
> scope for this table) at `search.gen_budget=320` (4× the b=80
> budget) with `prm=qwen_prm` instead of the b=80 table's default
> `llama_prm`. Two axes change at once — budget and PRM — so
> this table isn't a clean isolation of either; it answers
> "does the b=80 ranking across model family/size hold at a much
> larger search budget under qwen scoring," not "what does budget
> alone do." A matched-PRM (llama) b=320 row per model would be
> needed to separate the two effects.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, b=320, prm=qwen,
> **`llm.max_model_len=5000`**,
> tmpl=model-family default (native for Qwen, custom for Llama).
>
> ⚠️ **`max_model_len` is not uniform across these rows:**
> qwen-math-1.5b ran at **4096**, the other four at 5000 (verified
> in each cell's manifest). Qwen2.5-Math is architecturally capped
> at `max_position_embeddings=4096` and vLLM raises rather than
> clamps above it, so 5000 is unreachable for that model. Context
> window is therefore a second uncontrolled variable on the
> qwen-math row; it only bites on prompts approaching the cap, so
> treat it as a caveat for long-chain problems rather than a
> blanket confound. Details:
> [findings/coding-findings/qwen-math-4096-context-cap.md](findings/coding-findings/qwen-math-4096-context-cap.md).
>
> ✅ All 5 cells scored (2026-07-13), via `method=mcts_cnt_v01`
> (post-fix). Budget=320 is a 4× generation-count increase over
> the b=80 table — confirmed roughly 3-5× the b=80 per-trial
> wall-clock (e.g. qwen-7b gptq-int4 was 3.21 hr/trial at b=80,
> now 9.91 hr/trial at b=320).
>
> **W&B:** llama-1b `qlsp5tx6`, llama-3b `f32gf4ld`, qwen-3b
> `n1mez9rc`, qwen-7b gptq-int4 `4sm44tcf`, qwen-math-1.5b
> `uss9vu4b`.

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .7539<br>±.0270 | .5820<br>±.0309 | .5195<br>±.0313 | .4688<br>±.0312 | 9.39 |
| llama-3b fp16 | qwen | 2 | scored | .8711<br>±.0210 | .6875<br>±.0290 | .6758<br>±.0293 | .6523<br>±.0298 | 15.49 |
| qwen-3b fp16 | qwen | 2 | scored | .9297<br>±.0160 | .8125<br>±.0244 | .7734<br>±.0262 | .7539<br>±.0270 | 14.44 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .9492<br>±.0137 | .7969<br>±.0252 | .8203<br>±.0240 | .8047<br>±.0248 | 9.91 |
| qwen-math-1.5b fp16 | qwen | 2 | scored | .9453<br>±.0142 | .7891<br>±.0255 | .7969<br>±.0252 | .7891<br>±.0255 | 11.01 |

> **Analysis.** 5/5 cells scored. The b=80 ranking holds
> qualitatively: qwen-7b gptq-int4 is again the top pass@gb
> (.9492), llama-1b again the weakest (.7539) — model-family/size
> effects survive the 4× budget increase. Notably, absolute
> pass@gb rises across the board vs. the b=80/qwen-PRM table (e.g.
> llama-1b .7539 here vs its b=80 qwen-PRM row), consistent with
> more search budget helping regardless of model. hr/trial scales
> super-linearly for the two 3B-class models (llama-3b 15.49,
> qwen-3b 14.44 — both ~4.7-4.8× their b=80 figures) but sub-4×
> for qwen-7b gptq-int4 (9.91, ~3.1×) and qwen-math-1.5b (11.01);
> GPTQ's speed advantage narrows at b=320 relative to fp16 3B.
> **Limitations / follow-up:** budget and PRM both differ from the
> b=80 table at once; a matched-PRM b=320 row would isolate the
> budget effect alone.

### sem-mcts-v02

#### model family comparison (b=320, qwen PRM, lam=0.1, w_eff=10)
<!-- table-id: tbl-e144a5 -->
> **Compares:** a 5-model family/size sweep (llama-1b, llama-3b
> fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b — GPTQ
> variants llama-3b gptq and qwen-3b gptq-int4 excluded, out of
> scope for this table) as the `[gen_budget=80]` sem-mcts (qwen
> PRM) table above, but at `search.gen_budget=320` (4× the b=80
> budget) and at `lam=0.1, ds_alpha=3.16` (`w_eff =
> ds_alpha/sqrt(lam) = 10`) instead of that table's default point
> (`lam=0.01, ds_alpha=100`, i.e. `w_eff=1000`). Three axes move
> at once relative to that b=80 table — budget, lam, and
> ds_alpha — so this isn't a clean isolation of any one of them;
> paired with the `w_eff=100` table below (same budget, same lam,
> 10× ds_alpha) it does isolate `w_eff` at b=320.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=320, prm_batch_size=1, **`llm.max_model_len=5000`**,
> `ds_alpha_schedule=global`
> (default), `cov_update=sm`, `embeds_dim=512`/
> `embeds_proj=sparse` (defaults), tmpl=model-family default
> (native for Qwen, custom for Llama). **lam=0.1, ds_alpha=3.16**
> (`w_eff=10` — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row, same point used by the `sem-mcts-bl` w_eff=10
> table).
>
> ⚠️ **`max_model_len` is not uniform across these rows:**
> qwen-math-1.5b ran at **4096**, the other four at 5000 (verified
> in each cell's manifest). That is not a config slip —
> Qwen2.5-Math is architecturally capped at
> `max_position_embeddings=4096`, and vLLM raises rather than
> clamps above it, so 5000 is unreachable for that model. Context
> window is therefore a second uncontrolled variable on the
> qwen-math row; it only bites on prompts approaching the cap, so
> treat it as a caveat for long-chain problems rather than a
> blanket confound. Details:
> [findings/coding-findings/qwen-math-4096-context-cap.md](findings/coding-findings/qwen-math-4096-context-cap.md).
>
> ✅ All 5 cells scored (2026-07-13). Budget=320 is a 4×
> generation-count increase over the b=80 table; per-trial
> wall-clock landed at roughly 5-9× the b=80/w_eff=10 figures
> (well above the naive 4× expectation — see Analysis).
>
> **W&B:** llama-1b `pzu1ri27`, llama-3b `mek0jor8`, qwen-3b
> `pw6som82`, qwen-7b gptq-int4 `q6yjispf`, qwen-math-1.5b
> `lywharje`.

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .7383<br>±.0275 | .5898<br>±.0308 | .5508<br>±.0311 | .5195<br>±.0313 | 13.38 |
| llama-3b fp16 | qwen | 2 | scored | .8438<br>±.0227 | .6875<br>±.0290 | .6797<br>±.0292 | .6797<br>±.0292 | 16.54 |
| qwen-3b fp16 | qwen | 2 | scored | .9219<br>±.0168 | .7812<br>±.0259 | .7773<br>±.0261 | .7656<br>±.0265 | 18.02 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .9336<br>±.0156 | .8242<br>±.0238 | .7891<br>±.0255 | .7812<br>±.0259 | 9.77 |
| qwen-math-1.5b fp16 | qwen | 2 | scored | .9375<br>±.0152 | .8125<br>±.0244 | .8320<br>±.0234 | .8242<br>±.0238 | 15.35 |

> **Analysis.** 5/5 cells scored. Model-family/size ranking holds:
> qwen-7b gptq-int4 and qwen-math-1.5b lead pass@gb (.9336/.9375),
> llama-1b again weakest (.7383). hr/trial did not scale as the
> naive 4× estimate suggested — qwen-3b (18.02) and llama-3b
> (16.54) are close to 5× their expected b=80 figures, while
> qwen-7b gptq-int4 (9.77) stays well under 4×, again showing
> GPTQ's relative speed advantage widening rather than narrowing
> at this budget/lam point (contrast with the cnt-mcts b=320 table
> above, where GPTQ's edge narrowed).
> **Limitations / follow-up:** three axes (budget, lam, ds_alpha)
> differ from the b=80 default-point table above at once; no
> matched-budget b=80 row at this exact `lam=0.1, w_eff=10` point
> exists for sem-mcts (only for `sem-mcts-bl`), so a clean
> single-axis isolation isn't possible yet.

#### model family comparison (b=320, qwen PRM, lam=0.1, w_eff=100)
<!-- table-id: tbl-179d62 -->
> **Compares:** identical setup to the `w_eff=10` table above
> (same 5-model scope — GPTQ variants llama-3b gptq and qwen-3b
> gptq-int4 excluded, out of scope for this table), at
> `ds_alpha=31.6` instead of `3.16` (10× the diversity weight,
> same `lam=0.1`) — the b=320 counterpart of the `sem-mcts-bl`
> w_eff=100 table, and the paired point needed to isolate `w_eff`
> alone at this budget.
>
> **Fixed:** identical to the `w_eff=10` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> **`llm.max_model_len=5000`**,
> `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse`, tmpl=model-family
> default) except the diversity weight. **lam=0.1,
> ds_alpha=31.6** (`w_eff=100`).
>
> ⚠️ **`max_model_len` is not uniform across these rows:**
> qwen-math-1.5b ran at **4096**, llama-1b at 5000 (verified in
> both manifests) — and the same will hold for the three unrun
> rows. Qwen2.5-Math is architecturally capped at
> `max_position_embeddings=4096` and vLLM raises rather than
> clamps above it, so 5000 is unreachable for that model. This
> matters for the w_eff read below, which compares exactly these
> two models across tables: both carry their own window
> consistently at w_eff=10 and w_eff=100, so the `w_eff`
> comparison per model is unaffected — but llama-1b↔qwen-math
> comparisons within this table differ in window as well as model.
> Details:
> [findings/coding-findings/qwen-math-4096-context-cap.md](findings/coding-findings/qwen-math-4096-context-cap.md).
>
> ✅ 5 of 5 cells scored (llama-3b completed 2026-07-29). The
> earlier ⚠️ on this table said 2 of 5 and named llama-3b,
> qwen-3b and qwen-7b gptq-int4 as unrun; qwen-3b and qwen-7b
> were in fact filled after it was written and the note was never
> updated. llama-3b took two attempts (stalled on job 23376655
> with 1/2 trials, resumed and finished on 23419813).
>
> **W&B:** llama-1b `ufv99olb`, llama-3b `7c79wk6z`, qwen-3b
> `roclobp8`, qwen-7b gptq-int4 `c1ybuwgg`, qwen-math-1.5b
> `a2i0we44`.

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .7148<br>±.0283 | .5508<br>±.0311 | .4492<br>±.0311 | .4375<br>±.0311 | 16.18 |
| llama-3b fp16 | qwen | 2 | scored | .8320<br>±.0234 | .6680<br>±.0295 | .6406<br>±.0300 | .6211<br>±.0304 | 20.54 |
| qwen-3b fp16 | qwen | 2 | scored | .9336<br>±.0156 | .7969<br>±.0252 | .7539<br>±.0270 | .7461<br>±.0273 | 18.12 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .9375<br>±.0152 | .8086<br>±.0246 | .7891<br>±.0255 | .7734<br>±.0262 | 10.99 |
| qwen-math-1.5b fp16 | qwen | 2 | scored | .9453<br>±.0142 | .8242<br>±.0238 | .7773<br>±.0261 | .7773<br>±.0261 | 15.23 |

> **Analysis.** 5/5 cells scored. Comparing to the `w_eff=10`
> table for the models with both points: llama-1b pass@gb drops
> from .7383 (w_eff=10) to .7148 (w_eff=100) — a small decline
> consistent with the `sem-mcts-bl` finding that `w_eff=10`
> outperforms `w_eff=100`. qwen-math-1.5b is essentially flat
> (.9375 → .9453, within SEM), suggesting this model's pass@gb is
> already saturated and insensitive to the diversity weight at
> b=320.
>
> With the table complete, llama-3b lands at pass@gb .8320,
> between llama-1b (.7148) and the three ≥1.5b qwen models
> (.9336–.9453), so the model-family spread at this operating
> point is dominated by family rather than size: both llama rows
> sit well below every qwen row, including qwen-math-1.5b, which
> is the smallest model in the table. llama-3b's downstream
> metrics fall off more steeply than its pass@gb — maj@gb .6211
> against a .8320 ceiling is a 21-point gap, versus 12 points for
> qwen-7b gptq-int4 (.7734 vs .9375) — i.e. llama-3b generates a
> correct candidate but the PRM selects it less often. That is the
> qwen-PRM-on-llama-generator mismatch already noted elsewhere in
> this doc, and it is the widest instance of it here. At 20.54
> hr/trial llama-3b is also the slowest cell in the table.
> **Limitations / follow-up:** the pass-to-maj gap reading is a
> single-cell observation on 256 questions and llama-1b's own gap
> (.7148 → .4375, 27 points) is wider still, so "llama generators
> lose more at selection" needs the b=80 rows to confirm rather
> than these two points. Same caveats as the `w_eff=10` table
> above (three axes moved at once vs. the b=80 default-point
> table; no matched b=80 row at this exact point for sem-mcts).

---

## Run log (newest first)
> One dated block per run/comparison: hypothesis → result →
> follow-up. Append-only; newest at top.

### 2026-06-18 — cnt-mcts / llama-1b / custom / cpuct=2.0 / b=80
- **hypothesis:** baseline reference cell for the custom
  template; expect higher solution depth than native.
- **result:** pass@gb .648±.042, naive@gb .492±.044,
  wei@gb .469±.044, maj@gb .414±.044 (4 trials);
  ncomps 16.1±0.6, depth 8.6±0.1. W&B `fpgp2si1`.
- **follow-up:** sweep cpuct (0.5/1/2/4) on this anchor cell
  (`llm-reasoning-mcts-exp-todo` Track 1); backfill the
  other scored cnt-mcts pass@gb into the table above.

## Standing comparison questions
- Does sem-UCT beat cnt-UCT at matched budget? (needs
  sem-mcts runnable)
- Does the BL frontier protocol beat phase-based walks?
  (cnt-mcts-bl-v01 vs cnt-mcts @80, qwen PRM — no runs yet, see
  `#### model family, size, quantization comparison (qwen PRM)`
  under `### cnt-mcts-bl-v01`)
- Custom vs native template: consistent across algorithms?
- cpuct sensitivity: same optimum across algorithms? (no
  sweep for any algorithm yet)
- Does the cnt/sem (or BL) gap hold as gen_budget grows
  80→160→320? (the cross-budget question the Summary exists
  to answer)

## Links & connections
- Findings: [findings/exp-findings/prm-batch-size-throughput-memory.md](../findings/exp-findings/prm-batch-size-throughput-memory.md) —
  prm_bs sweep throughput/memory result + why the pass@gb
  gap isn't statistically real
- Findings index: [findings/README.md](../findings/README.md)
