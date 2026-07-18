# LLM Reasoning — MCTS Experiment Comparison — GSM8K

> **Provenance:** structure mirrored from [exp-comp-prm800k-level5.md](exp-comp-prm800k-level5.md) (the PRM800K level-5 doc) on 2026-07-13; every table reset to `planned` — no GSM8K runs exist yet. Launch commands are the level-5 counterparts' with `data=gsm8k` instead of `data=prm800k` (config hashes and run names follow automatically; GSM8K has no `level`, so `--level-N--` is omitted from run names — see `config_name()` in `utils/configs.py`). Unlike level-5 (a strict subset of level-4's grid, same dataset), **GSM8K has no prior grid to inherit** — it's a different dataset with no difficulty levels, no subject taxonomy, and much shorter reasoning traces than MATH (mean 286 vs. 439+ chars). Every table here is a hypothesis that the same knobs are worth tuning, not a confirmed finding carried over; treat all inherited `Compares`/`Fixed` prose as a starting design, not a result.
>
> **Scoring:** GSM8K runs through the same `compute_stats.py` pipeline as PRM800K, selected via `data=gsm8k` (`conf/data/gsm8k.yaml`, `ds_split=test_subset`, 256 questions). Ground-truth parsing and answer extraction are dataset-aware via `cfg.data.grader_name` (`"gsm8k"` here vs. `"math"` for PRM800K) — see `utils/parser.py::parse_ground_truth`'s `gsm8k` branch (splits the `answer` field on `"####"`) and `utils/metrics.py`, fixed 2026-07-13 to thread `grader_name` through instead of hardcoding `"math"`.

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on GSM8K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.

## Purpose
The four algorithm tracks (`llm-reasoning-mcts-exp`,
`llm-reasoning-mcts-bl-exp`, + the `sem` variants) own
*implementation* milestones. This doc owns *experiment tracking
and comparison*: per-algorithm tuning grids and the cross-algorithm
verdict — the views none of the per-algorithm tracks give.

**This is a living log, not a milestone doc.** No `progress: N/M`;
"done" isn't a state here.

## Structure and use
Two activities, two shapes:
- **Tuning *within* an algorithm, at a given budget and
  model** → a `## Tuning tables [gen_budget=N]` section per
  budget → nested `### algorithm` → `##### model` (or an `llm`
  column when several model tables share one comparison), so
  each table is just the config rows for one (budget,
  algorithm, model) cell. Different algorithms show different
  columns — no forced shared schema. `gen_budget`, algorithm,
  and model are subsection levels, not columns, since model
  size drives the GPU constraints and scaling behavior you're
  tuning around; larger budgets need less tuning, so those
  sections are sparser.
  → **Plan/run:** add a config row here; log hypothesis +
  follow-up in the Run log.
- **Comparing *across* algorithms at a fixed budget** →
  the sparse **Summary** table, one row per algorithm × model
  × budget, carrying the full metric set (pass/naive/weighted/
  maj@gb) — tuning tables carry pass@gb only. Everything else
  (depth, ncomps, timing) stays in W&B / the result dir,
  linked. The within-algorithm scaling curve (80→160→320) is
  read by scanning the `gen_budget=N` sections; the
  cross-algorithm-per-budget cut lives here.
  → **A config wins at a budget:** promote it to the Summary
  as that (algorithm, model, budget)'s best config — picked
  across **all tuning knobs jointly** (template, cpuct, …), not
  "best template" with other knobs held fixed — linking back to
  the tuning row. Don't promote a config from a different
  budget or LLM into a Summary row; that breaks the comparison.

---

## Cross-algorithm summary (QwenPRM)
> One table per model, one row per algorithm — pulled directly from
> each algorithm's own "model family, size, quantization comparison
> (QwenPRM)" table above/below (`cnt-mcts`, `sem-mcts`,
> `cnt-mcts-bl-v01`, `kube-mcts-bl-v01`, `kdepth-mcts-bl-v01`,
> `sem-mcts-bl-v01`). All rows fixed at b=80, bs-4, d-20,
> agg_strategy=`last`, tmpl=model-family default (native for Qwen,
> custom for Llama), prm=qwen. `cnt-mcts` row is method=`mcts_cnt_v01`
> (the only cnt-mcts entry point at this level — see the
> `### cnt-mcts` section above). `sem-mcts` row is `mcts_sem_v02` (PRM embeds),
> `ds_alpha=100` (w_eff not applicable — that knob is bl_sem-specific).
> `sem-mcts-bl-v01` row uses the `w_eff=100` table; see that
> algorithm's own section for the `w_eff=10` comparison point.
> `kube-mcts-bl-v01` (Fractional KUBE) and `kdepth-mcts-bl-v01`
> (depth-shaping) — see `docs/decisions/bl-kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/bl-kdepth-knapsack-bonus.md` for the
> algorithms.

**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kube-mcts-bl-v01 | — | planned | — | — | — | — | — |
| kdepth-mcts-bl-v01 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

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

#### model family, size, quantization comparison (RLHFlowPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=rlhflow, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).
>
> **W&B:** none yet (no GSM8K runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

#### model family, size, quantization comparison (QwenPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).
> Companion to the rlhflow-PRM table above; same 5 model/quant
> configs, different scoring PRM.
>
> **W&B:** llama-1b `1zihpeib`, llama-3b `2tqufnuv`, qwen-3b
> `8msypaw8`, qwen-7b gptq-int4 `hlbqb0ib`, qwen-math-1.5b
> `hy6vvm9b`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .9023<br>±.0131 | .8379<br>±.0163 | .7695<br>±.0186 | .7383<br>±.0194 | 2.34 |
| llama-3b fp16 | 2 | scored | .9668<br>±.0079 | .9316<br>±.0112 | .9121<br>±.0125 | .9082<br>±.0128 | 3.75 |
| qwen-3b fp16 | 2 | scored | .9707<br>±.0075 | .9414<br>±.0104 | .9277<br>±.0115 | .9199<br>±.0120 | 4.05 |
| qwen-7b gptq-int4 | 2 | scored | .9727<br>±.0072 | .9492<br>±.0097 | .9395<br>±.0106 | .9316<br>±.0112 | 2.44 |
| qwen-math-1.5b fp16 | 2 | scored | .9766<br>±.0067 | .9375<br>±.0107 | .9121<br>±.0125 | .8945<br>±.0136 | 4.00 |

> **Analysis.** All 5/5 cells scored (2 trials each). All far
> above their PRM800K level-5 counterparts (same table, model
> family/size/quant comparison, QwenPRM, `last` agg): llama-1b
> .3619→.9023, llama-3b .5522→.9668, qwen-3b .6978→.9707,
> qwen-7b gptq-int4 .7537→.9727, qwen-math-1.5b .7575→.9766 —
> consistent with GSM8K's grade-school arithmetic being much
> easier than MATH's competition problems. Ranking by pass@gb is
> qwen-math-1.5b > qwen-7b gptq-int4 ≈ qwen-3b > llama-3b >
> llama-1b — the two smallest/quantized models are essentially
> tied with the two largest, suggesting GSM8K may be near a
> ceiling for this search budget across model families.
> **Limitations / follow-up:** n=2 trials throughout — still
> first-look numbers. hr/trial is NOT monotonic in model size
> (qwen-3b's 4.05 > qwen-math-1.5b's 4.00 > llama-3b's 3.75),
> likely reflecting per-model trace-length differences rather
> than raw compute cost.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. `"last"` is every
> other table's fixed default; `"min"` and `"prod"` aren't yet
> reported anywhere in this doc.
>
> **Fixed:** method=`mcts_cnt_v01`, cpuct=2.0, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

### sem-mcts (v02)

#### embeds_strategy × scope sweep (QwenPRM)
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
> source) — the two `response` rows are **blocked**,
> shown for completeness.
> See [embeds-scope-design.md](decisions/embeds-scope-design.md)
> for the full explanation.
>
> **W&B:** none yet (no GSM8K runs).

| llm | prm | strategy | scope | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | last | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | response | — | — | — | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | response | — | — | — | — | planned | — | — | — | — | — |

> **Limitations / follow-up:** the two `response` rows are blocked
> on PRM-source `response_start_idx` support; queue them once the
> v02 core handles `embeds_scope=response` for `embeds_source=prm`.
> A v01 (policy-embeds) version of this table would unblock the
> `response` axis, since v01 supports it.

#### lam / ds_alpha joint sweep (llama-1b)
> **Compares:** whether `lam` and `ds_alpha` affect selection
> primarily through the effective diversity weight
> `w_eff = ds_alpha / sqrt(lam)`, or whether `lam` also has
> an independent effect on pass@gb. `lam` sets the initial
> ridge scale, `V_0 = lam * I`, and determines how quickly
> `V_inv` changes as embeddings accumulate.
>
> For each `w_eff` checkpoint, the table varies `lam` and
> `ds_alpha` jointly while keeping `w_eff` fixed. Stable
> results within a checkpoint would support `w_eff` as the
> main tuning variable. Differences would indicate a separate
> adaptation-rate effect from `lam`.
>
> The `w_eff = 0` row provides the no-diversity baseline.
> The `0.1` and `0.3` checkpoints probe the on-ramp below
> `w_eff = 1`, while `3` fills the interval between `1` and
> `10`. Together with the higher checkpoints, these values
> test whether pass@gb rises gradually from zero, changes
> sharply near `w_eff ≈ 1`, and eventually reaches the known
> (PRM800K) plateau near `w_eff = 100` — whether that plateau
> location transfers to GSM8K's much easier problem
> distribution is itself an open question this table answers.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | **1.0** | **10** | **10** | 2 | scored | .8887<br>±.0139 | .8398<br>±.0162 | .7871<br>±.0181 | .7480<br>±.0192 | 4.18 |
| llama-1b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .8965<br>±.0135 | .8398<br>±.0162 | .7969<br>±.0178 | .7539<br>±.0191 | 4.26 |
| llama-1b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .8965<br>±.0135 | .8555<br>±.0156 | .8105<br>±.0173 | .7871<br>±.0181 | 4.20 |
| llama-1b | qwen | 1.0 | 100 | 100 | 0/2 | running | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 31.6 | 100 | 0/2 | running | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 10 | 100 | 0/2 | running | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** 3/22 cells scored (2 trials each, n=256). The
> `w_eff=10` step is fully resolved: pass@gb .8887/.8965/.8965
> (`lam=1.0/0.1/0.01`) — all within 1 SEM of each other, no
> `lam`-dependence signal, matching the PRM800K-level5 pattern at
> this checkpoint. The `w_eff=100` step (3 cells) is currently
> running.
> **Limitations / follow-up:** `w_eff=100` in flight;
> `w_eff=0.1/0.3/1000` and the `w_eff=0` gap-closer remain
> unlaunched.

#### lam / ds_alpha joint sweep (llama-3b)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b table above, on llama-3b.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| llama-3b | qwen | **1.0** | **10** | **10** | 2 | scored | .9590<br>±.0088 | .9180<br>±.0121 | .9297<br>±.0113 | .9238<br>±.0117 | 4.88 |
| llama-3b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .9629<br>±.0084 | .9316<br>±.0112 | .9160<br>±.0123 | .9180<br>±.0121 | 5.33 |
| llama-3b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .9590<br>±.0088 | .9277<br>±.0115 | .9238<br>±.0117 | .9238<br>±.0117 | 5.44 |
| llama-3b | qwen | 1.0 | 100 | 100 | 2 | scored | .9648<br>±.0081 | .9414<br>±.0104 | .8809<br>±.0143 | .8711<br>±.0148 | 5.20 |
| llama-3b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .9688<br>±.0077 | .9316<br>±.0112 | .8984<br>±.0134 | .8906<br>±.0138 | 5.64 |
| llama-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .9648<br>±.0081 | .9316<br>±.0112 | .8984<br>±.0134 | .8984<br>±.0134 | 5.82 |
| llama-3b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** 6/22 cells scored (2 trials each, n=256). Both
> `w_eff=10` and `w_eff=100` steps are fully resolved: `w_eff=10`
> pass@gb .9590/.9629/.9590 (`lam=1.0/0.1/0.01`) — tightly
> clustered, no `lam`-dependence. `w_eff=100` similarly tight
> (.9648/.9688/.9648) — a hair higher than `w_eff=10` across the
> board, all within 1 SEM of each other and of `w_eff=10`. No
> `lam` effect visible at this budget/dataset.
> **Limitations / follow-up:** `w_eff=0.1/0.3/1000` and the
> `w_eff=0` gap-closer remain unlaunched; only 2 trials/cell.

#### lam / ds_alpha joint sweep (qwen-math-1.5b)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b tables above, on qwen-math-1.5b.
>
> **Fixed:** tmpl=model-family default (native), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | **1.0** | **10** | **10** | 2 | scored | .9785<br>±.0064 | .9395<br>±.0106 | .8809<br>±.0143 | .8672<br>±.0150 | 6.17 |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .9629<br>±.0084 | .9316<br>±.0112 | .8809<br>±.0143 | .8691<br>±.0149 | 5.93 |
| qwen-math-1.5b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .9688<br>±.0077 | .9336<br>±.0110 | .8984<br>±.0134 | .8809<br>±.0143 | 5.87 |
| qwen-math-1.5b | qwen | 1.0 | 100 | 100 | 2 | scored | .9531<br>±.0094 | .9121<br>±.0125 | .8320<br>±.0165 | .8125<br>±.0173 | 6.13 |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .9668<br>±.0079 | .9277<br>±.0115 | .8652<br>±.0151 | .8555<br>±.0156 | 5.92 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored | .9668<br>±.0079 | .9277<br>±.0115 | .8887<br>±.0139 | .8730<br>±.0147 | 5.86 |
| qwen-math-1.5b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** 6/22 cells scored (2 trials each, n=256).
> `w_eff=10` step: pass@gb .9785 (`lam=1.0`) is the highest of
> the three, vs. .9629/.9688 (`lam=0.1`/`0.01`) — a small
> `lam=1.0`-trending-higher pattern, opposite the direction seen
> at PRM800K-level5 for this model (`lam=1.0` trended lowest
> there). `w_eff=100` shows the reverse: `lam=1.0` .9531 is now
> the lowest of the three (vs. .9668 for both `lam=0.1`/`0.01`) —
> consistent with the level-5 direction at that checkpoint. All
> gaps are within ~1-2 SEM.
> **Limitations / follow-up:** `w_eff=0.1/0.3/1000` and the
> `w_eff=0` gap-closer remain unlaunched; only 2 trials/cell, so
> the direction flip between checkpoints is suggestive, not
> conclusive.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b/qwen-math-1.5b tables above, on qwen-7b
> gptq-int4.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | **1.0** | **10** | **10** | 2 | scored | .9688<br>±.0077 | .9414<br>±.0104 | .9434<br>±.0102 | .9414<br>±.0104 | 3.42 |
| qwen-7b gptq-int4 | qwen | 0.1 | 3.16 | 10 | 2 | scored | .9609<br>±.0086 | .9492<br>±.0097 | .9512<br>±.0095 | .9453<br>±.0101 | 3.71 |
| qwen-7b gptq-int4 | qwen | **0.01** | **1.0** | **10** | 2 | scored | .9766<br>±.0067 | .9570<br>±.0090 | .9473<br>±.0099 | .9473<br>±.0099 | 3.83 |
| qwen-7b gptq-int4 | qwen | 1.0 | 100 | 100 | 2 | scored | .9746<br>±.0070 | .9512<br>±.0095 | .9375<br>±.0107 | .9316<br>±.0112 | 3.54 |
| qwen-7b gptq-int4 | qwen | 0.1 | 31.6 | 100 | 2 | scored | .9707<br>±.0075 | .9492<br>±.0097 | .9316<br>±.0112 | .9219<br>±.0119 | 3.80 |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | 2 | scored | .9648<br>±.0081 | .9551<br>±.0092 | .9395<br>±.0106 | .9355<br>±.0109 | 3.94 |
| qwen-7b gptq-int4 | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** 6/22 cells scored (2 trials each, n=256). Both
> `w_eff=10` and `w_eff=100` steps land tightly clustered on
> pass@gb (.9609–.9766), the flattest spread of any model-family
> table in this GSM8K sweep so far — echoing the same flatness
> seen at PRM800K-level5 for this model. No clear `lam` effect at
> either checkpoint.
> **Limitations / follow-up:** `w_eff=0.1/0.3/1000` and the
> `w_eff=0` gap-closer remain unlaunched; only 2 trials/cell.

#### model family, size, quantization comparison (RLHFlowPRM)
> **Compares:** model family, size, and quantization jointly —
> same shape as cnt-mcts's table above, for cross-method
> comparability.
>
> **Fixed:** bs-4, d-20, b=80, tmpl=model-family default,
> method=`mcts_sem_v02` (PRM embeds), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm).
>
> **W&B:** none yet (no GSM8K runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** the same 5-model family/size/quantization sweep
> as the RLHFlowPRM table above, but scored with `prm=qwen`
> (Qwen-Math-7B-PRM) instead of the default `prm=rlhflow`
> (Llama-8B-PRM).
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), ds_alpha=100, ds_beta=1.0,
> prm_batch_size=1.
>
> **W&B:** none yet (no GSM8K runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.


#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. Scoring-side
> counterpart to the cnt-mcts table of the same name.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, ds_alpha=100.0, ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=10)
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
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=100)
> **Compares:** same as the `w_eff=10` table above, at the next
> `w_eff` checkpoint.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.1, ds_alpha=31.6 (w_eff=100),
> ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.


### cnt-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts's equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

### kube-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled.
>
> **Fixed:** method=`mcts_bl_kube_v01` (renamed 2026-07-16 from
> `mcts_bl_cnt_v02`), prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent, kube_affordable=true,
> bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family default
> (native for Qwen, custom for Llama). See
> `docs/decisions/bl-kube-bonus-schedule.md` for the schedule choice.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

### kdepth-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled.
>
> **Fixed:** method=`mcts_bl_kdepth_v01` (renamed 2026-07-17 from
> `mcts_bl_cnt_v03`), prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

### sem-mcts-bl

#### model family, size, quantization comparison (QwenPRM, w_eff=100)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled.
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
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

#### model family, size, quantization comparison (QwenPRM, w_eff=10)
> **Compares:** same 5-model/quant grid as the `w_eff=100` table
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
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

### cnt-mcts

#### model family comparison (b=320, QwenPRM)
> **Compares:** the same 5-model family/size/quantization sweep
> as the `[gen_budget=80]` table above, but at
> `search.gen_budget=320` (4× the b=80 budget) with
> `prm=qwen_prm` instead of the b=80 table's default
> `llama_prm`. Two axes change at once — budget and PRM — so
> this table isn't a clean isolation of either; it answers
> "does the b=80 ranking across model family/size/quantization
> hold at a much larger search budget under qwen scoring," not
> "what does budget alone do." A matched-PRM (llama) b=320 row
> per model would be needed to separate the two effects.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, b=320, prm=qwen,
> tmpl=model-family default (native for Qwen, custom for Llama).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; GSM8K's much
> shorter reasoning traces (§ top-of-doc) may make this scale
> differently than PRM800K's b=320 wall-clock did — no reference
> point exists yet.
>
> **W&B:** none yet (no GSM8K runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

### sem-mcts

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=10)
> **Compares:** the same 5-model family/size/quantization sweep
> as the `[gen_budget=80]` sem-mcts (QwenPRM) table above, but
> at `search.gen_budget=320` (4× the b=80 budget) and at
> `lam=0.1, ds_alpha=3.16` (`w_eff = ds_alpha/sqrt(lam) = 10`)
> instead of that table's default point (`lam=0.01,
> ds_alpha=100`, i.e. `w_eff=1000`). Three axes move at once
> relative to that b=80 table — budget, lam, and ds_alpha — so
> this isn't a clean isolation of any one of them; paired with
> the `w_eff=100` table below (same budget, same lam, 10×
> ds_alpha) it does isolate `w_eff` at b=320.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=320, prm_batch_size=1, `ds_alpha_schedule=global`
> (default), `cov_update=sm`, `embeds_dim=512`/
> `embeds_proj=sparse` (defaults), tmpl=model-family default
> (native for Qwen, custom for Llama). **lam=0.1, ds_alpha=3.16**
> (`w_eff=10` — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row, same point used by the `sem-mcts-bl` w_eff=10
> table).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; no GSM8K
> wall-clock reference point exists yet at any budget.
>
> **W&B:** none yet (no GSM8K runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=100)
> **Compares:** identical setup to the `w_eff=10` table above,
> at `ds_alpha=31.6` instead of `3.16` (10× the diversity
> weight, same `lam=0.1`) — the b=320 counterpart of the
> `sem-mcts-bl` w_eff=100 table, and the paired point needed to
> isolate `w_eff` alone at this budget.
>
> **Fixed:** identical to the `w_eff=10` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse`, tmpl=model-family
> default) except the diversity weight. **lam=0.1,
> ds_alpha=31.6** (`w_eff=100`).
>
> ⚠️ Entirely `planned` — no runs yet.
>
> **W&B:** none yet (no GSM8K runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No GSM8K data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-5 counterpart's command with `data=gsm8k`.

---
