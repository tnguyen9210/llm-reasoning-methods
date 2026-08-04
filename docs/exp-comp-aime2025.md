# LLM Reasoning — MCTS Experiment Comparison — AIME2025

> **Provenance:** structure mirrored from [exp-comp-gsm8k.md](exp-comp-gsm8k.md) (the GSM8K doc) on 2026-07-14; every table reset to `planned` — no AIME2025 runs exist yet. Launch commands are the GSM8K counterparts' with `data=aime2025` instead of `data=gsm8k` (config hashes and run names follow automatically; AIME2025 has no `level`, same as GSM8K, so `--level-N--` is omitted from run names — see `config_name()` in `utils/configs.py`). **AIME2025 has no prior grid to inherit** in the sense that matters most: it's not just a different dataset, it's a **much smaller one** — 30 questions total (one full AIME administration, I+II, 15 problems each) vs. GSM8K's 256-row test subset. Every cell's SEM will be markedly wider than the GSM8K table's as a direct consequence (a single flipped question moves pass@gb by ~3.3 percentage points on n=30, vs. ~0.4 on n=256) — treat all inherited `Compares`/`Fixed` prose as a starting design, not a result, and read early pass@gb numbers here with that sample-size caveat in mind throughout.
>
> **Scoring:** AIME2025 runs through the same `compute_stats.py` pipeline as GSM8K/PRM800K, selected via `data=aime2025` (`conf/data/aime2025.yaml`, `ds_split=train` — the only split that exists in the downloaded data, despite the name; see the file's contents, 30 rows). Ground-truth parsing and answer extraction are dataset-aware via `cfg.data.grader_name` (`"aime"` here vs. `"gsm8k"`/`"math"`). AIME's ground truth is always a bare integer 0–999 (a hard competition rule, not a formatting convention) — see `utils/parser.py::parse_ground_truth`'s `aime` branch (reads the `answer` field directly, no `\boxed{}` or `####` parsing needed for ground truth). Answer *extraction* from the model's own generation reuses the same `\boxed{}`-pulling logic as MATH/PRM800K unchanged, since the prompt convention ("put your final answer in `\boxed{}`") is the same regardless of dataset. Added 2026-07-14.
>
> **Contamination note:** AIME2025 problems were posted publicly (AoPS wiki, competition forums) within hours of the actual contest. Any model trained after early-to-mid 2025 may have seen these problems/solutions verbatim — treat pass@gb here as an upper bound on genuine reasoning ability for such models, not a clean measurement. AIME2024 (`data=aime2024`, not yet given its own tracking doc) is older and more likely contaminated for current-generation models; AIME2025 is the more useful of the two if a contamination-lite read matters.

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on AIME2025 — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.






<!-- toc:begin -- generated, do not hand-edit -->
## Contents

- [**Purpose**](#purpose)
- [**Structure and use**](#structure-and-use)
- [**Cross-algorithm summary (QwenPRM)**](#cross-algorithm-summary-qwenprm)
- [**Tuning tables \[gen_budget=80\]**](#tuning-tables-gen_budget80)
  - [cnt-mcts](#cnt-mcts)
    - [model family, size, quantization comparison (RLHFlowPRM)](#model-family-size-quantization-comparison-rlhflowprm) · `tbl-161a03`
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm) · `tbl-e742a7`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b)](#agg_strategy-comparison-qwen-3b-qwen-math-15b) · `tbl-6dad4f`
  - [sem-mcts-v02](#sem-mcts-v02)
    - [embeds_strategy × scope sweep (QwenPRM)](#embeds_strategy-scope-sweep-qwenprm) · `tbl-0c55e1`
    - [lam / ds_alpha joint sweep (llama-1b)](#lam-ds_alpha-joint-sweep-llama-1b) · `tbl-b1a6d9`
    - [lam / ds_alpha joint sweep (llama-3b)](#lam-ds_alpha-joint-sweep-llama-3b) · `tbl-d0ed2a`
    - [lam / ds_alpha joint sweep (qwen-math-1.5b)](#lam-ds_alpha-joint-sweep-qwen-math-15b) · `tbl-8bf48f`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4) · `tbl-ba8af1`
    - [embeds_center_mode comparison (lam=0.01/ds_alpha=1)](#embeds_center_mode-comparison-lam001ds_alpha1) · `tbl-4f8220`
    - [embeds_center_mode comparison (lam=0.01/ds_alpha=10)](#embeds_center_mode-comparison-lam001ds_alpha10) · `tbl-ddf79e`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=1)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam001ds_alpha1) · `tbl-6ef336`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=10)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam001ds_alpha10) · `tbl-4498f8`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha1) · `tbl-cfd7cf`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha10) · `tbl-878af9`
  - [sem-mcts-v02 \[cov_scope=local\]](#sem-mcts-v02-cov_scopelocal)
    - [lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-1b-embeds_refrelative) · `tbl-2ef3dc`
    - [lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-3b-embeds_refrelative) · `tbl-b94f3f`
    - [lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-3b-embeds_refrelative) · `tbl-435dd3`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4-embeds_refrelative) · `tbl-7a3760`
    - [lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-math-15b-embeds_refrelative) · `tbl-4ef506`
  - [cnt-mcts-bl-v01](#cnt-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-1) · `tbl-c7dd39`
  - [cnt-mcts-bl-v02](#cnt-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × cpuct) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-cpuct-qwen-3b-qwenprm) · `tbl-40a360`
  - [kube-mcts-bl-v01](#kube-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-2) · `tbl-d34700`
    - [kube_c sweep × model family (QwenPRM)](#kube_c-sweep-model-family-qwenprm) · `tbl-9d3944`
  - [kube-mcts-bl-v02](#kube-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × kube_c) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-kube_c-qwen-3b-qwenprm) · `tbl-bdeba2`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha08) · `tbl-bda3a8`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10) · `tbl-b3d812`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha00) · `tbl-7c1779`
    - [alpha × kube_c joint sweep (llama-3b, QwenPRM, parent_blend)](#alpha-kube_c-joint-sweep-llama-3b-qwenprm-parent_blend) · `tbl-86e0b6`
    - [gamma × kube_c joint sweep (qwen-3b, QwenPRM, path_decay)](#gamma-kube_c-joint-sweep-qwen-3b-qwenprm-path_decay) · `tbl-434915`
  - [kdepth-mcts-bl-v01](#kdepth-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-3) · `tbl-acca74`
    - [model family, size, quantization comparison (QwenPRM, depth_alpha=0.5)](#model-family-size-quantization-comparison-qwenprm-depth_alpha05) · `tbl-b429d0`
    - [model family, size, quantization comparison (QwenPRM, depth_alpha=2.0)](#model-family-size-quantization-comparison-qwenprm-depth_alpha20) · `tbl-3fbc68`
  - [kdepth-mcts-bl-v02](#kdepth-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-qwen-3b-qwenprm) · `tbl-59ccb9`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha08-1) · `tbl-d81f40`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10-1) · `tbl-288646`
  - [sem-mcts-bl-v01](#sem-mcts-bl-v01)
    - [model family comparison (QwenPRM, lam=0.01/ds_alpha=10, max_model_len=6000)](#model-family-comparison-qwenprm-lam001ds_alpha10-max_model_len6000) · `tbl-df1eeb`
    - [model family comparison (QwenPRM, lam=0.01/ds_alpha=1, max_model_len=6000)](#model-family-comparison-qwenprm-lam001ds_alpha1-max_model_len6000) · `tbl-b3f9bb`
  - [sem-mcts-bl-v02](#sem-mcts-bl-v02)
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10-lam001ds_alpha10) · `tbl-396f65`
- [**Tuning tables \[gen_budget=160, 320, …\] *(future)***](#tuning-tables-gen_budget160-320-future)
  - [cnt-mcts](#cnt-mcts-1)
    - [model family comparison (b=320, QwenPRM)](#model-family-comparison-b320-qwenprm) · `tbl-f31bf0`
  - [sem-mcts-v02](#sem-mcts-v02-1)
    - [model family comparison (b=320, QwenPRM, lam=0.1, w_eff=10)](#model-family-comparison-b320-qwenprm-lam01-w_eff10) · `tbl-b2d2d2`
    - [model family comparison (b=320, QwenPRM, lam=0.1, w_eff=100)](#model-family-comparison-b320-qwenprm-lam01-w_eff100) · `tbl-9d68e9`

*41 tables. Regenerate with `python scripts/gen_toc.py`.*
<!-- toc:end -->

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

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
<!-- table-id: tbl-161a03 -->
> **Fixed:** method=`mcts_cnt_v01`, prm=rlhflow, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama),
> **run.num_trials=4** (AIME2025's n=30 is small — 4 trials
> instead of the usual 2 to narrow the wide per-cell SEMs).
>
> **W&B:** none yet (no AIME2025 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-e742a7 -->
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama),
> **run.num_trials=4** (see the RLHFlowPRM table above).
> Companion to the rlhflow-PRM table above; same 5 model/quant
> configs, different scoring PRM.
>
> **W&B:** llama-1b `nvk5x5jc`, llama-3b `vetl9lf1`, qwen-3b
> `7m5nmmtj`, qwen-7b gptq-int4 `zxy77eld`, qwen-math-1.5b
> `n59svtr8`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | 0.78 |
| llama-3b fp16 | 4 | scored | .0750<br>±.0241 | .0500<br>±.0200 | .0333<br>±.0165 | .0167<br>±.0117 | 1.33 |
| qwen-3b fp16 | 4 | scored | .1417<br>±.0320 | .0667<br>±.0229 | .0917<br>±.0265 | .0917<br>±.0265 | 1.28 |
| qwen-7b gptq-int4 | 4 | scored | .2500<br>±.0397 | .1417<br>±.0320 | .1167<br>±.0294 | .1083<br>±.0285 | 1.25 |
| qwen-math-1.5b fp16 | 4 | scored | .2667<br>±.0405 | .1917<br>±.0361 | .1667<br>±.0342 | .1500<br>±.0327 | 1.02 |

> **Analysis.** 5/5 cells scored (4 trials each, n=30 questions).
> Ranking by pass@gb is qwen-math-1.5b (.2667) > qwen-7b
> gptq-int4 (.2500) > qwen-3b (.1417) > llama-3b (.0750) >
> llama-1b (.0083, essentially a floor — 1/120 correct) —
> broadly the same family ordering as GSM8K/PRM800K-level5's
> QwenPRM tables, but pass@gb is far lower across the board, as
> expected for a genuinely harder, likely-less-contaminated
> competition set.
> **Limitations / follow-up:** feeds
> `aime2025-cnt-model-family-qwen`. n=30 questions total means
> SEMs are wide (±.02 to ±.04) — treat rankings as directional,
> not conclusive. The RLHFlowPRM companion table above remains
> entirely planned.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
<!-- table-id: tbl-6dad4f -->
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. `"last"` is every
> other table's fixed default; `"min"` and `"prod"` aren't yet
> reported anywhere in this doc.
>
> **Fixed:** method=`mcts_cnt_v01`, cpuct=2.0, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> **run.num_trials=4** (see the model-family tables above).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

### sem-mcts-v02

#### embeds_strategy × scope sweep (QwenPRM)
<!-- table-id: tbl-0c55e1 -->
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
> proj=sparse512, cov_update=sm, ds_beta=1.0,
> **run.num_trials=4** (see the cnt-mcts tables above).
>
> ⚠️ `embeds_scope=response` is **not supported on v02** (PRM
> source) — the two `response` rows are **blocked**,
> shown for completeness.
> See [embeds-scope-design.md](decisions/embeds-scope-design.md)
> for the full explanation.
>
> **W&B:** none yet (no AIME2025 runs).

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
<!-- table-id: tbl-b1a6d9 -->
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
> location transfers to AIME2025's much harder problem
> distribution (harder than PRM800K level-5, not easier — unlike
> GSM8K) is itself an open question this table answers.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b, **run.num_trials=4**
> (see the cnt-mcts tables above).

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
| llama-1b | qwen | **1.0** | **10** | **10** | 4 | scored | .0167<br>±.0117 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.32 |
| llama-1b | qwen | 0.1 | 3.16 | 10 | 4 | scored | .0250<br>±.0143 | .0250<br>±.0143 | .0167<br>±.0117 | .0083<br>±.0083 | 1.33 |
| llama-1b | qwen | **0.01** | **1.0** | **10** | 4 | scored | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | .0000<br>±.0000 | 1.34 |
| llama-1b | qwen | 1.0 | 100 | 100 | 4 | scored | .0167<br>±.0117 | .0167<br>±.0117 | .0167<br>±.0117 | .0083<br>±.0083 | 1.32 |
| llama-1b | qwen | 0.1 | 31.6 | 100 | 4 | scored | .0250<br>±.0143 | .0167<br>±.0117 | .0083<br>±.0083 | .0083<br>±.0083 | 1.31 |
| llama-1b | qwen | 0.01 | 10 | 100 | 4 | scored | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.30 |
| llama-1b | qwen | 1.0 | 1000 | 1000 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.32 |
| llama-1b | qwen | 0.1 | 316.2 | 1000 | 4 | scored | .0333<br>±.0165 | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | 1.33 |
| llama-1b | qwen | 0.01 | 100 | 1000 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.34 |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** feeds
> `aime2025-sem-lam-dsalpha-sweep-llama1b`. Mostly planned;
> launch is the GSM8K counterpart's command with
> `data=aime2025`. The lam=0.01 w_eff=10/100 rows share runs
> with the centermode tables' `none` rows below.

#### lam / ds_alpha joint sweep (llama-3b)
<!-- table-id: tbl-d0ed2a -->
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b table above, on llama-3b.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b, **run.num_trials=4**
> (see the cnt-mcts tables above).

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
| llama-3b | qwen | **1.0** | **10** | **10** | 4 | scored | .0333<br>±.0165 | .0333<br>±.0165 | .0083<br>±.0083 | .0083<br>±.0083 | 1.83 |
| llama-3b | qwen | 0.1 | 3.16 | 10 | 4 | scored | .0750<br>±.0241 | .0583<br>±.0215 | .0417<br>±.0183 | .0250<br>±.0143 | 1.78 |
| llama-3b | qwen | **0.01** | **1.0** | **10** | 4 | scored | .0417<br>±.0183 | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | 1.78 |
| llama-3b | qwen | 1.0 | 100 | 100 | 4 | scored | .0500<br>±.0200 | .0417<br>±.0183 | .0333<br>±.0165 | .0333<br>±.0165 | 1.81 |
| llama-3b | qwen | 0.1 | 31.6 | 100 | 4 | scored | .0500<br>±.0200 | .0250<br>±.0143 | .0250<br>±.0143 | .0000<br>±.0000 | 1.84 |
| llama-3b | qwen | 0.01 | 10 | 100 | 4 | scored | .0583<br>±.0215 | .0167<br>±.0117 | .0167<br>±.0117 | .0167<br>±.0117 | 1.90 |
| llama-3b | qwen | 1.0 | 1000 | 1000 | 4 | scored | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | 1.80 |
| llama-3b | qwen | 0.1 | 316.2 | 1000 | 4 | scored | .0667<br>±.0229 | .0500<br>±.0200 | .0417<br>±.0183 | .0333<br>±.0165 | 1.78 |
| llama-3b | qwen | 0.01 | 100 | 1000 | 4 | scored | .0750<br>±.0241 | .0333<br>±.0165 | .0333<br>±.0165 | .0083<br>±.0083 | 1.84 |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** feeds
> `aime2025-sem-lam-dsalpha-sweep-llama3b`. Mostly planned;
> launch is the GSM8K counterpart's command with
> `data=aime2025`. The lam=0.01 w_eff=10/100 rows share runs
> with the centermode tables' `none` rows below.

#### lam / ds_alpha joint sweep (qwen-math-1.5b)
<!-- table-id: tbl-8bf48f -->
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b tables above, on qwen-math-1.5b.
>
> **Fixed:** tmpl=model-family default (native), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b, **run.num_trials=4**
> (see the cnt-mcts tables above).

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
| qwen-math-1.5b | qwen | **1.0** | **10** | **10** | 4 | scored | .2417<br>±.0392 | .1917<br>±.0361 | .1500<br>±.0327 | .1333<br>±.0312 | 1.46 |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | 4 | scored | .2250<br>±.0383 | .1417<br>±.0320 | .1250<br>±.0303 | .1333<br>±.0312 | 1.42 |
| qwen-math-1.5b | qwen | **0.01** | **1.0** | **10** | 4 | scored | .2583<br>±.0401 | .1833<br>±.0355 | .1583<br>±.0335 | .1500<br>±.0327 | 1.45 |
| qwen-math-1.5b | qwen | 1.0 | 100 | 100 | 4 | scored | .2583<br>±.0401 | .1667<br>±.0342 | .1583<br>±.0335 | .1417<br>±.0320 | 1.41 |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | 4 | scored | .2583<br>±.0401 | .1750<br>±.0348 | .1583<br>±.0335 | .1333<br>±.0312 | 1.41 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 4 | scored | .1917<br>±.0361 | .1417<br>±.0320 | .1333<br>±.0312 | .1250<br>±.0303 | 1.39 |
| qwen-math-1.5b | qwen | 1.0 | 1000 | 1000 | 4 | scored | .2333<br>±.0388 | .1750<br>±.0348 | .1417<br>±.0320 | .1167<br>±.0294 | 1.42 |
| qwen-math-1.5b | qwen | 0.1 | 316.2 | 1000 | — | failed | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 100 | 1000 | — | failed | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** feeds
> `aime2025-sem-lam-dsalpha-sweep-qwenmath15b`. Mostly planned;
> launch is the GSM8K counterpart's command with
> `data=aime2025`. The lam=0.01 w_eff=10/100 rows share runs
> with the centermode tables' `none` rows below.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
<!-- table-id: tbl-ba8af1 -->
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b/qwen-math-1.5b tables above, on qwen-7b
> gptq-int4.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4, **run.num_trials=4**
> (see the cnt-mcts tables above).

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
| qwen-7b gptq-int4 | qwen | **1.0** | **10** | **10** | — | failed | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 3.16 | 10 | 4 | scored | .2250<br>±.0383 | .1417<br>±.0320 | .1083<br>±.0285 | .1000<br>±.0275 | 1.57 |
| qwen-7b gptq-int4 | qwen | **0.01** | **1.0** | **10** | 4 | scored | .2250<br>±.0383 | .1167<br>±.0294 | .1333<br>±.0312 | .1250<br>±.0303 | 1.59 |
| qwen-7b gptq-int4 | qwen | 1.0 | 100 | 100 | 4 | scored | .2083<br>±.0372 | .1417<br>±.0320 | .0917<br>±.0265 | .0833<br>±.0253 | 1.62 |
| qwen-7b gptq-int4 | qwen | 0.1 | 31.6 | 100 | 4 | scored | .2083<br>±.0372 | .1500<br>±.0327 | .1250<br>±.0303 | .1167<br>±.0294 | 1.62 |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | 4 | scored | .3000<br>±.0420 | .1833<br>±.0355 | .1333<br>±.0312 | .1167<br>±.0294 | 1.65 |
| qwen-7b gptq-int4 | qwen | 1.0 | 1000 | 1000 | 4 | scored | .2000<br>±.0367 | .0917<br>±.0265 | .0750<br>±.0241 | .0667<br>±.0229 | 1.61 |
| qwen-7b gptq-int4 | qwen | 0.1 | 316.2 | 1000 | 4 | scored | .2583<br>±.0401 | .1167<br>±.0294 | .0750<br>±.0241 | .0667<br>±.0229 | 1.60 |
| qwen-7b gptq-int4 | qwen | 0.01 | 100 | 1000 | 4 | scored | .2750<br>±.0409 | .1417<br>±.0320 | .1167<br>±.0294 | .0917<br>±.0265 | 1.66 |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** feeds
> `aime2025-sem-lam-dsalpha-sweep-qwen7bgptq`. Mostly planned;
> launch is the GSM8K counterpart's command with
> `data=aime2025`. The lam=0.01 w_eff=10/100 rows share runs
> with the centermode tables' `none` rows below.

#### embeds_center_mode comparison (lam=0.01/ds_alpha=1)
<!-- table-id: tbl-4f8220 -->
> **Compares:** `embeds_center_mode="local"` (rep_exp-style
> sibling-group centering) against `embeds_center=false` (no
> centering — today's default). `"fixed"` mode isn't in this table
> yet — no precomputed held-out mean exists for AIME2025. See
> [rep-exp-elliptical-bonus-review.md](decisions/rep-exp-elliptical-bonus-review.md)
> follow-up #3 and
> [embeds-centering-design.md](decisions/embeds-centering-design.md)
> for the full discussion.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=80, proj=sparse512, cov_update=sm, cov_dtype=fp64 (default),
> ds_beta=1.0, prm_batch_size=1, tmpl=model-family default (native for
> Qwen, custom for Llama), **lam=0.01, ds_alpha=1.0** (`w_eff =
> ds_alpha/sqrt(lam) = 10`), **run.num_trials=4** (see the cnt-mcts
> tables above).
>
> **W&B:** llama-1b none `kusajt8w` / local `j3ktqdcj`, llama-3b
> none `llkxel7t` / local `afgb59jx`, qwen-3b none `gztx392f` /
> local `saxhgtii`, qwen-7b gptq-int4 none `yrcwjwpe` / local
> `cczd3l2r`, qwen-math-1.5b none `s87k7uxn` / local `jif17zx7`.

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | 4 | scored | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | .0000<br>±.0000 | 1.34 |
| llama-1b | qwen | local | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.28 |
| llama-3b | qwen | none | 4 | scored | .0417<br>±.0183 | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | 1.78 |
| llama-3b | qwen | local | 4 | scored | .0500<br>±.0200 | .0333<br>±.0165 | .0083<br>±.0083 | .0000<br>±.0000 | 1.82 |
| qwen-3b | qwen | none | 4 | scored | .1833<br>±.0355 | .1250<br>±.0303 | .1083<br>±.0285 | .1000<br>±.0275 | 1.71 |
| qwen-3b | qwen | local | 4 | scored | .1583<br>±.0335 | .1250<br>±.0303 | .1000<br>±.0275 | .0833<br>±.0253 | 1.77 |
| qwen-7b gptq-int4 | qwen | none | 4 | scored | .2250<br>±.0383 | .1167<br>±.0294 | .1333<br>±.0312 | .1250<br>±.0303 | 1.59 |
| qwen-7b gptq-int4 | qwen | local | 4 | scored | .2917<br>±.0417 | .1333<br>±.0312 | .1417<br>±.0320 | .1333<br>±.0312 | 1.69 |
| qwen-math-1.5b | qwen | none | 4 | scored | .2583<br>±.0401 | .1833<br>±.0355 | .1583<br>±.0335 | .1500<br>±.0327 | 1.45 |
| qwen-math-1.5b | qwen | local | 4 | scored | .2583<br>±.0401 | .1583<br>±.0335 | .1750<br>±.0348 | .1750<br>±.0348 | 1.47 |

> **Analysis.** 10/10 cells scored (4 trials each, n=30
> questions). `local` vs `none` splits both directions across
> models — llama-1b/qwen-3b/qwen-math-1.5b(mixed) trend slightly
> lower or flat under `local`, qwen-7b gptq-int4 trends higher
> (.2250→.2917) — no consistent direction, and every gap is well
> within 1 SEM given n=30's wide intervals. No clear
> centering-mode effect at this sample size.
> **Limitations / follow-up:** feeds
> `aime2025-sem-centermode-lam0.01-weff10`. n=30 questions means
> SEMs are wide (±.008 to ±.042) — treat any `local` vs `none`
> gap here as noise until more trials accumulate. A
> `"fixed"`-mode column remains a follow-up once a held-out mean
> exists for AIME2025.

#### embeds_center_mode comparison (lam=0.01/ds_alpha=10)
<!-- table-id: tbl-ddf79e -->
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=80, proj=sparse512,
> cov_update=sm, cov_dtype=fp64, ds_beta=1.0, prm_batch_size=1,
> tmpl=model-family default, run.num_trials=4) except
> **ds_alpha=10** (`w_eff=100`).
>
> **W&B:** llama-1b none `5dr4nnlo` / local `w121yxyb`, llama-3b
> none `vlsck1e4` / local `d0cs9hb6`, qwen-3b none `geje96pg` /
> local `u0wgl7xo`, qwen-7b gptq-int4 none `30pfsbnl` / local
> `js9umand`, qwen-math-1.5b none `bu8v110w` / local `5uwdjimk`.

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | 4 | scored | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.30 |
| llama-1b | qwen | local | 4 | scored | .0250<br>±.0143 | .0167<br>±.0117 | .0000<br>±.0000 | .0000<br>±.0000 | 1.35 |
| llama-3b | qwen | none | 4 | scored | .0583<br>±.0215 | .0167<br>±.0117 | .0167<br>±.0117 | .0167<br>±.0117 | 1.90 |
| llama-3b | qwen | local | 4 | scored | .0750<br>±.0241 | .0250<br>±.0143 | .0333<br>±.0165 | .0167<br>±.0117 | 1.86 |
| qwen-3b | qwen | none | 4 | scored | .1667<br>±.0342 | .1167<br>±.0294 | .0917<br>±.0265 | .0833<br>±.0253 | 1.77 |
| qwen-3b | qwen | local | 4 | scored | .2000<br>±.0367 | .1333<br>±.0312 | .0917<br>±.0265 | .0583<br>±.0215 | 1.77 |
| qwen-7b gptq-int4 | qwen | none | 4 | scored | .3000<br>±.0420 | .1833<br>±.0355 | .1333<br>±.0312 | .1167<br>±.0294 | 1.65 |
| qwen-7b gptq-int4 | qwen | local | 4 | scored | .2667<br>±.0405 | .1917<br>±.0361 | .1083<br>±.0285 | .1083<br>±.0285 | 1.68 |
| qwen-math-1.5b | qwen | none | 4 | scored | .1917<br>±.0361 | .1417<br>±.0320 | .1333<br>±.0312 | .1250<br>±.0303 | 1.39 |
| qwen-math-1.5b | qwen | local | 4 | scored | .2500<br>±.0397 | .1750<br>±.0348 | .1417<br>±.0320 | .1167<br>±.0294 | 1.43 |

> **Analysis.** 10/10 cells scored (4 trials each, n=30
> questions). Same pattern as the `ds_alpha=1` checkpoint above:
> `local` vs `none` splits both directions (llama-1b/llama-3b/
> qwen-3b/qwen-math-1.5b trend slightly higher under `local`,
> qwen-7b gptq-int4 trends lower), every gap within 1 SEM. No
> consistent centering-mode effect visible at this sample size.
> **Limitations / follow-up:** feeds
> `aime2025-sem-centermode-lam0.01-weff100`. n=30 questions
> means SEMs are wide (±.000 to ±.042) — treat gaps as noise
> until more trials accumulate.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=1)
<!-- table-id: tbl-6ef336 -->
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's per-step
> PRM scores collapse to one scalar — at `lam=0.01, ds_alpha=1.0`
> (`w_eff = ds_alpha/sqrt(lam) = 10`), the same checkpoint used in
> the `embeds_center_mode` tables above.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.01, ds_alpha=1.0 (w_eff=10),
> ds_beta=1.0, **run.num_trials=4** (see the cnt-mcts tables
> above).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-4498f8 -->
> **Compares:** same as the `ds_alpha=1.0` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.01, ds_alpha=10 (w_eff=100),
> ds_beta=1.0, **run.num_trials=4** (see the cnt-mcts tables
> above).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)
<!-- table-id: tbl-cfd7cf -->
> **Compares:** model family, size, and quantization jointly,
> scored with `prm=qwen` (Qwen-Math-7B-PRM), at `lam=0.01,
> ds_alpha=1.0` (`w_eff = ds_alpha/sqrt(lam) = 10`) — the same
> checkpoint used in the `embeds_center_mode` and `agg_strategy`
> tables above.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), lam=0.01, ds_alpha=1.0
> (w_eff=10), ds_beta=1.0, prm_batch_size=1, **run.num_trials=4**
> (see the cnt-mcts tables above).
>
> **W&B:** baselines cited from the `embeds_center_mode` table
> above (`none` rows) — llama-1b `kusajt8w`, llama-3b `llkxel7t`,
> qwen-3b `gztx392f`, qwen-7b gptq-int4 `yrcwjwpe`,
> qwen-math-1.5b `s87k7uxn`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | .0000<br>±.0000 | 1.34 |
| llama-3b fp16 | 4 | scored | .0417<br>±.0183 | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | 1.78 |
| qwen-3b fp16 | 4 | scored | .1833<br>±.0355 | .1250<br>±.0303 | .1083<br>±.0285 | .1000<br>±.0275 | 1.71 |
| qwen-7b gptq-int4 | 4 | scored | .2250<br>±.0383 | .1167<br>±.0294 | .1333<br>±.0312 | .1250<br>±.0303 | 1.59 |
| qwen-math-1.5b fp16 | 4 | scored | .2583<br>±.0401 | .1833<br>±.0355 | .1583<br>±.0335 | .1500<br>±.0327 | 1.45 |

> **Analysis.** 5/5 cells filled (cited from the
> `embeds_center_mode` table above — no new compute). Same
> ranking as the cnt-mcts QwenPRM table: qwen-math-1.5b >
> qwen-7b gptq-int4 ≈ qwen-3b > llama-3b > llama-1b.
> **Limitations / follow-up:** n=30 questions means SEMs are
> wide; treat as directional only.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-878af9 -->
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=80, proj=sparse512,
> cov_update=sm, ds_beta=1.0, prm_batch_size=1, tmpl=model-family
> default, run.num_trials=4) except **ds_alpha=10** (`w_eff=100`).
>
> **W&B:** baselines cited from the `embeds_center_mode` table
> above (`none` rows) — llama-1b `5dr4nnlo`, llama-3b `vlsck1e4`,
> qwen-3b `geje96pg`, qwen-7b gptq-int4 `30pfsbnl`,
> qwen-math-1.5b `bu8v110w`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.30 |
| llama-3b fp16 | 4 | scored | .0583<br>±.0215 | .0167<br>±.0117 | .0167<br>±.0117 | .0167<br>±.0117 | 1.90 |
| qwen-3b fp16 | 4 | scored | .1667<br>±.0342 | .1167<br>±.0294 | .0917<br>±.0265 | .0833<br>±.0253 | 1.77 |
| qwen-7b gptq-int4 | 4 | scored | .3000<br>±.0420 | .1833<br>±.0355 | .1333<br>±.0312 | .1167<br>±.0294 | 1.65 |
| qwen-math-1.5b fp16 | 4 | scored | .1917<br>±.0361 | .1417<br>±.0320 | .1333<br>±.0312 | .1250<br>±.0303 | 1.39 |

> **Analysis.** 5/5 cells filled (cited from the
> `embeds_center_mode` table above — no new compute). llama-1b
> is at floor (0/120 across all metrics). qwen-7b gptq-int4 is
> now the top pass@gb (.3000), ahead of qwen-math-1.5b (.1917) —
> a ranking flip vs. the `ds_alpha=1` checkpoint, though within
> ~1-2 SEM given n=30.
> **Limitations / follow-up:** n=30 questions means SEMs are
> wide; treat as directional only.


### sem-mcts-v02 [cov_scope=local]

> **Same implementation, two flags.** Everything in this section
> runs the same `mcts_sem_v02` code, `config_root`, and launcher
> as the section above, with `search.cov_scope=local` and
> `search.embeds_ref=relative` overridden — cells differ from
> their global twins only by those two overrides (and therefore
> by `config_hash`). The full mechanism argument lives in the
> PRM800K level-5 doc's `sem-mcts-v02 [cov_scope=local]`
> section; the short version:
>
> - **Local scope re-scales `w_eff`.** Global scope folds every
>   selection into one shared covariance, so the bonus has
>   decayed far below its nominal weight by mid-run; local scope
>   keeps each node's own fold count small, so the bonus stays
>   near `w_eff` for the whole run. The grid below is therefore
>   shifted down and denser at the low end than the global
>   sweeps above — **do not reuse the global operating point**.
> - **`embeds_ref=relative`** scores each child by its
>   displacement from the parent's embedding rather than by its
>   absolute position. On level-5 it matched or beat `absolute`
>   at the shared points on every qwen model and crossed sign on
>   both llama models (better at low `w_eff`, worse at high).
>   These tables sweep the `relative` arm directly; **no
>   `absolute` local sweeps exist for AIME2025**, so each table
>   reads against the global tables above, not against a paired
>   `absolute` twin.
> - **`lam` is held at 0.01**, inheriting the level-5 finding
>   that `lam` has no independent effect once `w_eff` is fixed.
>   The AIME2025 global sweeps' measured rows are consistent
>   with that within their wide error bars (largest spread:
>   qwen-7b at `w_eff=100`, .2083–.3000 across `lam`, ~1.6 SE).
> - **The transfer question.** Level-5 put the local+relative
>   optimum near `w_eff ≈ 1–3` on four of five models (qwen-math
>   is the exception, peaking at 10 among measured points) — an
>   order of magnitude below the global optimum of 10–100.
>   AIME2025 is much harder (pass@gb .03–.30 at b=80 against
>   .27–.78 on level-5), so whether the optimum location
>   transfers when the policy solves almost nothing is exactly
>   what this section asks.
> - **The `w_eff=0` rows are scope- and ref-independent.** With
>   `ds_alpha=0` the bonus is multiplied by zero, so each equals
>   the corresponding global sweep's (also unmeasured) `w_eff=0`
>   row — three hash-distinct configs, one number. Queue at most
>   one per model and cite it in all three places.
>
> Full section cost: 35 cells × 4 trials, ~230 GPU-hours. The
> `{0.3, 1, 3}` × {qwen-3b, qwen-7b, qwen-math} subset (9 cells,
> ~60 GPU-hours) brackets the level-5 optimum on the models with
> headroom to show an effect.

#### lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)
<!-- table-id: tbl-2ef3dc -->
> **Compares:** the level-5 relative sweep's grid (`tbl-ba6b11`)
> on AIME2025. llama-1b is where `embeds_ref` first crossed sign
> on level-5 — `relative` peaked at `w_eff` 1–3 (.3657/.3731)
> and fell by 10 (.3134) while `absolute` kept climbing. Whether
> any of that is visible here is doubtful: the model is at floor
> on AIME2025 (global `lam=0.01` anchors, `tbl-b1a6d9`: .0333
> pass@gb at `w_eff=10`, .0000 at 100), so this table is mostly
> a floor check, and the cheapest table in the section.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b, **lam=0.01**, data=aime2025,
> **run.num_trials=4** (see the cnt-mcts tables above).
>
> 6 of 7 cells **queued at priority 1.5** on 2026-08-03; the
> `w_eff=0` anchor stays `planned` (Tuan's call — no ledger
> entry until it is wanted). Hashes:
> `w_eff=0` `088d2e28`, `0.1` `1f36bdc6`, `0.3` `ec52f52b`,
> `1` `144f0dff`, `3` `a53d59bd`, `10` `86b819d7`, `100`
> `bdf1f032`. At ~1.4 hr/trial × 4 trials, ~39 GPU-hours.
>
> **W&B:** brw3pht0 (`w_eff=0.1`), x0g4ifht (`w_eff=0.3`),
> 15bl4u7o (`w_eff=1`), nciclzqg (`w_eff=3`),
> im2kla67 (`w_eff=10`), dlj079h6 (`w_eff=100`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | 4 | scored | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | .0000<br>±.0000 | 1.31 |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | 4 | scored | .0250<br>±.0143 | .0167<br>±.0117 | .0167<br>±.0117 | .0000<br>±.0000 | 1.27 |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 4 | scored | .0083<br>±.0083 | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | 1.29 |
| llama-1b | qwen | 0.01 | 0.3 | 3 | 4 | scored | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | 1.34 |
| llama-1b | qwen | 0.01 | 1.0 | 10 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.30 |
| llama-1b | qwen | 0.01 | 10 | 100 | 4 | scored | .0250<br>±.0143 | .0000<br>±.0000 | .0000<br>±.0000 | .0083<br>±.0083 | 1.31 |

> **Analysis.** Complete (6/7, closed 2026-08-03; only the
> `w_eff=0` anchor is unqueued). **The model is at the floor and
> the sweep is uninformative, exactly as predicted when this
> table was authored.** pass@gb reads .0083, .0250, .0083,
> .0083, .0083, .0250 across `w_eff` 0.1 → 100 — that is 1 or 3
> solved out of 120 graded, and every cell is within 1 SE of
> every other. maj@gb is .0000 in four of six cells. No `w_eff`
> can be ranked against any other here, and nothing about the
> level-5 llama-3b optimum at `w_eff=3` can be confirmed or
> refuted on this model.
> **What it does establish** is a negative control: local scope
> plus parent-relative embeddings does **not** rescue a weak
> policy on hard problems. The level-5 llama-1b relative sweep
> spans .2948–.3731 across the same grid; on AIME2025 the same
> configuration collapses to ~.01. Search geometry cannot
> substitute for a policy that almost never produces a correct
> chain — the diversity bonus only reallocates a budget among
> branches, and here no branch is right.
> **Limitations / follow-up:** at 1–3 solved of 120 the
> per-cell SE (~.008–.014) is larger than any plausible effect,
> so **do not read the two .0250 cells as a signal** — they are
> 3/120 against 1/120. The qwen tables below carry the actual
> transfer question; this one is now closed and needs no further
> compute. Feeds key: `tbl-2ef3dc`.

#### lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)
<!-- table-id: tbl-b94f3f -->
> **Compares:** the level-5 relative sweep's grid (`tbl-cf849a`)
> on AIME2025. Level-5 showed the llama-1b sign crossing
> **reproduces** on llama-3b (`tbl-7ee727`) — `relative` falls
> from .5784 at `w_eff=1` to .5485 at 10 while `absolute` rises
> — so the crossing is a llama-family property, and llama-3b is
> the family's only member with AIME2025 signal above floor.
> Global anchors (`tbl-d0ed2a`, `lam=0.01`): .0417 pass@gb at
> `w_eff=10`, .0583 at 100, .0750 at 1000 — a *late-peaking*
> global profile, so a local+relative optimum at `w_eff ≤ 3`
> here would be the clearest order-of-magnitude shift the
> section can show on a llama model.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b, **lam=0.01**, data=aime2025,
> **run.num_trials=4** (see the cnt-mcts tables above).
>
> 6 of 7 cells **queued at priority 1.5** on 2026-08-03; the
> `w_eff=0` anchor stays `planned` (Tuan's call — no ledger
> entry until it is wanted). Hashes:
> `w_eff=0` `84c25d6f`, `0.1` `d4e0f45f`, `0.3` `2dccd872`,
> `1` `d6134362`, `3` `2f426ea2`, `10` `a749ca4c`, `100`
> `83066656`. At ~1.9 hr/trial × 4 trials, ~53 GPU-hours — the
> most expensive table in the section.
>
> **W&B:** 2dlqe5yz (`w_eff=0.1`), 3epxxoh9 (`0.3`),
> kwziq7zb (`1`), m3q0x26b (`3`), mxcfrmy6 (`10`),
> lf5mom0m (`100`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | 4 | scored | .0667<br>±.0229 | .0250<br>±.0143 | .0250<br>±.0143 | .0083<br>±.0083 | 1.70 |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | 4 | scored | .0583<br>±.0215 | .0500<br>±.0200 | .0417<br>±.0183 | .0333<br>±.0165 | 1.69 |
| llama-3b | qwen | 0.01 | 0.1 | 1 | 4 | scored | .0667<br>±.0229 | .0333<br>±.0165 | .0250<br>±.0143 | .0167<br>±.0117 | 1.73 |
| llama-3b | qwen | 0.01 | 0.3 | 3 | 4 | scored | .0667<br>±.0229 | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | 1.79 |
| llama-3b | qwen | 0.01 | 1.0 | 10 | 4 | scored | .0500<br>±.0200 | .0417<br>±.0183 | .0167<br>±.0117 | .0167<br>±.0117 | 1.82 |
| llama-3b | qwen | 0.01 | 10 | 100 | 4 | scored | .0667<br>±.0229 | .0333<br>±.0165 | .0250<br>±.0143 | .0083<br>±.0083 | 1.84 |

> **Analysis.** Complete (6/7, closed 2026-08-03; only the
> `w_eff=0` anchor is unqueued). pass@gb is **flat across the
> entire grid**: .0667, .0583, .0667, .0667, .0500, .0667 for
> `w_eff` 0.1 → 100 — a total spread of .0167, well under one
> SE (~.022), and 6–8 solved of 120 in every cell. No optimum,
> no turnover, no ordering.
> **The level-5 result does not transfer.** On level-5 this
> model has the section's clearest interior optimum —
> `tbl-cf849a` rises .4440 → **.5821 at `w_eff=3`** → .5485 by
> 10, a +.138 span at ~4.5 SE. Here the same model on the same
> grid produces nothing distinguishable. The natural reading is
> that `w_eff` tuning only matters once the policy solves a
> non-trivial fraction of problems: at ~6 % there are too few
> correct branches for a diversity bonus to reallocate budget
> toward. Note the global AIME anchors (`tbl-d0ed2a`, .0417 at
> `w_eff=10`, .0583 at 100, .0750 at 1000) sit in the same band,
> so local+relative does not separate from global scope either.
> **Limitations / follow-up:** at 6–8 of 120 the per-cell SE
> (~.022) exceeds the whole observed spread, so this table can
> only report a null — it cannot bound how large a real effect
> would have to be. maj@gb is more scattered (.0083–.0333) but
> on 1–4 questions, so read nothing into it. More trials would
> be needed to say anything positive here; the qwen tables carry
> the transfer question. Feeds key: `tbl-b94f3f`.

#### lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)
<!-- table-id: tbl-435dd3 -->
> **Compares:** the level-5 relative sweep's grid (`tbl-b1cb82`)
> on AIME2025. On level-5, `relative` led `absolute` at both
> shared points (+.0112 at `w_eff=1`, +.0374 at 10) and was
> near-flat across 1–10 (.7164 → .7090), reading as robustness
> to over-weighted diversity rather than a higher peak. **No
> AIME2025 global qwen-3b sweep exists** — the only global
> anchors are the two family-table points (`tbl-cfd7cf` .1833 at
> `w_eff=10`, `tbl-878af9` .1667 at 100), so below `w_eff=10`
> this table is new territory for the model under either scope.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-3b, **lam=0.01**, data=aime2025,
> **run.num_trials=4** (see the cnt-mcts tables above).
>
> 6 of 7 cells **queued at priority 1.5** on 2026-08-03; the
> `w_eff=0` anchor stays `planned` (Tuan's call — no ledger
> entry until it is wanted). Hashes:
> `w_eff=0` `2ba68a6c`, `0.1` `762b7155`, `0.3` `270ad11f`,
> `1` `c2f78139`, `3` `642cb44d`, `10` `fc08cb02`, `100`
> `48c043bd`. At ~1.8 hr/trial × 4 trials, ~50 GPU-hours.
>
> **W&B:** 1si32174 (`w_eff=0.1`), t7wpqzcx (`0.3`),
> jko40dhd (`3`), 1ek80or7 (`10`), a6es7xcl (`100`);
> `w_eff=1` failed (preemption), no usable run.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| qwen-3b | qwen | 0.01 | 0.01 | 0.1 | 4 | scored | .2167<br>±.0378 | .1167<br>±.0294 | .1250<br>±.0303 | .1083<br>±.0285 | 1.61 |
| qwen-3b | qwen | 0.01 | 0.03 | 0.3 | 4 | scored | .1750<br>±.0348 | .1333<br>±.0312 | .0917<br>±.0265 | .0917<br>±.0265 | 1.65 |
| qwen-3b | qwen | 0.01 | 0.1 | 1 | — | running | — | — | — | — | — |
| qwen-3b | qwen | 0.01 | 0.3 | 3 | 4 | scored | .1917<br>±.0361 | .0917<br>±.0265 | .0417<br>±.0183 | .0500<br>±.0200 | 1.79 |
| qwen-3b | qwen | 0.01 | 1.0 | 10 | 4 | scored | .1417<br>±.0320 | .1167<br>±.0294 | .0750<br>±.0241 | .0833<br>±.0253 | 1.76 |
| qwen-3b | qwen | 0.01 | 10 | 100 | 4 | scored | .2083<br>±.0372 | .1333<br>±.0312 | .0833<br>±.0253 | .0917<br>±.0265 | 1.78 |

> **Analysis.** 5 of 7 cells measured (2026-08-03); `w_eff=0` is
> unqueued and **`w_eff=1` failed** (its allocation was
> preempted mid-run — see the ledger `history`). Measured:
> .2167 (`0.1`), .1750 (`0.3`), .1917 (`3`), .1417 (`10`),
> .2083 (`100`). Spread .0750 against SEs of ~.035, so the
> extremes are ~2 SE apart — suggestive but not separated, and
> the ordering is non-monotone (high at 0.1, dip at 0.3, second
> dip at 10, back up at 100). **Read it as no reliable `w_eff`
> effect**, with a weak hint that the low end is not worse than
> the high end.
> **The missing `w_eff=1` cell is the expensive one.** On
> level-5, qwen-3b's `relative` arm peaks at exactly `w_eff=1`
> (.7164) — so the one point this table would most want is the
> one the preemption took. Re-running it is the single highest-
> value cell in the AIME section.
> **Against the global anchors** (`tbl-cfd7cf` .1833 at
> `w_eff=10`, `tbl-878af9` .1667 at 100): local+relative is
> +.025 and +.042 at the shared points — same direction as
> level-5, but both inside 1 SE, so this is not yet evidence
> that local scope helps on AIME.
> **Limitations / follow-up:** n=120 graded, ~.035 SE per cell.
> Requeue `w_eff=1` before drawing any curve from this table;
> after that, the global qwen-3b low-end cells would make the
> comparison a scope effect rather than a low-`w_eff` effect
> (same caveat as level-5's qwen-3b sweep). Feeds key:
> `tbl-435dd3`.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)
<!-- table-id: tbl-7a3760 -->
> **Compares:** the level-5 relative sweep's grid (`tbl-5d64b1`)
> on AIME2025, on the strongest policy in the grid. On level-5,
> `relative` at `w_eff=1` (.7836) was this model's best number
> under any scope, decaying monotonically to .7537 by 100. The
> AIME2025 global anchors invert that shape (`tbl-ba8af1`,
> `lam=0.01`): .2250 at `w_eff=10` and **.3000 at 100** — the
> best AIME2025 pass@gb in the doc sits at a *high* `w_eff`.
> That makes this the section's sharpest transfer test: if
> local+relative moves the optimum an order of magnitude left
> here too, the best cell lands near `w_eff` 1–3; if AIME2025's
> hardness genuinely rewards heavier exploration, the peak stays
> right and the level-5 tuning does not transfer.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4, **lam=0.01**,
> data=aime2025, **run.num_trials=4** (see the cnt-mcts tables
> above).
>
> 6 of 7 cells **queued at priority 1.5** on 2026-08-03; the
> `w_eff=0` anchor stays `planned` (Tuan's call — no ledger
> entry until it is wanted). Hashes:
> `w_eff=0` `8831e73f`, `0.1` `eb21276e`, `0.3` `acad86d8`,
> `1` `d467e471`, `3` `9c16fdd5`, `10` `abe44056`, `100`
> `25367189`. At ~1.7 hr/trial × 4 trials, ~48 GPU-hours.
>
> **W&B:** vvtgz5hc (`w_eff=0.1`), 6tea8hre (`0.3`),
> 0pdey277 (`1`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | 4 | scored | .2583<br>±.0401 | .1583<br>±.0335 | .1333<br>±.0312 | .1500<br>±.0327 | 1.35 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | 4 | scored | .2667<br>±.0405 | .1750<br>±.0348 | .1833<br>±.0355 | .1500<br>±.0327 | 1.53 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | 4 | scored | .2833<br>±.0413 | .1667<br>±.0342 | .1667<br>±.0342 | .1583<br>±.0335 | 1.66 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3 | — | running | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 1.0 | 10 | — | running | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | — | running | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** queue this table first — it
> carries the transfer question, and the .3000 global anchor at
> `w_eff=100` is the number any local cell has to beat before
> the section can claim local scope helps on AIME2025. The
> `w_eff=100` cell doubles as the direct scope comparison at the
> global optimum. Feeds key: `tbl-7a3760`.

#### lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)
<!-- table-id: tbl-4ef506 -->
> **Compares:** the level-5 relative sweep's grid (`tbl-3a76ce`)
> on AIME2025. qwen-math-1.5b is the family outlier twice over:
> math-specialized embeddings, and the one level-5 model whose
> measured `relative` points peak at `w_eff=10` (.7612, vs
> .7425 at 1; its low end is still running). Global AIME2025
> anchors (`tbl-8bf48f`, `lam=0.01`): .2583 pass@gb at
> `w_eff=10`, .1917 at 100 — the earliest-declining global
> curve in the family, same as on level-5. Cheapest qwen table.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b, **lam=0.01**,
> data=aime2025, **run.num_trials=4** (see the cnt-mcts tables
> above).
>
> 6 of 7 cells **queued at priority 1.5** on 2026-08-03; the
> `w_eff=0` anchor stays `planned` (Tuan's call — no ledger
> entry until it is wanted). Hashes:
> `w_eff=0` `ad64a419`, `0.1` `ab915c9a`, `0.3` `db55f46c`,
> `1` `44397f4f`, `3` `819aebc2`, `10` `95395ca7`, `100`
> `602868da`. At ~1.45 hr/trial × 4 trials, ~41 GPU-hours.
>
> **W&B:** none yet (no AIME2025 local-scope runs).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.01 | 0.1 | — | running | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.03 | 0.3 | — | running | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.1 | 1 | — | running | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.3 | 3 | — | running | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 1.0 | 10 | — | running | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | — | running | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** level-5's low-end cells for this
> model (`tbl-3a76ce` `w_eff` 0.1/0.3/3/100) are still running;
> if they land before these are queued, re-read the Compares —
> a level-5 low-end peak would change which cells matter here.
> Feeds key: `tbl-4ef506`.


### cnt-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-c7dd39 -->
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts's equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama), **run.num_trials=4**.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

### cnt-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × cpuct) (qwen-3b, QwenPRM)
<!-- table-id: tbl-40a360 -->
> **Compares:** the two selectable v02 frontier scores head-to-head
> on one model. parent_blend arms sweep `alpha` (one-hop blend of a
> leaf's q with its parent's) at the file-default cpuct=2.0;
> path_decay arms sweep the full `gamma × cpuct` cross. Why cpuct
> is crossed with gamma rather than held fixed: path_decay's
> exploration term uses the AlphaZero shape
> `cpuct·sqrt(N_parent)/(1+N_leaf)` — no log damping, so at the
> default cpuct=2.0 it can swamp the `q_path` value term (range
> ~[0,1]) after a few backprops regardless of gamma. A flat gamma
> effect at cpuct=2.0 alone would be uninterpretable (gamma
> useless, or drowned out?). The cross separates the two stories:
> gamma mattering at cpuct=0.5 but not 2.0 confirms
> scale-domination; gamma inert at both scales is real evidence
> against full-path value reading. gamma semantics: 1.0 = plain
> path average, 0.8 = moderate decay, 0.5 = steep/near-local.
>
> **Fixed:** method=`mcts_bl_cnt_v02`, llm=qwen-3b fp16 (native
> tmpl), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | score_mode | alpha | gamma | cpuct | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.8 | — | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.6 | — | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | 0.5 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

### kube-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-d34700 -->
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled.
>
> **Fixed:** method=`mcts_bl_kube_v01` (renamed 2026-07-16 from
> `mcts_bl_cnt_v02`), prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent, kube_affordable=true,
> bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family default
> (native for Qwen, custom for Llama), **run.num_trials=4**. See
> `docs/decisions/bl-kube-bonus-schedule.md` for the schedule choice.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 0.8 |
| llama-3b fp16 | 4 | scored | .0667<br>±.0229 | .0250<br>±.0143 | .0250<br>±.0143 | .0250<br>±.0143 | 1.2 |
| qwen-3b fp16 | 4 | scored | .1000<br>±.0275 | .0833<br>±.0253 | .0833<br>±.0253 | .0833<br>±.0253 | 1.1 |
| qwen-7b gptq-int4 | 4 | scored | .2250<br>±.0383 | .1583<br>±.0335 | .1583<br>±.0335 | .1583<br>±.0335 | 1.05 |
| qwen-math-1.5b fp16 | 4 | scored | .1583<br>±.0335 | .1167<br>±.0294 | .1083<br>±.0285 | .1083<br>±.0285 | 0.9 |

> **Analysis.** AIME2025 is brutally hard for this scale: at
> b=80 the whole grid sits between 0.8% and 16% pass@gb, and the
> two Llama models are effectively at zero on every aggregated
> metric (llama-1b: `naive`/`wei`/`maj` all .0000 — its single
> pass@gb hit was never selected). Ordering is
> qwen-math-1.5b > qwen-3b > llama-3b > llama-1b, i.e. **math
> post-training beats parameter count** — the 1.5B math model
> doubles qwen-3b's aggregated accuracy at 80% of its cost/trial.
> The pass@gb-vs-maj@gb gap (llama-3b .0667 → .0250) says the
> searcher does find correct leaves it then fails to select;
> that selection gap, not generation, is the bottleneck here.
> **Limitations / follow-up:** qwen-7b gptq-int4 relaunched
> 2026-07-26 (its first attempt lost trials 2–4 to a cancelled
> allocation); with 30 questions × 4 trials the ±sem on a
> single-digit percentage is wide — treat sub-3% differences as
> noise.

#### kube_c sweep × model family (QwenPRM)
<!-- table-id: tbl-9d3944 -->
> **Compares:** sensitivity of kube-bl-v01 to the KUBE
> exploration coefficient `kube_c`, swept {0.1, 0.5, 2.0, 8.0}
> on the full 5-model/quant grid. 2.0 is the default (those 5
> cells are the exact runs of the model-family table above —
> reused, not re-run); 0.1/0.5 probe below it, 8.0 brackets
> above. Unlike the v02 alpha × kube_c joint sweep (llama-3b,
> parent_blend), this is a clean 1-D sweep — v01 has no
> score_mode/alpha knob, so kube_c is the only exploration
> scale and up/down bracketing is unaliased. Per-model read:
> whether the KUBE advantage over cnt-bl-v01 (seen at 2.0 on
> every model) is robust to the bonus scale, and whether the
> optimum shifts with model strength.
>
> **Fixed:** method=`mcts_bl_kube_v01`, prm=qwen,
> agg_strategy=`last`, kube_schedule=parent,
> kube_affordable=true, bs-4, d-20, b=80, prm_batch_size=1,
> level=5, tmpl=model-family default (native for Qwen, custom
> for Llama), **run.num_trials=4**.

| llm | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 0.1 | 4 | scored | .0167<br>±.0117 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 0.8 |
| llama-1b fp16 | 0.5 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 0.8 |
| llama-1b fp16 | 2.0 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 0.8 |
| llama-1b fp16 | 8.0 | 4 | scored | .0333<br>±.0165 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 0.8 |
| llama-3b fp16 | 0.1 | 4 | scored | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | .0167<br>±.0117 | 1.1 |
| llama-3b fp16 | 0.5 | 4 | scored | .0333<br>±.0165 | .0083<br>±.0083 | .0083<br>±.0083 | .0083<br>±.0083 | 1.2 |
| llama-3b fp16 | 2.0 | 4 | scored | .0667<br>±.0229 | .0250<br>±.0143 | .0250<br>±.0143 | .0250<br>±.0143 | 1.2 |
| llama-3b fp16 | 8.0 | 4 | scored | .0250<br>±.0143 | .0250<br>±.0143 | .0250<br>±.0143 | .0250<br>±.0143 | 1.2 |
| qwen-3b fp16 | 0.1 | 4 | scored | .1000<br>±.0275 | .0833<br>±.0253 | .0750<br>±.0241 | .0750<br>±.0241 | 1.1 |
| qwen-3b fp16 | 0.5 | 4 | scored | .1167<br>±.0294 | .1000<br>±.0275 | .1000<br>±.0275 | .1000<br>±.0275 | 1.1 |
| qwen-3b fp16 | 2.0 | 4 | scored | .1000<br>±.0275 | .0833<br>±.0253 | .0833<br>±.0253 | .0833<br>±.0253 | 1.1 |
| qwen-3b fp16 | 8.0 | 4 | scored | .1250<br>±.0303 | .1000<br>±.0275 | .1000<br>±.0275 | .0917<br>±.0265 | 1.2 |
| qwen-7b gptq-int4 | 0.1 | 4 | scored | .1917<br>±.0361 | .1667<br>±.0342 | .1000<br>±.0275 | .1083<br>±.0285 | 1.1 |
| qwen-7b gptq-int4 | 0.5 | 4 | scored | .1917<br>±.0361 | .1500<br>±.0327 | .1167<br>±.0294 | .1167<br>±.0294 | 1.0 |
| qwen-7b gptq-int4 | 2.0 | 4 | scored | .2250<br>±.0383 | .1583<br>±.0335 | .1583<br>±.0335 | .1583<br>±.0335 | 1.05 |
| qwen-7b gptq-int4 | 8.0 | 4 | scored | .2583<br>±.0401 | .1917<br>±.0361 | .1833<br>±.0355 | .1833<br>±.0355 | 1.1 |
| qwen-math-1.5b fp16 | 0.1 | 4 | scored | .2167<br>±.0378 | .1583<br>±.0335 | .1333<br>±.0312 | .1333<br>±.0312 | 0.9 |
| qwen-math-1.5b fp16 | 0.5 | 4 | scored | .1917<br>±.0361 | .1583<br>±.0335 | .1500<br>±.0327 | .1500<br>±.0327 | 0.9 |
| qwen-math-1.5b fp16 | 2.0 | 4 | scored | .1583<br>±.0335 | .1167<br>±.0294 | .1083<br>±.0285 | .1083<br>±.0285 | 0.9 |
| qwen-math-1.5b fp16 | 8.0 | 4 | scored | .1667<br>±.0342 | .1083<br>±.0285 | .1083<br>±.0285 | .0917<br>±.0265 | 0.9 |

> **Analysis.** 19/20 cells filled (2026-07-26). **The kube_c
> optimum is model-dependent, and it moves the opposite way from
> what the prm800k-level5 sweep suggested.** The two strongest
> models want *more* exploration: qwen-7b peaks at c=8.0
> (.2583 pass@gb vs .1917 at both 0.1 and 0.5 — a +35% relative
> jump, ~1.6σ) and qwen-3b trends the same way (.1250 at 8.0 vs
> .1000 at 0.1/2.0). qwen-math-1.5b inverts it, peaking at the
> *bottom* of the range (.2167 at c=0.1, falling to .1583 at
> 2.0) — a math-tuned policy proposes good children early, so
> bonus-driven width mostly buys noise. llama-3b's best is the
> default 2.0 and llama-1b is at floor everywhere (all
> aggregated metrics .0000 regardless of c) — below some
> capability threshold the exploration coefficient simply has
> nothing to trade off.
>
> Read against the aggregated columns the story tightens: for
> qwen-7b, c=8.0 lifts `maj@gb` from .1083 → .1833, so the extra
> exploration finds leaves that *survive selection*, not just
> pass@gb lottery tickets. Cost is flat across c (≤0.1 hr/trial
> spread within each model), so on AIME the coefficient is a
> free knob — worth tuning per model rather than fixing at 2.0.
>
> **Limitations / follow-up:** qwen-7b c=2.0 still running
> (relaunched 07-26), so its column has a hole exactly at the
> default — the c=8.0-beats-default claim for that model rests
> on the 0.1/0.5 comparison until it lands. 30 questions × 4
> trials means ±3–4 points on every cell; only the qwen-7b
> 8.0-vs-0.5 and qwen-math 0.1-vs-2.0 contrasts clear ~1.5σ.
> Worth re-running the winner cells at more trials before
> claiming a per-model tuning rule.

### kube-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × kube_c) (qwen-3b, QwenPRM)
<!-- table-id: tbl-bdeba2 -->
> **Compares:** the two selectable v02 frontier densities
> head-to-head, mirroring the cnt-mcts-bl-v02 score_mode sweep
> above cell-for-cell (same model, PRM, level, budget, arm grid) —
> so the two families' sweeps are directly comparable: under
> kube_schedule=parent (fixed here), kube's path_decay density is
> exactly bl_cnt v02's path_decay score divided by remaining cost,
> and kube's parent_blend bonus is exactly bl_cnt's PUCT bonus
> over cost. Any ranking difference between the families is
> attributable to the /cost division alone. The gamma × kube_c
> cross rationale is the same as the cnt table's (see its blurb):
> the AZ-shaped path_decay bonus has no log damping, so
> kube_c=2.0 may swamp the value term regardless of gamma — the
> cross separates "gamma useless" from "gamma drowned out".
>
> **Fixed:** method=`mcts_bl_kube_v02`, kube_schedule=`parent`,
> kube_affordable=true, llm=qwen-3b fp16 (native tmpl), prm=qwen,
> agg_strategy=`last`, bs-4, d-20, b=80, prm_batch_size=1,
> level=5, **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | score_mode | alpha | gamma | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | — | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.8 | — | — | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.6 | — | — | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | 0.5 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)
<!-- table-id: tbl-bda3a8 -->
> **Compares:** model family, size, and quantization jointly at
> the winning-candidate frontier score `score_mode=parent_blend`
> with `alpha=0.8` — same 5-model/quant grid as cnt-mcts-bl-v01's
> equivalent table above, so a direct kube_v02-vs-cnt-v01 (and,
> across the bl families' model-family tables) read is possible
> once filled. The qwen-3b cell is the **exact same run** as the
> `parent_blend/alpha=0.8` arm of the score_mode sweep above
> (planned for AIME2025) — reused, not re-run.
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=0.8**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
<!-- table-id: tbl-b3d812 -->
> **Compares:** the same 5-model/quant grid as the
> `parent_blend/alpha=0.8` table above, but at **alpha=1.0** — the
> exact-v01 control arm (no parent blend: `blended_q = q(leaf)`,
> recovering `BLMCTSKubeV01Config`'s kube_density exactly). Read
> against the alpha=0.8 table, this isolates whether the one-hop
> q-blend helps or hurts per model family. On qwen-3b the control
> currently wins (.6381 vs .6194), so this table tests whether
> that holds across models. qwen-3b reuses the score_mode sweep's
> alpha=1.0 arm (planned for AIME2025).
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=1.0**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.0)
<!-- table-id: tbl-7c1779 -->
> **Compares:** the same 5-model/quant grid at the **value-blindness
> extreme alpha=0.0** — the leaf's own q is ignored entirely and
> every child is scored by its parent's q alone
> (`blended_q = q(parent)`). Within a sibling set the value term is
> then identical for all children, so selection among siblings is
> driven purely by the exploration/visit term and the kube bonus.
> This is an ablation, not a tuning arm: it asks whether the leaf's
> own PRM signal matters at all under KUBE, bounding the alpha
> sweep from below. Read against the alpha=1.0 (pure own-q) and
> alpha=0.8 tables: a small 0.0→1.0 gap would mean per-leaf PRM
> discrimination contributes little beyond tree structure; a large
> gap confirms the leaf q is load-bearing. Unlike the 0.8/1.0
> tables, the qwen-3b cell is NOT reusable from the score_mode
> sweep (no alpha=0.0 arm exists there) — all 5 cells are net-new.
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=0.0**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### alpha × kube_c joint sweep (llama-3b, QwenPRM, parent_blend)
<!-- table-id: tbl-86e0b6 -->
> **Compares:** the parent_blend value-composition knob `alpha`
> jointly with the exploration scale `kube_c`, as a 3×3 factorial
> — NOT two sequential 1-D sweeps. The two parameters are
> partially aliased by construction: among siblings the parent-q
> term is a shared constant, so value discrimination is exactly
> `alpha·Δq` and behavior depends only on the ratio
> `kube_c/alpha`; alpha's genuinely new effect (importing the
> parent's q into cross-branch comparisons) is only identifiable
> against the factorial's interaction pattern. Reads: (1) kube_c
> main effect down the alpha=1.0 column; (2) alpha spread at
> kube_c=0.5/0.1 vs. at 2.0 — separation only at low kube_c
> confirms scale-domination (tune alpha in the low-c regime);
> flat everywhere is real evidence against one-hop blending;
> (3) the (a=1.0,c=2.0) vs (a=0.5,c=1.0-equivalent) diagonal is
> approximated by comparing constant-ratio pairs across the grid.
> kube_c grid {0.1, 0.5, 2.0} straddles the default from below —
> q ∈ [0,1] bounds sibling Δq to a few tenths, so the
> value↔bonus crossover plausibly sits below the default, not
> above (see the score_mode sweep's scale-domination rationale).
>
> **Fixed:** method=`mcts_bl_kube_v02`, llm=llama-3b fp16 (custom
> tmpl), **score_mode=parent_blend**, kube_schedule=`parent`,
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | alpha | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-3b | 1.0 | 2.0 | — | planned | — | — | — | — | — |
| llama-3b | 1.0 | 0.5 | — | planned | — | — | — | — | — |
| llama-3b | 1.0 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | 0.8 | 2.0 | — | planned | — | — | — | — | — |
| llama-3b | 0.8 | 0.5 | — | planned | — | — | — | — | — |
| llama-3b | 0.8 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | 0.5 | 2.0 | — | planned | — | — | — | — | — |
| llama-3b | 0.5 | 0.5 | — | planned | — | — | — | — | — |
| llama-3b | 0.5 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | 0.0 | 2.0 | — | planned | — | — | — | — | — |
| llama-3b | 0.0 | 0.5 | — | planned | — | — | — | — | — |
| llama-3b | 0.0 | 0.1 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### gamma × kube_c joint sweep (qwen-3b, QwenPRM, path_decay)
<!-- table-id: tbl-434915 -->
> **Compares:** the `path_decay` score_mode's two knobs jointly
> — the per-hop value decay `gamma` (rows) against the
> exploration-bonus coefficient `kube_c` (columns). The
> path_decay sibling of the alpha × kube_c parent_blend sweep
> above: same grid shape, the *other* v02 scorer. Reads whether
> gamma separates once kube_c is swept, and where the g0.5
> penalty (seen in the score_mode tables) sits across the bonus
> range.
>
> **Fixed:** method=`mcts_bl_kube_v02`, llm=qwen-3b fp16 (native
> tmpl), **score_mode=path_decay**, kube_schedule=`parent`,
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | gamma | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | 1.0 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | 1.0 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | 1.0 | 0.1 | — | planned | — | — | — | — | — |
| qwen-3b | 0.8 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | 0.8 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | 0.8 | 0.1 | — | planned | — | — | — | — | — |
| qwen-3b | 0.5 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | 0.5 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | 0.5 | 0.1 | — | planned | — | — | — | — | — |
| qwen-3b | 0.0 | 2.0 | — | planned | — | — | — | — | — |
| qwen-3b | 0.0 | 0.5 | — | planned | — | — | — | — | — |
| qwen-3b | 0.0 | 0.1 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

### kdepth-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-acca74 -->
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
> default (native for Qwen, custom for Llama), **run.num_trials=4**.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, depth_alpha=0.5)
<!-- table-id: tbl-b429d0 -->
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as the `depth_alpha=1.0` table above,
> but with a **concave** depth-bonus curve (`f(z)=1-z^0.5`, bonus
> stays high deeper into the tree). Read against the α=1.0 and
> α=2.0 tables, this isolates how the depth-bonus curvature
> interacts with model family/size at fixed `depth_beta=2.0`.
>
> **Fixed:** method=`mcts_bl_kdepth_v01`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, **depth_alpha=0.5**, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, depth_alpha=2.0)
<!-- table-id: tbl-3fbc68 -->
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as the `depth_alpha=1.0` table above,
> but with a **convex** depth-bonus curve (`f(z)=1-z^2`, bonus
> collapses fast → strongly favors shallow nodes). The α=2.0
> counterpart to the α=0.5 table, so the three tables
> (0.5/1.0/2.0) span the curvature axis at fixed
> `depth_beta=2.0`.
>
> **Fixed:** method=`mcts_bl_kdepth_v01`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, **depth_alpha=2.0**, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

### kdepth-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma) (qwen-3b, QwenPRM)
<!-- table-id: tbl-59ccb9 -->
> **Compares:** the two selectable v02 frontier densities
> head-to-head, mirroring the kube-mcts-bl-v02 score_mode sweep
> above — but on kdepth's density, whose exploration term is the
> fixed DEPTH bonus (`depth_beta*(1-depth_frac**depth_alpha)`),
> not a visit/clock bonus. So the score_mode blend here touches
> ONLY the value (q) term; the depth bonus and `/cost` are shared
> across both modes (see docs/decisions-log.md 2026-07-21). There
> is no `kube_c` axis (kdepth has no bonus coefficient to sweep —
> `depth_beta`/`depth_alpha` are held fixed), so path_decay varies
> gamma alone. The read: does path-aware value (one-hop parent
> blend, or gamma-decayed full path) beat plain own-q under a
> depth-shaped frontier?
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, depth_beta=2.0,
> depth_alpha=1.0, kube_affordable=true, llm=qwen-3b fp16 (native
> tmpl), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | score_mode | alpha | gamma | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.8 | — | — | planned | — | — | — | — | — |
| qwen-3b | parent_blend | 0.6 | — | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)
<!-- table-id: tbl-d81f40 -->
> **Compares:** model family, size, and quantization jointly at
> `score_mode=parent_blend` with `alpha=0.8` — same 5-model/quant
> grid as kdepth-mcts-bl-v01's and kube-mcts-bl-v02's equivalent
> tables, so a direct v01-vs-v02 (blend vs. no channel) and
> cross-family (kdepth vs. kube) read is possible once filled. The
> qwen-3b cell is the **exact same run** as the
> `parent_blend/alpha=0.8` arm of the score_mode sweep above
> (planned for AIME2025) — reused, not re-run.
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, **score_mode=parent_blend,
> alpha=0.8**, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
<!-- table-id: tbl-288646 -->
> **Compares:** the same 5-model/quant grid as the
> `parent_blend/alpha=0.8` table above, but at **alpha=1.0** — the
> exact-v01 control arm (no parent blend: `blended_q = q(leaf)`,
> recovering `BLMCTSKdepthV01Config`'s depth_density exactly). Read
> against the alpha=0.8 table, this isolates whether the one-hop
> q-blend helps or hurts per model family under a depth-shaped
> frontier. qwen-3b reuses the score_mode sweep's alpha=1.0 arm
> (planned for AIME2025).
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, **score_mode=parent_blend,
> alpha=1.0**, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

### sem-mcts-bl-v01

> **Note (2026-07-26):** the two `max_model_len`-default tables
> that used to head this section (`w_eff=100` and `w_eff=10`,
> tbl-065cf2 and the old tbl-b3f9bb config) were removed unrun.
> Every sem-bl llama cell on prm800k has died of context
> overflow at the 5000 default, so a default-window AIME grid
> would only reproduce that failure; both `w_eff` points are now
> tracked at `max_model_len=6000` below.

#### model family comparison (QwenPRM, lam=0.01/ds_alpha=10, max_model_len=6000)
<!-- table-id: tbl-df1eeb -->
> **Compares:** the SAME 5-model/quant grid as the
> `max_model_len=5000` table directly above (`tbl-c43f9b`), but
> with the vLLM context window raised to **6000** tokens
> (`llm.max_model_len=6000`). The question is diagnostic, not a
> new tuning axis: does the extra headroom let the sem-bl-v01
> **llama** cells run to completion? At the default 5000 those
> deep-search trajectories can overflow the context and the run
> raises (the same class of failure tracked in
> `docs/decisions/context-length-overflow-guard.md`). A
> row-by-row read against the 5000 table tells whether 6000 is
> enough headroom, and — where both complete — whether the
> larger window shifts pass@gb at all.
>
> **Fixed:** method=`mcts_bl_sem_v01` (PRM embeds — prm=qwen for
> both scoring AND diversity via `embeds_source=prm`, the schema
> default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_proj=sparse512` (`embeds_dim=512`), ds_beta=1.0,
> tmpl=model-family default (native for Qwen, custom for Llama), **run.num_trials=4**.
> **lam=0.01, ds_alpha=10** (`w_eff = ds_alpha/sqrt(lam) = 100`).
> **`max_model_len=6000`** (this table's whole point; hash-
> relevant, so every cell is a distinct config from the 5000
> table — nothing is shared).
>
> ⚠️ **qwen-math-1.5b cannot appear in this table at 6000:** its
> `max_position_embeddings=4096`, so vLLM rejects
> `max_model_len=6000` at engine construction. **Launched
> 2026-07-28 at the default 4096 window**
> (`sem-bl-v01-aime-mml4096-qwenmath15b`, `cfg-b82b9a25`): the
> row is NOT window-matched to the other four — read it as a
> diagnostic point, not as a like-for-like family comparison.
> Its `w_eff=10` counterpart is
> `sem-bl-v01-aime-ds1-mml4096-qwenmath15b` in the table below.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 2.22 |
| llama-3b fp16 | 4 | scored | .0500<br>±.0200 | .0167<br>±.0117 | .0000<br>±.0000 | .0000<br>±.0000 | 3.33 |
| qwen-3b fp16 | 4 | scored | .0667<br>±.0229 | .0250<br>±.0143 | .0167<br>±.0117 | .0250<br>±.0143 | 1.76 |
| qwen-7b gptq-int4 | 4 | scored | .1833<br>±.0355 | .1167<br>±.0294 | .0500<br>±.0200 | .0417<br>±.0183 | 1.71 |
| qwen-math-1.5b fp16 | 4 | scored | .1750<br>±.0348 | .0833<br>±.0253 | .0750<br>±.0241 | .0583<br>±.0215 | 1.53 |

> **Analysis.** Table complete (W&B me5s4zp4 for the
> qwen-math-1.5b row). The window question this table was built
> to answer is settled: both llama cells ran to completion at
> 6000, so the headroom is sufficient — no overflow failures.
> qwen-math-1.5b at its native 4096 window lands at .1750
> pass@gb, statistically level with qwen-7b's .1833 and well
> above the two llamas.
> **Limitations / follow-up:** the qwen-math-1.5b row is NOT
> window-matched (4096 vs 6000), so it is a diagnostic point,
> not part of the like-for-like family comparison. Whether 6000
> vs 5000 shifts pass@gb cannot be read here — the 5000 table's
> llama cells never completed, which is precisely why this
> table exists.

#### model family comparison (QwenPRM, lam=0.01/ds_alpha=1, max_model_len=6000)
<!-- table-id: tbl-b3f9bb -->
> **Compares:** same grid as the `lam=0.01/ds_alpha=10` table
> above, at one order of magnitude lower effective diversity
> weight — the two tables together give a first (coarse) read on
> whether the model-family ranking is sensitive to `w_eff` for
> this algorithm, ahead of a proper `w_eff` sweep. Anchored to
> sem-mcts-v02's `lam=0.01/ds_alpha=1` table above (`w_eff=10`).
>
> **Fixed:** identical to the `lam=0.01/ds_alpha=10` table above
> (method=`mcts_bl_sem_v01`, prm=qwen, agg_strategy=`last`, bs-4,
> d-20, b=80, prm_batch_size=1, `ds_alpha_schedule=global`,
> `cov_update=sm`, `embeds_proj=sparse512`, ds_beta=1.0, tmpl=
> model-family default, **`llm.max_model_len=6000`**) except the
> diversity weight, **run.num_trials=4**.
> **lam=0.01, ds_alpha=1.0** (`w_eff = ds_alpha/sqrt(lam) = 10`).
>
> **Why `max_model_len=6000` (changed 2026-07-26):** this table
> was authored at the 5000 default and re-specified at 6000
> before any cell ran, so nothing is lost and every cell keeps a
> single config hash. Rationale is the sem-bl overflow record on
> prm800k: every llama sem-bl cell has failed with
> `decoder prompt + output > max_model_len` at 5000 (level-4 and
> level-5, v01 and v02), while no qwen cell has. 6000 is the
> same headroom the level-5 `tbl-9f7cda` table uses.
>
> ⚠️ **qwen-math-1.5b cannot appear in this table at 6000:** its
> `max_position_embeddings=4096`, so vLLM rejects
> `max_model_len=6000` at engine construction. Same exclusion as
> the `ds_alpha=10` table above. Its `w_eff` point is therefore
> measured at 4096 instead (see the level-5 b=320 mml4096
> diagnostics). **Launched 2026-07-28 at the default 4096
> window** (`sem-bl-v01-aime-ds1-mml4096-qwenmath15b`,
> `cfg-29c1f655`): the row is marked `mml4096` and is NOT
> window-matched to the other four — read it as a diagnostic
> point, not as a like-for-like family comparison.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 4 | scored | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | .0000<br>±.0000 | 1.76 |
| llama-3b fp16 | 4 | scored | .0417<br>±.0183 | .0250<br>±.0143 | .0083<br>±.0083 | .0000<br>±.0000 | 2.25 |
| qwen-3b fp16 | 4 | scored | .1250<br>±.0303 | .0917<br>±.0265 | .1000<br>±.0275 | .0833<br>±.0253 | 1.75 |
| qwen-7b gptq-int4 | 4 | scored | .2083<br>±.0372 | .1583<br>±.0335 | .1667<br>±.0342 | .1583<br>±.0335 | 1.69 |
| qwen-math-1.5b fp16 | 4 | scored | .2083<br>±.0372 | .1583<br>±.0335 | .1667<br>±.0342 | .1667<br>±.0342 | 1.42 |

> **Analysis.** Table complete (W&B ofcu3huf for the
> qwen-math-1.5b row). Read against the `ds_alpha=10` table
> above, the lower diversity weight is better for the two Qwens
> and neutral for the Llamas: qwen-3b .0667 → .1250, qwen-7b
> .1833 → .2083, qwen-math-1.5b .1750 → .2083 pass@gb, with the
> aggregated metrics moving in the same direction and by more
> (qwen-math-1.5b maj@gb .0583 → .1667). qwen-math-1.5b ties
> qwen-7b exactly on pass@gb at ~1/5 the parameters and the
> lowest hr/trial in the table.
> **Limitations / follow-up:** the qwen-math-1.5b row runs at a
> 4096 window against the others' 6000, so its comparison to
> qwen-7b conflates model and context length. The `w_eff`
> direction here (lower is better) is the opposite of what the
> b=320 sem-mcts-v02 tables show, where `w_eff` made no
> difference at all — worth checking whether that is a b=80 vs
> b=320 effect or a sem-bl vs sem-mcts one.

### sem-mcts-bl-v02

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-396f65 -->
> **Compares:** model family, size, and quantization jointly for
> sem-mcts-bl-v02 at its selectable frontier value term fixed to
> `score_mode=parent_blend, alpha=1.0` (the no-blend control:
> `q_term = q(leaf)`, byte-identical to `score_mode=own`), with the
> diversity knobs pinned to the same `lam=0.01/ds_alpha=10`
> (`w_eff=100`) checkpoint as the sem-mcts-bl-v01 table above and
> sem-mcts-v02's tables — so bl_sem_v02-vs-bl_sem_v01 and, cell-
> for-cell, bl_sem_v02-vs-kube_v02 (both at parent_blend/alpha=1.0)
> reads are possible once filled.
>
> **Fixed:** method=`mcts_bl_sem_v02`, **score_mode=parent_blend,
> alpha=1.0**, embeds_source=prm, **lam=0.01, ds_alpha=10**
> (`w_eff=100`), ds_beta=1.0 (default), ds_alpha_schedule=global
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama), **run.num_trials=4**.
>
> ⚠️ Entirely planned — no AIME2025 runs yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the prm800k-level5 counterpart's command with `data=aime2025`.

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

### cnt-mcts

#### model family comparison (b=320, QwenPRM)
<!-- table-id: tbl-f31bf0 -->
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
> tmpl=model-family default (native for Qwen, custom for Llama),
> **run.num_trials=4** (see the [gen_budget=80] tables above).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; AIME2025's much
> longer reasoning traces (§ top-of-doc contamination/size note)
> than PRM800K may make this scale differently than PRM800K's
> b=320 wall-clock did — no reference point exists yet, and n=30
> means even a 4x budget increase may not narrow the wide SEMs
> much.
>
> **W&B:** none yet (no AIME2025 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 4 | scored | .0333<br>±.0165 | .0167<br>±.0117 | .0167<br>±.0117 | .0000<br>±.0000 | 3.23 |
| llama-3b fp16 | qwen | 4 | scored | .1333<br>±.0312 | .0500<br>±.0200 | .0333<br>±.0165 | .0250<br>±.0143 | 5.31 |
| qwen-3b fp16 | qwen | 4 | scored | .2833<br>±.0413 | .1167<br>±.0294 | .1000<br>±.0275 | .0833<br>±.0253 | 5.16 |
| qwen-7b gptq-int4 | qwen | 4 | scored | .4000<br>±.0449 | .1417<br>±.0320 | .1667<br>±.0342 | .1667<br>±.0342 | 4.79 |
| qwen-math-1.5b fp16 | qwen | 4 | scored | .3833<br>±.0446 | .2000<br>±.0367 | .2000<br>±.0367 | .2000<br>±.0367 | 4.00 |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

### sem-mcts-v02

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=10)
<!-- table-id: tbl-b2d2d2 -->
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
> table). **run.num_trials=4** (see the [gen_budget=80] tables
> above).
>
> Measured 4.3–6.5 hr/trial across the five models.
>
> **W&B:** 5zu4q5qs (qwen-3b), teeguzv7 (qwen-7b). llama-1b,
> llama-3b and qwen-math-1.5b were scored in an earlier cycle.

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 4 | scored | .0417<br>±.0183 | .0083<br>±.0083 | .0000<br>±.0000 | .0000<br>±.0000 | 5.24 |
| llama-3b fp16 | qwen | 4 | scored | .1250<br>±.0303 | .0333<br>±.0165 | .0500<br>±.0200 | .0333<br>±.0165 | 6.50 |
| qwen-3b fp16 | qwen | 4 | scored | .3167<br>±.0426 | .1333<br>±.0312 | .1667<br>±.0342 | .1583<br>±.0335 | 6.32 |
| qwen-7b gptq-int4 | qwen | 4 | scored | .3083<br>±.0423 | .1833<br>±.0355 | .1667<br>±.0342 | .1750<br>±.0348 | 4.28 |
| qwen-math-1.5b fp16 | qwen | 4 | scored | .3667<br>±.0442 | .1667<br>±.0342 | .1917<br>±.0361 | .1917<br>±.0361 | 4.93 |

> **Analysis.** Table complete. AIME2025 splits the models into
> two tiers with nothing in between: the two Llamas reach
> .0417/.1250 pass@gb while the three Qwens cluster at
> .3083–.3667. qwen-math-1.5b leads on pass@gb despite being the
> smallest Qwen, and llama-1b's wei/maj are exactly .0000 — it
> never once agreed with itself on a correct answer.
> **Limitations / follow-up:** pass@gb is 2–3× the aggregated
> metrics on every Qwen row, so the search finds correct
> solutions that none of the three aggregators can select. That
> gap, not the model ranking, is the actionable finding here.

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=100)
<!-- table-id: tbl-9d68e9 -->
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
> default, **run.num_trials=4**) except the diversity weight.
> **lam=0.1, ds_alpha=31.6** (`w_eff=100`).
>
> Measured 5.4–7.7 hr/trial across the five models.
>
> **W&B:** ua58yrf0 (llama-1b), 3tabmsr6 (llama-3b), tq780e5r
> (qwen-3b), 1whjad9v (qwen-7b), yzl0mbhl (qwen-math-1.5b).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 4 | scored | .0333<br>±.0165 | .0083<br>±.0083 | .0167<br>±.0117 | .0083<br>±.0083 | 5.64 |
| llama-3b fp16 | qwen | 4 | scored | .1500<br>±.0327 | .0583<br>±.0215 | .0500<br>±.0200 | .0333<br>±.0165 | 7.69 |
| qwen-3b fp16 | qwen | 4 | scored | .3250<br>±.0429 | .1167<br>±.0294 | .0917<br>±.0265 | .0917<br>±.0265 | 6.81 |
| qwen-7b gptq-int4 | qwen | 4 | scored | .3083<br>±.0423 | .1500<br>±.0327 | .1500<br>±.0327 | .1500<br>±.0327 | 6.04 |
| qwen-math-1.5b fp16 | qwen | 4 | scored | .3667<br>±.0442 | .1333<br>±.0312 | .1500<br>±.0327 | .1583<br>±.0335 | 5.40 |

> **Analysis.** Table complete, and paired with the `w_eff=10`
> table above this isolates `w_eff` at b=320. pass@gb is
> essentially unchanged by the 10× diversity weight — the
> largest move on any model is .0167 (llama-1b .0417 → .0333,
> qwen-3b .3167 → .3250), all well inside one standard error.
> The aggregated metrics drift down slightly more (qwen-3b
> wei@gb .1667 → .0917, maj@gb .1583 → .0917), but at ±.03 those
> are also not separable.
> **Limitations / follow-up:** with 4 trials on 30 questions,
> ±.03–.04 is the resolution floor here, and the whole `w_eff`
> effect lives below it. Deciding this axis on AIME2025 would
> need either many more trials or a metric with less variance;
> the level-5 sweeps are the better place to tune it.

---
