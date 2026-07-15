# LLM Reasoning — MCTS Experiment Comparison — AIME2025

> **Provenance:** structure mirrored from [exp-comp-gsm8k.md](exp-comp-gsm8k.md) (the GSM8K doc) on 2026-07-14; every table reset to `planned` — no AIME2025 runs exist yet. Launch commands are the GSM8K counterparts' with `data=aime2025` instead of `data=gsm8k` (config hashes and run names follow automatically; AIME2025 has no `level`, same as GSM8K, so `--level-N--` is omitted from run names — see `config_name()` in `utils/configs.py`). **AIME2025 has no prior grid to inherit** in the sense that matters most: it's not just a different dataset, it's a **much smaller one** — 30 questions total (one full AIME administration, I+II, 15 problems each) vs. GSM8K's 256-row test subset. Every cell's SEM will be markedly wider than the GSM8K table's as a direct consequence (a single flipped question moves pass@gb by ~3.3 percentage points on n=30, vs. ~0.4 on n=256) — treat all inherited `Compares`/`Fixed` prose as a starting design, not a result, and read early pass@gb numbers here with that sample-size caveat in mind throughout.
>
> **Scoring:** AIME2025 runs through the same `compute_stats.py` pipeline as GSM8K/PRM800K, selected via `data=aime2025` (`conf/data/aime2025.yaml`, `ds_split=train` — the only split that exists in the downloaded data, despite the name; see the file's contents, 30 rows). Ground-truth parsing and answer extraction are dataset-aware via `cfg.data.grader_name` (`"aime"` here vs. `"gsm8k"`/`"math"`). AIME's ground truth is always a bare integer 0–999 (a hard competition rule, not a formatting convention) — see `utils/parser.py::parse_ground_truth`'s `aime` branch (reads the `answer` field directly, no `\boxed{}` or `####` parsing needed for ground truth). Answer *extraction* from the model's own generation reuses the same `\boxed{}`-pulling logic as MATH/PRM800K unchanged, since the prompt convention ("put your final answer in `\boxed{}`") is the same regardless of dataset. Added 2026-07-14.
>
> **Contamination note:** AIME2025 problems were posted publicly (AoPS wiki, competition forums) within hours of the actual contest. Any model trained after early-to-mid 2025 may have seen these problems/solutions verbatim — treat pass@gb here as an upper bound on genuine reasoning ability for such models, not a clean measurement. AIME2024 (`data=aime2024`, not yet given its own tracking doc) is older and more likely contaminated for current-generation models; AIME2025 is the more useful of the two if a contamination-lite read matters.

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on AIME2025 — per-algorithm tuning tables grouped
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
> `cnt-mcts-bl-v01`, `cnt-mcts-bl-v02`, `cnt-mcts-bl-v03`,
> `sem-mcts-bl-v01`). All rows fixed at b=80, bs-4, d-20,
> agg_strategy=`last`, tmpl=model-family default (native for Qwen,
> custom for Llama), prm=qwen. `cnt-mcts` row is method=`mcts_cnt_v01`
> (the only cnt-mcts entry point at this level — see the
> `### cnt-mcts` section above). `sem-mcts` row is `mcts_sem_v02` (PRM embeds),
> `ds_alpha=100` (w_eff not applicable — that knob is bl_sem-specific).
> `sem-mcts-bl-v01` row uses the `w_eff=100` table; see that
> algorithm's own section for the `w_eff=10` comparison point.
> `cnt-mcts-bl-v02` (Fractional KUBE) and `cnt-mcts-bl-v03`
> (depth-shaping) — see `docs/decisions/kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/depth-shaping-knapsack-bonus.md` for the
> algorithms.

**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
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
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama),
> **run.num_trials=4** (see the RLHFlowPRM table above).
> Companion to the rlhflow-PRM table above; same 5 model/quant
> configs, different scoring PRM.
>
> **W&B:** none yet (no AIME2025 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`. n=30
> questions total (vs. GSM8K's 256) means SEMs here will be much
> wider than the GSM8K table's — a single flipped question moves
> pass@gb by ~3.3 percentage points, so treat early cells as
> noisy until more trials accumulate.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
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
| llama-1b | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-1b | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### lam / ds_alpha joint sweep (llama-3b)
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
| llama-3b | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### lam / ds_alpha joint sweep (qwen-math-1.5b)
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
| qwen-math-1.5b | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
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
| qwen-7b gptq-int4 | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 1000 | 1000 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 316.2 | 1000 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### embeds_center_mode comparison (lam=0.01/ds_alpha=1)
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
> **W&B:** none yet (no AIME2025 runs).

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | — | planned | — | — | — | — | — |
| llama-1b | qwen | local | — | planned | — | — | — | — | — |
| llama-3b | qwen | none | — | planned | — | — | — | — | — |
| llama-3b | qwen | local | — | planned | — | — | — | — | — |
| qwen-3b | qwen | none | — | planned | — | — | — | — | — |
| qwen-3b | qwen | local | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | none | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | local | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | none | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | local | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### embeds_center_mode comparison (lam=0.01/ds_alpha=10)
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=80, proj=sparse512,
> cov_update=sm, cov_dtype=fp64, ds_beta=1.0, prm_batch_size=1,
> tmpl=model-family default, run.num_trials=4) except
> **ds_alpha=10** (`w_eff=100`).
>
> **W&B:** none yet (no AIME2025 runs).

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | — | planned | — | — | — | — | — |
| llama-1b | qwen | local | — | planned | — | — | — | — | — |
| llama-3b | qwen | none | — | planned | — | — | — | — | — |
| llama-3b | qwen | local | — | planned | — | — | — | — | — |
| qwen-3b | qwen | none | — | planned | — | — | — | — | — |
| qwen-3b | qwen | local | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | none | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | local | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | none | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | local | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=1)
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
> **W&B:** none yet (no AIME2025 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=80, proj=sparse512,
> cov_update=sm, ds_beta=1.0, prm_batch_size=1, tmpl=model-family
> default, run.num_trials=4) except **ds_alpha=10** (`w_eff=100`).
>
> **W&B:** none yet (no AIME2025 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.


### cnt-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts's equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama),
> **run.num_trials=4** (see the cnt-mcts tables above).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

### cnt-mcts-bl-v02

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled.
>
> **Fixed:** method=`mcts_bl_cnt_v02`, prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent, kube_affordable=true,
> bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family default
> (native for Qwen, custom for Llama), **run.num_trials=4** (see
> the cnt-mcts tables above). See
> `docs/decisions/kube-bonus-schedule.md` for the schedule choice.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

### cnt-mcts-bl-v03

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled.
>
> **Fixed:** method=`mcts_bl_cnt_v03`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama), **run.num_trials=4**
> (see the cnt-mcts tables above).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
> ratio matters). **run.num_trials=4** (see the cnt-mcts tables
> above).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
> model-family default, **run.num_trials=4**) except the
> diversity weight. **lam=0.1, ds_alpha=3.16** (`w_eff =
> ds_alpha/sqrt(lam) = 10` — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
> table). **run.num_trials=4** (see the [gen_budget=80] tables
> above).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; no AIME2025
> wall-clock reference point exists yet at any budget.
>
> **W&B:** none yet (no AIME2025 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

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
> default, **run.num_trials=4**) except the diversity weight.
> **lam=0.1, ds_alpha=31.6** (`w_eff=100`).
>
> ⚠️ Entirely `planned` — no runs yet.
>
> **W&B:** none yet (no AIME2025 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No AIME2025 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the GSM8K counterpart's command with `data=aime2025`.

---
