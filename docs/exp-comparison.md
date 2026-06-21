# LLM Reasoning — MCTS Experiment Comparison

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on PRM800K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.

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

## Algorithm name ↔ code mapping
> Row labels are conceptual names; `method=` is what
> `config_name()` emits into
> `results/<dataset>/<method>--level-N--...--b-NNN--.../`.

| Concept | Code `method=` | Core module | Status in code |
|---|---|---|---|
| cnt-mcts | `mcts_cnt` | `mcts_cnt_search_v05_00_00` | on `ExpConfig`, run |
| cnt-mcts-bl | `mcts_bl_cnt_v01` | `mcts_bl_cnt_search_v01_00_00` | on `ExpConfig`, run |
| | `mcts_bl_cnt_v02`? | `..._v02_00_00` | flat, not migrated |
| sem-mcts (policy) | `mcts_sem_v01` | `mcts_sem_search_v01_00_00` | on `ExpConfig`, **runnable (no runs yet)** |
| sem-mcts (PRM) | `mcts_sem_v02` | `mcts_sem_search_v02_00_00` | on `ExpConfig`, **runnable (no runs yet)** |
| sem-mcts-bl | `mcts_sem_bl` *(pending)* | — (not built) | **not implemented** |

> **sem-mcts is now runnable** (rename + `ExpConfig` migration
> landed 2026-06-18; launcher `generate_mcts_sem.py`, configs
> `conf/(search/)mcts_sem_v0{1,2}*`). It splits into two
> embedding *sources* — `v01` pools from the **policy** (2nd
> vLLM engine), `v02` from the **PRM** — a clean source
> ablation, same diversity algorithm. v02 also supports optional
> sparse projection (`embeds_proj=sparse`, embeds_dim=512) and
> `cov_update=exact|sherman_morrison`. `sem-mcts-bl` is still
> not built (`llm-reasoning-mcts-bl-exp` backlog,
> `mcts_bl_embeds`). Listed so the target grid is visible.

---

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

| algorithm | model | budget | trials | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| cnt-mcts | Llama-3.2-1B | 80 | 4 | .648±.042 | .492±.044 | .469±.044 | .414±.044 |
| cnt-mcts | Llama-3.2-3B | 80 | 3 | .747±.022 | .503±.026 | .586±.025 | .581±.025 |
| cnt-mcts | Qwen-2.5-3B | 80 | 3 | .875±.017 | .685±.024 | .737±.023 | .721±.023 |
| cnt-mcts | Qwen-2.5-Math-1.5B | 80 | 2 | .879±.020 | .746±.027 | .770±.026 | .758±.027 |
| cnt-mcts-bl | Llama-3.2-1B | 80 | 4 | .492±.022 | .395±.022 | .383±.022 | .381±.022 |
| cnt-mcts-bl | Qwen-2.5-Math-1.5B | 80 | 3 | .654±.024 | .578±.025 | .573±.025 | .576±.026 |
| sem-mcts (policy) | — | 80 | — | *planned* | — | — | — |
| sem-mcts (PRM) | — | 80 | — | *planned* | — | — | — |
| sem-mcts-bl | — | 80 | — | *not built* | — | — | — |

> Winning config per row (cpuct fixed at 2.0 throughout — no
> sweep yet, so template is the only knob currently in play;
> see tuning tables for the full grid): cnt-mcts — Llama-3.2-1B
> **custom** (.648 > native .566, the only model with both
> scored); Llama-3.2-3B **custom** (.747 > native .732, both
> now scored); Qwen-2.5-3B **native** (only scored);
> Qwen-2.5-Math-1.5B **native** (custom is scored at .906 but
> template-bug, n=1 — see tuning table; not a valid winner yet).
> cnt-mcts-bl — Llama-3.2-1B **custom**,
> Qwen-2.5-Math-1.5B **native** (each only scored). `sem-*`
> blocked (rename / not built).
>
> **What the numbers say (budget 80):**
> - **cnt-bl loses to cnt where they overlap.** Llama-3.2-1B:
>   cnt .648 vs cnt-bl .492. Qwen-2.5-Math-1.5B: cnt .879 vs
>   cnt-bl .654. The best-first frontier protocol is
>   *underperforming* the phase-based baseline at this budget
>   on both shared models — worth a hard look before
>   investing more in BL.
> - Qwen-2.5-Math-1.5B-native: cnt has 2 scored trials, cnt-bl
>   has 3 — ⚠ not a matched comparison yet
>   (`llm-reasoning-mcts-bl-exp-todo` M3).
> - Within cnt, model size/family dominates template: both
>   Qwen models hit .879 pass@gb, well above Llama
>   (.648/.747).
> - **Custom now beats native on both Llama sizes** (1B:
>   .648 > .566; 3B: .747 > .732) — but the 3B Qwen custom
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

##### custom vs native template comparison
> Isolates `tmpl`, the only varying knob — all other knobs held
> at default and dropped as columns. All four models' tables,
> grouped here so the custom-vs-native picture reads in one
> place instead of being interleaved with other cnt-mcts
> subsections (e.g. the prm_batch_size sweep, moved out below).
> Kept as separate tables (one per model) rather than merged
> into one, so each model's caveats/footnotes stay attached to
> its own rows.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, prm_batch_size=2.

| llm | tmpl | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | **custom** | 4 | scored | **.648±.042** | .492±.044 | .469±.044 | .414±.044 |
| Llama-3.2-1B | native | 2 | scored | .566±.031 | .371±.030 | .348±.030 | .313±.029 |

---

| llm | tmpl | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | **custom** | 3 | scored | **.747±.022** | .503±.026 | .586±.025 | .581±.025 |
| Llama-3.2-3B | native | 4 | scored | .732±.020 | .520±.022 | .547±.022 | .529±.022 |

---

| llm | tmpl | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| Qwen-2.5-3B | native | 3 | scored | .875±.017 | .685±.024 | .737±.023 | .721±.023 |

> 4 raw trials exist; trial 2's scoring never ran (raw
> `.jsonl` present, no scored output) — row above is trials
> {0,1,3}. Score trial 2 to get the full 4-trial number.

---

| llm | tmpl | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| Qwen-2.5-Math-1.5B | custom | 1 | scored ⚠ | .906±.026 | .781±.037 | .773±.037 | .773±.037 |
| Qwen-2.5-Math-1.5B | native | 2 | scored | .879±.020 | .746±.027 | .770±.026 | .758±.027 |

> ⚠ **custom = template-bug, n=1 — NOT comparable to native.**
> This run applies the hardcoded Llama-3.1 `custom_chat_template`
> to Qwen's tokenizer (force-override regardless of `llm=`),
> which Qwen was never trained on — completions leak raw
> `<|eot_id|>`-style markup after the boxed answer
> (`llm-reasoning-mcts-exp-todo` Track 1). Accuracy isn't
> tanked (the answer lands before the leak) but this is *not* a
> clean custom-vs-native signal, and ± is within-trial only
> (n=1, bootstrapped over 128 Qs), not across-trial variance.
> The leaked text also ~3.5×'d completion length → the single
> trial took ~50 min to score. Re-run after the per-model
> template fix before treating custom as a real number; W&B run
> `kk32i2lp`. **Fixed 2026-06-19** — `gen.use_custom_template`
> now defaults per model family (Qwen → native, else → custom),
> so a fresh run needs no override and won't hit this bug;
> re-run to get a clean comparable custom row.

##### prm_batch_size sweep
> Isolates the in-loop PRM scoring micro-batch
> (`search.prm_batch_size`, [utils/configs.py](../utils/configs.py))
> from accuracy — same search config otherwise, so pass@gb
> should be ~flat across rows (modulo sampling noise); the
> point is the **runtime/throughput and memory** delta, not a
> new accuracy result. rlhflow/qwen prm_bs∈{1,4} rows are
> 2-trial runs scored 2026-06-21; prm_bs=2 not explicitly run
> yet (the existing baseline cell uses a different trial
> count/path, so it's left out rather than presented as a
> matched data point).
>
> **Fixed:** Llama-3.2-1B, tmpl=custom, cpuct=2.0, bs-4, d-20,
> b=80.

| prm | prm_bs | trials | status | pass@gb | time/trial (hr) | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| rlhflow | 1 | 2 | scored | .617±.030 | 2.51 | 30.23 |
| rlhflow | 2 | — | not run | — | — | — |
| rlhflow | 4 | 2 | scored | .641±.030 | 2.38 | 31.68 |
| qwen | 1 | 2 | scored | .633±.030 | 2.35 | 27.49 |
| qwen | 4 | 2 | scored | .676±.029 | 2.31 | 28.68 |

> Within the n=2 rows, pass@gb is flat within ~1 SEM across
> prm_bs (rlhflow: .617/.641; qwen: .633/.676) — no accuracy
> regression from larger micro-batches, as expected; time/trial
> also flat (~2.3-2.5 hr), so this sweep shows no throughput win
> at this model/budget scale. **Peak GPU mem is NOT flat,
> though** — `prm_bs=4` consistently costs ~1.2-1.5 GB more than
> `prm_bs=1` (rlhflow: 30.23→31.68 GB; qwen: 27.49→28.68 GB),
> pulled from W&B's auto-logged
> `system.gpu.0.memoryAllocatedBytes` (max over each run's
> history; no explicit code instrumentation — not in
> `timing_state.json` or `wandb.log()` calls in
> [generate_mcts_cnt.py](../generate_mcts_cnt.py)). So `prm_bs=1`
> is the safer default if memory headroom is the binding
> constraint (V100S 32GB): same accuracy, same speed, less
> memory pressure, at this model/budget scale. W&B: rlhflow
> prm_bs=1 `1c9026yj`, prm_bs=4 `wb2un007`; qwen prm_bs=1
> `8vvw5usb`, prm_bs=4 `u9itrf7k`. Full writeup incl. why the
> pass@gb gap isn't real and the trial count needed to actually
> resolve it:
> [findings/exp-findings/prm-batch-size-throughput-memory.md](findings/exp-findings/prm-batch-size-throughput-memory.md).

##### model / precision comparison
> Cross-model, cross-precision sweep at fixed budget — isolates
> how model family, size, and quantization trade off against
> pass@gb and runtime. Unlike the template comparison (one knob)
> this varies model+precision jointly per row, so `llm` is a
> single combined string (model-precision) rather than split
> columns.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, b=80, tmpl=model-family
> default (native for Qwen, custom for Llama, per the
> 2026-06-19 per-family default fix above).

| llm | trials | status | pass@gb | time/trial (hr) |
|---|---|---|---|---|
| Llama-3.2-1B fp16 | — | planned | — | — |
| Llama-3.2-3B fp16 | — | planned | — | — |
| Llama-3.2-3B gptq | — | planned | — | — |
| Qwen-2.5-3B fp16 | — | planned | — | — |
| Qwen-2.5-3B gptq-int4 | — | planned | — | — |
| Qwen-2.5-7B gptq-int4 | — | planned | — | — |
| Qwen-2.5-Math-1.5B fp16 | — | planned | — | — |

### sem-mcts
> **Runnable as of 2026-06-18** (rename + migration landed).
> Two methods = two embedding sources: `mcts_sem_v01` (policy
> embeds, 2nd vLLM engine) and `mcts_sem_v02` (PRM embeds, no
> 2nd engine). knobs beyond template/cpuct: ds_alpha, ds_beta,
> lam, embeds_strategy (last/avg), embeds_normalize, and for
> v02 embeds_proj (none/sparse, dim 512) + cov_update
> (exact/sherman_morrison). Defaults in conf/search/mcts_sem_v0*.
> Run v01 and v02 at matched model/level/trials vs. cnt-mcts —
> the comparison the project exists for.

##### embeds_proj × cov_update sweep (v02, cpuct=2.0)
> A 2×2-per-model grid instead of two single-knob sweeps.
> `embeds_proj`: `none` feeds the PRM's raw 4096-dim hidden
> state into the covariance bonus; `sparse512` JL-projects it
> to 512 first (~2.5× speed win, accuracy cost untested).
> `cov_update`: `exact` recomputes V^-1 each step;
> `sherman_morrison` (sm) updates it incrementally
> (path-identical to exact, proven; the question here is
> whether that holds at scale). pass@gb should match within
> noise across cov_update (same path) but may differ across
> embeds_proj (lossy projection); time/trial is the throughput
> axis. method=`mcts_sem_v02` only — proj/cov_update don't
> exist on v01.

| llm | proj | cov_update | trials | status | pass@gb | time/trial (hr) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | none | exact | — | planned | — | — |
| Llama-3.2-1B | none | sm | — | planned | — | — |
| Llama-3.2-1B | sparse512 | exact | — | planned | — | — |
| Llama-3.2-1B | sparse512 | sm | 2 | scored | .5938±.0308 | 4.27 |
| Llama-3.2-1B | sparse512 | sm (prmbs-2) | 1 | scored ⚠ | .6094±.0433 | 4.38 |
| Qwen-2.5-Math-1.5B | none | exact | — | planned | — | — |
| Qwen-2.5-Math-1.5B | none | sm | — | planned | — | — |
| Qwen-2.5-Math-1.5B | sparse512 | exact | 2 | scored | .8711±.0210 | 4.34 |
| Qwen-2.5-Math-1.5B | sparse512 | sm | 2 | scored | .8633±.0215 | 4.32 |

> Other @gb metrics: Llama-1B sparse512×sm (2 trials) — naive
> .4453±.0311, weighted .4297±.0310, maj .4141±.0308, ncomps
> 14.2±0.8, depth 8.7±0.2, nphases 44.5±11.0, ndepths 9.4±0.2.
> W&B `kqn1lj13`. The prmbs-2 row (n=1, ⚠ within-trial SEM
> only) is a separate run at a different in-loop PRM batch
> size — kept alongside rather than merged, since
> prm_batch_size shouldn't affect accuracy but isn't a tracked
> column here; W&B `ttsp0a0g`.
>
> Qwen-1.5B sparse512 (exact vs sm, 2 trials each): naive
> .7383±.0275 vs .7617±.0267; weighted .7500±.0271 (both); maj
> .7539±.0270 vs .7422±.0274 — all within ~1 SEM, no
> systematic effect from cov_update. ncomps 24.6±1.1 vs
> 23.6±1.0, nphases 16.1±4.1 vs 14.6±3.9 — n=2 trials, SEMs
> wide, treat as preliminary. W&B `lkltpzc1` (exact) /
> `vlri1uw0` (sm).

##### Llama-3.2-1B
| method | tmpl | cpuct | trials | status | pass@gb |
|---|---|---|---|---|---|
| sem v01 (policy) | custom | 2.0 | — | *planned* | — |
| sem v02 (PRM) | custom | 2.0 | — | *planned* | — |

> Match cnt-mcts Llama-1B (custom, 4 trials) for the head-to-head.

##### Llama-3.2-3B
| method | tmpl | cpuct | trials | status | pass@gb |
|---|---|---|---|---|---|
| sem v01 (policy) | custom | 2.0 | — | *planned* | — |
| sem v02 (PRM) | custom | 2.0 | — | *planned* | — |

##### Qwen-2.5-Math-1.5B
| method | tmpl | cpuct | trials | status | pass@gb |
|---|---|---|---|---|---|
| sem v01 (policy) | native | 2.0 | — | *planned* | — |
| sem v02 (PRM) | native | 2.0 | — | *planned* | — |

> Use native (cnt-mcts Qwen-Math custom has the template bug;
> match the clean native cnt row, 2 trials, for comparability).

### cnt-mcts-bl
> knobs: template, cpuct (bs-4, d-20 fixed). method=`mcts_bl_cnt_v01`.
> No cpuct sweep yet — every row is the default 2.0. Same
> selection rule as cnt-mcts: the Summary above promotes
> whichever row scores highest on **pass@gb** across all
> knobs jointly. (`num_phases` cap exists but isn't a tuned
> knob yet — open backlog question in
> `llm-reasoning-mcts-bl-exp-todo` on whether to keep it,
> replace it, or remove it.)

##### Llama-3.2-1B
| tmpl | cpuct | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| custom | 2.0 | 4 | scored | .492±.022 | .395±.022 | .383±.022 | .381±.022 |

##### Qwen-2.5-Math-1.5B
| tmpl | cpuct | trials | status | pass@gb | naive@gb | weighted@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| native | 2.0 | 3 | scored | .654±.024 | .578±.025 | .573±.025 | .576±.026 |

> ⚠ trial-count mismatch vs. cnt-mcts for Qwen-2.5-Math-1.5B
> native (bl=3, cnt=2) — reconcile before the head-to-head
> (`llm-reasoning-mcts-bl-exp-todo` M3).

### sem-mcts-bl
*Not implemented (no `mcts_sem_bl` in code).*

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

---

## Run log (newest first)
> One dated block per run/comparison: hypothesis → result →
> follow-up. Append-only; newest at top.

### 2026-06-18 — cnt-mcts / Llama-1B / custom / cpuct=2.0 / b=80
- **hypothesis:** baseline reference cell for the custom
  template; expect higher solution depth than native.
- **result:** pass@gb .648±.042, naive@gb .492±.044,
  weighted@gb .469±.044, maj@gb .414±.044 (4 trials);
  ncomps 16.1±0.6, depth 8.6±0.1. W&B `fpgp2si1`.
- **follow-up:** sweep cpuct (0.5/1/2/4) on this anchor cell
  (`llm-reasoning-mcts-exp-todo` Track 1); backfill the
  other scored cnt-mcts pass@gb into the table above.

## Standing comparison questions
- Does sem-UCT beat cnt-UCT at matched budget? (needs
  sem-mcts runnable)
- Does the BL frontier protocol beat phase-based walks?
  (cnt-mcts-bl vs cnt-mcts @80 — data exists, pending the
  trial-count fix + write-up)
- Custom vs native template: consistent across algorithms?
- cpuct sensitivity: same optimum across algorithms? (no
  sweep for any algorithm yet)
- Does the cnt/sem (or BL) gap hold as gen_budget grows
  80→160→320? (the cross-budget question the Summary exists
  to answer)

## Links & connections
- Findings: [findings/exp-findings/prm-batch-size-throughput-memory.md](findings/exp-findings/prm-batch-size-throughput-memory.md) —
  prm_bs sweep throughput/memory result + why the pass@gb
  gap isn't statistically real
- Findings index: [findings/README.md](findings/README.md)
