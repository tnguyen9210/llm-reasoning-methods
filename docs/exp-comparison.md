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
| cnt-mcts-bl | `mcts_bl_cnt_v01` | `mcts_bl_cnt_search_v01_00_00` | runs logged |
| cnt-mcts-bl v02 | `mcts_bl_cnt_v02` | `mcts_bl_cnt_search_v02_00_00` | runnable, no runs yet |
| sem-mcts-bl | `mcts_sem_bl` *(pending)* | — (not built) | not implemented |

> Every sem-mcts row elsewhere in this doc is **v02** (PRM-sourced
> embeddings) — v01 (policy-sourced, via a 2nd vLLM pooling
> engine) is wired up on `ExpConfig` but has no runs yet. v01 vs.
> v02 is a clean embedding-*source* ablation on the same
> diversity algorithm; v02 additionally supports
> `embeds_proj=none|sparse` (sparse = JL projection to 512-dim,
> ~2.5x faster) and `cov_update=exact|sherman_morrison`.
> `mcts_bl_cnt_v02` has a launcher + config now (previously
> flat/unmigrated) but hasn't been run. `sem-mcts-bl` remains
> unbuilt (`llm-reasoning-mcts-bl-exp` backlog, `mcts_bl_embeds`).

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
| cnt-mcts-bl | llama-1b | 80 | 4 | .492<br>±.022 | .395<br>±.022 | .383<br>±.022 | .381<br>±.022 |
| cnt-mcts-bl | qwen-math-1.5b | 80 | 3 | .654<br>±.024 | .578<br>±.025 | .573<br>±.025 | .576<br>±.026 |
| sem-mcts-bl | — | 80 | — | *not built* | — | — | — |

> Winning config per row (cpuct fixed at 2.0 throughout — no
> sweep yet, so template is the only knob currently in play;
> see tuning tables for the full grid): cnt-mcts — llama-1b
> **custom** (.648 > native .566, the only model with both
> scored); llama-3b **custom** (.744 > native .732, both
> now scored); qwen-3b **native** (only scored);
> qwen-math-1.5b **native** (custom is scored at .894 over 2
> trials but template-bug — see tuning table; not a valid
> winner yet).
> cnt-mcts-bl — llama-1b **custom**,
> qwen-math-1.5b **native** (each only scored). `sem-*`
> blocked (rename / not built).
>
> **What the numbers say (budget 80):**
> - **cnt-bl loses to cnt where they overlap.** llama-1b:
>   cnt .648 vs cnt-bl .492. qwen-math-1.5b: cnt .879 vs
>   cnt-bl .654. The best-first frontier protocol is
>   *underperforming* the phase-based baseline at this budget
>   on both shared models — worth a hard look before
>   investing more in BL.
> - qwen-math-1.5b-native: cnt has 2 scored trials, cnt-bl
>   has 3 — ⚠ not a matched comparison yet
>   (`llm-reasoning-mcts-bl-exp-todo` M3).
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

| prm | prm_bs | trials | status | pass@gb | hr/trial | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| rlhflow | 1 | 2 | scored | .617<br>±.030 | 2.51 | 30.23 |
| rlhflow | 2 | — | not run | — | — | — |
| rlhflow | 4 | 2 | scored | .641<br>±.030 | 2.38 | 31.68 |
| qwen | 1 | 2 | scored | .633<br>±.030 | 2.35 | 27.49 |
| qwen | 4 | 2 | scored | .676<br>±.029 | 2.31 | 28.68 |

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

| prm | prm_bs | trials | status | pass@gb | hr/trial | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| rlhflow | 1 | — | to rerun | — | — | — |
| rlhflow | 2 | — | to rerun | — | — | — |
| rlhflow | 4 | — | to rerun | — | — | — |
| qwen | 1 | — | to rerun | — | — | — |
| qwen | 4 | — | to rerun | — | — | — |

#### rlhflow vs qwen PRM comparison

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

| llm | prm | enforce_eager | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-3b | rlhflow | False (default) | — | to rerun | — | — | — | — | — |
| llama-3b | rlhflow | True | — | to rerun | — | — | — | — | — |

#### model family, size, quantization comparison
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

### sem-mcts
> **Runnable as of 2026-06-18** (rename + migration landed).
> Two methods = two embedding sources: `mcts_sem_v01` (policy
> embeds, 2nd vLLM engine) and `mcts_sem_v02` (PRM embeds, no
> 2nd engine). knobs beyond template: ds_alpha, ds_beta,
> lam, embeds_strategy (last/avg), embeds_normalize, and for
> v02 embeds_proj (none/sparse, dim 512) + cov_update
> (exact/sherman_morrison). Defaults in conf/search/mcts_sem_v0*.
> Run v01 and v02 at matched model/level/trials vs. cnt-mcts —
> the comparison the project exists for. sem-mcts has no
> `cpuct` knob (selection is q-value-only on first visit, then
> a ds_alpha/ds_beta-weighted diversity bonus on later visits;
> see `core/mcts_sem_search_v02_00_00.py:select_child`).

#### embeds_proj × cov_update sweep (v02)
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
> `avg` rows at both checkpoints are new and queued.
>
> **`lam=0.01` addendum (2026-07-08):** same two `w_eff` checkpoints
> at the table's default `lam=0.01` (`w_eff=10 → ds_alpha=1.0`,
> `w_eff=100 → ds_alpha=10`). The `last` rows reuse already-scored
> cells from the `lam / ds_alpha joint sweep (v02, llama-3b)` table
> (`cfg-23f6c64a`, `cfg-baa5b18e`) — no new run. The `avg` rows are
> new and queued.

| strategy | scope | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| last | full | 0.01 | 100 | 1000 | 2 | done (see ds_alpha=100) | — | — | — | — | — |
| last | full | 0.01 | 1.0 | 10 | 2 | scored (see lam/ds_alpha joint sweep) | .7500<br>±.0271 | .6562<br>±.0297 | .6562<br>±.0297 | .6523<br>±.0298 | — |
| last | full | 0.01 | 10 | 100 | 2 | scored (see lam/ds_alpha joint sweep) | .7695<br>±.0264 | .6797<br>±.0292 | .6445<br>±.0300 | .6211<br>±.0304 | — |
| last | full | 0.1 | 3.16 | 10 | 2 | scored (see lam/ds_alpha joint sweep) | .7578<br>±.0268 | .6719<br>±.0294 | .6602<br>±.0297 | .6289<br>±.0303 | — |
| last | full | 0.1 | 31.6 | 100 | 2 | scored (see lam/ds_alpha joint sweep) | .7812<br>±.0259 | .6562<br>±.0297 | .6211<br>±.0304 | .5938<br>±.0308 | — |
| avg | full | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |
| avg | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| avg | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| avg | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| avg | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| last | response | — | — | — | — | blocked | — | — | — | — | — |
| avg | response | — | — | — | — | blocked | — | — | — | — | — |

> **Analysis.** No new data yet for `avg`. `last`×`full` is fully
> covered at all four `w_eff` checkpoints tested so far (1000, 10,
> 100 at both `lam=0.01` and `lam=0.1`) via reuse — no new runs
> needed for `last`. The five new `avg`×`full` cells (default
> `lam=0.01,ds_alpha=100` plus the `w_eff∈{10,100}` checkpoints at
> both `lam=0.01` and `lam=0.1`) are the genuinely new+runnable work
> here — the key read is whether mean-pooling changes pass@gb vs.
> the last-token default at matched `w_eff`, and whether that answer
> is consistent across `w_eff` levels and across `lam` the way
> `last` already is.
> **Limitations / follow-up:** 5 of 12 cells are genuinely
> new+runnable (`avg`×`full` at all `w_eff`/`lam` checkpoints,
> queued — `experiments.yaml` group `sem-mcts`, feeds
> `sem-mcts/embeds-strategy-scope`). The two `response` rows are
> blocked on PRM-source `response_start_idx` support; queue them
> once the v02 core handles `embeds_scope=response` for
> `embeds_source=prm`. A v01 (policy-embeds) version of this table
> would unblock the `response` axis, since v01 supports it.

#### ds_alpha sweep (v02)
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
| llama-1b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .6172<br>±.0304 | .5391<br>±.0312 | .5117<br>±.0313 | .4727<br>±.0313 | — |
| llama-1b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .6133<br>±.0305 | .5195<br>±.0313 | .5312<br>±.0312 | .4961<br>±.0313 | — |
| llama-1b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .5938<br>±.0308 | .5508<br>±.0311 | .5234<br>±.0313 | .4688<br>±.0312 | — |
| llama-1b | qwen | **1.0** | **10** | **10** | **2** | **scored (step 1)** | **.6172<br>±.0304** | **.5273<br>±.0313** | **.4766<br>±.0313** | **.4375<br>±.0311** | — |
| llama-1b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .6133<br>±.0305 | .5156<br>±.0313 | .4766<br>±.0313 | .4375<br>±.0311 | — |
| llama-1b | qwen | **0.01** | **1.0** | **10** | **2** | **scored (step 1)** | **.6250<br>±.0303** | **.5469<br>±.0312** | **.5039<br>±.0313** | **.4648<br>±.0312** | — |
| llama-1b | qwen | 1.0 | 100 | 100 | 2 | scored | .6289<br>±.0303 | .5312<br>±.0312 | .4375<br>±.0311 | .3438<br>±.0297 | — |
| llama-1b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .6094<br>±.0306 | .5117<br>±.0313 | .4531<br>±.0312 | .4219<br>±.0309 | — |
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
| llama-3b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .7227<br>±.0280 | .6523<br>±.0298 | .6484<br>±.0299 | .6367<br>±.0301 | — |
| llama-3b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .7422<br>±.0274 | .6758<br>±.0293 | .6680<br>±.0295 | .6602<br>±.0297 | — |
| llama-3b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .7305<br>±.0278 | .6641<br>±.0296 | .6836<br>±.0291 | .6602<br>±.0297 | — |
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
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored (see qwen-PRM ds_alpha=10 above) | .8789<br>±.0204 | .7969<br>±.0252 | .7891<br>±.0255 | .7695<br>±.0264 | 3.98 |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .8867<br>±.0198 | .7930<br>±.0254 | .7578<br>±.0268 | .7500<br>±.0271 | — |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .8672<br>±.0213 | .7891<br>±.0255 | .7656<br>±.0265 | .7422<br>±.0274 | — |

> **Analysis.** Both `lam=0.1` cells (w_eff=10: .8867, w_eff=100:
> .8672) are within ~1 SEM of the `lam=0.01, w_eff=100` baseline
> (.8789) — consistent with the llama-1b/llama-3b tables' finding
> that `lam` has no independent effect once `w_eff` is matched, now
> extended to a third model. Unlike llama-1b (which showed
> maj@gb/wei@gb degrading noticeably at `w_eff=100`), qwen-math-1.5b
> stays flat across all four @gb metrics between `w_eff=10` and
> `w_eff=100` — no sign of the same weighted/majority-vote
> degradation at high diversity weight for this model.
> **Limitations / follow-up:** no `lam=1.0` arm run for this model,
> so this isn't a full step-1 replication (only one `lam` value
> tested per `w_eff`) — treat as a spot-check, not confirmation.
> Completing the `lam=1.0` arm at `w_eff=10` and `100` would make
> this a proper step-1 check like the other two tables.

#### model family, size, quantization comparison
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
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No data yet.
> **Limitations / follow-up:** all 12 cells are new; none queued in
> `experiments.yaml`. Lowest priority among open sem-mcts threads —
> the `lam=0.01` version of this table already showed `agg_strategy`
> is flat within SEM (and cnt-mcts's identical sweep agrees), so
> this is a robustness check on an already-flat finding, not a
> likely source of new signal.

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
| qwen-3b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-3b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | rlhflow | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No data yet.
> **Limitations / follow-up:** all 12 cells are new; none queued in
> `experiments.yaml`. Same low-priority rationale as the `w_eff=10`
> table above.

#### LLM vs PRM embeds comparison
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

### cnt-mcts-bl
> knobs: template, cpuct (bs-4, d-20 fixed). method=`mcts_bl_cnt_v01`.
> No cpuct sweep yet — every row is the default 2.0. Same
> selection rule as cnt-mcts: the Summary above promotes
> whichever row scores highest on **pass@gb** across all
> knobs jointly. (`num_phases` cap exists but isn't a tuned
> knob yet — open backlog question in
> `llm-reasoning-mcts-bl-exp-todo` on whether to keep it,
> replace it, or remove it.)

##### llama-1b
> **Compares:** cnt-mcts-bl's only run config for this model —
> not yet a sweep, just the baseline reference cell.
>
> **Fixed:** tmpl=custom, cpuct=2.0, bs-4, d-20, b=80.
>
> **W&B:** not yet recorded here — see Run log below
> (2026-06-18 entry).

| tmpl | cpuct | trials | status | pass@gb | naive@gb | wei@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| custom | 2.0 | 4 | scored | .492<br>±.022 | .395<br>±.022 | .383<br>±.022 | .381<br>±.022 |

> **Analysis.** Single config, no comparison axis yet — this is
> the reference cell promoted to the Summary table.
> **Limitations / follow-up:** no template or cpuct sweep run
> yet for this model.

##### qwen-math-1.5b
> **Compares:** cnt-mcts-bl's only run config for this model —
> not yet a sweep, just the baseline reference cell.
>
> **Fixed:** tmpl=native, cpuct=2.0, bs-4, d-20, b=80.
>
> ⚠️ trial-count mismatch vs. cnt-mcts for qwen-math-1.5b native
> (bl=3, cnt=2) — not yet reconciled.
>
> **W&B:** not yet recorded here.

| tmpl | cpuct | trials | status | pass@gb | naive@gb | wei@gb | maj@gb |
|---|---|---|---|---|---|---|---|
| native | 2.0 | 3 | scored | .654<br>±.024 | .578<br>±.025 | .573<br>±.025 | .576<br>±.026 |

> **Analysis.** Single config, no comparison axis yet.
> **Limitations / follow-up:** reconcile the trial-count mismatch
> against cnt-mcts (`llm-reasoning-mcts-bl-exp-todo` M3) before
> using this row in any head-to-head with cnt-mcts.

#### model family, size, quantization comparison (qwen PRM)
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts (updated)'s equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled. All 7 cells are new for bl_cnt_v01 with the
> qwen PRM — the two existing `#####` cells above (llama-1b,
> qwen-math-1.5b) used the rlhflow PRM, so they don't populate
> this table directly and would need a qwen-PRM rerun to join
> it.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (the
> new default — see `generate_mcts_bl_cnt_v01.py`/
> `BLMCTSCntConfig` alignment fix), tmpl=model-family default
> (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| llama-3b gptq | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No data yet.
> **Limitations / follow-up:** all 7 cells are new; none queued
> in `experiments.yaml`. Priority depends on whether bl_cnt_v01
> is still an active comparison target vs. cnt-mcts — check
> `llm-reasoning-mcts-bl-exp-todo` before launching a full sweep
> here.

### sem-mcts-bl
> **Compares:** nothing yet — `mcts_sem_bl` is not implemented
> in code (no core module, no launcher, no config). Listed here
> only so the target algorithm grid is visible.
>
> **Limitations / follow-up:** blocked on
> `llm-reasoning-mcts-bl-exp` backlog item `mcts_bl_embeds`.

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

### cnt-mcts

#### model family comparison (b=320, qwen PRM)
> **Compares:** the same 7-model family/size/quantization sweep
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
> generation-count increase over the b=80 table; expect roughly
> 4× the per-trial wall-clock of the corresponding b=80 row
> (e.g. qwen-7b gptq-int4 was 3.21 hr/trial at b=80).
>
> **W&B:** none yet (no runs exist).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b gptq | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No data yet — nothing to take away. Once
> filled, the key read is whether the b=80 table's ranking
> (qwen-7b gptq-int4 best, GPTQ trading accuracy for speed)
> holds at b=320, and whether qwen-PRM scoring shifts the
> absolute levels relative to the b=80/llama-PRM table.
> **Limitations / follow-up:** all 7 cells are planned (see
> `experiments.yaml`, group `cnt-mcts`, feeds
> `cnt-mcts/model-family-b320-qwen`). Budget and PRM both
> differ from the b=80 table at once; a matched-PRM b=320 row
> would isolate the budget effect alone.

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
  (cnt-mcts-bl vs cnt-mcts @80 — data exists, pending the
  trial-count fix + write-up)
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
