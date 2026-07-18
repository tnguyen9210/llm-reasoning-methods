# `mcts_bl_cnt_v03` renamed to `mcts_bl_kdepth_v01`: its own algorithm family, not a same-family sibling version

*2026-07-17*

Records the full migration when Tuan asked to rename
`mcts_bl_cnt_search_v03_00_00` to match the naming convention of the
previous day's [bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md)
migration. Same underlying pattern: a mechanical-sounding rename
request that turns out to have real reach, since the config
`method`/`algo` string is embedded in result-dir names and the
identity hash of every already-scored run
([manifest-runid-resume-design.md](manifest-runid-resume-design.md)).
This file is the pointer target for every "renamed 2026-07-17... see
bl-cnt-to-bl-kdepth-rename.md" note left across the codebase and docs
during the migration.

## Why rename at all, and why a new family instead of staying `v03`

Tuan asked what name would best describe a protocol that "combines
knapsack-based allocation with depth-based exploration" — this
variant's selection criterion is exactly that: the same fractional-
knapsack cost normalization (`/ cost(x)`) shared with the KUBE
variant, but with KUBE's UCB confidence bonus replaced by a
deterministic depth-decay function `f_a(z) = 1 - z^depth_alpha`
(no visit counts, no confidence bound, no exploration guarantee of
any kind).

Before deciding on a name, the same scoping question that drove the
KUBE rename was raised explicitly: should this stay `mcts_bl_cnt`'s
v03, or move to its own family like KUBE did the day before? The
variant's own module docstring already argued the case for the
latter — it describes itself as "a deliberately different theoretical
basis (no bandit/regret guarantee)... not a bugfix or refinement" of
anything in the bl_cnt/bl_kube lineage, and "cnt" in `mcts_bl_cnt`
specifically denotes count-based (visit-count) exploration, which this
variant has none of at all. Keeping it filed under `mcts_bl_cnt` was
the same category mismatch the KUBE rename had already fixed for the
sibling variant. Tuan confirmed: **new family**, `mcts_bl_kdepth`, with
this variant becoming v01 of that family (not v03 — first member of a
new lineage, not a continuation of the old one).

**The name itself**: Tuan proposed `kdepth` directly ("something like
mcts_bl_kdepth_v01, kdepth or something similar") — a compressed
coinage of knapsack + depth that mirrors `kube`'s shape and length
(both four-to-six-letter single tokens naming the selection
mechanism), reads naturally as "knapsack-cost-normalized depth-
shaping," and by its very shape signals a new family rather than a
`bl_cnt` sibling — satisfying both the naming question and the family-
scope question in one answer.

## What changed

### Code / config layer

| Old | New |
|---|---|
| `core/mcts_bl_cnt_search_v03_00_00.py` | `core/mcts_bl_kdepth_search_v01_00_00.py` |
| `utils/configs.py::BLMCTSCntV03Config` | `utils/configs.py::BLMCTSKdepthV01Config` |
| `method: str = "mcts_bl_cnt_v03"` | `method: str = "mcts_bl_kdepth_v01"` |
| `conf/mcts_bl_cnt_v03_prm800k.yaml` | `conf/mcts_bl_kdepth_v01_prm800k.yaml` |
| `conf/search/mcts_bl_cnt_v03.yaml` | `conf/search/mcts_bl_kdepth_v01.yaml` |
| schema name `mcts_bl_cnt_v03_schema` | schema name `mcts_bl_kdepth_v01_schema` |
| `algo_dict["mcts_bl_cnt_v03"]` (`generate_mcts_bl_cnt.py`) | `algo_dict["mcts_bl_kdepth_v01"]` |
| `_METHOD_TO_GROUP["mcts_bl_cnt_v03"] = "cnt-mcts-bl"` (`status.py`) | `_METHOD_TO_GROUP["mcts_bl_kdepth_v01"] = "kdepth-mcts-bl"` |

The group-label change (`cnt-mcts-bl` → `kdepth-mcts-bl`) follows the
same reasoning as the KUBE rename's group-label change: it exists so
`status.py --group <g>` filtering and the doc's `###` subsection names
agree, and leaving it collapsed onto the shared `cnt-mcts-bl` bucket
would contradict the family split. Doc-side row/section labels were
updated to `kdepth-mcts-bl-v01` to match.

`generate_mcts_bl_cnt.py`, `compute_stats.py`, and `status.py`'s
`ConfigStore` registrations, dispatch dicts, and `_METHOD_TO_ROOT` /
`_METHOD_TO_LAUNCHER` / `_METHOD_TO_GROUP` maps were all updated to
match. Checked whether `mcts_bl_cnt_search_v01_00_00.py`'s and
`mcts_bl_kube_search_v01_00_00.py`'s own module docstrings reference
v03 by name (mirroring how v01/v03 were checked during the KUBE
rename) — **neither does**, so this half of the migration was a
no-op: no sibling-docstring edits were needed there. Two pre-existing,
unrelated issues were fixed opportunistically while touching the core
file anyway: a typo ("depth-shaping-knapbe-bonus.md" → "...-knapsack-
bonus.md") appearing twice in the module docstring, which had been
pointing at a filename that never existed.

### Result dirs / manifests (the irreversible-feeling part, again)

Same mechanism and same care as the KUBE migration: because `method`
is hashed and dir-named, every already-scored run needed physical
migration, not just a forward-looking code change. 5 result
directories existed, all under
`results/prm800k/mcts_bl_cnt_v03--level-4/` (one per model family:
Llama3.2-1B, Llama3.2-3B, Qwen2.5-3B, Qwen2.5-7B-GPTQ-Int4,
Qwen2.5-Math-1.5B). For each, the identical procedure from the KUBE
migration was reused (see
[bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md) for the
full step-by-step): mutate `config_identity.search.method`, recompute
`config_hash` by reproducing `config_hash()`'s exact algorithm
(`json.dumps(sort_keys=True, separators=(",",":"))` → SHA1 → first 8
hex chars) against the mutated identity, rebuild `config_name`, rename
the directory, atomically rewrite `manifest.json`, carry `varied` and
`run_id` over unchanged. One hash was independently spot-verified by
hand outside the migration script (`e8c9626f`, matched). Ran as a
dry-run first (printed the full plan, no writes) before applying.

| Model | Old hash | New hash | W&B run_id |
|---|---|---|---|
| Llama3.2-1B | `8dda8957` | `e8c9626f` | `3yyni5fn` |
| Llama3.2-3B | `3978c92d` | `ed731d7b` | `kl22af6y` |
| Qwen2.5-3B | `e1311880` | `e74406ac` | `asn77q7e` |
| Qwen2.5-7B-GPTQ-Int4 | `4a99b754` | `883e5265` | `g0jll3ts` |
| Qwen2.5-Math-1.5B | `12183f3b` | `cbbbeccb` | `i8qyv1gv` |

### Ledger (`experiments.yaml`)

5 entries had `config_root: mcts_bl_cnt_v03_prm800k`. Each was
updated: `config_root` → `mcts_bl_kdepth_v01_prm800k`, `group:`
→ `kdepth-mcts-bl` (matching the `_METHOD_TO_GROUP` change above), and
the `note:` field's embedded old dir-name/hash citation was updated to
the new dir-name/hash, with the old values preserved inline in the
note (same append-safe pattern as the KUBE ledger edits). `recorded:
true` and the scored `pass@gb` values were left untouched — the
numbers didn't change, only where they live on disk.

### Docs

- `docs/algorithms.md`: the kdepth variant's row moved out of the
  `## BL-MCTS` table section into its own `## BL-KDEPTH-MCTS` section
  (mirroring `## BL-KUBE-MCTS`, which mirrors `## BL-Sem-MCTS` — a
  peer section, not nested under `## CNT-MCTS`).
- `docs/decisions/depth-shaping-knapsack-bonus.md` → renamed to
  `bl-kdepth-knapsack-bonus.md` (its title and filename encoded the
  old identity, same situation `kube-bonus-schedule.md` was in the
  day before). `docs/decisions/kube-affordability-restriction.md` and
  `docs/decisions/strip-and-reappend-separator.md` were checked and
  confirmed to have **no** v03 references at all — genuinely nothing
  to fix in either.
- `docs/decisions/bl-cnt-path-aware-frontier-score-design.md`'s §7.2
  and §7.3 (written 2026-07-16, before this rename) had their
  file-path permalinks updated to the new filename and corrected line
  numbers.
- All 4 `docs/exp-comp-*.md` files (`prm800k-level4`, `prm800k-level5`,
  `gsm8k`, `aime2025`): the table-row/section token `cnt-mcts-bl-v03`
  → `kdepth-mcts-bl-v01` (35 occurrences total, only
  `prm800k-level4.md` carrying real scored numbers), plus the
  `method=mcts_bl_cnt_v03` "Fixed:" caveat lines and
  `depth-shaping-knapsack-bonus.md` links in each.
- `docs/decisions-log.md`: same append-only treatment as the KUBE
  rename — genuinely historical entries were left with their original
  narrative; a same-session forward-pointing note was added where
  needed.

## What was deliberately left alone

- **Genuinely historical `decisions-log.md` entries** keep their
  original "v03" terminology in prose, per append-only convention.
- **Filenames of individual result files inside each migrated run
  dir** (trial `.jsonl`, `.done` markers, score `.txt` files) still
  carry the old `mcts_bl_cnt_v03--...` name — accurate to when they
  were written.
- **The v02/v03 docstring-vs-code discrepancy** flagged in
  [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  §7.3 (both files' Algorithm blocks say "Add non-terminal children to
  leaf_nodes" but the code appends all children) — still live after
  this rename; only the file-path citation pointing at it was updated,
  the underlying discrepancy itself was not touched.

## Connections

- [bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md) — the
  previous day's rename this one directly follows the pattern of;
  contains the full step-by-step manifest-migration procedure reused
  here without re-deriving it.
- [bl-kdepth-knapsack-bonus.md](bl-kdepth-knapsack-bonus.md) — this
  variant's own design decision (the depth-shaping formula, the sign
  error caught before implementation), predating this rename.
- [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  §7.2 — the analysis that established this variant reads no visit
  counts at all, the fact underlying the family-split decision here.
- `docs/decisions-log.md`, 2026-07-17 entry (top of file) — the
  short-form pointer to this file.
