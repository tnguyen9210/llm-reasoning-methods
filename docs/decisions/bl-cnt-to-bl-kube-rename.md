# `mcts_bl_cnt_v02` renamed to `mcts_bl_kube_v01`: its own algorithm family, not a same-family sibling version

*2026-07-16*

Records the full migration when Tuan asked to rename
`mcts_bl_cnt_search_v02_00_00` to `mcts_bl_kube_search_v01_00_00` — a
mechanical request that turned out to have real reach, since the
config `method`/`algo` string is embedded in result-dir names and the
identity hash of every already-scored run
([manifest-runid-resume-design.md](manifest-runid-resume-design.md)).
This file is the pointer target for every "renamed 2026-07-16... see
bl-cnt-to-bl-kube-rename.md" note left across the codebase and docs
during the migration.

## Why rename at all, and why a new family instead of staying `v02`

The fractional-KUBE variant (`docs/decisions/bl-kube-bonus-schedule.md`,
`docs/decisions/kube-affordability-restriction.md`) had lived as
`mcts_bl_cnt_v02` since 2026-07-09 — a same-family sibling version of
`mcts_bl_cnt_v01`'s PUCT selection, following the repo's `v01`/`v02`/
`v03` pattern for variants that share a launcher and core skeleton but
swap one component. On reflection this framing undersold the
difference: KUBE isn't PUCT-plus-a-tweak, it's a different selection
theory entirely (cost-normalized knapsack density vs. plain UCB), and
"cnt" in the family name (`mcts_bl_cnt`) specifically refers to
count-based exploration — which KUBE's confidence bonus still uses,
but wrapped in a cost-division term PUCT never had. Tuan decided this
warranted its own family name, `mcts_bl_kube`, with the renamed
variant becoming v01 of that family (not v02, since it's the first
member of a new lineage, not a continuation of the old one).

`mcts_bl_cnt_v03` (depth-shaping knapsack) was NOT renamed or
renumbered as part of THIS migration — it stayed in the `bl_cnt`
family that day, even though it shares KUBE's knapsack skeleton and
`kube_affordable` feasibility logic. Its own docstring already framed
it as "a deliberately different theoretical basis... not a
refinement" of the KUBE variant, and it has no visit-count/confidence-
bound term at all (a static function of tree position) — closer in
spirit to a heuristic bl_cnt variant than to KUBE's bandit-theoretic
lineage. This asymmetry (v02 renamed, v03 not) was a deliberate
choice, not an oversight, given what was in scope on 2026-07-16.

**Update 2026-07-17:** the "revisit if v03 is ever felt to belong in
its own family too" question above was in fact revisited the very
next day — v03 has since been renamed to `mcts_bl_kdepth_v01`, its own
family, for exactly the reasoning anticipated here. See
[bl-cnt-to-bl-kdepth-rename.md](bl-cnt-to-bl-kdepth-rename.md) for
that migration. This section is left as written for historical
accuracy (it correctly describes the 2026-07-16 scope), not corrected
in place.

## What changed

**Two questions were asked and answered before any file moved**
(via `AskUserQuestion`, both confirmed by Tuan):

1. **Does the config `method`/`algo` string change too, or only
   file/class names?** → **Yes, the string changes too**
   (`mcts_bl_cnt_v02` → `mcts_bl_kube_v01`). This was the
   consequential choice: `config_hash` is a SHA1 over the full
   `search` config dict including `method`
   ([utils/configs.py::config_identity](../../utils/configs.py)), so
   changing the string changes the hash of every existing run, and
   `level_dir(cfg) = f"{cfg.search.method}{level_str}"` means the
   result-dir *name itself* is derived from `method` — so this choice
   forced the disk migration in the next section, not just a code
   rename.
2. **Should sibling files' own docstrings (v01, v03) be updated to
   the new name, or left referencing the old name with a note?** →
   **Update sibling docstrings** to the new name, so all three
   `bl_cnt`/`bl_kube` core files agree on the current name going
   forward.

### Code / config layer

| Old | New |
|---|---|
| `core/mcts_bl_cnt_search_v02_00_00.py` | `core/mcts_bl_kube_search_v01_00_00.py` |
| `utils/configs.py::BLMCTSCntV02Config` | `utils/configs.py::BLMCTSKubeV01Config` |
| `method: str = "mcts_bl_cnt_v02"` | `method: str = "mcts_bl_kube_v01"` |
| `conf/mcts_bl_cnt_v02_prm800k.yaml` | `conf/mcts_bl_kube_v01_prm800k.yaml` |
| `conf/search/mcts_bl_cnt_v02.yaml` | `conf/search/mcts_bl_kube_v01.yaml` |
| schema name `mcts_bl_cnt_v02_schema` | schema name `mcts_bl_kube_v01_schema` |
| `algo_dict["mcts_bl_cnt_v02"]` (`generate_mcts_bl_cnt.py`) | `algo_dict["mcts_bl_kube_v01"]` |
| `_METHOD_TO_GROUP["mcts_bl_cnt_v02"] = "cnt-mcts-bl"` (`status.py`) | `_METHOD_TO_GROUP["mcts_bl_kube_v01"] = "kube-mcts-bl"` |

The group-label change (`cnt-mcts-bl` → `kube-mcts-bl`) followed
directly from the "own family" decision: that label exists so
`status.py --group <g>` filtering and the doc's `###` subsection
names agree
([exp-record-results](../../.claude/skills/exp-record-results/SKILL.md)
§2 table-first scoping depends on this), and leaving it collapsed
onto the shared `cnt-mcts-bl` bucket would have contradicted the
family split. Doc-side row/section labels were updated to
`kube-mcts-bl-v01` to match (see below).

`generate_mcts_bl_cnt.py`, `compute_stats.py`, and `status.py`'s
`ConfigStore` registrations, dispatch dicts, and `_METHOD_TO_ROOT` /
`_METHOD_TO_LAUNCHER` / `_METHOD_TO_GROUP` maps were all updated to
match. `mcts_bl_cnt_search_v01_00_00.py` and
`mcts_bl_cnt_search_v03_00_00.py`'s own module docstrings (sibling
cross-references) were updated to name the variant by its current
file/class name rather than the old ones.

### Result dirs / manifests (the irreversible-feeling part)

Because `method` is hashed and dir-named, every already-scored run
needed physical migration, not just a forward-looking code change —
otherwise `find_run_dir` would search under the *new* `level_dir`
(`mcts_bl_kube_v01--level-4`) for manifests recording the *old* hash,
and the 5 existing scored runs would silently become unfindable
(`status.py --done --not-recorded` would show them as if they'd
vanished, and any future `compute_stats.py` re-verification would
fail to match).

5 result directories existed, all under
`results/prm800k/mcts_bl_cnt_v02--level-4/` (one per model family:
Llama3.2-1B, Llama3.2-3B, Qwen2.5-3B, Qwen2.5-7B-GPTQ-Int4,
Qwen2.5-Math-1.5B). For each:

1. Loaded the existing `manifest.json`'s recorded `config_identity`
   (the exact dict `config_identity(cfg)` produced at run time).
2. Mutated `config_identity["search"]["method"]` from
   `"mcts_bl_cnt_v02"` to `"mcts_bl_kube_v01"`.
3. Recomputed `config_hash` by reproducing `config_hash()`'s own
   algorithm exactly — `json.dumps(identity, sort_keys=True,
   separators=(",",":"), default=str)` → SHA1 → first 8 hex chars —
   against the mutated identity. This is not an approximation: since
   the mutated identity is byte-identical to what a real `cfg` with
   only `search.method` changed would produce, the hash is exactly
   what `config_hash(cfg)` would return for that config. Independently
   re-verified for one run by hand outside the migration script
   (`bc421e24`, matched).
4. Rebuilt `config_name` from the same template `config_name()` uses
   (`{algo}--level-{n}--{llm}--{prm}--d-{depth}--bs-{batch}--b-{budget}
   --cfg-{hash}`) with the new algo string and new hash.
5. `os.rename`'d the whole run directory to the new name (preserving
   every file inside untouched — trial `.jsonl`/`.done` markers,
   `timing_state.json`, scored `.txt` files — those still carry the
   *old* `mcts_bl_cnt_v02--...` name in their own filenames, which is
   correct: they were genuinely written under that name at generation
   time, and nothing in the codebase's file-discovery logic depends on
   inner filenames matching the containing directory's name).
6. Atomically rewrote `manifest.json` (temp file + `os.replace`, same
   pattern `write_manifest` itself uses) with the new `config_name`,
   `config_hash`, and mutated `config_identity`; `varied` and `run_id`
   (the W&B run id) carried over unchanged — all 5 W&B runs
   (`5hlr1c61`, `rwqjq7fl`, `vimdoh1b`, `a0nlicyf`, `ub5wqvva`) stay
   correctly linked to their result dirs.
7. Removed the now-empty old level-dir
   (`results/prm800k/mcts_bl_cnt_v02--level-4/`).

Ran as a dry-run first (printed the full rename plan, no writes) before
applying — see the migration's scratchpad script for the exact
transformation; not checked into the repo since it's a one-shot tool,
not reusable infra.

| Model | Old hash | New hash | W&B run_id |
|---|---|---|---|
| Llama3.2-1B | `4586b6a1` | `bc421e24` | `5hlr1c61` |
| Llama3.2-3B | `55643f02` | `06feb79e` | `rwqjq7fl` |
| Qwen2.5-3B | `e1360dda` | `f0dcc67f` | `vimdoh1b` |
| Qwen2.5-7B-GPTQ-Int4 | `0ef9c289` | `aade1550` | `a0nlicyf` |
| Qwen2.5-Math-1.5B | `497d2e67` | `7dd65d04` | `ub5wqvva` |

### Ledger (`experiments.yaml`)

5 entries had `config_root: mcts_bl_cnt_v02_prm800k`. Each was
updated: `config_root` → `mcts_bl_kube_v01_prm800k`, `group:`
→ `kube-mcts-bl` (matching the `_METHOD_TO_GROUP` change above), and
the `note:` field's embedded old dir-name/hash citation was updated
to the new dir-name/hash (with the old values preserved inline in the
note, so the ledger entry still shows its own migration history
rather than silently overwriting it). `recorded: true` and the scored
`pass@gb` values were left untouched — the numbers didn't change,
only where they live on disk.

### Docs

- `docs/algorithms.md`: the KUBE variant's row moved out of the
  BL-MCTS table section into its own `## BL-KUBE-MCTS` section
  (mirroring how `## BL-Sem-MCTS` is already a peer section, not
  nested under `## CNT-MCTS`) — consistent with treating it as its own
  family rather than a BL-MCTS sub-bullet.
- `docs/decisions/kube-bonus-schedule.md` → renamed to
  `bl-kube-bonus-schedule.md` (its title and filename encoded the old
  identity). `docs/decisions/kube-affordability-restriction.md` and
  `docs/decisions/depth-shaping-knapsack-bonus.md` were NOT renamed
  (their titles are neutral / already scoped to v03 respectively) but
  had internal code-references and links updated.
- `docs/decisions/bl-cnt-path-aware-frontier-score-design.md`'s §7
  (written earlier the same day, before this rename) had its
  file-path permalinks updated to the new filename and corrected line
  numbers (the module docstring grew ~10 lines from the rename's own
  edits, shifting every downstream anchor).
- All 4 `docs/exp-comp-*.md` files (`prm800k-level4`,`prm800k-level5`,
  `gsm8k`, `aime2025`): the table-row/section token `cnt-mcts-bl-v02`
  → `kube-mcts-bl-v01` (37 occurrences total, only `prm800k-level4.md`
  carrying real scored numbers, others all `planned` placeholder
  rows), plus the `method=mcts_bl_cnt_v02` "Fixed:" caveat lines and
  `kube-bonus-schedule.md` links in each.
- `docs/decisions-log.md`: genuinely historical entries (2026-07-09,
  predating this rename by a week) were left with their original
  narrative — rewriting "v02" to "bl_kube_v01" in a record of a
  decision that was actually made about something called `v02` at the
  time would falsify the historical record, violating the file's own
  append-only/newest-first convention. The one exception: a link to
  `decisions/kube-bonus-schedule.md` was updated to the renamed
  filename, since a stale link serves no one and fixing it doesn't
  change what the entry says happened. The same-day 2026-07-16 entry
  (written earlier in this session, before this rename) got an inline
  forward-pointing note instead of narrative rewriting, since it's a
  same-session pointer to code, not multi-week-old history.

## What was deliberately left alone

- **`mcts_bl_cnt_v03`** stays in the `bl_cnt` family (see "Why rename"
  above) — not renamed, not renumbered.
- **Genuinely historical `decisions-log.md` entries** (2026-07-09 and
  earlier) keep their original "v02" terminology in prose, per
  append-only convention.
- **Filenames of individual result files inside each migrated run
  dir** (trial `.jsonl`, `.done` markers, score `.txt` files) still
  carry the old `mcts_bl_cnt_v02--...` name — accurate to when they
  were written, and nothing depends on them matching the containing
  directory's new name.
- **The pre-existing `MCTSBLCntConfig` typo** in
  `docs/decisions/set-gen-budget-for-mcts-search.md` (should read
  `BLMCTSCntConfig`) — unrelated to this rename (it's about v01's
  class name), left as a separate, smaller cleanup item.
- **The v02/v03 docstring-vs-code discrepancy** flagged in
  [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  §7.3 (both files' Algorithm blocks say "Add non-terminal children to
  leaf_nodes" but the code appends all children) — still live after
  this rename; only the file-path citations pointing at it were
  updated, the underlying discrepancy itself was not touched (out of
  scope for a naming migration).

## Connections

- [bl-kube-bonus-schedule.md](bl-kube-bonus-schedule.md),
  [kube-affordability-restriction.md](kube-affordability-restriction.md)
  — the KUBE variant's own design decisions, predating this rename.
- [bl-cnt-path-aware-frontier-score-design.md](bl-cnt-path-aware-frontier-score-design.md)
  §7 — the same-day analysis that first needed to distinguish
  `mcts_bl_cnt_v02` from `mcts_bl_cnt_v03` in detail, written just
  before this rename made the family split official.
- [manifest-runid-resume-design.md](manifest-runid-resume-design.md)
  — the manifest/`find_run_dir` design this migration had to respect
  to keep the 5 existing runs findable.
- `docs/decisions-log.md`, 2026-07-16 entry (top of file) — the
  short-form pointer to this file.
