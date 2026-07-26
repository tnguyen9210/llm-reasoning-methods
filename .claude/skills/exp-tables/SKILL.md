# exp-tables

Loaded when: Tuan asks to create or adapt a comparison table in
one of the `docs/exp-comp-*.md` tracking docs and register its
experiments — "add a table comparing X across Y", "make a
ds_alpha sweep table", "adapt the model-family table for
kube-v02 at alpha=0.5", `/exp-tables`. Replaces the old
`exp-new-comparison-table` skill (workflow v2: per-doc ledgers,
5-state lifecycle). Sibling of `exp-run` (launches) and
`exp-check` (scores/records). Design:
[docs/decisions/experiment-workflow-v2.md](../../../docs/decisions/experiment-workflow-v2.md).

Creates the doc-side artifact from one intent statement:
1. a `####` comparison table in the right `docs/exp-comp-*.md`,
   every row's `status` reflecting repo reality, and
2. for cells that ALREADY have a ledger entry anywhere, a
   targeted `feeds:` append on that entry — nothing else.

**Net-new cells get NO ledger entry.** A `planned` table row is
the plan; the ledger tracks only runs Tuan has decided to run.
He adds an entry when he is ready for the cell to run, not when
the table is authored (Tuan's rule, 2026-07-24). So a fresh table
is normally 100% doc-side writes.

It does NOT launch runs (`exp-run`) and does NOT commit.

---

## 0. Invariants (the judgment steps — do not skip)

- **--dedup every cell BEFORE writing anything.** Never
  hand-derive a hash; never assume a config composes.
- **Never duplicate-append.** If `--dedup` shows a match in ANY
  ledger, reuse that entry: append this table's feeds key to its
  `feeds:` list (targeted Edit) and carry its status + numbers
  into the new table. One config = one entry, forever.
- **Never create a ledger entry for a net-new (unrun) cell.**
  The doc row says `planned`; the ledger stays untouched. Tuan
  queues a cell when he wants it run — that is when its entry is
  born (`status: inqueue` + `priority`). Writing `planned`
  entries en masse floods the ledger with cells nobody intends
  to run and makes `planned` meaningless.
- **THE SYNC RULE: any ledger status change is followed, in the
  same turn, by**
  ```
  python orchestration/status.py --sync-doc <stem> --apply
  ```
  The table's status cell is derived from the ledger, never
  hand-maintained in parallel. This applies to every transition
  (`planned`->`inqueue` on queueing, `inqueue`->`running` on
  launch, `->failed`, `->scored`). Skipping it silently desyncs
  the doc: on 2026-07-24 three queued cells still read `planned`
  and 30 launched cells still read `planned`/`inqueue` across
  three docs. Read the dry-run first (no `--apply`) and check
  `mismatches=0` before applying.
- **Detect the mixed case.** A "new" table often has cells
  already on disk or already running — their rows get the real
  status (`running`, `scored` + numbers), not `planned`.
- **Statuses are exactly**: `planned | inqueue | running |
  scored | failed` — bare word in the status cell, no
  parentheticals (Tuan's rule, 2026-07-22).
- **Match per-model template conventions** (§3).
- **Report, don't auto-commit.**

## 1. Inputs to resolve (infer, then confirm with Tuan)

The compared question; llm group(s); prm group; method/
config_root (§2); fixed vs varied knobs; budget (default
`search.gen_budget=80`); trials (default 2); target doc =
where the data group + level says it goes (level5/level4/
gsm8k/aime2025 — same mapping as the ledger split). For an
**adapted table**: which existing table is the template, which
axis is swapped; copy its Fixed block and column layout, then
proceed identically — adaptation only changes where the spec
comes from, never the procedure.

## 2. Method -> launcher / config_root

| family | launcher | config_root |
|---|---|---|
| cnt-mcts | generate_mcts_cnt.py | mcts_cnt_prm800k |
| sem-mcts v01/v02 | generate_mcts_sem.py | mcts_sem_v01/v02_prm800k |
| bon | generate_bon.py | bon_prm800k |
| cnt-bl v01/v02 | generate_mcts_bl_cnt.py | mcts_bl_cnt_v01/v02_prm800k |
| kube-bl v01/v02 | generate_mcts_bl_cnt.py | mcts_bl_kube_v01/v02_prm800k |
| kdepth-bl v01/v02 | generate_mcts_bl_cnt.py | mcts_bl_kdepth_v01/v02_prm800k |
| sem-bl v01/v02 | generate_mcts_bl_cnt.py | mcts_bl_sem_v01/v02_prm800k |

`group:` = the family name as used in existing entries
(cnt-mcts, sem-mcts, kube-mcts-bl-v02, ...). Dataset switch via
`data=gsm8k` / `data=aime2025` / `data.level=5` overrides.

## 3. Per-model template convention (easy to get wrong)

`use_custom_template` is baked into each `conf/llm/*.yaml`
(qwen native, llama custom). Selecting a whole `llm=` group is
enough — do NOT pass `llm.use_custom_template` unless
deliberately overriding a field on an already-selected group.
When in doubt `--dedup` both ways and see which hash matches
sibling tables.

## 4. Procedure

1. **Resolve cells** (model × varied values + fixed overrides).
2. **Dedup every cell:**
   ```
   python orchestration/status.py --dedup <config_root> <key=val> ...
   ```
   Record `hash, on_disk, n_done, n_scored, matches`. All new
   hashes must be distinct (a within-set collision = spec bug;
   stop). run.num_trials is NEVER an override (hash-excluded;
   it's the entry's `trials`).
3. **Show Tuan the resolved-cell table** and get go-ahead:
   `| cell | hash | on_disk | match (ledger/id/status) | -> action |`
   action = `doc row only (no ledger entry)` |
   `reuse <id> (status, k/n scored)`.
4. **Write the doc `####` section** (house style §5, placement
   per the doc's `## Tuning tables [gen_budget=N]` hierarchy).
   Reused scored cells get their numbers copied from the doc
   cell the matched entry's `feeds` points at (token-cheap, no
   recompute); other reused cells get the entry's live status.
5. **Ledger writes** in `orchestration/ledgers/<stem>.yaml` —
   there is only ONE case that writes:
   - **matched cell** -> targeted Edit on the existing entry:
     append this table's feeds key to its `feeds:` list. Nothing
     else.
   - **net-new cell** -> **no ledger write at all.** The doc row
     carries `status: planned` and that is the whole record. Do
     not append an entry, not even a `planned` one.

   Keep each net-new cell's `hash` from step 2 in the *report*
   (§8) so Tuan can queue it later without re-deriving. When he
   says "queue cell X", that is when its entry is appended —
   bottom, never reorder — and **the doc row must be re-synced in
   the same turn** (see the sync rule below) so the table shows
   `inqueue`, not a stale `planned`:
   ```yaml
   - id: <human-id, queue-style>
     launcher: <launcher>
     hash: "<hash verbatim from --dedup>"
     config_root: <config_root>
     overrides: {<dict, no run.* keys>}
     trials: <n>
     feeds: [<this table's feeds-key>]
     group: <family>
     status: inqueue
     priority: <as Tuan specifies>
     expected_hr: <estimate>
     note: <short label>
   ```
6. **Mint the table's stable ID:** `python
   orchestration/status.py --mint-table-ids --apply` stamps a
   `<!-- table-id: tbl-xxxxxx -->` line under the new heading
   (immutable, survives retitles; see
   docs/decisions/stable-table-ids.md). Prefer that tbl-id in
   the new entries' `feeds:` (with a `# <table>` comment); the
   human feeds key still works but is only a label.
7. **Re-verify:** `python orchestration/status.py --verify
   --ledger <stem>` (composes + global id/hash uniqueness) and
   eyeball `python orchestration/status.py --sync-doc <stem>`
   (dry-run) — it should propose no changes to your fresh
   table.
8. **Report**: N cells (new/reused/scored), the feeds key, and
   — for every net-new cell — its `hash` and derived launch
   command, since no ledger entry holds them yet. Then state what
   to do next: nothing is queued or launchable until Tuan names
   the cells he wants run; on that request, append their entries
   (`status: inqueue` + `priority`, per step 5) and `/exp-run`
   drains them.

## 5. Doc house style

Unchanged from the old skill — blockquote preamble
(**Compares** / **Fixed** / ⚠ / **W&B**), the table, Analysis
blockquote with **Limitations / follow-up** naming the feeds
key. Wrap ~72 chars; scored cells `.NNNN<br>±.NNNN`; planned
rows all `—`. The Limitations line MUST name the feeds key in
backticks — `--sync-doc` and exp-check row-matching key off it
(or off the `<!-- table-id -->`; either identifier works, the
tbl-id is the durable one).

## 6. Failure modes

- **Duplicate-append** (the worst): a matched cell appended as
  new. §0 / step 5.
- **Appending `planned` entries for net-new cells** — the ledger
  tracks intent-to-run, not the space of authored table cells.
  A fresh table adds ZERO new ledger entries. §0 / step 5.
- **Stale-status row**: writing `planned` for a cell that's on
  disk or running. Step 2's dedup output prevents it.
- **feeds-key drift**: key not findable in the Limitations line
  -> the table becomes invisible to --sync-doc. Step 6's
  dry-run catches it (table listed "no feeds key").
- **Wrong template** (§3); **budget in wrong doc section**;
  **run.* keys leaking into overrides** (they change nothing
  and desync trials).
