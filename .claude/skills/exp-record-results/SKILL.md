# exp-record-results

Loaded when: Tuan asks to record / transcribe finished runs
into one of the `docs/exp-comp-*.md` tracking docs (e.g.
`exp-comp-prm800k-level4.md`, `exp-comp-prm800k-level5.md`,
`exp-comp-gsm8k.md`) — e.g. "record the done runs", "run the
recorder", "update the tables with finished results", "what's
done but not in the doc yet". Sibling of
`exp-new-comparison-table`.

**Two entry points, same operation.** This fires whether Tuan
enters *ledger-first* ("record the done runs") or *table-first*
("update / refresh / check the `<table name>` table", "fill in
what's done for this table", with a table selected or named).
Table-first is identical — a table IS a `feeds` value — it just
scopes the worklist to one table's cells (see §3a).

This skill closes the gap between a run being **done** (all
trials finished, on disk + W&B) and its number being **in the
doc**. Those are two separate states; the gap between them is
the worst kind of drift (results exist but the tables don't
show them). `recorded: false` on a `done` entry IS that gap.

It does **not** commit — it reports what it recorded and what
needs Tuan's attention, then waits (his standing rule: show
changes, confirm before any git commit/push).

---

## 0. The one idea that makes this a recorder, not a transcriber

Naively, "write the number into the cell" means *overwrite
whatever's there*. That quietly corrupts the table the first
time reality and the doc disagree. This skill upgrades that
step to **reconcile**:

> Before writing, look at what's already in the cell.
> - empty (`—`)  → first fill, just write.
> - has a number → compare it to your freshly-computed number.
>     - agrees (within rounding) → fine, proceed, flip recorded.
>     - disagrees → **DO NOT overwrite.** Collect the mismatch
>       and surface it; leave `recorded` as-is for Tuan to
>       adjudicate.

A mismatch is **not an error to auto-fix** — it's a question
for Tuan. Three real causes, all of which happen here:
1. **`recorded` out of sync** — flag says `false` but a number
   was already hand-entered; you're about to clobber it.
2. **A re-run moved the number** — e.g. the cell shows a
   2-trial mean, but the run is now done at 4 trials; the new
   number is *correctly* different. Surfacing it tells Tuan the
   table moved and why.
3. **The cited source died** — the cell names a W&B run that
   was since deleted (this happened: a cell cited `ctmgmcrp`,
   a deleted run). Recomputing forces "run not found" to
   surface instead of being silently skipped.

This audit is the step that caught the dead-W&B-run citation
on its own, without anyone noticing by eye. It is the whole
point of the skill.

---

## 1. The system this fits into

Three layers (vault guide
`research-coding-practices-guides/tracking-experiment-status`):
- `experiments.yaml` — append-only ledger; `recorded` is the
  one mutable field and the ONLY thing this skill writes there.
- `status.py` — the reconciler; tells you what's
  `done --not-recorded` and emits per-cell verification.
- `docs/exp-comp-*.md` — the report layer (one file per
  dataset/level, e.g. `exp-comp-prm800k-level4.md`,
  `exp-comp-gsm8k.md`); this skill writes `done` numbers into
  its cells.

The `feeds` key on each entry names which doc cell(s) the
number goes into. A run can feed several tables (one entry's
`feeds` is a list) — record into *every* cell it names.

---

## 2. The authoritative number source (do not guess this)

**`compute_stats.py` is the source of the summary numbers** —
NOT the raw per-question `.txt` files (those are per-trial 0/1
indicators, not the cross-trial mean). Invoke it with the SAME
overrides as the ledger entry:

```
python compute_stats.py --config-name <config_root> \
    <the entry's overrides...> run.num_trials=<trials>
```

It prints `result_dir`, `config_name`, then the summary. The
dict keys `metrics.compute_stats_basics` returns (each value is
a `(mean, sem)` tuple) are **underscore form** — NOT the `@gb`
form used as the doc *column* heading. Don't confuse them:

| dict key (what you index) | doc column (what you write under) |
|---|---|
| `pass_gb` | pass@gb |
| `naive_gb` | naive@gb |
| `weighted_gb` | wei@gb |
| `maj_gb` | maj@gb |
| `ncomps`, `depth`, `nphases`, `ndepths` | (context, not usually in the table) |

> ⚠️ Indexing `s["pass@gb"]` raises `KeyError` — the key is
> `s["pass_gb"]`. The `@gb` strings are doc-column labels only.

Format in the doc as `.NNNN<br>±.NNNN` (4 dp), matching
existing scored cells.

> ⚠️ **`hr/trial` does NOT come from compute_stats.** Per the
> doc's own notes it's read from `timing_state.json` (newer
> runs) or the mean of `time_per_trial_hr` over logged trials
> in W&B (older runs). Source it separately; if unavailable,
> leave the existing value or `—` and say so — don't fabricate.

> ⚠️ **Side effect:** `compute_stats.py` also writes `eval/*`
> to the run's W&B summary (`resume="must"`). This is
> idempotent (overwrite-only, never appends a series), so
> re-running is safe — but know that "computing stats" also
> refreshes W&B. If a run has no `wandb_run_id.txt` it skips
> W&B silently and still prints the numbers.

---

## 3. Procedure — step 0: table-first scoping (if a table was named)

If the request is "update/refresh/check the `<table>` table"
rather than "record the done runs," do this first, then
continue with the numbered steps below:

1. **Map the table → its `feeds` key.** Read the table's `####`
   heading + its `**Limitations / follow-up:**` line in the
   relevant `docs/exp-comp-*.md` doc (it usually names the
   feeds key, e.g. `sem-mcts/ds_alpha-sweep-qwen`). That key is
   the scope.
2. **Check feeds coverage of EVERY cell — the gap ledger-first
   misses.** For each row of the table, confirm a ledger entry
   exists whose `feeds` includes this table's key. A cell can
   be *done on disk* yet invisible to the worklist because its
   entry's `feeds` doesn't name this table (this happened: a
   `last×full` cell was done but only fed `ds_alpha-sweep-qwen`,
   not `embeds-strategy-scope`). For such a cell: `--check` its
   config to find the backing entry, then **add this table's
   feeds key** to that entry's `feeds` list (append-only-safe
   ledger edit) so the recorder sees it. Only then proceed.
3. **Scope the worklist** to this table: run step 1 below with
   `--group <group>` and mentally filter to entries whose
   `feeds` includes the table key (or eyeball `status.py
   --group <g>` for the relevant rows). From here, the
   procedure is identical.

### Numbered steps

1. **List the worklist:**
   ```
   python status.py --done --not-recorded
   ```
   (Add `--group <g>` to scope, e.g. just `cnt-mcts`.) Each row
   is a done run whose number isn't in the doc yet. If empty —
   nothing to record; say so and stop.

2. **For each entry, compute real stats** (§2). Capture the
   four summary metrics' mean±SEM — dict keys `pass_gb /
   naive_gb / weighted_gb / maj_gb` (§2 maps them to the
   pass@gb/naive@gb/wei@gb/maj@gb doc columns).

3. **Locate the target cell(s)** via the entry's `feeds`
   key(s). Find the matching table + row in the relevant
   `docs/exp-comp-*.md` doc. One entry may feed several cells
   (e.g. a default-config run that backs ds_alpha-sweep AND
   model-family AND prm-comparison) — handle each.

   - If `feeds` names a cell you can't find (heading renamed,
     row missing): **don't guess** — surface it as "feeds key
     `X` has no matching cell" and skip writing that one.

4. **AUDIT each cell before writing** (§0):
   - cell empty (`—`) → mark for write.
   - cell has a number → compare to computed. Within rounding
     → mark for write (consistent). Differs → **collect a
     mismatch** `{cell, existing, computed, entry}`, do NOT
     write, do NOT flip recorded.
   - cell cites a W&B run that compute_stats reported as
     gone/not-found → collect as a mismatch (dead source).

5. **Verify the trial count** the doc claims vs. reality. The
   `status.py` row shows `[n_done/trials]`; the doc cell's
   `trials` column should match `n_done`. If the doc says a
   different trial count, that's a mismatch to surface (often
   the re-run case).

6. **Write** the agreed cells (empty + consistent): fill
   pass/naive/wei/maj, set `status` to `scored` (or `scored ⚠`
   if a documented caveat applies), set `trials` to the real
   count, fill `hr/trial` if sourced. Update the table's
   `**W&B:**` line with the run_id if it lists them.

7. **Flip `recorded: true`** in `experiments.yaml` for entries
   whose every fed cell was written cleanly. Leave `recorded`
   untouched for any entry with an unresolved mismatch.

8. **Re-check** nothing broke:
   ```
   python status.py --done --not-recorded   # shrinks toward empty
   python status.py --verify
   ```

9. **Announce** (§4).

---

## 4. What to report

Two separate lists — never blur them:

**Recorded** (the clean path):
> Recorded N runs into M cells:
> - `<entry note>` → `<table>` row `<row>`: pass .X naive .Y
>   wei .Z maj .W (n trials). recorded ✓

**Needs your attention** (the audit's output — the valuable
part):
> ⚠️ K mismatches — NOT written, recorded NOT flipped:
> - `<table>` row `<row>`: doc shows .OLD, computed .NEW
>   (likely: re-run from 2→4 trials / hand-entered value /
>   dead W&B run `<id>`). Your call: keep doc, take computed,
>   or investigate.

Then stop and wait — do not commit, do not auto-resolve
mismatches.

---

## 5. Append-only / write-discipline (do not violate)

- The ONLY writes this skill makes: numbers into a
  `docs/exp-comp-*.md` doc's cells, and `recorded: false → true`
  in `experiments.yaml`. Nothing else in the ledger changes; no
  entries added, removed, or reordered.
- `recorded` only flips `true` for a *cleanly written* entry.
  A mismatch leaves it `false` (so the run stays on the
  worklist until adjudicated — that's correct, not a bug).
- Never overwrite a populated cell to make a number "match" —
  surface the conflict instead. Silent overwrite is the exact
  failure this skill exists to prevent.

---

## 6. Failure modes specific to this skill

- **Silent overwrite** (the cardinal sin): replacing an
  existing cell value without comparing. §0/§4 prevent it.
- **Wrong number source**: reading the per-question `.txt`
  instead of `compute_stats.py`. §2.
- **Fabricated `hr/trial`**: it's not in compute_stats; source
  it or leave it. §2.
- **Premature `recorded` flip**: flipping before all of an
  entry's `feeds` cells are written, or flipping on a
  mismatch. §3.7 / §5.
- **Dead W&B run treated as missing data**: if recompute says
  "run not found," that's a mismatch to surface (the cell
  cites a deleted run), not a reason to blank the cell.
