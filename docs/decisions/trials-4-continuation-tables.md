# Trials 2→4 continuations: same-file n=4 tables, feeds move, frozen n=2 snapshots

*2026-08-12 — undecided; proposal recorded, nothing executed.
Open: adopt or revise the same-file plan; which tables get n=4;
whether the queued `w_eff=∞` cells bump to `trials: 4`.*

Records a proposal for where 4-trial versions of existing 2-trial
comparison tables live, and how the ledger tracks a trial-count
continuation of an already-scored config. Triggered by the plan
to extend the
`cov_scope=local` / `embeds_ref=relative` sweeps from 2 to 4
trials while keeping the 2-trial results visible as their own
tables.

## The question

Raising `run.num_trials` 2→4 and resuming generates only trials
2–3 — cheap, and exactly the intent. But Tuan wants *separate*
tables for the n=2 and n=4 readings, and
`docs/exp-comp-prm800k-level5.md` is ~4,300 lines, so the obvious
move was a fresh file. Should the n=4 tables go to a new doc?

## What the machinery forces

Four facts pin the shape of the answer:

1. **`run.num_trials` is hash-excluded.** The continuation lands
   in the same `cfg-<hash>` result dir; the searcher's resume
   skips finished trials. No new config, no re-run of trials 0–1.
2. **One ledger entry per hash, ever** (the `--dedup` invariant).
   So a continuation cannot be a new entry: the existing entry is
   edited in place — `trials: 2→4`, `status: scored→inqueue` —
   and `--queue` then emits the launch command with
   `run.num_trials=4`.
3. **`--sync-doc` derives the status cell of every row an entry
   feeds.** If one entry fed both the n=2 and n=4 tables, the
   "frozen" n=2 table would flip back to `inqueue`/`running` on
   the next sync. A frozen table therefore must stop being fed:
   the entry's `feeds` key *moves* to the n=4 table's tbl-id.
4. **The doc↔ledger map is hard-coded 1:1**
   (`orchestration/status.py` `DOC_MAP`, ~line 88), and hash
   uniqueness is global across ledgers. A second synced doc means
   either extending status.py or splitting the ledger — and a
   split breaks shared-feed entries (the `w_eff` 1/10 cells feed
   both the sweep tables and the `embeds_ref` comparison tables
   inside the same doc).

## Proposal

- **n=4 tables live in the same doc**,
  `docs/exp-comp-prm800k-level5.md`, under a new mirrored
  section (`## Tuning tables [trials=4]`, per-budget
  subsections), with fresh tbl-ids and cross-links to their n=2
  twins in both directions.
- **The n=2 tables freeze.** Each gains one ⚠ line — "frozen at
  n=2, superseded by `tbl-XXXXXX`; no longer ledger-fed" — and
  its entry's `feeds` key is replaced by the n=4 tbl-id. The
  frozen table drops into `--sync-doc`'s unsynced list, which is
  harmless and expected.
- **Long-form prose can leave the tracked doc.** Cross-table
  narrative that needs no status cells may live in a
  hand-written `docs/analysis-*.md`; sync-doc ignores it. Tables
  stay where the tooling can reach them.
- n=2 numbers stay reproducible forever via
  `compute_stats.py ... run.num_trials=2` (reads trials 0–1
  only); the frozen table is a verifiable snapshot, not a relic.

## Rejected alternatives

- **Separate synced doc** — needs status.py surgery or a ledger
  split (fact 4); the split orphans shared-feed comparisons.
- **Dual-feeding both tables from one entry** — the frozen
  table's statuses flip back on every sync (fact 3).
- **Forcing a new hash with a dummy override** — creates a new
  result dir, so trials 0–1 re-run from scratch: 2 wasted trials
  per cell at 5–30 hr each.

## Consequences and follow-ups

- **W&B summary drift:** `compute_stats.py` refreshes the run's
  `eval/*` summary in place, so after the n=4 rescore the W&B
  ids cited in a frozen n=2 table show 4-trial numbers on W&B.
  The doc snapshot becomes the only 2-trial record; the frozen
  table's ⚠ line should say so.
- **No exp-check special-casing needed:** after trials 2–3
  generate, the entry reads `finished, scored 2/4` → the normal
  scoring-pass path fills 2–3, then `compute_stats` with
  `run.num_trials=4` writes the n=4 cells.
- **Cost gates table selection:** +2 trials ≈ +9–14 h/cell at
  b=80 (~65–70 GPU-h per 7-row table) and +18–58 h/cell at
  b=320 (~250+ GPU-h per table). Which tables get n=4 is decided
  per table, not globally.
- **The ten `w_eff=∞` cells queued 2026-08-12 carry
  `trials: 2`.** If n=4 is the target for their tables, bump
  them before they launch; the one already running (llama-1b
  b=80) would simply resume later for two more trials.
- `hr/trial` in an n=4 table is the 4-trial average from
  `timing_state.json`; the frozen table keeps the 2-trial value.
  Expected, no action.
