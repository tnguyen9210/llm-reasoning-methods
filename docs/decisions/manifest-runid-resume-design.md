# Manifest, run-id, and resume: one lifecycle across five decisions

Five log entries trace how a run's `manifest.json`, its W&B run-id, and
its resume-on-restart behavior got their current shape — and how a
real bug (the resume-fragmentation bug) broke the invariant this
lifecycle depends on. This doc covers the thread in order;
[decisions-log.md](../decisions-log.md) has each entry's full original
text (2026-06-17, 2026-06-21 ×2, 2026-06-24).

## Current mechanism (verified 2026-07-07)

`utils/configs.py`:

- **`write_manifest(result_dir, cfg, varied=None, run_id=None)`** —
  writes `{config_name, config_hash, config_identity, varied, run_id}`
  to `{result_dir}/manifest.json` via atomic temp-write + `os.replace`
  (a crash mid-write never leaves a half-written file visible under
  the real name). Called **twice per launch**: once before
  `wandb.init` (`run_id=` whatever `load_wandb_run_id` returned for
  this dir — see below, **not unconditionally `None`**), once after
  (`run_id=wandb_run.id`).
- **`load_wandb_run_id(result_dir)`** — reads `manifest.json["run_id"]`
  first; falls back to the legacy standalone `wandb_run_id.txt`
  sidecar for any dir not yet migrated to the folded-in field.
- **Launcher call order** (`generate_mcts_cnt.py` and siblings):
  `run_id = load_wandb_run_id(result_dir)` happens **first**, then
  `write_manifest(result_dir, cfg, run_id=run_id)` (pre-`init`), then
  `wandb.init(id=run_id, resume="allow")`, then
  `write_manifest(result_dir, cfg, run_id=wandb_run.id)` (post-`init`).
  Reading before the first write is the fix for the bug below — get
  this ordering wrong and every resume mints a fresh W&B run instead
  of reattaching.
- **Trial-body write order**, per trial: (1) dump raw results
  (temp-write + `os.replace`, same atomic pattern), (2) fold timing
  into the running average and `wandb.log` it, (3) write the
  per-trial `.done` marker, (4) score (`build_scored_dataset`, wrapped
  in `try`/`except` so a scoring failure never discards the raw
  generation already on disk). Resume skips any trial whose `.done`
  marker already exists.
- **`timing_state.json` stays a separate file**, not folded into
  `manifest.json` — it's written once *per trial* (a hot-loop
  write), while `manifest.json`'s `run_id` field is written exactly
  twice per *run*.

## How it got here

### 2026-06-17 — introduce the W&B sidecar; establish resume + write-order discipline

**Problem:** generation and post-processing are separate processes,
and multi-trial runs on preemptible GPUs get killed mid-run; re-running
from scratch both wastes completed trials and mints a duplicate W&B
run. **Fix:** a standalone `wandb_run_id.txt` sidecar so post-processing
could reattach via `wandb.init(id=..., resume="must")`; resume skips
any trial with an existing `.done` marker; the trial body is ordered
**dump → log timing → write marker → score** specifically so the
marker means exactly "generation finished, raw results safely on
disk" — a crash before it leaves nothing and the trial reruns cleanly,
a crash after it leaves valid results resume correctly skips. This
entry's own caveat already flagged the seed of the 2026-06-18
`config_name` risk: resume keys off the `.done` marker in the
`config_name` dir, so any result-affecting knob missing from
`config_name` lets an unrelated run resume-skip a trial it shouldn't
(see [decisions/config-name-design.md](config-name-design.md)).

### 2026-06-21 — fold the sidecar into `manifest.json`

**Problem:** the result-dir naming rework (same date, see
[decisions/config-name-design.md](config-name-design.md)) gave every
run dir a `manifest.json` for identity — having a *second* one-line
sidecar file just for the run-id was redundant. **Fix:** add a
`run_id` field to `manifest.json`; `write_manifest` takes an optional
`run_id` and is called twice per launch (before/after `wandb.init`, as
described above); `load_wandb_run_id` reads the manifest field first,
falling back to the legacy sidecar. **Why this preserves crash-safety:**
writing the manifest *before* `wandb.init` means a crash during the
(network-dependent) `wandb.init()` call still leaves a locatable,
identity-recorded dir — `find_run_dir` matches on `config_hash`/
`config_identity`, both written in that same first call, independent
of whether `run_id` is `None` yet. **Migration:** backfilled `run_id`
into all 42 existing manifests from their sidecars (zero mismatches),
then deleted the 42 now-redundant sidecar files.

### 2026-06-21 — keep `timing_state.json` separate, don't fold it in too

**Problem:** having just folded `run_id` in, considered folding
`timing_state.json` (the per-trial running-average sidecar) into
`manifest.json` too. **Decision:** don't — kept separate. **Why:** the
two sidecars have incompatible write lifecycles. `run_id` is written
exactly twice per *run* — set-once-then-frozen, safe to share a file
with the mostly-static identity fields. `timing_state.json` is written
once **per trial**, in the generator's hot loop; folding it in would
mean every trial completion does a read-modify-write of the *entire*
manifest (identity fields included) just to bump three timing numbers,
and raises write-contention risk if a `compute_stats.py`/
`prepare_scored_dataset.py` post-process ever runs concurrently with a
still-generating trial loop — two atomic-replace writers on the same
file instead of two independent ones. The split isn't incidental
structure; it's "identity, rarely written" kept apart from "per-trial
telemetry, written every trial" because the two have genuinely
different concurrency and write-frequency needs.

### 2026-06-24 — the resume-fragmentation bug: reading run_id too late nulled it

**The original ordering was backwards:** `write_manifest(cfg)` (no
`run_id` argument, i.e. `run_id=None`) → *then* `load_wandb_run_id` →
`wandb.init(id=run_id, resume="allow")` → `write_manifest(cfg,
run_id=wandb_run.id)`.

**The bug:** `write_manifest` writes the *whole* payload via atomic
replace — so that first call, passing `run_id=None`, **overwrote the
already-saved `run_id` with null one line before `load_wandb_run_id`
even ran**. Every resume attempt therefore loaded `None` regardless of
what had been saved previously, `wandb.init(id=None)` minted a
*fresh* W&B run every time, and the original run was silently
orphaned. **Observed live:** a stalled run `mfs5klyg` resumed as a
brand-new `aum658fp`; another, `7ccy14de`, resumed as `lzqhvfj6` —
fragmenting one logical experiment across multiple empty W&B runs, and
leaving any doc/ledger citation of the old id dangling (the same
failure class as a prior deleted-run citation the reconciler had
separately caught).

**The fix:** read `load_wandb_run_id` **before** the first
`write_manifest` call, and pass the result through explicitly:
`run_id = load_wandb_run_id(result_dir)` then
`write_manifest(result_dir, cfg, run_id=run_id)`. The pre-`init` write
now *preserves* whatever id already existed instead of nulling it.
This restores the invariant the 2026-06-21 fold-in decision had
assumed but never enforced: `run_id` is written twice per run but
should never be *cleared* by either write. `run_id` is not part of
`config_identity`/the hash (`_HASH_EXCLUDE` — see
[decisions/config-name-design.md](config-name-design.md)), so the fix
touches nothing the identity/resolution mechanism reconciles.

**Verified:** re-running the two stalled configs after the fix kept
their *original* ids (`mfs5klyg`, `7ccy14de`) in the manifest instead
of minting new ones; the two orphaned runs (`aum658fp`, `lzqhvfj6`)
were deleted from W&B, both confirmed empty and uncited anywhere.

## Why this lineage matters together, not just the final fix

Each entry's fix was locally correct and each one's specific
assumption is exactly what the next entry either builds on or (in the
2026-06-24 case) accidentally violated: the 2026-06-17 write-order
discipline defined what "safely resumable" means; the 2026-06-21
fold-in preserved that discipline's crash-safety property while
consolidating two files into one; but folding `run_id` into a
whole-payload atomic-replace file created a new failure mode (a write
meant to preserve identity could silently clear a *different* field)
that hadn't existed when `run_id` lived in its own single-purpose
sidecar. The bug wasn't a regression introduced carelessly — it was
the specific new risk the consolidation itself introduced, one call
ordering away from being safe.

## Revisit if

Any future `write_manifest` caller has a legitimate reason to *clear*
`run_id` (today, no such caller exists — every write either preserves
a loaded id or supplies a freshly-minted one) — that would need an
explicit flag rather than relying on "whatever `load_wandb_run_id`
returned" as the only source of truth, or the exact bug this thread
just fixed could resurface in a new call site. Also revisit if
per-run file count itself becomes a bottleneck (very many small result
dirs) — that's the stated trigger for reconsidering the
`timing_state.json`-stays-separate choice.
