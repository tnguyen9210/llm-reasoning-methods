# Experiment workflow v2: per-doc ledgers, 5-state lifecycle, 3 verbs

*2026-07-22 — designed and implemented with Tuan (plan approved
same day). Supersedes the queue.yaml half of
`hpc-idle-gpu-orchestration.md`; the srun-overlap launch
mechanics described there live on unchanged inside the exp-run
skill.*

## Why

Two structural failures in the v1 split (experiments.yaml ledger
+ orchestration/queue.yaml):

1. **Queue-only blocks bypassed the ledger.** Tables were created
   with queue entries but no ledger entries (all of AIME2025, the
   kube/cnt-v02 blocks), so the recording skill — whose worklist
   came from the ledger — could not see them. Recording regressed
   to manual work.
2. **The verify→delete loop starved.** The queue's contract was
   "Tuan deletes entries after verifying" — in practice 147 of
   151 entries sat `running` indefinitely, and doc statuses
   drifted (a 299-cell manual reconcile was needed 2026-07-22).

Tuan's actual routine is three verbs — create tables, run
experiments, check experiments — so v2 gives each verb one skill
and one source of truth.

## The design

### Files

```
experiments/<stem>.yaml   <->  docs/exp-comp-<stem>.md
  (prm800k-level5, prm800k-level4, gsm8k, aime2025[, misc])
orchestration/jobs.yaml        allocation pool (unchanged)
```

One entry = one experiment, cradle to grave. Fields: id,
launcher, hash (written once, audited by --verify), config_root,
overrides (dict, never run.*), trials, feeds, group, **status**,
priority, expected_hr, launch, history, note. Commands are
derived (status.py launch_command), never stored. `recorded` is
gone — `status: scored` subsumes it.

### Lifecycle

```
planned -> inqueue -> running -> scored
                         └----> failed
```

| transition | actor |
|---|---|
| (new) -> planned | exp-tables |
| planned -> inqueue (+priority) | Tuan's decision |
| inqueue -> running (+launch) | exp-run |
| running -> scored | exp-check (stats + audited cell write) |
| running -> failed (+history) | exp-check (crash/stall/missing) |
| failed -> inqueue | Tuan's explicit request ONLY |

Failed runs are **reported, never auto-relaunched** — a
deliberate choice (2026-07-22) after repeat-preemption episodes:
a run on its third death should cross Tuan's desk, not silently
burn a fourth GPU slot. Relaunch-resume is cheap when he does
requeue (result dirs are hash-addressed; finished trials skip).

The ledger is status truth; doc tables mirror it. Doc status
vocabulary is the same five bare words (no parentheticals).

### status.py (still the one compose/hash source of truth)

New subcommands (all read-only on ledgers except --sync-doc,
which writes DOC status cells only):

- `--queue` — inqueue entries, priority-sorted, with launch
  commands. The exp-run worklist.
- `--check-running [--ledger STEM]` — verdict per running entry:
  `missing` (no dir) | `finished` (n_done>=trials; reports
  scored=k/n + wandb) | `still-running` (W&B alive — NEVER
  touched) | `stalled`. The exp-check worklist.
- `--dedup ROOT k=v...` — --check plus every ledger entry
  sharing the hash (ledger file, id, status, feeds). The
  exp-tables dedup step.
- `--sync-doc STEM [--apply]` — patch doc status cells from
  ledger truth. Conservative: only tables whose Limitations line
  names a feeds key, only bijectively matched rows, never
  downgrades a scored cell, dry-run by default.
- `--ledger STEM` filter; global id/hash uniqueness in --verify;
  legacy fallback (reads experiments.yaml if experiments/ is
  absent) until retirement.

### Skills (3 verbs + the independent smoke test)

- **exp-tables** — create/adapt a table; --dedup every cell;
  reuse existing entries (append feeds key) or append
  status: planned; never duplicate a config.
- **exp-run** — one drain cycle: jobs.yaml refresh, 0%/0MiB
  probe, walltime guard, nohup srun --overlap, fire-and-forget;
  flips inqueue->running.
- **exp-check** — verdict every running entry; score finished
  ones into their tables (compute_stats is the only number
  source; no-silent-overwrite audit carried over from
  exp-record-results); mark failures failed + report; launch
  GPU scoring (prepare_scored_dataset needs CUDA) onto idle
  pooled GPUs fire-and-forget.
- **exp-smoke-test** — unchanged and deliberately OUTSIDE the
  workflow: a post-code-change feasibility check (isolated
  results_subdir=smoketest, W&B disabled, no ledger entry),
  run before an experiment ever enters the system.

## Migration (2026-07-22, scripts/migrate_ledger.py)

225 ledger entries + 151 queue commands -> 339 entries across 4
per-doc files (37 hash-matched, 114 queue-only — the measured
size of the bypass bug). Status seeded from disk truth: 133
scored (== the old recorded:true count exactly), 181 running,
21 planned, 4 inqueue (== the queue's planned entries). 147
launch blocks preserved. Known wrinkle: 4 pre-rename mcts_cnt
entries carry stored hashes that no longer recompose (their dirs
were relinked via from_dir); --verify reports them until Tuan
retires or re-annotates those legacy entries.

Old files (experiments.yaml, orchestration/queue.yaml) are
retired only after the running backlog is drained and Tuan
signs off (git history preserves them).

## First fruits

The very first --check-running pass auto-detected
`kdepth-bl-v01-l5-da2-qwen7bgptq` as `missing` (queued 07-21,
never produced a dir) — the exact class of silent failure the
old workflow could only catch by hand.
