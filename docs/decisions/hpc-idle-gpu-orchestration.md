# HPC idle-GPU experiment orchestration: on-demand cycle over queue.yaml + auto-maintained jobs.yaml

*2026-07-10 — design decided (details below settled with Tuan).
2026-07-14 — reverted from recurring cron to manual-trigger only
(see "Scheduler"); crontab entries removed, `cron_stop_at.txt`/
`cron_output.log`/`run_cycle.sh` are now historical artifacts of
the cron attempt, not part of the live design.
2026-07-22 — **queue.yaml SUPERSEDED by workflow v2** (see
`experiment-workflow-v2.md`): experiment state moved into
per-doc ledgers `experiments/*.yaml` with a 5-state lifecycle;
the drain cycle is now the `exp-run` skill reading
`status.py --queue`. The srun-overlap launch mechanics, idle
probe, walltime guard, and jobs.yaml design below remain the
live reference — only the queue-file half is retired.*

Records the design of the orchestration system that launches
queued experiments onto idle GPUs inside Tuan's existing SLURM
allocations, and the choices made where the design forked. Each
cycle is now triggered manually by Tuan (via `/exp-run-priority-queue`
or a request like "run the experiments in the queue"), not on a
fixed schedule.

## Context

The manual flow this automates was exercised end-to-end on
2026-07-10: probe a jupyter allocation's GPU with
`srun --jobid=<id> --overlap nvidia-smi` (cgroup-scoped — reads
only that job's GPU, even on shared nodes), and if idle, launch a
generation run inside that allocation with
`nohup srun --jobid=<id> --overlap python <launcher> ... &`
(first such run: level-5 llama-1b cnt-mcts, W&B `05lky8bc`, on
job `22814236`). The orchestrator repeats this on a fixed cadence
against a user-curated experiment queue.

This supersedes (for SLURM clusters) the earlier vault todo about
an SSH-based multi-node orchestrator — same pattern, with
`srun --overlap` into existing allocations replacing `ssh` into
raw nodes, and SLURM job IDs replacing a hand-kept node list.

## Files (all under `orchestration/`)

1. **`queue.yaml` — the experiment queue.** Thin and
   user-curated; deliberately NOT `experiments.yaml` (the ledger
   has ~130 append-only entries whose statuses are *derived* from
   disk by `status.py`; "planned in the ledger" ≠ "run this
   week", and level-5 runs may be queued before they have ledger
   entries). Per entry: unique `id`, full launch `command`,
   optional `expected_hr`, `status`, and an orchestrator-written
   `launch:` block. Ownership split:
   - **Tuan**: appends entries (`status: planned`), reorders to
     reprioritize (queue is drained top-first), deletes entries
     after manually verifying completion. The orchestrator never
     marks anything done.
   - **Orchestrator**: flips `planned` → `running` and fills
     `launch: {job_id, node, pid, at}` at launch time, via
     targeted edits only (never a full rewrite), so comments and
     ordering survive.
2. **`jobs.yaml` — the allocation pool. AUTO-MAINTAINED.**
   Refreshed at every cycle from `squeue -u tnguyen9210`
   (running-state GPU-partition jobs): expired/pending/vanished
   entries pruned, newly appeared allocations added. Because the
   job list itself is overwritten each cycle, manual edits to it
   don't stick; the file carries an `exclude:` list (preserved
   across refreshes) as the one manual control — add a job id
   there to keep a session off-limits (e.g. a notebook about to
   be used interactively, whose GPU would otherwise look idle
   and get claimed).
No per-cycle log file (removed 2026-07-14 — Tuan found the
written log unnecessary once cycles are manual and reported
directly in the conversation). The cycle's outcome — pool refresh
result, idle probes, (job, experiment) pairs launched, skips and
why — is reported straight to Tuan in the triggering conversation
instead of written to disk.

## The cycle (run manually, on Tuan's request)

1. Refresh `jobs.yaml` from `squeue` (R-state GPU jobs, minus
   `exclude`).
2. Reload `queue.yaml`; collect `status: planned` entries in file
   order. Both files are re-read from scratch every cycle — no
   state is assumed to persist between cycles (Tuan may have
   added/removed entries or allocations meanwhile).
3. For each pooled job not already claimed this cycle, probe:
   `srun --jobid=<id> --overlap nvidia-smi
   --query-gpu=utilization.gpu,memory.used --format=csv,noheader`.
   **Idle ⇔ utilization == 0% AND memory == 0 MiB** (an active
   vLLM run always holds GBs, so the memory clause alone rules
   out between-batch dips; a fresh launch occupies its GPU within
   ~1–2 min of being started).
4. **Walltime guard**: skip a job whose remaining time
   (`squeue -j <id> -o %L`) is less than the candidate entry's
   `expected_hr` — launching a 6-hour run into a 2-hours-left
   allocation wastes the whole run. Entries without
   `expected_hr` launch anyway.
5. Launch the next planned entry inside the chosen allocation:
   `cd <repo> && nohup srun --jobid=<id> --overlap <command>
   > /dev/null 2>&1 &` — stdout discarded because W&B is the
   run's log of record (per Tuan; established convention).
6. Mark the entry `running` + write its `launch:` block; claim
   the job id for the rest of this cycle (no double-booking a
   GPU whose new process hasn't loaded yet).
7. Repeat 3–6 until no idle pooled GPUs remain or the queue has
   no planned entries. Report the outcome to Tuan directly.

## Explicitly out of scope (Tuan's call, 2026-07-10)

- **No success/failure monitoring.** The orchestrator launches
  and marks `running`; it does not verify startup, detect
  stalls, or retry. A run that dies at any point simply sits as
  `running` with no results and is caught during Tuan's manual
  verification (W&B shows the crash). Rationale: keep the loop
  simple and never burn GPU-days re-crashing on a deterministic
  bug (cf. the llama-3b context-overflow crashes,
  [context-length-overflow-guard.md](context-length-overflow-guard.md)).
  The earlier vault todo's retry-up-to-3 policy is dropped for
  now; revisit only if silent failures become a recurring cost.
- **No completion marking.** Manual, after verification — the
  queue entry is then deleted by Tuan. Ledger recording
  (`experiments.yaml` + `status.py --backfill` picking up the
  orphan dirs) continues through the existing
  exp-record-results flow, untouched by this system.

## Scheduler

The per-cycle procedure is codified as the project skill
[.claude/skills/exp-run-priority-queue/SKILL.md](../../.claude/skills/exp-run-priority-queue/SKILL.md)
(user-invocable as `/exp-run-priority-queue`, or any equivalent
request — "run the experiments in the queue", "run an
orchestrator cycle", "check job ids and run planned
experiments"). One invocation = one cycle, draining `planned`
entries in `priority` order (1 before 2 before 3…, file order
breaking ties). **Manual only, no recurring schedule.**

**History (reverted 2026-07-14):** the original design ran a
system crontab (`orchestration/run_cycle.sh`) firing
`claude -p` every 45 min (a Claude-managed `CronCreate` at
`*/15 * * * *` was the first choice, but was never actually armed
this way — the crontab fallback was used instead). That crontab
was disabled early via `cron_stop_at.txt`'s fixed stop timestamp
and had been firing no-op cycles (logging "stop time reached" and
exiting) for the rest of its life; Tuan asked 2026-07-14 to stop
recurring entirely rather than re-arm it, so the crontab entries
were removed outright. `run_cycle.sh`, `cron_stop_at.txt`, and
`cron_output.log` are kept as historical record of that attempt,
not part of the live flow — do not re-add crontab entries or
treat those files as needing maintenance going forward. If
`/loop`-style periodic triggering is wanted again later, that is
a new decision to make explicitly with Tuan, not a default to
restore.

## Assumptions / limits

- One GPU per pooled allocation (true of the current jupyter
  jobs: 1× V100S each). A multi-GPU allocation would need
  per-GPU accounting this design doesn't have.
- The launch environment (repo path, `py311` env, HF/W&B creds)
  is inherited from the login-node shell that `srun --overlap`
  is invoked from — the same environment this design was
  validated in.
- Cycles are assumed non-overlapping (a cycle takes seconds to a
  couple of minutes, and is now triggered manually rather than on
  a fixed cadence, so overlap is unlikely in practice). If two
  cycles ever did run concurrently, the per-cycle claim set plus
  re-probing makes it mostly harmless, but no lock is taken —
  accepted risk.
