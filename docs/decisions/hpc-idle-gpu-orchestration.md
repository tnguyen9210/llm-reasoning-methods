# HPC idle-GPU experiment orchestration: 15-minute cycle over queue.yaml + auto-maintained jobs.yaml

*2026-07-10 — design decided (details below settled with Tuan);
not yet armed. Go-live = scheduling the cycle (see "Scheduler").*

Records the design of the recurring orchestration system that
launches queued experiments onto idle GPUs inside Tuan's existing
SLURM allocations, and the choices made where the design forked.

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
3. **`log.md` — append-only cycle log.** One entry per cycle:
   timestamp, pool refresh result, idle probes, (job, experiment)
   pairs launched, skips and why. This is the audit trail for
   "what happened overnight".

## The cycle (every 15 minutes, aligned :00/:15/:30/:45)

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
   ~1–2 min, well inside the 15-min cadence).
4. **Walltime guard**: skip a job whose remaining time
   (`squeue -j <id> -o %L`) is less than the candidate entry's
   `expected_hr` — launching a 6-hour run into a 2-hours-left
   allocation wastes the whole run. Entries without
   `expected_hr` launch anyway, with a note in the log.
5. Launch the next planned entry inside the chosen allocation:
   `cd <repo> && nohup srun --jobid=<id> --overlap <command>
   > /dev/null 2>&1 &` — stdout discarded because W&B is the
   run's log of record (per Tuan; established convention).
6. Mark the entry `running` + write its `launch:` block; claim
   the job id for the rest of this cycle (no double-booking a
   GPU whose new process hasn't loaded yet).
7. Repeat 3–6 until no idle pooled GPUs remain or the queue has
   no planned entries. Append the cycle's log entry either way.

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
[.claude/skills/exp-orchestrate-cycle/SKILL.md](../../.claude/skills/exp-orchestrate-cycle/SKILL.md)
(user-invocable as `/exp-orchestrate-cycle`; also the contract a
scheduled/headless invocation executes — one invocation = one
cycle).

Decided: **Claude-managed cron (`CronCreate`, `*/15 * * * *`) if
it executes locally on this login node; otherwise system crontab
running headless `claude -p "<run one orchestrator cycle>"`.**
CronCreate's locality must be verified at go-live (its skill
description says "cloud agents", and a cloud runner has no
`squeue`/`srun` — if so, the crontab fallback applies). Either
way the cycle survives closed editors and ended sessions, and
`*/15` gives the exact :00/:15/:30/:45 alignment requested.
A `/loop` in a live session was considered and rejected as the
primary mechanism (dies with the session) but remains useful for
a supervised first day. A no-Claude pure-Python cron script was
considered (cheapest, most rigid) and rejected for now — Tuan
wants the agent in the loop.

## Assumptions / limits

- One GPU per pooled allocation (true of the current jupyter
  jobs: 1× V100S each). A multi-GPU allocation would need
  per-GPU accounting this design doesn't have.
- The launch environment (repo path, `py311` env, HF/W&B creds)
  is inherited from the login-node shell that `srun --overlap`
  is invoked from — the same environment this design was
  validated in.
- Cycles are assumed non-overlapping (a cycle takes seconds to
  a couple of minutes ≪ 15 min). If a cycle ever runs long, the
  per-cycle claim set plus re-probing makes a concurrent cycle
  mostly harmless, but no lock is taken — accepted risk at this
  cadence.
