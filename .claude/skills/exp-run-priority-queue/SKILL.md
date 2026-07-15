# exp-run-priority-queue

Loaded when: Tuan wants one cycle of the idle-GPU experiment
orchestrator executed — "run the experiments (in the queue)",
"run an orchestrator cycle", "fill the idle GPUs from the
queue", "check job ids and run planned experiments",
`/exp-run-priority-queue`. **Manual-trigger only — not on a
cron/recurring schedule** (the prior cron setup was removed
2026-07-14; Tuan runs this himself whenever he has new
allocations or wants the queue drained). Design:
[docs/decisions/hpc-idle-gpu-orchestration.md](../../../docs/decisions/hpc-idle-gpu-orchestration.md).
Sibling of `exp-smoke-test` / `exp-record-results` (this skill
launches real experiments; those validate and record them).

One invocation = ONE full cycle: refresh the allocation pool,
probe for idle GPUs, launch as many `planned` queue entries as
there are usable idle GPUs — **lowest `priority` number first**
(1 before 2 before 3…), file order breaking ties within the same
priority — mark them `running`. Then stop — no waiting, no
watching.

---

## 0. What this skill is and is NOT

**IS:** the mechanical drain-the-queue step. Launch onto idle
GPUs, exit.

**IS NOT:**
- a monitor — it does NOT verify startup, detect stalls, check
  W&B, or retry failures. A run that crashes stays `running`
  until Tuan's manual verification catches it (his explicit
  call, 2026-07-10; W&B is the run log of record).
- a completion tracker — it NEVER marks anything done and NEVER
  deletes queue entries. Tuan deletes entries after verifying.
- a scorer or ledger writer — no `compute_stats.py`, no
  `experiments.yaml` edits. Recording happens later through the
  normal exp-record-results / `status.py --backfill` flow.

## 1. Files (all under `orchestration/`)

- `queue.yaml` — list of entries: `id` (unique), `command`
  (full launch command, run from the repo root), `expected_hr`
  (optional), `priority` (integer, lower = launched sooner —
  treat a missing `priority` as the lowest priority, i.e. sort
  after every entry that has one), `status` (`planned` |
  `running`), and — written by this skill at launch — `launch:
  {job_id, node, pid, at}`. Drained in **priority order (1, 2,
  3, … ascending)**, file order breaking ties within the same
  priority; Tuan sets/edits `priority` to reprioritize.
- `jobs.yaml` — `exclude:` (list of job ids to never touch —
  PRESERVE across refreshes) and `jobs:` (auto-rewritten every
  cycle from squeue).

Both yaml files are re-read fresh at the start of every cycle —
never assume state from a previous cycle or from conversation
memory. Tuan may have edited either file at any time.

## 2. Refresh the allocation pool

```
squeue -u tnguyen9210 -h -t R -o "%i %P %N %L"
```

Keep rows whose partition starts with `gpu`. Rewrite
`jobs.yaml`'s `jobs:` list from this (one `- <jobid>   # <node>`
line each), carrying `exclude:` over unchanged. Jobs that
vanished are thereby pruned; new allocations are thereby added.
Pending (`PD`) jobs never appear (`-t R`) — correct: you cannot
`srun --overlap` into an unstarted allocation.

Capture the `%L` (time remaining, `D-HH:MM:SS` or `HH:MM:SS`)
per job for the walltime guard in §5.

## 3. Read the queue

Parse `orchestration/queue.yaml` (`yaml.safe_load`). Work list =
entries with `status: planned`, sorted by `priority` ascending
(1 first, then 2, …), file order breaking ties within the same
priority; an entry with no `priority` sorts after every entry
that has one, in its original file order. If none: tell Tuan
"queue empty" and stop.

## 4. Probe for idle GPUs

For each pooled job id NOT in `exclude` and NOT already claimed
this cycle:

```
srun --jobid=<id> --overlap nvidia-smi \
  --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

**Idle ⇔ `0 %` utilization AND `0 MiB` memory.** Both clauses:
the memory clause is the strong signal (an active vLLM run holds
GBs even between batches; a momentary 0%-util dip alone is not
idle). Any srun/probe error → treat that job as unavailable this
cycle and mention it in your summary to Tuan; do not retry the
probe.

These are 1-GPU allocations, and the probe is cgroup-scoped — it
reads only that job's GPU, even on shared nodes. One idle job =
capacity for exactly one launch.

## 5. Walltime guard

Before pairing an idle job with the next planned entry: if the
entry has `expected_hr` and the job's remaining time (§2's `%L`)
is less, **skip this job for this entry** (try the next idle
job; the entry stays at the head of the work list). A 6-hour run
in a 2-hours-left allocation wastes the whole run. Entries
without `expected_hr` launch anywhere; mention it in your
summary to Tuan.

## 6. Launch + mark

From the repo root
(`/home/u20/tnguyen9210/tnn1/LLMs/llm-reasoning-methods`):

```
nohup srun --jobid=<id> --overlap <command> > /dev/null 2>&1 &
disown
```

- stdout to /dev/null is deliberate: W&B is the run's log.
- Record the nohup'd pid.
- Immediately claim `<id>` for the rest of the cycle (a fresh
  launch takes ~1-2 min to occupy its GPU — never re-probe a
  claimed job and double-book it).
- Then edit that queue entry — **targeted Edit on that entry
  only, never a full-file rewrite** (preserves Tuan's comments
  and ordering):
  - `status: planned` → `status: running`
  - append the launch block:
    ```
    launch:
      job_id: <id>
      node: <node>
      pid: <pid>
      at: <YYYY-MM-DD HH:MM>
    ```
- Fire-and-forget: do NOT wait for W&B init, do NOT check the
  process again. (No `wandb:` field — capturing it would mean
  waiting ~1 min per launch; Tuan matches runs in the W&B UI.)

Repeat §4-§6 pairing until idle jobs or planned entries run out.

## 7. Report + validate

No log file — report the cycle's outcome directly to Tuan in
your response: pool size (excluded/pruned/added), which jobs were
idle, each (job, experiment) pair launched with its pid, any
skips and why, and the queue's before/after planned count.

After all edits: `yaml.safe_load` both yaml files to confirm
they still parse. If either fails, fix before ending the turn —
a malformed queue breaks every future cycle.

## 8. Failure modes / cautions

- **Half-edited queue**: if anything goes wrong mid-cycle, leave
  entries you launched marked `running` (they ARE running) and
  entries you didn't touch as `planned`; never roll back a
  status for a process you started. Report what happened to Tuan.
- **Launch command typo'd / dies instantly**: by design, not
  detected. It surfaces as a `running` entry with no result dir
  growth and a dead/short W&B run at verification time.
- **Same experiment queued twice** (same composed config): both
  launches share a config hash → same result dir; the second
  run's trials resume/collide. Not this skill's job to dedupe —
  flag it to Tuan if two planned entries have byte-identical
  commands, launch only the first.
- **A GPU idles between cycles because its run finished**: fine
  — next cycle's probe sees 0%/0MiB and refills it. That is the
  system working, not a bug. Do NOT conclude the finished run
  "failed"; completion status is Tuan's to judge.
- **`exclude` is load-bearing**: a jupyter session Tuan just
  opened looks idle. If he says a session is his, add it to
  `exclude` before the next cycle, not just this one.
- **Never launch outside the pool**: only job ids in
  `jobs.yaml`'s post-refresh `jobs:` list, minus `exclude`.
