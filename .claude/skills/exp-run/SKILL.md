# exp-run

Loaded when: Tuan wants one cycle of the experiment orchestrator
executed — "run the experiments", "run an orchestrator cycle",
"fill the idle GPUs from the queue", `/exp-run`. One attended
cycle, watched. (For launching on a timer over a fixed window,
that is the `exp-cron` skill.) Design:
[docs/decisions/experiment-workflow-v2.md](../../../docs/decisions/experiment-workflow-v2.md)
(supersedes the queue.yaml design in
hpc-idle-gpu-orchestration.md).

One invocation = ONE full cycle: refresh the allocation pool,
probe for idle GPUs, launch as many `inqueue` ledger entries as
there are usable idle GPUs — **lowest `priority` number first**
— mark them `running`. Then stop: no waiting, no watching.

---

## 0. What this skill is and is NOT

**IS:** the mechanical drain-the-queue step. Launch onto idle
GPUs, flip ledger status, exit.

**IS NOT:**
- a monitor — it does NOT verify startup, detect stalls, check
  W&B, or retry failures. That is the `exp-check` skill's job
  (and even exp-check only *reports* failures; it never
  relaunches on its own).
- a completion tracker — it NEVER marks anything scored/failed.
- a queue editor — Tuan decides what is `inqueue` and at what
  priority; this skill only flips `inqueue -> running`.

## 1. Files

- `orchestration/ledgers/*.yaml` — per-doc ledgers (workflow
  v2). The ONLY fields this skill touches, via targeted Edit on
  one entry at a time (never a full-file rewrite): `status:
  inqueue -> running` and the appended
  `launch: {job_id, node, pid, at}` block.
- `orchestration/runtime/jobs.yaml` — allocation pool.
  `exclude:` (jobs to never touch — PRESERVE across refreshes)
  and `jobs:` (auto-rewritten every cycle from squeue).

Never read the raw ledgers for the worklist — that is what
`python orchestration/status.py --queue` is for (compact,
priority-sorted, with ready-to-run commands).

## 2. Refresh the allocation pool

```
squeue -u tnguyen9210 -h -t R -o "%i %P %N %L"
```

Keep rows whose partition starts with `gpu`. Rewrite
`jobs.yaml`'s `jobs:` list (one `- <jobid>   # <node> (<%L> left)`
line each), carrying `exclude:` over unchanged. Pending (`PD`)
jobs never appear (`-t R`) — correct: you cannot
`srun --overlap` into an unstarted allocation. Capture `%L`
(time remaining) per job for the walltime guard.

## 3. Read the worklist

```
python orchestration/status.py --queue
```

Each entry prints as a `# id ledger prio expected_hr hash`
comment plus the exact launch command. Empty -> tell Tuan
"queue empty" and stop. Drain order is priority ascending
(ties: ledger, file order) — the tool sorts; do not re-sort.

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
idle). Any probe error -> that job is unavailable this cycle;
mention it in the summary; do not retry.

These are 1-GPU allocations and the probe is cgroup-scoped. One
idle job = capacity for exactly one launch.

## 5. Walltime guard

If the entry has `expected_hr` and the job's remaining time is
less, skip this job for this entry (try the next idle job; the
entry stays at the head). Entries without `expected_hr` launch
anywhere; mention it.

## 6. Launch + mark

From the repo root, using the command `--queue` printed
verbatim:

```
nohup srun --jobid=<id> --overlap <command> > /dev/null 2>&1 &
disown
```

- stdout to /dev/null is deliberate: W&B is the run's log.
- Record the nohup'd pid; claim `<id>` for the rest of the
  cycle (a fresh launch takes ~1-2 min to occupy its GPU —
  never re-probe a claimed job).
- Then a **targeted Edit on that entry in its ledger file**
  (`orchestration/ledgers/<ledger>.yaml`; `--queue` printed
  the ledger):
  - `status: inqueue` -> `status: running`
  - append:
    ```
    launch:
      job_id: <id>
      node: <node>
      pid: <pid>
      at: <YYYY-MM-DD HH:MM>
    ```
    (If the entry already has a `launch:` block from an earlier
    attempt, replace it; prior attempts live in `history:`,
    written by exp-check.)
- Fire-and-forget: do NOT wait for W&B init, do NOT check the
  process again.

Repeat §4-§6 until idle jobs or inqueue entries run out.

## 7. Report + validate

Report directly: pool size (excluded/pruned/added), idle jobs,
each (job, entry-id) launch with pid, skips and why, and the
inqueue count before/after. Then
`python -c "import yaml; ..."`-validate every ledger file you
edited plus jobs.yaml still parse. A malformed ledger breaks
every future cycle — fix before ending the turn.

## 8. Failure modes / cautions

- **Half-edited cycle**: entries you launched stay `running`
  (they ARE running); entries you didn't touch stay `inqueue`.
  Never roll back a status for a process you started.
- **Launch dies instantly**: by design not detected here; it
  surfaces as a `failed`/`missing` verdict in the next
  exp-check cycle.
- **Duplicate configs**: `--verify`'s global hash-uniqueness
  check guards the ledgers; if `--queue` ever shows two entries
  with the same hash, launch only the first and flag it.
- **`exclude` is load-bearing**: a jupyter session Tuan just
  opened looks idle. If he says a session is his, add it to
  `exclude` before the next cycle.
- **Never launch outside the pool**: only job ids in the
  post-refresh `jobs:` list, minus `exclude`.
