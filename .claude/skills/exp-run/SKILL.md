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

## 5. Walltime guard, then cancel-the-useless guard

For each idle job J, place the **highest-priority inqueue entry
whose `expected_hr` fits inside J's `%L` remaining**. An entry
that does not fit is skipped for J and stays at the head (try it
on the next idle job). Entries without `expected_hr` fit
anywhere; mention it.

If **no** inqueue entry fits J — i.e. even the smallest
`expected_hr` in the queue exceeds J's remaining time — then J
can never host work before it expires. **`scancel` it
immediately:**

```
scancel <jobid>
```

Then drop it from the pool for this cycle (the next `squeue`
refresh drops it from `jobs.yaml` automatically) and report the
cancellation with the numbers that justified it (`%L` left vs.
the queue's minimum `expected_hr`).

**Cancellation preconditions — ALL must hold.** This is the only
destructive action in the skill; a wrong `scancel` throws away an
allocation that took days to schedule.
- J probed **idle** in §4 (`0 %` AND `0 MiB`). Never cancel on a
  util-only signal.
- J is **not** in `exclude:`.
- J was **not claimed this cycle** (a job you launched onto
  reads 0/0 for ~1-2 min — cancelling it would kill the run you
  just started).
- The queue is **non-empty**. With nothing `inqueue` there is no
  yardstick, so cancel nothing and say so — an idle allocation
  with hours left is an asset once Tuan queues more work.
- **No** inqueue entry lacks `expected_hr` (an unsized entry
  fits anywhere, so nothing is ever un-hostable).

Safe because 0 MiB means the process is gone: `generate_*.py`
holds both the vLLM engine and the PRM resident through scoring
and the dataset write, and frees the GPU only at process exit —
there is no live phase that reads 0 MiB (verified 2026-07-31).

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

## 7. Sync the docs, report + validate

**Sync first.** For every ledger you edited this cycle, the doc's
status cells are now stale (`inqueue` rows that are actually
`running`). Run, per affected ledger stem:
```
python orchestration/status.py --sync-doc <stem>          # dry-run
python orchestration/status.py --sync-doc <stem> --apply  # if mismatches=0
```
The table's status cell is DERIVED from the ledger — never
hand-edit it in parallel, and never skip this step. Omitting it on
2026-07-24 left 30 launched cells still reading `planned`/
`inqueue` across three docs.

Then report directly: pool size (excluded/pruned/added), idle
jobs, each (job, entry-id) launch with pid, skips and why, **any
jobs `scancel`led and the numbers that justified it**, the
inqueue count before/after, and the sync result (`patched=N`).

**Always close the report with the occupancy table:**
```
python orchestration/status.py --running
```
One row per live allocation — `job | wandb | experiment | family |
elapsed | est. left | job left` — sorted by `est. left` ascending,
so the next GPU to free is the top row. `elapsed` is measured from
`launch.at`; `est. left` is `expected_hr - elapsed` and shows
`OVER <hr>` past the estimate (a soft signal — but `OVER` with
`job left` near zero means the allocation will cut the run off).
`wandb` is the run id from the result manifest, paste-ready for the
W&B API; `?` means no manifest matched the entry's hash yet (normal
for the first ~2 min of a launch). Treat it as a pointer, not a
liveness check — W&B mislabels live runs `crashed` on a dropped
heartbeat, and a row can be a FINISHED run whose allocation
persists (its GPU probes 0%/0MiB — §4's probe is the liveness
ground truth; such a row is a completion for exp-check). Paste
the table verbatim.

The table also prints `idle=N` (live jobs no `running` entry
claims — cross-check §4's probe) and `stale_running=N` (entries
marked `running` whose job ended or was reused, i.e. unresolved
completions). Report `stale_running` as a count and hand it to
`exp-check`; this skill never marks anything scored/failed (§0).

Then `python -c "import yaml; ..."`-validate every ledger file you
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
