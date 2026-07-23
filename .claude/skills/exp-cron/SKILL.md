# exp-cron

Loaded when: Tuan wants experiments launched onto idle GPUs
automatically on a timer for a bounded stretch — "run
experiments every 30 minutes for 10 hours", "launch queued
experiments every 20 min for the next 4 hours", `/exp-cron
INTERVAL DURATION`.

One purpose: on a cron schedule, probe the GPU pool for idle
allocations and launch eligible `inqueue` experiments onto them.
A recurring `CronCreate` job fires every INTERVAL; each firing
checks for idle GPUs and starts what fits; the job removes
itself when DURATION elapses.

## Requirements (operational facts)

1. **Runs on the HPC login node.** Launching needs
   `squeue`/`srun`, which run on `r5u*.puma.hpc.arizona.edu`.
   This Claude session is on that node, so cron firings can
   probe and launch.
2. **Lives for the session.** The cron job is in-memory and
   ends when this Claude session ends. Keep the session open for
   the window; to survive an SSH/laptop disconnect, launch
   Claude inside `screen` (`/usr/bin/screen`) on the login node
   first.
3. **Launches real jobs unattended, every firing.** No human
   confirms each launch. The idle probe below (0 % AND 0 MiB,
   claim-after-launch) prevents double-booking within a firing;
   the bounded DURATION keeps the whole run finite.

## 1. Parse the request

Extract:
- **INTERVAL** = minutes between firings (`every 30 min`,
  `20-min`).
- **DURATION** = how long to keep going (`for 10 hours`, `for
  the next 4 h`), OR a cycle count to convert
  (`10 cycles` -> `DURATION = cycles · INTERVAL`, report the
  implied duration).

If INTERVAL or DURATION is missing/unparseable, ask once — never
guess. Constraints: **INTERVAL ≥ 1 min**; **DURATION ≤ 7 days**
(warn if longer); pick an off-minute cron expression (§2).

Sanity note: cells take ~3-16 h, so a GPU frees only every few
hours. A firing every 30 min mostly finds nothing newly idle —
harmless (it launches nothing and exits), but a 5-10 min cadence
is overkill; suggest 20-60 min unless Tuan wants tighter refill
latency. Cron jitter (≤10% of period) makes the firing count
approximate — say so.

## 2. Create the cron job

Build a **unique sentinel** so each firing can find and delete
its own job via CronList (CronCreate returns the id only to the
creating turn; firing turns self-identify by prompt text). Tag
with the launch time, e.g. `EXP-CRON-<HHMM>`.

Cron expression from INTERVAL (off-minute — the tool asks for
it):
- `INTERVAL ≤ 59`: `*/INTERVAL * * * *`.
- `INTERVAL == 60`: `7 * * * *`.
- `INTERVAL` a multiple of 60: `7 */H * * *`, `H = INTERVAL/60`.
Round a non-clean INTERVAL to the nearest expressible cadence
and tell Tuan what you rounded to.

Write the **stop-time** (now + DURATION, absolute local time) to
`orchestration/runtime/exp_cron_stop.txt` (gitignored dir). Second line:
the cron job id (for clean manual teardown). This file is the
bound the firings read.

Then `CronCreate`:
- `cron`: the expression.
- `recurring`: true.
- `prompt`:
  ```
  /exp-cron TICK sentinel=EXP-CRON-<HHMM> stop_at=<YYYY-MM-DD HH:MM>
  ```

## 3. Launch once now, then let cron drive

Do NOT wait for the first firing:
1. Run one launch pass now (§5). Prefix `exp-cron launch 1
   (window ends <stop_at>)`.
2. Confirm to Tuan: cron id + expression + human cadence, the
   window end, the ~N firings implied, the 7-day cap, that it
   ends when the session ends. Turn ends; cron takes over.

## 4. On a TICK firing

Woken by `/exp-cron TICK sentinel=S stop_at=T`:
1. **Check the clock FIRST.** If now ≥ T (window over):
   - `CronList` → find the job whose prompt contains sentinel
     `S` → `CronDelete` it. (Fallback: id on line 2 of
     `orchestration/runtime/exp_cron_stop.txt`.)
   - Clear `orchestration/runtime/exp_cron_stop.txt`.
   - Final report (§6). Do NOT launch again.
2. Else (inside the window): run one launch pass (§5), report
   prefixed `exp-cron launch (window ends T)`. Leave the cron
   job alone — it re-fires itself.

Never create a second cron job on a TICK — the job already
recurs; a second would double the cadence and race launches.

## 5. The launch pass (idle probe + launch)

Each firing does exactly this:

1. **Refresh the pool.**
   ```
   squeue -u tnguyen9210 -h -t R -o "%i %P %N %L"
   ```
   Keep rows whose partition starts with `gpu`. Rewrite
   `orchestration/runtime/jobs.yaml`'s `jobs:` list (one
   `- <jobid>   # <node> (<%L> left)` line each), carrying
   `exclude:` over unchanged (it is load-bearing — a jupyter
   session Tuan opened looks idle). Capture `%L` (time
   remaining) per job for the walltime guard.
2. **Read the worklist.**
   ```
   python orchestration/status.py --queue
   ```
   Each entry prints as a `# id ledger prio expected_hr hash`
   comment plus its exact launch command, priority-sorted.
   Empty -> nothing to launch this firing; exit the pass.
3. **Probe each pooled job** NOT in `exclude` and NOT already
   claimed this firing:
   ```
   srun --jobid=<id> --overlap nvidia-smi \
     --query-gpu=utilization.gpu,memory.used --format=csv,noheader
   ```
   **Idle ⇔ `0 %` utilization AND `0 MiB` memory** (an active
   vLLM run holds GBs even between batches; a momentary 0%-util
   dip alone is not idle). Any probe error -> that job is
   unavailable this firing; note it; do not retry.
4. **Walltime guard.** If the entry has `expected_hr` and the
   job's `%L` remaining is less, skip this job for this entry
   (try the next idle job; the entry stays at the head).
5. **Launch + mark**, using the command `--queue` printed
   verbatim, from the repo root:
   ```
   nohup srun --jobid=<id> --overlap <command> > /dev/null 2>&1 &
   disown
   ```
   - Record the nohup'd pid; claim `<id>` for the rest of the
     firing (a fresh launch takes ~1-2 min to occupy its GPU —
     never re-probe a claimed job and double-book it).
   - Targeted Edit on that entry in its ledger file
     (`orchestration/ledgers/<ledger>.yaml`; `--queue` printed the
     ledger): `status: inqueue -> running`, and append
     ```
     launch:
       job_id: <id>
       node: <node>
       pid: <pid>
       at: <YYYY-MM-DD HH:MM>
     ```
     (If the entry already has a `launch:` block, replace it.)
   - Fire-and-forget: do NOT wait for startup, do NOT re-check
     the process.
   Repeat 3-5 until idle jobs or inqueue entries run out.
6. **Validate**: `yaml.safe_load` every ledger file you edited
   plus `jobs.yaml` still parse. Fix before ending the turn — a
   malformed ledger breaks every future firing.

Report the firing tersely: pool size, idle jobs, each
(job → entry-id, pid) launched, skips and why, inqueue
before/after.

## 6. Teardown + final report

At window end (§4.1), after deleting the job:
> exp-cron done: window elapsed (~N firings at INTERVAL-min
> cadence). Launched across the run: <entry -> job/pid>, one
> list. Ended with inqueue=<k>, running=<m>. <if k>0: "queue
> not fully drained — <k> inqueue, no idle GPUs the last
> firing.">

If Tuan says "stop the cron" mid-window: `CronList` →
`CronDelete` the sentinel's id → clear the stop-file → confirm.
End any session with `CronList` if a job was running — a
lingering job is the thing to prevent (session exit clears it
too).

## 7. Failure modes / cautions

- **Leaving cron running past the window** — the cardinal sin;
  launches jobs Tuan didn't intend. §4.1 clock-check +
  self-delete. If the CronDelete fails, RETRY before ending the
  turn; report if it still won't die.
- **Session ends mid-window** → GPUs go idle. Not preventable
  from inside the skill; the requirements note warns Tuan up
  front (use `screen` to harden).
- **Second cron job on a firing** — doubles the rate and races
  launches. §4: firings never create; only §2 creates.
- **Double-booking a GPU** — claim `<id>` at launch and never
  re-probe it that firing (§5.5); a fresh launch is invisible
  for ~1-2 min.
- **`exclude` dropped on refresh** — preserve it verbatim; a
  jupyter session Tuan opened reads as idle (§5.1).
- **Launching outside the pool** — only job ids in the
  post-refresh `jobs:` list, minus `exclude`.
- **Un-clean interval** — round INTERVAL to an expressible
  cadence and say what you rounded to.
