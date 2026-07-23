# exp-check

Loaded when: Tuan asks to check the experiments — "check the
experiments", "which runs finished/failed", "score the finished
runs", "update the tables with finished results", `/exp-check`.
Absorbs the old `exp-record-results` skill (its audit rules live
on in §4). Sibling of `exp-run` (launches) and `exp-tables`
(creates tables + ledger entries). Design:
[docs/decisions/experiment-workflow-v2.md](../../../docs/decisions/experiment-workflow-v2.md).

One invocation: verdict every `running` ledger entry, then per
verdict — score finished runs into their doc tables, mark
crashed/stalled/missing runs `failed` (REPORT, never relaunch),
leave live runs untouched. This closes the gap between "done on
disk" and "number in the doc" AND the gap between "queue says
running" and "nothing is actually running" — the two drifts the
old workflow starved on.

---

## 0. The two prime rules

1. **A run still alive on W&B is NEVER touched** — whatever the
   disk looks like. `still-running` verdicts are left exactly
   as they are, every cycle, no exceptions.
2. **Reconcile, don't transcribe** (inherited from
   exp-record-results): before writing any table cell, look at
   what's already there. Empty -> write. Same number -> confirm.
   Different number -> DO NOT overwrite; surface the mismatch
   for Tuan. A mismatch is a question, not an error to auto-fix
   (real causes: re-run moved the number; hand-entered value;
   dead W&B citation).

## 1. Input — one command, never raw ledgers

```
python status.py --check-running [--ledger STEM]
```

One line per `running` entry:
`id  verdict  n_done/trials  scored=k/n  wandb=<state>  dir=<..>`

Verdicts: `finished | still-running | stalled | missing`.
Batch with `--ledger` when draining a large backlog (bounds W&B
calls and scoring fan-out).

## 2. Per-verdict actions

### still-running -> untouched. Count them for the report.

### finished, scored < trials -> launch scoring, stay running
`prepare_scored_dataset.py` REQUIRES CUDA + loads the PRM —
never run it on the login shell. Reuse exp-run's pool mechanics
(squeue refresh -> idle probe, §4-§5 of exp-run/SKILL.md) and
fire-and-forget one idle GPU per entry:

```
nohup srun --jobid=<id> --overlap python prepare_scored_dataset.py \
  --config-name <config_root> <overrides> run.num_trials=<trials> \
  > /dev/null 2>&1 &
```

(bon configs need `+prm=<prm>` — no prm group of their own.)
The entry stays `running`; the NEXT exp-check cycle sees the
scored files and completes it. No idle GPU -> report those
entries as "blocked: needs GPU scoring".

### finished, scored == trials -> score, record, flip
1. `python compute_stats.py --config-name <config_root>
   <overrides> run.num_trials=<trials>` — CPU-only; THE number
   source (§3). Grading is 48-way parallel per run since
   2026-07-22 (`+num_proc=N` to override) — one run takes
   ~1-2 min, dominated by W&B sync, so **run them one after
   another**, not concurrently. If you must run several at
   once, pass `+num_proc=1` each and cap at ~12 concurrent:
   the shell lives INSIDE a SLURM job's cgroup (ssh sessions
   are adopted into the newest GPU allocation — ~22 cores,
   ~110 GiB), and a 45-way batch got ~21 processes silently
   cgroup-OOM-killed (no traceback) on 2026-07-22. One log
   file per process, never a shared path (concurrent writers
   race). Verify every log ends with its summary line before
   reading numbers.
2. Audit-write every doc cell the entry's `feeds` names (§4).
3. Targeted Edit in the entry's ledger: `status: running ->
   scored`.
4. After all entries of a doc: `python status.py --sync-doc
   <stem> --apply` to settle remaining status cells
   (conservative; its report lists anything it wouldn't touch).

### stalled / missing -> mark failed, REPORT (never relaunch)
**Probe before flipping** (2026-07-22 lesson): W&B `crashed` is
a heartbeat verdict, not proof of death — a live searcher was
once marked failed while mid-trial-2. If the entry has a
`launch:` block, first check the process:
```
srun --jobid=<launch.job_id> --overlap bash -c \
  'nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
   xargs -r ps -o pid,etime,cmd -p' | grep <config_root>
```
A live matching process -> the entry STAYS running (note the
wandb lapse); only a dead/absent process earns `failed`.
Targeted Edit on the entry:
- append to `history:` (create if absent):
  ```
  history:
    - {at: <now>, job_id: <launch.job_id>, outcome: <verdict>,
       trials_done: <n_done>}
  ```
- `status: running -> failed`
Update the doc row's status cell to `failed` (via --sync-doc or
directly if unambiguous). Report these entries PROMINENTLY with
their history (how many prior deaths — a 3rd preemption reads
differently than a 1st). Requeue (`failed -> inqueue`) ONLY when
Tuan says so; resume then skips finished trials automatically
(the result dir is hash-addressed).

## 3. The authoritative number source (do not guess)

`compute_stats.py` prints the summary; the dict keys are
underscore form, the doc columns are @gb form:

| dict key | doc column |
|---|---|
| `pass_gb` | pass@gb |
| `naive_gb` | naive@gb |
| `weighted_gb` | wei@gb |
| `maj_gb` | maj@gb |

Format `.NNNN<br>±.NNNN` (4 dp). **`hr/trial` does NOT come from
compute_stats** — read `timing_state.json` in the result dir
(`avg_time_per_trial_hr`), else W&B `time_per_trial_hr`; if
unavailable leave `—`, never fabricate. Side effect: compute_stats
also refreshes the run's W&B `eval/*` summary (idempotent).

## 4. The cell audit (verbatim from exp-record-results)

Per cell the entry feeds:
- cell empty (`—`) -> write; set `trials` to real n_done, status
  to `scored`, add the W&B run id to the table's `**W&B:**` line
  if it keeps one.
- cell has a number -> compare (within rounding). Agrees ->
  confirm, proceed. Differs -> collect `{cell, existing,
  computed, entry}`; do NOT write; do NOT flip status.
- `feeds` names a cell you can't find -> "feeds key X has no
  matching cell"; skip that write, still count the entry's other
  cells.
- One entry can feed several cells — handle each; flip
  `status: scored` only when EVERY fed cell was written cleanly.

## 5. Report

Three lists, never blurred, then stop (no commits — standing
rule):

> **Scored** (N entries -> M cells): id -> table/row, the four
> numbers, trials.
> **Failed** (K entries): id, verdict, n_done/trials, history
> depth, last job/node. Ask which to requeue.
> **Untouched**: still-running (count + ids), blocked-on-GPU
> scoring (ids).
> **⚠ Mismatches** (separate, the audit's output): cell, doc
> value, computed value, likely cause. Tuan adjudicates.

## 6. Failure modes

- **Silent overwrite** — the cardinal sin. §0.2 / §4.
- **Marking failed what W&B says is alive** — §0.1; the verdict
  rules already order wandb==running above stalled, trust them.
- **Scoring on the login shell** — prepare_scored_dataset OOMs
  or hogs; always srun --overlap onto a pooled idle GPU.
- **Premature scored flip** — only after every fed cell is
  written cleanly (§4).
- **Auto-requeue** — never. `failed` is a terminal state until
  Tuan speaks.
- **Stale statuses after a partial pass** — if the turn must
  end early, ledger statuses you already flipped are correct
  and stay; report exactly how far you got.
