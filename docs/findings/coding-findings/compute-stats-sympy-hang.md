# compute_stats.py can hang forever on a pathological boxed answer

*2026-07-07*

**Observation:** `utils/metrics.py`'s grading helpers
(`run_with_timeout`, `_grade_pred`) bound `grader2.math_equal` with
`signal.alarm`, on the assumption that a timeout would abort a slow
symbolic comparison. This assumption is false whenever the hang is
inside `sympy`'s C-level code: `SIGALRM` only gets delivered between
Python bytecode instructions, so a call stuck deep in `sympy`
(parsing, simplification, matrix comparison) can block signal
delivery indefinitely. The nominal 2s timeout does nothing in that
case — the process spins at ~100% CPU with no output until killed
externally.

Reproduced live: `compute_stats.py --config-name mcts_cnt_prm800k
llm=qwen_3b_gptq_int4 prm=llama_prm run.num_trials=2` hung for 6+
minutes (99% CPU, one process) before being killed; retried, hung
identically. Isolated (bisecting record-by-record, then
completion-by-completion within the hanging record) to
`test/precalculus/920.json`, `completion[4]`, whose boxed "answer"
was a whole equation rather than a value:
`\mathbf{A}^{27} + \mathbf{A}^{31} + \mathbf{A}^{40} = \mathbf{I}`.
Comparing this string against the ground-truth matrix inside
`grader2.symbolic_equal` hung so completely that even a **fresh**
`signal.alarm(6)` in a brand-new, minimal isolation script — nothing
to do with the original call stack — could not interrupt it. That
rules out a tunable-timeout-value fix; the mechanism itself cannot
bound this class of hang.

## The fix

`grader2.py` already had a working hard-kill alternative, unused by
`metrics.py`: `math_equal(..., timeout=True)` routes symbolic
comparison through `call_with_timeout(symbolic_equal_process, ...)`,
which runs the comparison in a `multiprocessing.Process` and calls
`.terminate()` if it's still alive after the timeout — a real OS-level
kill, not a signal that the target code can fail to receive. Both
`metrics.py` call sites were calling `grader2.math_equal(...)` with
the default `timeout=False` (in-process, signal-only). Fixed by
passing `timeout=True` at both:

```python
# run_with_timeout
result = fn_grade(c_answer, gt_answer, timeout=True)

# _grade_pred
return grader2.math_equal(pred_answer, gt_answer, timeout=True)
```

The outer `signal.alarm` in both functions is left in place — it
still usefully bounds `extract_answer` (pure Python, not sympy) and
acts as a secondary guard around the whole call.

## Verification

- The previously-hanging `completion[4]` now resolves in ~1s
  (`False`, correctly) instead of hanging forever.
- Both trial files for the affected run (`qwen-3b gptq-int4` +
  rlhflow, `cfg-0ad81fba`) replayed record-by-record end-to-end with
  no hangs — including a second copy of the same
  `test/precalculus/920.json` record in the other trial file, which
  now takes ~10s (not instant, not infinite).
- The real `compute_stats.py` invocation for this cell now completes
  in ~1 minute and logs correctly to its W&B run (`bigbjzi4`).

## Cost of the fix

`multiprocessing.Process` spawn + join has real overhead per call —
several records that used to return in <0.1s now take 0.5-10s (a
handful of multi-second symbolic comparisons that were previously
either instant or infinite are now just slow). Net effect on a
256-record (2-trial) run: total wall-clock went from "never finishes"
to ~1 minute — an unambiguous win, but future `compute_stats.py` runs
on other configs may take noticeably longer than pre-fix runs did
*when they didn't hit a poison record*, since every grading call now
pays subprocess overhead, not just the ones that used to hang.

## Implication

Any historical `compute_stats.py` run that appeared to "just take a
long time" (rather than visibly hang) may have been silently stuck in
this exact failure mode without anyone noticing, if it was eventually
killed for an unrelated reason (job timeout, node preemption) before
producing output — there was no way to distinguish "slow" from "hung
forever" without this fix, since neither one printed anything before
completing. Any config whose scored data includes a candidate that
boxes a non-value expression (an equation, an unresolved symbolic
expression mixing matrices and huge literal powers, etc.) is a
candidate for this — not unique to `qwen-3b gptq-int4`/rlhflow; that
combination is simply the first one to have surfaced it.

## Connections

- `utils/metrics.py::run_with_timeout`, `_grade_pred` — the fixed
  call sites.
- `utils/grader2.py::math_equal`, `call_with_timeout`,
  `symbolic_equal_process` — the pre-existing hard-kill path this fix
  now actually uses.
- [exp-comparison.md](../../exp-comparison.md) — cnt-mcts (updated)
  `model family, size, quantization comparison` (rlhflow) table,
  `qwen-3b gptq-int4` row — the cell this was discovered while
  filling in.
