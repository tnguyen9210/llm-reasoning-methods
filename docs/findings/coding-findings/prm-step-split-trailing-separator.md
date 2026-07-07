# PRM step-splitting silently corrupts agg_strategy="last" on non-terminal candidates

*2026-07-06*

**Observation:** `QwenPRM._build_prompt` and
`RLHFlowPRM._build_conversations` (`core/reward_models.py`) both
split a candidate answer into steps with `answer.split("\n\n")`,
with no `removesuffix` first. vLLM's `include_stop_str_in_output=
True` with `stop=["\n\n"]` means **non-terminal** candidates —
generation cut mid-search by the stop string, not by EOS or length
— keep a trailing `"\n\n"`. A plain split on that trailing
separator produces a bogus empty final "step", which gets its own
scored `<extra_0>` position. Under `agg_strategy="last"`
(`core/scoring.py::aggregate_scores`), this bogus score silently
replaces the trajectory's true last-step score. Terminal
candidates (EOS/length-terminated, no trailing separator) were
never affected — the corruption is specific to mid-search
candidates, i.e. every internal node of the search tree.

Same root cause as the generation-side separator bug
([library-version-trajectory-completeness.md](library-version-trajectory-completeness.md)),
but on the scoring side rather than the prompt-construction side,
and a distortion rather than a collapse — nothing crashes or
degenerates, the score is just wrong.

## The fix

Added a shared static helper on the `PRM` base class:

```python
@staticmethod
def _split_steps(answer: str) -> list[str]:
    return answer.removesuffix("\n\n").split("\n\n")
```

Both subclasses call `self._split_steps(answer)` instead of
`answer.split("\n\n")` directly. No-op for terminal candidates
(nothing to strip).

## What the bogus score actually was

Verified live against the loaded `QwenPRM` model
(`unittests/examine_prm_scores_qwenprm_v1.ipynb`, Examples 5–7),
using `unittest.mock.patch.object(PRM, "_split_steps", ...)` to
reproduce the pre-fix behavior for a single `prm.score()` call
without touching any source file (auto-restores on block exit,
even on error).

Four non-terminal candidates, each scored pre-fix: the full
trajectory, and the same trajectory **cut right after its bad
step** (the search-critical case — a candidate whose last
completed step is bad should be valued by that step's low score
under `agg="last"`).

| | bogus score | true/bad step it replaced |
|---|---|---|
| wrong algebra trajectory, full | 0.3000 | 0.5142 (recovered last step) |
| wrong algebra trajectory, cut after bad step | 0.0115 | 0.0103 (the bad step itself) |
| real generated trajectory, full | 0.9731 | 0.9526 (true last step) |
| real generated trajectory, cut after bad step | 0.0861 | 0.0593 (the bad step itself) |

**The bogus score tracks trajectory quality, not the last step.**
Cut right after a bad step, the bogus score sits right next to
that bad step's own score (0.0115 vs 0.0103; 0.0861 vs 0.0593) —
no masking of a just-failed branch. The divergence shows up only
on the *full* trajectories, where the bogus score (0.30) falls
between an early bad step (0.01) and a later recovered step
(0.51) — i.e. it reads like a holistic trajectory-level
P(correct), not a per-step judgment. So the bug substituted
**trajectory-level value for last-step value** on every
non-terminal candidate: systematic, but correlated with quality
rather than arbitrary — it never made a bad branch look healthy
at exactly the point it went bad.

## Implication

Every non-terminal MCTS candidate scored with `agg_strategy=
"last"` before this fix received a holistic-trajectory score
instead of a true last-step score. The direction of the error is
bounded (never masks a just-failed branch), but the magnitude is
real (~0.02 on a healthy trajectory, ~0.21 on a bad one in the
cases measured) and it applies to **every internal search node**,
not just occasional candidates — a much larger blast radius than
a single edge case. Whether this measurably changed downstream
search/accuracy numbers is open; no before/after accuracy
comparison has been run. If ds_alpha or model-family comparisons
that used `agg="last"` come under question, this is the mechanism
to check first.

Related: `unittests/examine_prm_scores_qwenprm_v1.ipynb`
(Examples 1–2 batching/preamble sanity checks are unaffected by
this bug — they use complete, terminal trajectories).
