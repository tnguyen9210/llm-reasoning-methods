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

Verified live against both loaded PRMs
(`unittests/examine_prm_scores_qwenprm_v1.ipynb` and
`examine_prm_scores_rlhflowprm_v1.ipynb`, Examples 5–7 in each),
using `unittest.mock.patch.object(PRM, "_split_steps", ...)` to
reproduce the pre-fix behavior for a single `prm.score()` call
without touching any source file (auto-restores on block exit,
even on error).

Four non-terminal candidates per PRM, each scored pre-fix: the
full trajectory, and the same trajectory **cut right after its
bad step** (the search-critical case — a candidate whose last
completed step is bad should be valued by that step's low score
under `agg="last"`).

| | | bogus score | true/bad step it replaced |
|---|---|---|---|
| QwenPRM | wrong algebra, full | 0.3000 | 0.5142 (recovered last step) |
| QwenPRM | wrong algebra, cut after bad step | 0.0115 | 0.0103 (the bad step itself) |
| QwenPRM | real generated, full | 0.9731 | 0.9526 (true last step) |
| QwenPRM | real generated, cut after bad step | 0.0861 | 0.0593 (the bad step itself) |
| RLHFlowPRM | wrong algebra, full | 0.9668 | 0.8579 (recovered last step) |
| RLHFlowPRM | wrong algebra, cut after bad step | **0.8130** | **0.2394** (the bad step itself) |
| RLHFlowPRM | real generated, full | 0.8398 | 0.9619 (true last step) |
| RLHFlowPRM | real generated, cut after bad step | 0.5391 | 0.4961 (the bad step itself) |

**The bogus score behaves like a holistic trajectory-level
P(correct), substituted for the true per-step value — but whether
that masks a bad branch is PRM-specific.** In both models, on the
full trajectories the bogus score sits *between* an early bad
step and a later recovered step (Qwen: 0.30 between 0.01 and
0.51; RLHFlow: 0.97 between 0.86 recovered and higher, 0.84
between 0.50 and 0.96) — consistent with a holistic judgment
rather than a per-step one.

Where the two PRMs diverge is the search-critical cut-after-bad-
step case:

- **QwenPRM tracks the bad step tightly** (0.0115 vs 0.0103,
  0.0861 vs 0.0593) — no masking; a bad branch scored just after
  its bad step still reads as bad.
- **RLHFlowPRM masks it.** Cut right after the wrong division, the
  bogus score is **0.8130** against the bad step's own **0.2394**
  — a bad branch scored *healthy* at exactly the point search
  should have pruned it. The real-trajectory case shows the same
  direction, more mildly (0.5391 vs 0.4961).

So the original claim that the bug "never made a bad branch look
healthy" does **not** generalize across PRMs — it held for Qwen,
but RLHFlowPRM's pre-fix bogus score could actively hide a
just-failed non-terminal candidate from being pruned.

## Implication

Every non-terminal MCTS candidate scored with `agg_strategy=
"last"` before this fix received a holistic-trajectory score
instead of a true last-step score, on **both** PRM families in
the codebase (`QwenPRM`, `RLHFlowPRM`) — this was never a
single-model edge case, since `_split_steps` is shared on the
`PRM` base class and both subclasses called the unguarded split.
The severity differs by PRM: QwenPRM's distortion is bounded
(never masks a just-failed branch, ~0.02–0.21 magnitude in the
cases measured); RLHFlowPRM's is not bounded the same way — it
can score a just-failed branch as healthy (0.81 vs the bad step's
0.24), which is a real risk to search quality wherever
RLHFlowPRM + `agg="last"` were used together on non-terminal
candidates. Whether this measurably changed downstream
search/accuracy numbers is open; no before/after accuracy
comparison has been run. If ds_alpha or model-family comparisons
that used `agg="last"` come under question — especially ones
using RLHFlowPRM — this is the mechanism to check first.

Bonus, unrelated to this bug: on the same real trajectory,
RLHFlowPRM scores the actually-wrong Step 1 at 0.50 (near-maximum
uncertainty) where QwenPRM scores it at 0.06 (confident flag) —
worth knowing if PRM quality is ever compared across families,
independent of the splitting bug.

Related: `unittests/examine_prm_scores_qwenprm_v1.ipynb`,
`examine_prm_scores_rlhflowprm_v1.ipynb` (Examples 1–2
batching/preamble sanity checks are unaffected by this bug in
either notebook — they use complete, terminal trajectories).
