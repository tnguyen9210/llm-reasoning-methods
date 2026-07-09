# bl_cnt_v01's best-first frontier search leaves ~18% of questions with zero completions at gen_budget=80

*2026-07-08*

**Observation:** a 2-question smoke test of
`mcts_bl_cnt_search_v01_00_00` (llama-1b, rlhflow PRM,
`gen_budget=80`, `max_depth=20`) produced
`completions: []` for **both** questions, despite spending the full
generation budget on each (`q_total_gens=[80, 80]`). This initially
looked like a regression from an unrelated infra-alignment change
(see `docs/decisions-log.md` 2026-07-08 entry porting `build_prm`/
`quantization`/`prm_batch_size`/timing-state fixes into this file) —
it is not; it is a real, pre-existing property of the algorithm,
confirmed to already affect the existing 128-question recorded run.

## What's actually happening

Traced live via a monkey-patched `create_child` (logs every
`stop_reason` seen): across 95 expansions in a debug run (llama-1b,
`gen_budget=25`), **every single one** stopped at the `"\n\n"` step
separator — none via `EOS`/`length`, so `create_child`'s terminal
check (`stop_reason in ("EOS", "length")`) never fired and nothing
was added to `completed_nodes`. This is *expected* mid-solution
behavior (each step boundary is supposed to stop at `"\n\n"`); the
question is why the tree never reached an actual end-of-answer.

Depth trace for that debug run: `[0,0,0,0, 1,1,1,1, 2,2,2,2, ...,
7,7,7,7]` — roughly linear, ~3.5 generations consumed per depth
level. At that rate, reaching `max_depth=20` needs ~70 generations,
leaving essentially no margin within `gen_budget=80` for a solution
that needs many steps, or for the search to backtrack and revisit
a shallower node with better PUCT. The two smoke-tested questions
apparently needed more depth than the budget allowed.

## Confirmed via the existing 128-question recorded run

`results/prm800k/mcts_bl_cnt_v01--level-4/mcts_bl_cnt_v01--level-4--
Llama3.2-1B--tmpl-custom--bs-4--d-20--b-080--cpuct-2.0/generate_...
--trial-000.jsonl`:

| | `mcts_bl_cnt_v01` (frontier, best-first) | `mcts_cnt_v01` (phase-based, root-to-leaf) |
|---|---|---|
| questions | 128 | 128 |
| zero-completion questions | **23 (18%)** | 4 (3%) |
| `gen_budget` | 80 | 80 |

Both use the identical `gen_budget=80`, `max_depth=20`. So the
2-question smoke test wasn't a fluke *result* (0/2 zero-completion
questions is well within the expected range for a small sample of a
population where ~18% of questions hit this), but the underlying
mechanism is real and disproportionate compared to `mcts_cnt_v01`.

## Why the two algorithms differ here

`mcts_cnt_v01` walks root-to-leaf every phase and always generates
one more step from wherever the walk currently is — every phase
makes forward progress toward a terminal state (or gets forced
terminal at `max_depth`), so it can't "waste" a phase on a branch
that doesn't advance depth.

`mcts_bl_cnt_v01` selects globally by PUCT across the entire
`leaf_nodes` frontier each phase. A freshly-expanded batch of 4
children (`batch_size=4`) all start at `visit_count=1` with no
exploration bonus (`u=0` when `parent_visits==0` or `visits==0`,
`MCTSNode.puct`), competing directly against deeper, more-visited
nodes for the next selection. On a hard problem, this lets the
search spend generations on shallow-but-locally-attractive branches
instead of monotonically deepening the one branch closest to a real
answer — bounded by `gen_budget`, this can exhaust the budget with
no branch ever reaching depth 20 or an early EOS/length stop.

## Is this a bug?

No — it's an inherent tradeoff of the frontier-based algorithm as
designed (the module's own docstring already frames this as the
"Key difference from mcts_cnt_search_v01_00_00"), not a defect
introduced by any recent change. Nothing here needs fixing; this
finding exists to explain why a small (1-2 question) smoke test of
this specific algorithm can legitimately show 0/N completions
without indicating a regression, and to give the ~18% zero-completion
rate a documented baseline for future comparison.

## Revisit if

- The zero-completion rate changes meaningfully after any future
  edit to `mcts_bl_cnt_search_v01_00_00`'s selection or expansion
  logic — 18% (at `gen_budget=80`, llama-1b) is the baseline to
  compare against.
- `gen_budget` is ever raised for this method specifically — worth
  checking whether the zero-completion rate drops as budget grows
  (would confirm it's purely a budget-exhaustion effect) or plateaus
  (would suggest some questions get structurally stuck regardless of
  budget).
