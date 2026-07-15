# Consolidated summary: rep_exp review + embeds_center_mode + cov_dtype (2026-07-14)

> Single-file index of every question asked, finding verified, and
> implementation detail from the 2026-07-14 session covering (A) the
> `verl-recipe/rep_exp` review, (B) building `embeds_center_mode`, and
> (C) building `cov_dtype`. This is a **cross-reference**, not a new
> source of truth:
> - Prose writeups: [rep-exp-elliptical-bonus-review.md](rep-exp-elliptical-bonus-review.md)
>   (thread A), [embeds-centering-design.md](embeds-centering-design.md)
>   (thread B, local-mean section), [covariance-precision.md](covariance-precision.md)
>   (thread C).
> - Verbatim Q&A backup: [rep-exp-elliptical-bonus-review-transcript.md](rep-exp-elliptical-bonus-review-transcript.md)
>   (threads A/B only — thread C not yet appended there).
> - Chronological decisions: [decisions-log.md](../decisions-log.md),
>   2026-07-14 entries.
> Where this doc and those disagree, the source doc wins — this is a
> map, not an independent record.

---

## Thread A — `rep_exp` repo review

### Questions asked, in order

1. Thorough review request: trajectory generation, LLMs used, system
   prompts, chat templates, prompt/answer formatting, scoring/eval
   metrics, implementation details relevant to our pipeline.
2. Document the discussion into `docs/decisions/`.
3. Verify all "Implementation details relevant to your pipeline"
   points actually made it into the new doc.
4. Make a verbatim backup copy of the discussion.
5. Explain local-centering in simpler language.
6. Clarify: are the "eight rollouts" in the sibling group from the
   *same* prompt (matching our `bs=4` same-prompt candidates), or
   different prompts?
7. Add that clarification discussion to the transcript.
8. Make embedding-centering configurable, implementing local
   centering as a strategy (→ becomes thread B).
9. Double-check the exact step order: pool → [subtract sibling
   mean] → project → fold into V → bonus — is this the real order,
   and does it match how we should implement local centering?
10. Add that step-order discussion to the transcript.
11. Explain, in simpler language: (a) leverage vs. leave-one-out,
    (b) the `persist_covariance=True` "stale means" code comment.
12. Add that discussion to the transcript.
13. Is our own sem-mcts already a form of leverage-based exploration
    (worked A/B/C root-node example)?
14. Do they use different implementations for test-time vs.
    post-training? Clarify the full per-training-step sequence (one
    prompt vs. batch of prompts; how 8 candidates are generated; how
    `cov_inv` is initialized, folded, reset/carried; single-batch vs.
    sequential folding; motivation).
15. Add both (13) and (14) to the transcript.
16. *(this request)* Consolidate all questions/findings/details from
    every conversation so far into a separate file.

### Findings verified against source (verl pinned `b9bd00ef`)

| # | Claim checked | Verdict | Where |
|---|---|---|---|
| 1 | Sibling group = 8 rollouts of the **same prompt** (`uid`-keyed), same structural shape as our `bs=4` same-prompt candidates | **Confirmed** | `rep_exp_trainer.py` main step: `uid` assigned once per prompt, then `gen_batch.repeat(repeat_times=n, interleave=True)` |
| 2 | Step order is pool → center → project → fold → bonus | **Wrong as stated.** Real order: pool → **project** (line 316) → float64 upgrade (319) → **center** (361) → SM-fold (368-371) → bonus (376). Centering is *after* projection, not before | `elliptical_reward_model_worker.py::compute_rm_score` |
| 3 | Our `_extract_embeds`'s documented order ("pool → project → center → normalize") already matches their real order | **Confirmed** — no reordering needed in `_maybe_center_local`'s placement | same file, cross-checked against `mcts_sem_search_v02_00_00.py` |
| 4 | The 8 sibling vectors are folded into `cov_inv` "in a single batch" | **Wrong as stated.** It's a **sequential Python for-loop** over exactly 8 items (`for hidden_state in final_mean_hidden_states: cov_inv = cov_inv - ...`), not one batched linear-algebra op | `elliptical_reward_model_worker.py`, SM-fold loop |
| 5 | `reward_type=leverage` (their actual paper setting) computes the bonus using the fully-folded, self-inclusive `cov_inv` | **Confirmed** — line 376 calls `_compute_bonuses(...)` with `cov_inv` already folded with all 8 group vectors (loop at 368-371 runs first) | same file |
| 6 | `leave_one_out` is the yaml default but not what the paper's script runs | **Confirmed** | same file, other branch of `_compute_bonuses` |
| 7 | `persist_covariance=True` branch deliberately skips centering, with an explicit code comment warning about stale means | **Confirmed** — line 351: `# NOTE: we don't center here since otherwise the covariance will accumulate stale means`. Independent code-level confirmation of the same caveat already reasoned about for our own accumulated-`V` design | same file |
| 8 | Test-time and post-training use **different implementations** | **Wrong premise.** It's **one implementation**, gated by a batch-key check: `EllipticalRewardManager.__call__` — `if "rm_scores" not in data.batch: return super().__call__(...)` (bonus bypassed). Training and validation both build the reward manager identically via `load_reward_manager(...)` | `elliptical_reward_manager.py`; `main_rep_exp.py` lines 320-364 |
| 9 | `_validate()` never touches the elliptical bonus | **Confirmed** — only calls `self.val_reward_fn`, never `self.rm_wg` | `rep_exp_trainer.py::_validate()`, lines 88-227 |
| 10 | Our sem-mcts is already a form of leverage-based exploration (root A/B/C example) | **Confirmed** — `select_child`'s covariance update is unconditional and runs immediately after selection, before returning; `V_inv` is shared/accumulated across the whole tree, so a later sibling's diversity score at the same node is computed against a `V_inv` that already contains an earlier-selected sibling's embedding — structurally identical to their leverage example | `mcts_sem_search_v02_00_00.py::select_child` |
| 11 | `persist_covariance=False` (their real, paper setting): `cov_inv` is per-prompt, per-training-step, freshly reinitialized to `I/lamb`, never carried across prompts/steps/batches | **Confirmed** — zero cross-time contamination risk under the setting they actually ran | `elliptical_reward_model_worker.py::__init__` + `compute_rm_score` |

### Corrections made to the curated review doc

- Added a mapping-table row for per-dataset answer-instruction
  differences (GSM8K native `####` vs. our fixed `\boxed{}` prompt via
  `core/reward_models.py::QWEN_SYSTEM_PROMPT`).
- Sharpened the `lamb=0.01` / `beta=0.01` row: `lamb` is a genuine
  external anchor (independent group, same ridge scale); `beta` is
  explicitly **not** transferable as a number (different bonus
  formula, different units).
- Updated the centering mapping-table row to same-group-*structure*
  parity: same-prompt siblings on both sides, but theirs recomputes
  the mean fresh every training step and never carries it forward,
  vs. ours (pre-2026-07-14) centering globally/online across steps.
- Marked follow-up #3 (local centering) as **BUILT 2026-07-14**.

### Follow-up candidates (still open, none but #3 executed)

1. Unblock `avg`+`response` pooling via v01 policy embeds (direct
   evidence this cell is worth running before v02 PRM support).
2. Projection-dim ablation at 32/128 vs. our 512.
3. Local (sibling-group) centering — **built**, see thread B.
4. Signal-gated diversity (suppress `q_diversity` when all PRM scores
   are uniformly low — analog of their none-correct gating).
5. float64 for the `V^-1` math (cheap insurance against SM drift).
6. (Deferred) last-300-chars extraction clip for GSM8K-style regexing.

Deliberately not adopted: their unbiased pass@k estimator (would
change the meaning of every recorded `pass@gb` cell); their
redrawn-per-batch sparse projection (our fixed-matrix reproducibility
reasoning stands, though it's now known to be a choice, not a
necessity).

---

## Thread B — `embeds_center_mode` implementation

### Questions asked, in order

1. Make embedding-centering configurable; implement local centering
   as a strategy.
2. *(clarify)* Can we reuse the existing `embeds_center` options
   instead of adding a new knob?
3. *(decision)* "please go with C" — new `embeds_center_mode` flag
   (vs. B: sentinel value reuse in `embeds_mean_dir`).
4. *(scoping, via AskUserQuestion)* "Let's focus on v02 first. Don't
   do anything the other two [v01, bl_sem cores]."
5. *(scoping, via AskUserQuestion)* "Exclude-if-default. Could you
   check whether we already have options for embeddings strategies?"

### Design decisions

- **Scope: v02 core only.** `v01`/`bl_sem` cores deliberately ignore
  the flag; field comment says so explicitly.
- **Mechanism: new flag, not sentinel reuse.** Option B (sentinel in
  `embeds_mean_dir`) had zero hash risk but hid a stringly-typed mode
  inside a path field. Option C (new `embeds_center_mode` field) was
  chosen for clarity, at the cost of needing a new hash-safety
  mechanism.
- **Hash safety: `_HASH_EXCLUDE_IF_DEFAULT`.** New reusable mechanism
  in `utils/configs.py` — a field is dropped from the config identity
  dict iff it equals a pinned neutral value (`"fixed"`, frozen
  forever, independent of the dataclass default). Lets a field be
  added to a hashed group without changing any existing hash.

### Implementation (files touched)

- `utils/configs.py` — added `embeds_center_mode: str = "fixed"` to
  `MCTSSemV01Config` only (inherited by `MCTSSemV02Config`, not
  redefined there — corrected from an earlier draft of this doc that
  said "both"); added `_HASH_EXCLUDE_IF_DEFAULT = {"search":
  {"embeds_center_mode": "fixed"}}`; updated `config_identity()`'s
  dict comprehension to also drop keys matching the neutral default.
- `core/mcts_sem_search_v02_00_00.py`:
  - Module docstring: documented the new flag and updated the
    `embeds_mean` line to note it's required only for `mode="fixed"`.
  - `_extract_embeds`: added mode validation (raises on unknown
    mode) and a `defer_local` flag that skips center+normalize when
    `mode="local"` (deferred to group level, since the group mean
    needs the whole sibling batch, not just one vector).
  - New function `_maybe_center_local(embeds, sc)`: subtracts the
    group's own mean, then L2-normalizes — preserving the pipeline
    invariant that centering happens in linear space before the
    non-linear normalize step. `batch_size=1` edge case: centered
    vector is exactly zero → zero bonus, SM-fold of a zero vector is
    a no-op.
  - Wired into both return points of `_embed_candidates` (policy
    branch and prm branch).
- `generate_mcts_sem.py` — mean-`.npy` load now gated on
  `cfg.search.embeds_center and cfg.search.embeds_center_mode ==
  "fixed"` (local mode needs no precomputed file).
- `conf/search/mcts_sem_v02.yaml` — added `embeds_center_mode: fixed`
  with inline comment citing the rep_exp review doc.

### Verification performed

- `python status.py --check ... ` at 3 points:
  - baseline (no `embeds_center_mode` override) → hash `c371341f`
  - `+embeds_center_mode=local` → hash `1aa258a0` (distinct, as
    expected)
  - explicit `embeds_center_mode=fixed` override → hash `c371341f`
    (unchanged from baseline, confirms exclude-if-default works)
- `python status.py --group sem-mcts` → full ledger (111 done, 6
  planned, 1 partial), **0 orphaned/not-found entries** after the
  config change.

### Coherence caveat (documented, not resolved)

rep_exp pairs local centering with a **fresh-per-group** covariance
(`persist_covariance=False`); our `V` **accumulates** across the
whole search. Local centering + accumulated `V` means each group's
vectors enter `V` with a different affine offset each time —
structurally the same incoherence concern already raised for the
(unbuilt) online-centering mode. This makes `embeds_center_mode=
"local"` an **ablation arm, not a clean transplant** of rep_exp's
method. The faithful transplant (fresh `V` per expansion) would be a
separate, larger change.

### Status: crashed in production, reimplemented, now verified (2026-07-15)

The first implementation (above) was launched live via the idle-GPU
orchestrator and **crashed instantly on every job**:
`ValueError: "MCTS" object has no field "cov_dtype"` (a sibling
feature built the same day — see Thread C below — but the same root
cause applied to both, since neither had been smoke-tested before
launch). Root cause: `MCTS` is a **pydantic `BaseModel`**, which
raises on `self.attr = value` for any undeclared field.
`embeds_center_mode` itself wasn't the crashing field, but was
reverted alongside `cov_dtype` (`git restore`) as part of the same
incident, then reimplemented identically 2026-07-15 with one
opportunistic fix: a shared `_is_local_center(sc)` predicate so
`_extract_embeds`'s defer decision and `_maybe_center_local`'s gate
can't silently disagree (flagged in an earlier code review, fixed
during the redo). **Verified 2026-07-15 with a live end-to-end smoke
test** this time (`WANDB_MODE=offline`, `1q/1trial`,
`results_subdir=smoketest`, `embeds_center=true
embeds_center_mode=local`): ran the complete pipeline with no crash,
scored dataset written (`cfg-59968b28`, 63.8s total). Not committed
or pushed. Full incident writeup:
[decisions-log.md #2026-07-15](../decisions-log.md#2026-07-15--search-configs-cov_dtype--embeds_center_mode--reimplemented-after-a-pydantic-field-declaration-bug).

---

## Thread C — `cov_dtype` (fp32 vs. fp64 covariance precision)

### Questions asked, in order

1. Research the benefits of float64 vs. float16/float32 for
   covariance-matrix computations — does fp64 actually help?
2. Would it be reasonable to make covariance precision configurable
   (keep current behavior as default, add an fp64 option)?
3. "Could you please do (b)" — the "add a real ablation flag" option
   (vs. an un-gated silent upgrade).

### Key finding that reframed the task

Before any code was touched, tracing the actual dtype flow revealed
the premise needed correcting: **`V`/`V_inv` were already float64**
by NumPy default (`np.eye`/`np.linalg.solve` with no `dtype=` upcast
to float64 automatically) — only the embeddings (`u`, `q_embeds`,
sourced from `_extract_embeds`'s `.float()` cast to float32) were
float32, silently promoted to float64 at every combined op (`V_inv @
u`, `einsum`). So there was no live fp32 covariance path to compare
against — it was implicit fp64, undocumented. Confirmed with Tuan
(via AskUserQuestion) before implementing: make it **explicit**
(`cov_dtype: "fp32"|"fp64"`, default `"fp64"` preserving current
behavior, with `"fp32"` now casting embeds + seed matrices uniformly)
rather than the narrower alternative (only control the embeds-upcast
step, leaving `V`/`V_inv` fixed at float64 always).

### Implementation (files touched)

- `utils/configs.py` — added `cov_dtype: str = "fp64"` to
  `MCTSSemV01Config` (inherited by v02, same pattern as
  `embeds_center_mode`); added `"cov_dtype": "fp64"` to
  `_HASH_EXCLUDE_IF_DEFAULT["search"]`.
- `core/mcts_sem_search_v02_00_00.py`:
  - New module-level `_COV_DTYPES = {"fp32": np.float32, "fp64":
    np.float64}` lookup.
  - `MCTS.__init__`: validates `config.search.cov_dtype` (raises on
    unknown value), stores `self.cov_dtype`, passes it as `dtype=` to
    all three `np.eye(...)` seed calls for `V`/`V_inv`.
  - `select_child`: casts `u` (the selected child's embedding) to
    `self.cov_dtype` before any covariance op; the exact-update
    branch's `np.eye(...)` identity also takes `dtype=self.cov_dtype`.
  - `_diverse_select`: new `cov_dtype` kwarg (default `np.float64`
    for standalone callers), casts `q_embeds` before the einsum.
  - `_select_by_diversity`: passes `cov_dtype=self.cov_dtype` through
    to `_diverse_select`.
  - Module docstring: added the flag to the config-flags list.
- `conf/search/mcts_sem_v02.yaml` — added `cov_dtype: fp64` with
  inline comment.

### Verification performed

- `python status.py --check ...` at 3 points:
  - baseline (no override) → hash `c371341f` (unchanged from before
    this flag existed)
  - `+cov_dtype=fp32` → hash `573c095f` (distinct, as expected)
  - explicit `cov_dtype=fp64` override → hash `c371341f` (matches
    baseline, confirms exclude-if-default works)
- `python status.py --group sem-mcts` → full ledger, 0 orphans or
  mismatches.
- `_COV_DTYPES` sanity-checked directly: `"fp32" -> numpy.float32`,
  `"fp64" -> numpy.float64`, unknown keys correctly absent (guarded
  by the `ValueError` raise in `__init__`).

### Scope note (corrects an assumption from Thread B's writeup)

`cov_dtype` is defined once on `MCTSSemV01Config` and inherited by
v02 — **not** duplicated on both classes. This corrects an inaccurate
claim in Thread B's original summary (now fixed above) that
`embeds_center_mode` was added "to both" classes; checking the actual
file showed it too is defined once on `MCTSSemV01Config` only.
**v02 core only** for the actual wiring: v01's own `MCTS.__init__`
and `_diverse_select` are separate code (different `V`/`V_inv`
handling entirely) and do not read `cov_dtype` — the field is
inherited but inert there, matching the same v02-only scope Tuan set
for the centering work.

### Status: crashed in production, reimplemented, now verified (2026-07-15)

Hash/wiring verification (above) passed, but the code was never
smoke-tested before being launched live via the idle-GPU
orchestrator — every launched job **crashed instantly**:
`ValueError: "MCTS" object has no field "cov_dtype"`. Root cause:
`MCTS` (`core/mcts_sem_search_v02_00_00.py`) is a **pydantic
`BaseModel`**; `__init__` only assigned `self.cov_dtype = ...`
without declaring it as a class-level field first (unlike `V`/`V_inv`,
which are declared) — pydantic rejects `self.attr = value` for any
undeclared attribute. Fixed by `git restore`-reverting both this and
`embeds_center_mode` entirely, then reimplementing with
`cov_dtype: Any = np.float64` declared as a proper field on `MCTS`,
alongside `V`/`V_inv`/`completed_nodes`. **Verified 2026-07-15 with a
live end-to-end smoke test** (`WANDB_MODE=offline`, `1q/1trial`,
`results_subdir=smoketest`, `cov_dtype=fp32` explicit): ran the
complete pipeline — model load, full `num_phases=1000` search,
scoring, scored dataset written — with no crash. Hash values
unchanged from the first pass (`c371341f` baseline stable). Still
open: no live A/B comparing `fp32` vs. `fp64` selected-child
*sequences* on an actual run — that remains the natural next step to
empirically test whether precision is actually mattering in practice.
Full writeup: [covariance-precision.md](covariance-precision.md).

---

## Cross-thread pattern worth naming

Every finding in Thread A that got corrected (step order, batch vs.
sequential folding, "different implementations" framing) was caught
by Tuan asking for source-level re-verification rather than accepting
a prior summary — each time, the fix came from re-reading the pinned
verl commit's actual code, not from patching the wording. This
consolidated doc's "verdict" column exists specifically so future
sessions don't have to re-derive these corrections from scratch.
