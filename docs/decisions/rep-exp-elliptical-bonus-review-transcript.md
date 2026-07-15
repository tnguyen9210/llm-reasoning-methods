# Raw discussion transcript: rep_exp repo review (2026-07-14)

> Verbatim copy of the conversational review that
> [rep-exp-elliptical-bonus-review.md](rep-exp-elliptical-bonus-review.md)
> was distilled from — kept as a backup per Tuan's request. The
> curated doc supersedes this file where they disagree; this copy
> preserves the full original wording, including phrasing that was
> compressed out of the curated version.

---

# Review: `verl-recipe/rep_exp` — "Representation-Based Exploration for Language Models"

**Bottom line:** this is the official post-training code for Tuyls,
Foster, Krishnamurthy & Ash (arXiv 2510.11686) — the closest
published relative of your sem-mcts diversity bonus. It computes an
**elliptical (leverage-score) exploration bonus over mean-pooled
policy hidden states** and adds it to the correctness reward inside
**GRPO** (verl, pinned commit `b9bd00ef`). The repo covers the
*post-training* half of the paper; the test-time half is not in this
recipe. Several of their design choices land exactly on cells in your
`embeds_strategy × scope` tables — including the one your v02
currently blocks.

## 1. How trajectories are generated

No search, no tree — plain **GRPO rollouts via vLLM**
(`rollout.name=vllm`, sync mode): `n=8` samples per prompt at
`temperature=1.0, top_p=1, top_k=-1`, batch of 1024 prompts (512 for
DAPO/AIME). Max lengths: 1024/1024 prompt/response for MATH & GSM8K;
2048/8192 for DAPO-AIME. Rollouts are grouped per-prompt by a `uid`;
the group of 8 is both the GRPO advantage-normalization group and the
covariance group for the bonus. KL coef = 0 and entropy coef = 0 —
**the elliptical bonus is the only exploration mechanism**.

## 2. Which LLMs

One model: **Qwen2.5-7B-Instruct**, in two roles — the policy, and
(a frozen copy as the "reward model" worker) the **representation
extractor**. No PRM anywhere; correctness is rule-based.

## 3–5. System prompts and chat templates

- **No custom chat template and no explicit system prompt.** Each
  example is a single user message; verl's `RLHFDataset` calls the
  tokenizer's stock `apply_chat_template(...,
  add_generation_prompt=True)` with no overrides
  (`reward_model.model.input_tokenizer=null`, so the extractor
  consumes the actor's exact tokens — template-switching code exists
  but is disabled).
- Since only Qwen2.5-7B-Instruct is used, there are **no cross-model
  template differences** — nothing analogous to your
  native-vs-custom `tmpl` axis. The only effective system prompt is
  Qwen's own template default ("You are Qwen, created by Alibaba
  Cloud...") that the tokenizer inserts when no system message is
  present.

## 6. Prompt / reasoning / answer formatting

Instruction is appended to the question inside the user turn, and —
notably — it is **dataset-specific**:

| dataset | instruction appended | ground truth |
|---|---|---|
| GSM8K | `Let's think step by step and output the final answer after "####".` | int extracted from `#### N` |
| MATH (lighteval) | `Let's think step by step and output the final answer within \boxed{}.` | `remove_boxed(last_boxed_only_string(solution))` |
| DAPO + AIME24 | same `\boxed{}` instruction | bare answer string |

Reasoning steps are free-form (no step delimiters enforced, nothing
like your step-level PRM segmentation). Splits: MATH dev = the
MATH-500 indices (500 problems), test = the remaining ~4500; GSM8K
dev = random 512 of test; AIME24 dev = `MathArena/aime_2024_I` +
`_II` (30 problems — note: a different HF source from your
`Maxwell-Jia/AIME_2024`), train = 4096 problems sampled from a
deduplicated DAPO set.

## 7–8. Scoring and evaluation metrics

**Correctness (extrinsic reward)** dispatches on `data_source` — the
exact analog of your `grader_name`:
- **GSM8K:** strict regex `#### (\-?[0-9\.\,]+)` on the **last 300
  chars only**, take last match, strip commas, exact string match →
  1.0 / 0.
- **MATH:** last `\boxed{}` extraction + Hendrycks-style
  `strip_string` normalization + string equality → 1.0 / 0.
- **DAPO/AIME:** HuggingFace **math-verify** — ground truth wrapped
  as `\boxed{gt}`, symbolic parse+verify, exceptions → 0.

**Validation metrics** (`metric_utils.py`): sample **n = 128
(train-time val) / 256 (final eval)** responses per question at
temperature 1.0 (not greedy), then:
- **pass@k** for k = 1, 2, 4, ..., n via the unbiased combinatorial
  estimator `1 − C(n−c, k)/C(n, k)` (computed as a stable product) —
  the Codex/HumanEval estimator, *not* empirical best-of-k;
- **maj@k** via bootstrap resampling of size-k subsets with majority
  vote over extracted predictions;
- checkpoint selection = **best pass@1 on dev**. During validation
  the elliptical bonus is bypassed entirely (manager falls back to
  the plain correctness reward).

## The elliptical bonus itself (the part that matters for you)

Pipeline per training batch, per prompt-group of 8
(`elliptical_reward_model_worker.py`):

1. **Representation:** frozen backbone forward (`AutoModel`, bf16,
   no LM head) → `last_hidden_state`; **mean over response tokens
   only** (prompt excluded). In your vocabulary:
   `embeds_strategy=avg`, `embeds_scope=response` — **exactly the
   cell your v02 blocks** for PRM-source embeds.
2. **Projection:** sklearn `SparseRandomProjection` to
   **`sparse_dim` = 32 (MATH/GSM8K) or 128 (AIME)** — far more
   aggressive than your `sparse512`. With
   `randomize_sparse_matrix=True` (their paper setting) the matrix
   is **redrawn every batch**.
3. **Precision:** projected features are upgraded to **float64** for
   all covariance math.
4. **Covariance:** with `persist_covariance=False` (paper setting)
   the covariance is **per-group and per-step**: center the 8
   features on the group mean, init `V⁻¹ = I/λ` with **`lamb=0.01`**,
   then fold all 8 in via **Sherman–Morrison rank-1 updates** — same
   update as your `cov_update=sm`.
5. **Bonus:** `reward_type=leverage` → bonus_i = `h̃ᵢᵀ V_full⁻¹ h̃ᵢ`
   (self included — a leverage score). A `leave_one_out` variant
   exists (SM *downdate* with the `1 − xᵀV⁻¹x` denominator, then
   quadratic form) — it's the yaml default but **not** what the
   paper's script runs. A `persist_covariance=True` variant maintains
   per-prompt running `V⁻¹` and mean across steps.
6. **Placement:** the bonus is written on the **last response token**
   of the token-level reward tensor — trajectory-level, not per-step.
7. **Combination:** `total = α·correctness + β·bonus` with **α=1.0,
   β=0.01**, no normalization (`normalization=none`; rnd/z-score
   options exist).
8. **Gating** — the subtle one:
   `turn_off_elliptical_if_none_correct=True` — if **zero of the 8
   rollouts are correct, the bonus is zeroed for the whole group**.
   Under GRPO this prevents the bonus from becoming the *only*
   advantage signal and reinforcing diverse-but-wrong trajectories.
   Flags for some-correct/all-correct/incorrect-rollout gating exist,
   all off.

## Implementation details relevant to your pipeline

1. **The blocked cell has published support.** Their entire method
   runs on `avg` pooling over **response-only** scope — the
   combination your `embeds_strategy × scope` tables mark as blocked
   (v02, PRM-source). Their extractor is the *policy* backbone
   though, i.e. your v01 policy-embeds path, which does support
   response scope. That's a concrete argument for prioritizing the
   v01 version of that sweep table.
2. **Projection dimension:** 32–128 vs your 512, and redrawn per
   batch. If their results hold at d=32, your 512 may be paying
   covariance-conditioning and compute costs for nothing — a cheap
   ablation (`embeds_dim=32/128`) on an existing sweep point would
   tell you.
3. **Group-centering before the covariance fold.** They subtract the
   sibling-group mean before updating `V⁻¹`. Your `embeds_mean` knob
   is adjacent, but theirs is *local* centering (per expansion group,
   per step), which makes the bonus measure within-group dispersion
   rather than distance from a global mean. For MCTS expansion
   batches (your `bs=4` candidates) this is directly transplantable.
4. **float64 covariance math.** Cheap insurance at d=32–512; your
   V100S constraint (no bf16) makes fp32→fp64 for just the d×d ops
   very affordable.
5. **Signal-gated diversity.** Their `turn_off_if_none_correct`
   translates naturally to your setting as: suppress the diversity
   bonus when the PRM scores of all candidates are uniformly low (no
   signal worth diversifying around). Could be a one-line experiment
   in sem-mcts selection.
6. **λ=0.01 as their operating point** matches one of your standing
   `lam` sweep values — a nice external anchor, though their β=0.01
   weight lives on a different scale (leverage scores of centered,
   32-dim, float64 features ∈ [0,1]-ish since leverage ≤ 1) than
   your `ds_alpha` bonus.
7. **Grading discipline differs from yours on GSM8K:** they prompt
   for and grade the native `####` format; your pipeline prompts
   GSM8K with the boxed convention and grades via `grader_name=gsm8k`
   on ground truth but boxed extraction on predictions. Their "last
   300 chars only" clip and "take the *last* match" rules are also
   small robustness details your `extract_answer` mostly shares
   (last-match) but the clip may be worth borrowing for very long
   traces.
8. **Their pass@k is the unbiased estimator**, computed from n=256
   samples at temperature 1.0. Your `pass@gb` is an empirical pass
   over the search's gb completions — not directly comparable to
   their numbers; if you ever benchmark against this paper, match
   the estimator.
9. **AIME source difference:** they use `MathArena/aime_2024_I/II`;
   you just built `aime2024` from `Maxwell-Jia/AIME_2024`. Same 30
   problems, but field names and answer formatting may differ —
   worth remembering if you ever cross-check numbers.

---

# Follow-up: verifying the exact pool -> center -> project -> bonus order (2026-07-14)

> Second verbatim exchange, appended the same day after Tuan asked to
> double-check a specific claim from the discussion above against the
> actual source rather than trusting the earlier summary.

**Tuan's question:** "Could you double-check whether the
Representation-Based Exploration repository implements the following
sequence? '1. pool the hidden states -> one vector per candidate
2. [NEW] subtract the mean of this candidate's own sibling group
3. project (sparse random projection) 4. fold into V via
Sherman-Morrison (same update rule you already use) 5. bonus =
x^T V^-1 x (theirs) vs sqrt(x^T V^-1 x) (yours)' Please verify the
exact order from the relevant code and explain whether this is the
correct way to implement local centering in our repository. Also
note any important differences between their pipeline and ours."

**Response (re-read `workers/elliptical_reward_model_worker.py` line
by line, `compute_hidden_states` -> `compute_rm_score`):**

## The actual order, verified line-by-line

1. **Pool** (`compute_hidden_states`, `_forward_micro_batch`): mean
   over response tokens -> `mean_hidden_states`, one vector per
   candidate. Matches step 1.
2. **Sparse project** (`compute_rm_score` line 316):
   `mean_hidden_states = mean_hidden_states @ sparse_matrix` -- happens
   to the **whole batch at once**, on the *raw, un-centered* pooled
   vectors.
3. **Upgrade to float64** (line 319) -- right after projection, before
   centering.
4. **Group-center** (line 361, non-persist branch):
   `centered_mean_hidden_states = filtered_mean_hidden_states -
   filtered_mean_hidden_states.mean(dim=0)` -- runs on the
   **already-projected, already-float64** vectors, filtered down to
   one `uid` group at a time.
5. **Sherman-Morrison fold** (line 368-371): folds the *centered*
   vectors into `cov_inv`.
6. **Bonus** (`_compute_bonuses`, line 279): `x^T V^-1 x` on the same
   centered vectors.

**So the real order is: pool -> project -> center -> fold -> bonus.**
The earlier summary listed "subtract mean" as step 2 (before
projection) -- **that was wrong.** Centering happens **after**
projection, not before.

## Why this matters for the implementation

This is exactly our own pipeline's documented order already --
`_extract_embeds`'s five-step comment says explicitly: *"pool ->
project -> center -> normalize,"* with the center step's own comment
stating *"the mean must be subtracted in the linear space"* (i.e.
after any linear transform like projection, since projection is
itself linear and commutes cleanly with translation only if you
center first in the *same* space you're comparing in -- but
critically, centering has to happen in whichever space `V` actually
lives, which is post-projection).

So on this specific point, **our `_maybe_center_local` implementation
is already correct and already matches their real order** -- it
centers after `_extract_embeds` has finished pooling+projecting (it
operates on the *already-projected* embeddings returned by
`_embed_candidates`), which is the true post-projection space,
matching their `compute_rm_score` doing the group-subtraction on the
already-projected `filtered_mean_hidden_states`. No code change
needed on this point; it was the prose description that had the step
order wrong, not the implementation.

## Other real differences confirmed/sharpened by this re-read

1. **Projection is computed once per micro-batch across ALL groups,
   not per-group.** The sparse projection (line 316) runs on the
   *entire* `mean_hidden_states` tensor for the whole training batch
   (potentially many different prompts/uids at once) -- grouping by
   `uid` only happens starting at line 324, *after* projection.
   Centering, in contrast, is done per-`uid`-group (line 331 `mask`,
   line 361). So projection is global-batch, centering is per-group.
   This matches our pipeline's shape too (our sparse projection
   matrix is fixed/global across the whole run; our local centering
   is per-expansion-group).
2. **The bonus formula uses the *fully-folded* `cov_inv` (leverage,
   includes self), not a leave-one-out `cov_inv`.** Line 376 calls
   `_compute_bonuses(final_mean_hidden_states, cov_inv, ...)` with the
   `cov_inv` that has *already* had all of this group's vectors
   folded in (the loop at 368-371 runs first, then line 376). This
   confirms `reward_type=leverage` (their actual paper setting)
   computes the bonus using the same-group-inclusive `V^-1`, not
   excluding the candidate itself -- as opposed to `leave_one_out`,
   which explicitly excludes it via the SM-downdate math in
   `_compute_bonuses`'s other branch.
3. **`persist_covariance=True`'s explicit comment is a direct warning
   our accumulated-`V` setup should heed:** line 351, `# NOTE: we
   don't center here since otherwise the covariance will accumulate
   stale means` -- they *deliberately skip centering* in the
   persist-covariance branch, precisely because their own team
   recognized that centering-then-accumulating-across-time corrupts
   the covariance with stale offsets. This is the exact same caveat
   flagged in `embeds-centering-design.md` ("local centering feeds
   our *accumulated* V ... ablation arm, not clean transplant") -- but
   now it's not just inference, it's their own code comment making
   the identical point about their own persist-covariance path.
   Stronger evidence than the earlier reasoning alone.

## Direct answers

- **Is the 5-step sequence in the question correct?** No -- step 2
  ("subtract mean") and step 3 ("project") are swapped. The real
  order is **pool -> project -> float64 -> center (per-group) -> fold
  via SM -> bonus**.
- **Is `_maybe_center_local`'s implementation correctly ordered
  relative to this corrected sequence?** Yes -- it operates on
  vectors that have already gone through pool+project inside
  `_extract_embeds`, and centers before the SM-fold happens (which
  occurs later, in the tree-selection code, not inside
  `_maybe_center_local` itself). No change needed.
- Offered to add a short "pool -> project -> center -> fold -> bonus"
  note plus the stronger "their own code warns about this" citation
  to the curated review doc and `embeds-centering-design.md` --
  awaiting Tuan's go-ahead before editing either.

---

# Follow-up: plain-language explanation of two dense points (2026-07-14)

> Third verbatim exchange, appended the same day after Tuan asked for
> a simpler-language, more detailed explanation of two specific
> bullets from the follow-up above (the leverage-vs-leave-one-out
> point, and the persist_covariance centering-warning point).

**Tuan's question:** "could you please clarify in more details in
simpler language these two points 'The bonus formula uses the
fully-folded cov_inv (leverage, includes self), not a leave-one-out
cov_inv. Line 376 calls _compute_bonuses(final_mean_hidden_states,
cov_inv, ...) with the cov_inv that has already had all of this
group's vectors folded in (the loop at 368-371 runs first, then line
376). This confirms my earlier note that reward_type=leverage (their
actual paper setting) computes the bonus using the
same-group-inclusive V^-1, not excluding the candidate itself — as
opposed to leave_one_out, which explicitly excludes it via the
SM-downdate math in _compute_bonuses's other branch.

persist_covariance=True's explicit comment is a direct warning that
your accumulated-V setup should heed: line 351, # NOTE: we don't
center here since otherwise the covariance will accumulate stale
means — they deliberately skip centering in the persist-covariance
branch, precisely because their own team recognized that
centering-then-accumulating-across-time corrupts the covariance with
stale offsets. This is the exact same caveat I flagged in your
embeds-centering-design.md doc ("local centering feeds our
accumulated V ... ablation arm, not clean transplant") — but now it's
not just my inference, it's their own code comment making the
identical point about their own persist-covariance path. Worth citing
directly since it's stronger evidence than my own reasoning.'"

**Response:**

Let me break both of these down with a concrete worked example, since
the math-heavy phrasing was hiding a fairly simple idea in each case.

## Point 1: "leverage" vs "leave-one-out" -- does a candidate's own bonus include itself?

**The core question:** when you compute "how novel is this
candidate?", do you compare it against *only the other candidates*,
or against *itself plus the others*? These give different numbers,
and their code supports both -- but only one is what they actually
ran.

**Concrete example.** Say you have 3 sibling candidates in a group,
with embeddings A, B, C. The covariance-building step folds all three
into `cov_inv` one at a time: fold A in, then fold B in, then fold C
in. After this loop finishes, `cov_inv` is a single matrix that
"knows about" all three vectors.

Now -- when you compute candidate A's bonus, which version of
`cov_inv` do you use?

- **`leverage` (what they actually ran):** use the *final* `cov_inv`
  -- the one that already includes A, B, *and* C. So A's own bonus is
  computed using a matrix that A itself helped build. This is like
  asking "how novel does A look, using a notion of 'typical' that A
  itself contributed to defining?"
- **`leave_one_out` (the config default, but NOT what they ran):** for
  A's bonus, first *mathematically remove* A's own contribution from
  `cov_inv` (this is what the "SM-downdate" math does --
  Sherman-Morrison has both an "add a vector" update and its inverse,
  "remove a vector" downdate), then compute A's bonus against that
  A-excluded matrix. This is "how novel does A look, compared only to
  B and C, without A being allowed to inflate its own novelty score."

**Why this matters intuitively:** leave-one-out is the more
principled choice -- it answers "is A different from its *siblings*,"
which is closer to what you actually want to measure. Leverage
(self-included) can be gamed a little: if A is genuinely a huge
outlier, folding A into the matrix itself pulls the matrix's "notion
of variance" toward A's direction, which can actually *shrink* A's own
computed bonus relative to what leave-one-out would give it (the
matrix partially "explains away" A using A). It's a subtle
self-referential wrinkle.

**Why I flagged it:** their published paper results used `leverage`
(self-included) even though their own code's *default* config value
is `leave_one_out`. That's a real mismatch between "what the yaml
says is default" and "what they actually ran to get their numbers" --
worth knowing if you ever try to reproduce their exact setup, since
copying the yaml defaults verbatim would silently give you the other
variant.

**How this relates to your repo:** your own `sqrt(x^T V^-1 x)` bonus
in `_diverse_select` is computed the same way structurally -- you use
whatever `V_inv` is current when you score a candidate, and that
`V_inv` gets updated with the *chosen* child's embedding only *after*
selection, not before. So your bonus for an unvisited candidate is
naturally "leave-it-out" already (nothing has been folded in yet for
that specific pick), which is a different mechanism from either of
theirs, but it's worth being precise that theirs is a
batch-of-siblings comparison while yours is a sequential
tree-selection comparison.

---

## Point 2: their own code has a comment warning about the exact risk your local-centering mode has

**The setup:** their `persist_covariance` flag has two modes:
- `False` (what the paper actually used): every training step, throw
  away the old covariance matrix and start completely fresh, using
  only *this step's* group of 8 embeddings. Since it's rebuilt from
  scratch each time, centering them on their own group-mean first is
  safe -- there's no "old" data lingering to get contaminated.
- `True` (an alternative, unused-in-the-paper mode): *keep* the same
  covariance matrix running across many training steps, continuously
  accumulating more and more embeddings into it over time -- much
  closer to what your `V` does across an entire MCTS search.

**The comment they wrote, in the `True` branch:** `# NOTE: we don't
center here since otherwise the covariance will accumulate stale
means`

**In plain language, what this comment is saying:** "We deliberately
do NOT subtract a group mean in this persistent-covariance mode,
because if we did, we'd be mixing centered vectors from many
different time-points into the same long-lived matrix -- and each of
those group-means was computed from a *different* group of
embeddings, at a *different* moment. The matrix ends up being an
average of apples centered-around-apples-mean, oranges
centered-around-oranges-mean, all jammed into one comparison. That's
incoherent -- it doesn't measure anything consistent."

**Concrete illustration.** Imagine at step 1, you center 8 embeddings
around their mean, call it mu_1, and fold the results into `V`. At
step 50, you have 8 *new* embeddings (maybe for a completely
different prompt/topic by now), center them around a *different*
mean mu_50, and fold *those* into the same `V`. Now `V` contains a
mixture: some vectors are "distance from mu_1," others are "distance
from mu_50" -- two different reference points, both silently living
inside the same matrix. When you later ask `V^-1` "how spread out is
this new vector?", the answer is contaminated by this history of
shifting reference points that have nothing to do with each other.
That's the "stale means" problem -- old, no-longer-relevant mean
values keep exerting influence long after they should have expired.

**Why this is directly relevant to what we just built:** your `V` in
sem-mcts is *exactly* this kind of long-lived, continuously-
accumulating matrix -- it persists and grows across the *entire* MCTS
search, not just one expansion. So the new
`embeds_center_mode="local"` you just added has precisely the
structure they warned about: each expansion group gets centered
around *its own* fresh mean (computed only from that group's 4
siblings), and then those centered vectors get folded into the *same,
long-running* `V` that also holds centered vectors from every other
expansion in the search, each with its own different mean. Same
"mixing different reference points into one matrix" problem, just at
MCTS-expansion granularity instead of training-step granularity.

**Why I called this "stronger evidence than my own reasoning":** I
had already written this exact caveat into your
`embeds-centering-design.md` doc based on my own mathematical
reasoning about what *should* go wrong. Finding their code comment
means an independent team, building essentially the same mechanism,
hit this exact issue themselves and left a warning about it -- that's
a second, independent confirmation of the concern, not just my own
inference. It doesn't mean local-centering-with-accumulated-V is
broken or unusable -- it just means it's a genuine, known risk (not a
hypothetical one I invented), so it's worth treating this mode
explicitly as "an ablation arm to empirically test," not "assumed to
be an improvement," exactly as the doc already frames it.

Offered to fold either of these plainer explanations into the docs
(the curated review, the transcript, or `embeds-centering-design.md`)
in place of the denser original phrasing -- awaiting Tuan's
direction.

---

# Follow-up: is our own sem-mcts already a form of leverage-based exploration? (2026-07-14)

> Fourth verbatim exchange. Tuan connected Point 1 above (leverage
> vs. leave-one-out) to our own repo's actual selection code and
> asked whether our mechanism already qualifies as leverage-based.

**Tuan's question:** "Regarding Point 1 in 'Raw discussion transcript:
rep_exp repo review (2026-07-14)', could our current llm-reasoning
implementation already be considered a form of leverage-based
exploration? For context, V_inv is updated only after a child is
selected, using that child's embedding, and the matrix is accumulated
throughout the entire search. Suppose the root node has three
candidates: A, B, and C. During the first traversal, the algorithm
selects A and updates V_inv using A's embedding. It then continues
searching until it reaches a terminal node and eventually returns to
the root. On the next visit to the root, the exploration scores for
A, B, and C are computed using a V_inv that already contains
information from A. This appears to match the leverage-based example
you described earlier. Is this interpretation correct?"

**Response (re-verified against `core/mcts_sem_search_v02_00_00.py`
line by line):**

Confirmed against the actual code. The interpretation is **correct**.

## Verified sequence: root visits A, B, C

1. **First visit to root** (`node.visit_count() == 1`): dispatches to
   `_select_by_q_value` (line 685) -- this branch never reads
   `V_inv` at all. It picks the highest-PRM-score child by pure
   q-value argmax. Say it picks A.
2. **Immediately after** (still inside `select_child`, line 692's
   comment literally says "UNCONDITIONAL: it runs on BOTH branches"):
   A's embedding is folded into `V_inv` via the Sherman-Morrison
   update (line 703-706). This happens **before returning from
   `select_child`** -- so by the time the tree descends further from
   A, `V_inv` already contains A.
3. Search descends from A all the way to a terminal node, then
   backprops back up to the root.
4. **Second visit to root** (`node.visit_count()` is now 2, so `> 1`):
   dispatches to `_select_by_diversity` (line 687) ->
   `_diverse_select` (line 672), which is passed **`self.V_inv`** --
   the *same instance attribute*, now containing A's contribution
   from step 2.

So when scoring A, B, and C on this second visit, the diversity bonus
for **all three candidates, including A itself**, is computed against
a `V_inv` that A already helped build. That is exactly the "leverage"
(self-included) structure from Point 1 -- not leave-one-out.

## One nuance worth being precise about

- Folding a vector into `V_inv` via Sherman-Morrison specifically
  **shrinks** the quadratic form along that vector's own direction
  (that's the whole point of the update -- it's what makes `V_inv`
  encode "we've already explored this direction, don't over-reward
  it again"). So if A gets re-visited later, A's own bonus term would
  be suppressed by its own prior inclusion -- same self-referential
  effect as the rep_exp `leverage` case.
- But in this specific example, A, B, and C are **three different
  children being compared at the same decision point**, and only A
  has been folded in so far. So the effect isn't "A discounts itself"
  in this specific comparison -- it's "A's *presence in* `V_inv`
  changes the bonuses computed for B and C too" (since `V_inv` is one
  shared matrix, not per-candidate). B and C's bonuses on this second
  visit are being measured relative to a `V_inv` that already "knows
  about" A's direction, which is the intended exploration behavior --
  reward B and C more if they point in directions unlike A, less if
  they're similar to A.

## Conclusion: yes, leverage-based, with one structural difference from theirs worth naming

Our mechanism and their `leverage` mode share the same core property
-- a shared, currently-accumulated `V_inv`/`cov_inv` that includes
prior picks, used to score everything compared against it, with no
leave-one-out downdate anywhere in the loop. But there's a real
structural difference in *when* folding happens:

- **Theirs:** all 8 siblings are generated together, then **all 8 are
  folded into `cov_inv`** (the loop at line 368-371 in their code),
  and *then* every one of the 8 gets its bonus computed against that
  fully-folded matrix -- so by the time bonuses are computed, every
  sibling already "sees" every other sibling, including itself.
- **Ours:** candidates are folded in **sequentially, one at a time,
  across separate tree visits** -- only the *actually selected* child
  gets folded in at each visit, and unselected siblings from that
  same expansion (B and C in the example) never get folded in on
  their own account at all (unless later revisited and selected). So
  our `V_inv` at any point in time reflects the *history of past
  selections*, not "everything generated so far" -- many candidates
  that were generated but never chosen contribute nothing to `V_inv`,
  ever.

That's a genuinely different exploration structure from theirs, even
though both are "leverage-style, self-inclusive, no leave-one-out."
Theirs measures novelty within one synchronous batch of same-prompt
siblings; ours measures novelty against an asynchronous,
path-dependent history of prior tree-selection choices -- closer in
spirit to a sequential bandit (LinUCB-style, which the code's own
comments elsewhere already describe it as) than to their one-shot
batch comparison.

Offered to add this Q&A to the transcript file -- Tuan asked for the
next exchange to be included together with this one instead (see
below), so both are captured in this same append.

---

# Follow-up: the full training-step sequence, corrected (2026-07-14)

> Fifth verbatim exchange. Tuan asked whether test-time compute and
> post-training use different implementations in the repo, and for a
> much more detailed, corrected walk-through of exactly how/when the
> covariance is built, reset, and whether it persists -- since the
> earlier phrase "all eight are folded into cov_inv in a single
> batch" was ambiguous about batching vs. sequential looping.

**Tuan's question:** "First, could you check whether their repository
uses different implementations for test-time compute and
post-training? I am confused about the statement: All eight sibling
trajectories are generated together, and then all eight are folded
into cov_inv in a single batch. Could you explain the full sequence
more clearly, starting from the beginning of each training epoch or
cycle? In particular, please clarify: whether they process one
prompt at a time or a batch of prompts; how the eight candidate
trajectories are generated for each prompt; how the initial V_inv or
cov_inv is constructed; whether all eight sibling embeddings are
added to the covariance matrix simultaneously or sequentially; how
cov_inv is updated after processing one prompt; whether the updated
cov_inv is carried over to subsequent prompts, batches, or training
steps; when and why cov_inv is reset; and how this process differs
between post-training and test-time inference. Please also explain
why they would add all eight sibling trajectories to cov_inv at once.
I would like to understand the motivation and consequences of their
approach."

**Response (re-verified against `rep_exp_trainer.py`,
`main_rep_exp.py`, `reward_manager/elliptical_reward_manager.py`, and
`workers/elliptical_reward_model_worker.py`, all re-read line by
line):**

## Do they use different implementations for test-time vs. post-training?

Not really -- there's only one implementation, and test-time
inference in this repo simply doesn't call it. This repo (`rep_exp`)
is exclusively the post-training half of the paper (the README says
so explicitly; no separate inference-time diversity-search code
exists anywhere in the tree). The "test-time" side of the paper's
title isn't shipped here at all.

What does exist inside this one repo is a training-vs-validation
distinction: `EllipticalRewardManager.__call__`'s very first line
checks `if "rm_scores" not in data.batch:` -- during training this is
False (the trainer computed `rm_scores` just before calling the
reward manager, via `rm_wg.compute_hidden_states` +
`rm_wg.compute_rm_score`, `rep_exp_trainer.py` lines 493-500); during
validation (`_validate()`, lines 88-227) `self.rm_wg` is never called
at all, so `rm_scores` is never present, and the manager silently
falls back to plain correctness-only scoring. Not two
implementations -- one implementation that training invokes and
validation skips, via a batch-key check.

## The full sequence, one training step at a time

Do they process one prompt at a time, or a batch of prompts? A batch
of many prompts at once -- `data.train_batch_size` (1024 for
MATH/GSM8K, 512 for DAPO-AIME) distinct prompts per step.

Step by step:

1. **Load a batch of N distinct prompts** from the dataloader.
2. **Assign a `uid`** to each of the N prompts -- one UUID per prompt.
3. **Replicate each prompt 8 times**:
   `gen_batch.repeat(repeat_times=8, interleave=True)`. Now `8N` rows,
   `N` distinct `uid` values (each appears 8 times).
4. **Generate**: vLLM samples one continuation for each of the `8N`
   rows (temperature 1.0) in one generation call for all N prompts at
   once.
5. **Compute hidden states** (`rm_wg.compute_hidden_states`): a single
   forward pass of the frozen backbone over all `8N` sequences at
   once (micro-batched for memory), mean-pooling each -> `8N` vectors.
6. **Compute the bonus** (`rm_wg.compute_rm_score`) -- expanded below.
7. **Combine with correctness**
   (`EllipticalRewardManager.__call__`): `total = alpha*correctness +
   beta*bonus`, none-correct gating applied per prompt-group.
8. **GRPO advantage + PPO update**, `uid` as the group key.
9. Loop back to step 1 for the next batch.

## Inside step 6 -- corrected detail

**Sparse projection:** with `randomize_sparse_matrix=True` (their
actual setting), `self.sparse_matrix` stays `None` forever, so a
brand-new random projection matrix is constructed from scratch on
*every single call* to `compute_rm_score` and never persisted. This
one matrix is applied to the *entire batch's* `8N` vectors at once --
shared across every prompt in this step, not per-prompt.

**`cov_inv` is one per prompt, not shared across the batch.** The
loop (`for i in range(len(data))`, `uid = ...`, skip already-seen
`uid`s) processes each distinct `uid` exactly once, using only its
own 8 rows (`mask = uid == this_uid`).

**How `cov_inv` is initialized and the 8 siblings folded in
(`persist_covariance=False`, their actual setting):**
```python
centered = filtered_mean_hidden_states - filtered_mean_hidden_states.mean(dim=0)
cov_inv = torch.eye(d, dtype=torch.float64) * self.lamb**-1   # LOCAL var, fresh every time
for hidden_state in centered:      # loop over exactly 8 vectors
    cov_inv = cov_inv - cov_inv @ x @ middle @ x.t() @ cov_inv   # sequential SM update
```
`cov_inv` is a **local Python variable**, reinitialized to `I/lamb`
from scratch for *every single prompt, every single training step* --
it is not an attribute on `self`. The 8 siblings are folded in **one
at a time, sequentially, in a Python for-loop** -- not as one batched
linear-algebra operation. Correction from the earlier summary: "all
eight are folded into cov_inv in a single batch" was imprecise; it's
sequential looping over 8 items, just all happening within one
prompt's processing before moving to the next prompt.

**Does `cov_inv` carry over to the next prompt, batch, or step?** No,
not in the paper's actual setting. The local variable is
garbage-collected the instant that prompt's loop body finishes. The
next prompt (same batch or a later step) gets its own fresh
`I/lamb`. Nothing about the covariance is retained across prompts or
steps in this mode.

**When and why is it reset?** Under `persist_covariance=False` there
is no "reset" event needed -- nothing ever persists, so there's
nothing to reset; every prompt starts from `I/lamb` by construction.
Under the alternative `persist_covariance=True` mode (not used for
the paper's results), `self.cov_inv_dict[prompt_index]` DOES persist
and accumulate across steps for the same prompt -- and no code
anywhere in the file ever clears these dicts, so in that mode it
would accumulate for the entire lifetime of the training run. That
mode is simply never invoked in `train_elliptical.sh`.

## Why fold all 8 siblings into a freshly-initialized-per-prompt covariance -- motivation and consequences

**Motivation:** the bonus answers "relative to the other trajectories
the model produced for *this same problem, right now*, how different
is this one?" -- a within-problem, within-instant question that
doesn't need history from other prompts or other steps. Reinitializing
`cov_inv` fresh per prompt, per step, is the most literal version of
"measure novelty against my own present siblings only": not just
locally-centered (as discussed in the point-2 follow-up above), but
locally-reinitialized in its entirety.

**Consequences -- stronger and cleaner than the earlier discussion
implied.** Because `cov_inv` never persists in their actual setting,
there is no cross-time contamination at all: no stale means, no
drifting covariance -- none of the coherence concerns flagged for our
own `V` (which persists across an entire MCTS search) or even for
their own `persist_covariance=True` variant apply to the setting that
produced their real published numbers. The tradeoff: this bonus can
only ever measure "diversity within these 8 samples of one problem,
right now" -- it structurally cannot accumulate any broader notion of
"which parts of representation space has this policy already
explored across many problems/steps," because it's rebuilt from zero
every time. A deliberate scope restriction, not an oversight --
matching the narrow question GRPO's per-prompt advantage
normalization already asks.

This strengthens, rather than weakens, the earlier point: their
paper setting (`persist_covariance=False`) is the fully local, fully
non-accumulating version -- no covariance state persists at all,
anywhere, ever. Our `V` in sem-mcts, by contrast, is always the
accumulating kind (it has to be, for sequential exploration across
an entire tree search) -- so the coherence concern raised for our new
`embeds_center_mode="local"` doesn't have a clean analog in their
actual paper setting at all. It only has an analog in their
`persist_covariance=True` mode -- the one they built but explicitly
chose not to use for their reported results. Worth stating plainly:
the mode that ships their real numbers has zero persistence-related
risk; the risk flagged earlier only applies if comparing against
their other, unused, more speculative mode.

Offered to add this corrected trace to the transcript file --
confirmed, appended here.
