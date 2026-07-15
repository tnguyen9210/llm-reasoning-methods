# Related work: verl-recipe `rep_exp` (Representation-Based Exploration) — review and implications

*Reviewed 2026-07-14 —
[decisions-log.md #2026-07-14](../decisions-log.md#2026-07-14--related-work-rep_exp-elliptical-bonus-review--follow-up-candidates)*

Review of the official post-training code for **"Representation-Based
Exploration for Language Models: From Test-Time to Post-Training"**
(Tuyls, Foster, Krishnamurthy, Ash — arXiv 2510.11686), published as
the `rep_exp` recipe in
[verl-project/verl-recipe](https://github.com/verl-project/verl-recipe/tree/main/rep_exp).
It is the closest published relative of our sem-mcts diversity bonus:
an **elliptical (leverage-score) exploration bonus over mean-pooled
LLM hidden states**, sparse-projected, maintained via Sherman-Morrison
rank-1 updates — the same mathematical family as our
`q_diversity = sqrt(x^T V^-1 x)` term. Key difference in setting:
they add the bonus to the *reward* inside **GRPO post-training**
(no search, no PRM); we add it to *selection* inside test-time MCTS.
The repo covers only the post-training half of their paper; the
test-time half is not in this recipe.

This doc records what their pipeline does (verified from source, verl
pinned commit `b9bd00ef`), the point-by-point mapping onto our knobs,
and the follow-up candidates it motivates. Nothing here is executed
yet — each follow-up is a candidate, not a commitment.

## Their pipeline (facts, from source)

**Setting.** GRPO via verl + vLLM: one model
(**Qwen2.5-7B-Instruct**) is both the policy and (as a frozen copy in
a dedicated FSDP worker) the representation extractor. No PRM.
`n=8` rollouts per prompt at temperature 1.0; the group of 8 is both
the GRPO advantage group and the covariance group. KL coef and
entropy coef are both 0 — the elliptical bonus is the *only*
exploration mechanism. Tasks: MATH (lighteval), GSM8K,
DAPO-train/AIME24-dev.

**Prompt format.** Single user message, question + a dataset-specific
instruction appended; **no custom chat template, no explicit system
prompt** (the tokenizer's stock `apply_chat_template` runs with
defaults, so Qwen's built-in default system prompt applies). GSM8K
keeps its native `#### N` answer convention; MATH and DAPO/AIME use
`\boxed{}`. Only one model → nothing analogous to our
native-vs-custom `tmpl` axis.

**Bonus computation** (`workers/elliptical_reward_model_worker.py`),
per training batch, per prompt-group of 8:

1. **Representation:** frozen backbone forward (`AutoModel`, bf16, no
   LM head) → `last_hidden_state`, **mean-pooled over response tokens
   only** (prompt excluded). In our vocabulary:
   `embeds_strategy=avg` + `embeds_scope=response`.
2. **Projection:** sklearn `SparseRandomProjection` to
   **`sparse_dim=32`** (MATH/GSM8K) or **128** (DAPO-AIME); with
   their paper setting `randomize_sparse_matrix=True` the matrix is
   **redrawn every batch** (vs. ours: fixed matrix, dim 512 — see
   [sparse-random-projection.md](sparse-random-projection.md)).
3. **Precision:** projected features upgraded to **float64** for all
   covariance math.
4. **Covariance:** paper setting `persist_covariance=False` → the
   covariance is **per-group and per-step**: center the 8 features on
   the group mean, init `V^-1 = I/lamb` with **`lamb=0.01`**, fold
   all 8 in via Sherman-Morrison rank-1 updates (same update as our
   `cov_update=sm` — see
   [sherman-morrison-covariance-update.md](sherman-morrison-covariance-update.md)).
   A `persist_covariance=True` variant (per-prompt running `V^-1` +
   running-mean adjustment across steps) exists but is off.
5. **Bonus:** `reward_type=leverage` → `bonus_i = h_i^T V^-1 h_i`
   with self included (a leverage score), computed *after* folding
   the whole group. A `leave_one_out` variant (SM downdate, then
   quadratic form) is the yaml default but NOT what the paper's
   script runs. Note theirs is the squared form; ours takes `sqrt`.
6. **Placement:** bonus written on the last response token of the
   token-level reward tensor — trajectory-level, not per-step.
7. **Combination:** `total = alpha * correctness + beta * bonus`,
   `alpha=1.0`, `beta=0.01`, `normalization=none` (rnd / z-score
   options exist, unused).
8. **Gating:** `turn_off_elliptical_if_none_correct=True` — if zero
   of the 8 rollouts are correct, the bonus is zeroed for the whole
   group. Under GRPO this stops the bonus from becoming the only
   advantage signal and reinforcing diverse-but-wrong rollouts.
   (Flags for some-correct / all-correct / per-rollout gating exist,
   all off.)

**Scoring** (dispatch on `data_source` — the analog of our
`grader_name`): GSM8K = strict `#### (\-?[0-9\.\,]+)` regex on the
**last 300 chars only**, last match, exact string match. MATH = last
`\boxed{}` + Hendrycks `strip_string` equality. DAPO/AIME =
HuggingFace **math-verify** (gold wrapped as `\boxed{gt}`,
exceptions → 0).

**Eval metrics:** n=128 (dev) / 256 (final) samples per question at
temperature 1.0, then **pass@k via the unbiased combinatorial
estimator** `1 - C(n-c,k)/C(n,k)` for k = 1, 2, 4, ..., n, plus
maj@k via bootstrap. Checkpoint selection = best dev pass@1. During
validation the bonus is bypassed entirely.

**AIME source:** `MathArena/aime_2024_I` + `_II` (30 problems) —
different HF repo from our `aime2024` (`Maxwell-Jia/AIME_2024`); same
problems, different field names/formatting. Remember this if
cross-checking numbers.

## Mapping onto our knobs

| theirs | ours | delta |
|---|---|---|
| policy backbone `last_hidden_state` | `embeds_source=prm` (v02) or policy (v01) | they never use a PRM |
| mean over response tokens | `embeds_strategy=avg` + `embeds_scope=response` | exactly the cell v02 blocks for PRM-source ([embeds-scope-design.md](embeds-scope-design.md)) |
| sparse proj, dim 32/128, redrawn per batch | `embeds_proj=sparse`, dim 512, fixed matrix | 4-16x smaller dim; fresh matrix each step |
| group-mean centering, per prompt-group (8 same-prompt rollouts, `uid`-keyed, one `V^-1` fold per training step) | `embeds_mean` modes ([embeds-centering-design.md](embeds-centering-design.md)); our `bs=4` candidates are likewise 4 samples of the identical prompt at one MCTS node (`generate_mcts_sem.py`: `current_templated * config.search.batch_size`) | same group *structure* on both sides (same-prompt siblings) — theirs recomputes the mean fresh every training step and never carries it forward, ours currently centers globally/online across steps, not per-node |
| `lamb=0.01` ridge init | `lam` (same role) | same operating point as one of our sweep values — an external anchor: an independent group shipped the same ridge scale |
| `beta=0.01` reward weight | `ds_alpha` (selection weight) | NOT transferable as a number: their bonus is a leverage score of group-centered fp64 features (bounded ~[0,1]), ours is `sqrt(x^T V^-1 x)` of uncentered features — different scale and different units ([tuning-semantic-score-weights-and-lambda.md](tuning-semantic-score-weights-and-lambda.md)) |
| per-dataset answer instruction (GSM8K native `####`, MATH/AIME `\boxed{}`) | one fixed `\boxed{}` system prompt for every dataset (`core/reward_models.py::QWEN_SYSTEM_PROMPT`) | our GSM8K runs ask for boxed and rely on `extract_answer`'s boxed-first/last-number-fallback; theirs formats and grades GSM8K natively |
| float64 covariance math | fp32 | numerical-care delta |
| bonus gated on group having >=1 correct | ungated | no analog in our selection |
| pass@k unbiased estimator, n=256 samples | empirical pass@gb over search completions | NOT directly comparable |

## Follow-up candidates (prioritized, none executed)

1. **Unblock avg+response via v01 (policy embeds).** Their whole
   method runs on the exact pooling/scope combination our v02 blocks
   for PRM-source embeds. v01 (policy-source) already supports
   `embeds_scope=response` — the published result is direct evidence
   this cell of the `embeds_strategy x scope` sweep is worth running
   first, on the v01 path, rather than waiting on v02
   `response_start_idx` support for PRM embeds.
2. **Projection-dim ablation (`embeds_dim=32/128`).** If their
   results hold at d=32, our 512 may be paying conditioning + compute
   cost for nothing. Cheap: rerun one existing sem-mcts sweep point
   at 32 and 128.
3. **Local (sibling-group) centering.** Center each expansion batch's
   embeddings on the batch mean before folding into `V` — measures
   within-group dispersion instead of distance from a global mean.
   Directly transplantable to our `bs=4` expansion groups.
   **BUILT 2026-07-14** as `embeds_center_mode="local"` (v02 core
   only) — see the local-mean section of
   [embeds-centering-design.md](embeds-centering-design.md); no runs
   yet.
4. **Signal-gated diversity.** Their none-correct gating translates
   to: suppress `q_diversity` when all candidates' PRM scores are
   uniformly low (nothing worth diversifying around). One-line
   change in `_diverse_select`; needs a threshold choice.
5. **float64 for the `V^-1` math.** At d=32-512 the d x d ops are
   cheap even on the V100S; fp64 there is near-free insurance
   against SM-update drift on long searches.
6. **(Deferred) last-300-chars extraction clip.** Their GSM8K
   extractor only regexes the tail of the generation — a small
   robustness/speed trick for very long traces; our sympy-hang
   finding ([compute-stats-sympy-hang.md](../findings/coding-findings/compute-stats-sympy-hang.md))
   is adjacent motivation.

Not adopted, deliberately: their pass@k estimator for our tables
(our pass@gb measures the search's own output set, not i.i.d.
samples — switching would change the meaning of every recorded
cell); their redrawn-per-batch projection (our fixed-matrix decision
in [sparse-random-projection.md](sparse-random-projection.md) was
made for cross-node reproducibility, and nothing here overturns
that reasoning — but it is now known to be a choice, not a
necessity).

## Sources

- Repo: <https://github.com/verl-project/verl-recipe/tree/main/rep_exp>
  (reviewed at HEAD of `main`, 2026-07-14; verl pinned commit
  `b9bd00efba253ea90072555c45692054cf703de2`)
- Paper: Tuyls, Foster, Krishnamurthy, Ash. "Representation-Based
  Exploration for Language Models: From Test-Time to Post-Training."
  arXiv:2510.11686. <https://arxiv.org/abs/2510.11686>
- Key files: `workers/elliptical_reward_model_worker.py` (bonus),
  `reward_manager/elliptical_reward_manager.py` (combination +
  gating), `metric_utils.py` (pass@k/maj@k), `data_preprocess/*.py`
  (prompt formats), `train_elliptical.sh` (paper hyperparameters).
