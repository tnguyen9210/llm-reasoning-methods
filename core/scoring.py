"""Score search completions and assemble a Hugging Face Dataset.

Vendored from sal.utils.{score,math} so this project has no runtime
dependency on the sal config/scoring stack. The answer-extraction
parser lives in core.qwen_math_parser (a verbatim copy of sal's);
keep both in sync with upstream if sal changes.

Two layers:

- score_dataset(): the per-row aggregation + prediction driver.
  Takes plain args (agg_strategy, n, num_proc) instead of a
  sal.config.Config. Adds agg_scores and pred_{weighted,maj,naive}@gb.
- build_scored_dataset(): the orchestration that turns one search
  trial's raw results (batched per question) into a per-question
  scored Dataset and writes it to disk. Called by generate_mcts_cnt
  after each trial, and re-runnable standalone via
  prepare_scored_dataset.
"""

import math
import signal
from collections import defaultdict
from multiprocessing import Manager
from typing import Any, Literal

from datasets import Dataset
from latex2sympy2 import latex2sympy
from sympy import latex, simplify

from core.qwen_math_parser import extract_answer, strip_string

# Keys build_scored_dataset attaches itself (or that aren't
# per-question stats); every *other* per-question list in `results`
# is auto-attached as a stat column under its raw key. This keeps
# the scorer method-agnostic: mcts_cnt brings c_step_cnts/c_depths/
# ndepths_arr/..., bon brings completion_ntokens, etc.
RESERVED_RESULT_KEYS = {"completions", "scores"}


# ---------------------------------------------------------------
# Math-equivalence canonicalization (vendored from sal.utils.math).
# ---------------------------------------------------------------

class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException


_manager = Manager()
_shared_cache = _manager.dict()


def memoized_canonical_form(
    expression: str, timeout_seconds: int = 3
) -> str:
    """Canonicalize a LaTeX math expression via sympy, with a shared
    cross-process cache and a hard timeout. Falls back to a stripped
    string form on timeout or parse error."""
    if expression in _shared_cache:
        return _shared_cache[expression]

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        parsed_expr = latex2sympy(expression)
        simplified_expr = simplify(parsed_expr)

        signal.alarm(0)

        canonical_form = latex(simplified_expr)
        _shared_cache[expression] = canonical_form
        return canonical_form
    except TimeoutException:
        fallback = strip_string(expression)
        _shared_cache[expression] = fallback
        return fallback
    except Exception:
        fallback = strip_string(expression)
        _shared_cache[expression] = fallback
        return fallback
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------
# Per-row prediction helpers (vendored from sal.utils.math). Each
# operates on one dataset row's lists for a subset size n.
# ---------------------------------------------------------------

def subsample_completions(
    x: dict[str, list[Any]], n: int
) -> dict[str, list[Any]]:
    completions = x["completions"]
    agg_scores = x["agg_scores"]
    if len(completions) != len(agg_scores):
        raise ValueError(
            "completions and agg_scores must match: got "
            f"{len(completions)} vs {len(agg_scores)}."
        )
    # n is the subset label "gb" (gen-budget): the single use-all
    # subset this pipeline emits — every completion at the run's full
    # generation budget.
    return {
        f"completions@{n}": completions,
        f"agg_scores@{n}": agg_scores,
    }


def extract_completion_answers(
    x: dict[str, list[Any]], n: int | None = None
) -> dict[str, list[str]]:
    if n is None:
        return {
            "preds": [extract_answer(p, "math") for p in x["completions"]]
        }
    return {
        f"preds@{n}": [
            extract_answer(p, "math") for p in x[f"completions@{n}"]
        ]
    }


def compute_naive_pred(
    x: dict[str, list[Any]], n: int
) -> dict[str, str]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    if len(preds) == 0:
        return {f"pred_naive@{n}": "\\boxed{NA}"}
    preds = [
        (p, s)
        for p, s in sorted(
            zip(preds, scores), key=lambda t: t[1], reverse=True
        )
    ]
    return {f"pred_naive@{n}": "\\boxed{" + preds[0][0] + "}"}


def compute_weighted_pred(
    x: dict[str, list[Any]], n: int
) -> dict[str, str]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    if len(preds) == 0:
        return {f"pred_weighted@{n}": "\\boxed{NA}"}
    return {
        f"pred_weighted@{n}": "\\boxed{"
        + find_answer_with_largest_sum(preds, scores)
        + "}"
    }


def compute_maj_pred(x: dict[str, list[Any]], n: int) -> dict[str, str]:
    preds = x[f"preds@{n}"]
    if len(preds) == 0:
        return {f"pred_maj@{n}": "\\boxed{NA}"}
    return {f"pred_maj@{n}": "\\boxed{" + find_majority_answer(preds) + "}"}


def find_answer_with_largest_sum(
    answers: list[str], scores: list[float]
) -> str:
    """Group answers by canonical form, return the original answer of
    the group with the largest cumulative score (weighted vote)."""
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    canonical_groups = defaultdict(float)
    canonical_to_original = {}

    for answer, score in zip(answers, scores):
        canonical_form = memoized_canonical_form(answer)
        canonical_groups[canonical_form] += score
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    max_canonical = max(canonical_groups, key=canonical_groups.get)
    return canonical_to_original[max_canonical]


def find_majority_answer(answers: list[str]) -> str:
    """Group answers by canonical form, return the original answer of
    the largest group (majority vote); first group wins ties."""
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    canonical_groups = defaultdict(int)
    canonical_to_original = {}

    for answer in answers:
        canonical_form = memoized_canonical_form(answer)
        canonical_groups[canonical_form] += 1
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            return canonical_to_original[canonical_form]


# ---------------------------------------------------------------
# Aggregation + the dataset-level scoring driver.
# ---------------------------------------------------------------

def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    if agg_strategy == "prod":
        return math.prod(scores)
    if agg_strategy == "last":
        return scores[-1]
    raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score_dataset(
    dataset: Dataset,
    agg_strategy: str = "last",
    n: str = "gb",
    num_proc: int = 1,
) -> Dataset:
    """Add agg_scores and pred_{weighted,maj,naive}@{n} to a dataset
    that already has `completions` and per-step `scores`.

    Config-free port of sal.utils.score.score. Only one subset is
    emitted — the use-all subset, suffixed `gb` (gen-budget: every
    completion produced at the run's full generation budget). So the
    prediction fields are pred_weighted@gb / pred_maj@gb / pred_naive@gb.
    """
    dataset = dataset.map(
        lambda x: {
            "agg_scores": [
                aggregate_scores(s, agg_strategy) for s in x["scores"]
            ]
        }
    )
    for subset in [n]:
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": subset},
            num_proc=num_proc,
            desc=f"Subsample {subset}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": subset},
            num_proc=num_proc,
            desc=f"Extract answers {subset}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": subset},
            num_proc=num_proc,
            desc=f"Weighted pred {subset}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": subset},
            num_proc=num_proc,
            desc=f"Majority pred {subset}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": subset},
            num_proc=num_proc,
            desc=f"Naive pred {subset}",
        )
        dataset = dataset.remove_columns(
            [f"completions@{subset}", f"agg_scores@{subset}",
             f"preds@{subset}"]
        )
    return dataset


# ---------------------------------------------------------------
# Orchestration: raw search results -> scored per-question Dataset.
# ---------------------------------------------------------------

def build_scored_dataset(
    results: dict[str, list],
    dataset: Dataset,
    prm,
    result_dir: str,
    run_name: str,
    trial_idx: int,
    agg_strategy: str = "last",
    n: str = "gb",
    num_proc: int = 1,
    batch_size: int = 4,
) -> Dataset:
    """Turn one trial's raw search results into a scored Dataset and
    write it to {result_dir}/{run_name}--trial-{trial_idx:03d}.jsonl.

    `results` holds per-question lists batched over all questions (the
    object generate_*.jsonl stores). `dataset` is the matching question
    split (sliced to the same questions). The output row schema is the
    dataset's base fields (problem/solution/answer/subject/level/
    unique_id) plus completions, scores, agg_scores,
    pred_{weighted,maj,naive}@gb, and whatever per-question stats the
    method produced.

    Stats are method-agnostic: every per-question list in `results`
    other than `completions` is attached as a column under its raw key
    (mcts_cnt -> c_step_cnts/c_depths/ndepths_arr/...; bon ->
    completion_ntokens). A list is treated as per-question when its
    length equals the number of questions; anything else is skipped.
    """
    completions = results["completions"]
    num_questions = len(completions)

    # Align the dataset rows with the questions actually searched.
    if len(dataset) != num_questions:
        dataset = dataset.select(range(num_questions))

    questions = [dataset[i]["problem"] for i in range(num_questions)]

    # PRM-score every candidate: scores[q][answer][step].
    scores = prm.score(questions, completions, batch_size=batch_size)

    dataset = dataset.add_column("completions", completions)
    dataset = dataset.add_column("scores", scores)

    # Auto-attach every other per-question stat list under its raw key.
    for key, value in results.items():
        if key in RESERVED_RESULT_KEYS:
            continue
        if isinstance(value, list) and len(value) == num_questions:
            dataset = dataset.add_column(key, value)
        else:
            print(f"skip non-per-question result field: {key!r}")

    dataset = score_dataset(
        dataset, agg_strategy=agg_strategy, n=n, num_proc=num_proc,
    )

    out_path = f"{result_dir}/{run_name}--trial-{trial_idx:03d}.jsonl"
    dataset.to_json(out_path)
    print(f"wrote scored dataset -> {out_path}")
    return dataset
