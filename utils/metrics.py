"""Performance metrics for scored search-result datasets.

Operates on the scored per-question jsonl that
prepare_scored_dataset / build_scored_dataset writes: each row has
`completions`, per-step `scores`, `agg_scores`,
`pred_{naive,weighted,maj}@gb`, and the Scheme-C search stats
(`comp_depth`, `comp_gen`, `q_last_phase`, `phase_depths`, ...).
Answer correctness is graded against the ground truth via the
vendored parser + math-equality grader.

The `@gb` suffix = gen-budget: the prediction uses every completion
produced at the run's full generation budget (the single use-all
subset the scoring step emits).

Key metrics (per question, then averaged over questions x trials):
  pass@gb     any candidate completion is correct (best-of-n oracle)
  naive/weighted/maj@gb   correctness of the corresponding pred field
  ncomps      number of candidate completions
  depth       mean per-completion finish depth (comp_depth)
  nphases     final MCTS phase reached (q_last_phase)
  ndepths     mean per-phase depth (phase_depths)

Metric dict keys use underscores (pass_gb, ...) so they are safe as
.txt filename stems; the printed summary reads them as @gb metrics.
"""

import signal

import numpy as np
np.set_printoptions(precision=4)

from datasets import load_dataset
from utils import parser, grader2

import logging


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(
    fn_extract_answer, fn_grade, completion, gt_answer, timeout=2
):
    """Extract an answer from `completion` and grade it against
    `gt_answer`, aborting if it runs past `timeout` seconds (the
    sympy grader can hang on pathological strings)."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        c_answer = fn_extract_answer(completion, 'math')
        result = fn_grade(c_answer, gt_answer)
    except TimeoutException:
        print(f"Timeout: {completion}")
        c_answer = None
        result = None
    finally:
        signal.alarm(0)
    return c_answer, result


def _grade_pred(pred_field, gt_answer, timeout=2):
    """Grade a single pred_*@gb field string against the ground
    truth. Returns bool (False on timeout)."""
    pred_answer = parser.extract_answer(pred_field, 'math')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return grader2.math_equal(pred_answer, gt_answer)
    except TimeoutException:
        return False
    finally:
        signal.alarm(0)


# --------------------------------------------------------------- #
# Per-dataset correctness evaluation                              #
# --------------------------------------------------------------- #

def evaluate_correctness(dataset, timeout=2):
    """Per-question metrics for one trial's scored dataset.

    Returns a dict of 1-D arrays (one entry per question):
      pass_gb, naive_gb, weighted_gb, maj_gb  (correctness in {0,1})
      ncomps, depth, nphases, ndepths     (search-cost stats)
    """
    n = len(dataset)
    out = {
        "pass_gb": np.zeros(n),
        "naive_gb": np.zeros(n),
        "weighted_gb": np.zeros(n),
        "maj_gb": np.zeros(n),
        "ncomps": np.zeros(n),
        "depth": np.zeros(n),
        "nphases": np.zeros(n),
        "ndepths": np.zeros(n),
    }

    for q_idx, data in enumerate(dataset):
        _, gt_answer = parser.parse_ground_truth(data, 'math')
        completions = data["completions"]

        # No completions: the search produced nothing for this
        # question. Correctness is 0 (a genuine failure — it stays in
        # the denominator), but the search-cost stats are undefined
        # (mean over an empty list), so mark them nan to drop them
        # from the nan-aware averages rather than fabricate a 0.
        if len(completions) == 0:
            out["pass_gb"][q_idx] = 0.0
            out["naive_gb"][q_idx] = 0.0
            out["weighted_gb"][q_idx] = 0.0
            out["maj_gb"][q_idx] = 0.0
            out["ncomps"][q_idx] = 0
            out["depth"][q_idx] = np.nan
            out["nphases"][q_idx] = np.nan
            out["ndepths"][q_idx] = np.nan
            continue

        # Aggregation-based predictions (naive / weighted / maj).
        out["naive_gb"][q_idx] = _grade_pred(
            data["pred_naive@gb"], gt_answer, timeout)
        out["weighted_gb"][q_idx] = _grade_pred(
            data["pred_weighted@gb"], gt_answer, timeout)
        out["maj_gb"][q_idx] = _grade_pred(
            data["pred_maj@gb"], gt_answer, timeout)

        # pass@gb: oracle best-of-n — correct if ANY completion is.
        pass_gb_correct = False
        for completion in completions:
            _, is_correct = run_with_timeout(
                parser.extract_answer, grader2.math_equal,
                completion, gt_answer, timeout,
            )
            if is_correct is True:
                pass_gb_correct = True
                break
        out["pass_gb"][q_idx] = pass_gb_correct

        # Search-cost stats (Scheme-C keys).
        out["ncomps"][q_idx] = len(completions)
        out["depth"][q_idx] = np.mean(data["comp_depth"])
        out["nphases"][q_idx] = data["q_last_phase"]
        out["ndepths"][q_idx] = np.mean(data["phase_depths"])

    return out


def _load_trials(result_dir, config_name, num_trials):
    """Load + evaluate each trial, returning per-metric arrays
    concatenated over all trials x questions."""
    per_trial = []
    for trial_idx in range(num_trials):
        path = (
            f"{result_dir}/{config_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        dataset_res = load_dataset(
            "json", data_files=path, split='train',
        )
        per_trial.append(evaluate_correctness(dataset_res))

    keys = per_trial[0].keys()
    return {
        k: np.concatenate([t[k] for t in per_trial]) for k in keys
    }


def _mean_sem(arr):
    """Mean and standard error of the mean (nan-aware)."""
    mean = np.nanmean(arr)
    sem = np.nanstd(arr, ddof=1) / np.sqrt(len(arr))
    return mean, sem


# --------------------------------------------------------------- #
# Top-level: basic statistics                                     #
# --------------------------------------------------------------- #

def compute_stats_basics(result_dir, config_name, num_trials):
    """Aggregate correctness + search-cost stats across trials,
    save the per-question correctness arrays, and print a summary
    line. Returns the stats dict (metric -> (mean, sem))."""
    stats = _load_trials(result_dir, config_name, num_trials)

    # Persist the raw per-question correctness for downstream tests.
    for name in ("pass_gb", "naive_gb", "weighted_gb", "maj_gb"):
        np.savetxt(
            f"{result_dir}/{name}_{config_name}.txt", stats[name],
        )

    order = [
        "pass_gb", "naive_gb", "weighted_gb", "maj_gb",
        "ncomps", "depth", "nphases", "ndepths",
    ]
    summary = {k: _mean_sem(stats[k]) for k in order}

    corr = ", ".join(
        f"{summary[k][0]:0.4f} (±{summary[k][1]:0.4f})"
        for k in ("pass_gb", "naive_gb", "weighted_gb", "maj_gb")
    )
    cost = ", ".join(
        f"{summary[k][0]:0.1f} (±{summary[k][1]:0.1f})"
        for k in ("ncomps", "depth", "nphases", "ndepths")
    )
    print(f"{corr}, {cost}")
    return summary


# --------------------------------------------------------------- #
# Correctness-vs-budget curves                                    #
# --------------------------------------------------------------- #

def max_with_index(arr):
    max_score = arr[0]
    max_idx = 0
    for i, val in enumerate(arr):
        if val > max_score:
            max_score = val
            max_idx = i
    return max_score, max_idx


def compute_correctness_curve_budget(dataset, step_budget, timeout=2):
    """Per-question best-of-n correctness as a function of generation
    budget. Uses comp_gen (generation count when each completion
    finished) as the budget axis; the running argmax over agg_scores
    is graded and held flat between budget steps."""
    peak_correctness = np.zeros((len(dataset), step_budget))
    peak_idxes = np.zeros((len(dataset), step_budget))

    for q_idx, data in enumerate(dataset):
        completions = data["completions"]
        comp_gen = data["comp_gen"]
        agg_scores = data["agg_scores"]
        if len(completions) == 0:
            continue

        _, gt_answer = parser.parse_ground_truth(data, 'math')

        max_correctness_list = []
        max_score = float('-inf')
        max_is_correct = False
        max_step_cnt = -1
        max_overlap = False

        for completion, step_cnt, score in zip(
            completions, comp_gen, agg_scores
        ):
            if score > max_score:
                max_score = score
                if step_cnt == max_step_cnt:
                    max_overlap = True
                max_step_cnt = step_cnt
                _, max_is_correct = run_with_timeout(
                    parser.extract_answer, grader2.math_equal,
                    completion, gt_answer, timeout,
                )

            if max_overlap:
                max_correctness_list[-1] = max_is_correct
            else:
                max_correctness_list += (
                    [max_is_correct]
                    * (step_cnt - len(max_correctness_list))
                )
            max_overlap = False

        if len(max_correctness_list) < step_budget:
            max_correctness_list += (
                [max_correctness_list[-1]]
                * (step_budget - len(max_correctness_list))
            )
        peak_correctness[q_idx, :] = max_correctness_list[:step_budget]

    return peak_correctness, peak_idxes


def compute_stats_correctness_curve_budget(
    result_dir, config_name, num_trials, step_budget
):
    """Average the budget curve over trials x questions; print and
    return the peak mean correctness with its SEM."""
    all_curves = []
    for trial_idx in range(num_trials):
        path = (
            f"{result_dir}/{config_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        dataset_res = load_dataset(
            "json", data_files=path, split='train',
        )
        curve, _ = compute_correctness_curve_budget(
            dataset_res, step_budget,
        )
        all_curves.append(curve)

    all_curves = np.concatenate(all_curves)
    nsamples = len(all_curves)
    curve_mean = np.mean(all_curves, axis=0)
    curve_sem = np.std(all_curves, axis=0, ddof=1) / np.sqrt(nsamples)

    peak_mean, peak_idx = max_with_index(curve_mean)
    print(f"peak_gb_score = {peak_mean:0.4f} "
          f"(±{curve_sem[peak_idx]:0.4f})")
    return curve_mean, curve_sem
