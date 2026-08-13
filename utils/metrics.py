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

import os
import signal
from concurrent.futures import ProcessPoolExecutor

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
    fn_extract_answer, fn_grade, completion, gt_answer,
    grader_name='math', timeout=2,
):
    """Extract an answer from `completion` and grade it against
    `gt_answer`, aborting if it runs past `timeout` seconds.

    `grader_name` selects the utils/parser.py `data_name` vocabulary
    ("math", "gsm8k", ...) fn_extract_answer branches on.

    The grading call passes timeout=True so grader2.math_equal routes
    symbolic comparison through its multiprocessing hard-kill path
    (call_with_timeout / symbolic_equal_process) instead of comparing
    in-process: sympy can hang on pathological strings (e.g. a boxed
    equation instead of a value) in ways signal.alarm cannot interrupt
    -- signals only fire between Python bytecode instructions, and a
    stuck sympy call can sit entirely in C. The signal.alarm here still
    bounds fn_extract_answer (pure-Python, not sympy) and acts as a
    second guard around the whole call."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        c_answer = fn_extract_answer(completion, grader_name)
        result = fn_grade(c_answer, gt_answer, timeout=True)
    except TimeoutException:
        print(f"Timeout: {completion}")
        c_answer = None
        result = None
    finally:
        signal.alarm(0)
    return c_answer, result


def _grade_pred(pred_field, gt_answer, grader_name='math', timeout=2):
    """Grade a single pred_*@gb field string against the ground
    truth. Returns bool (False on timeout). See run_with_timeout for
    why grading passes timeout=True (hard-kill subprocess, not
    signal.alarm)."""
    pred_answer = parser.extract_answer(pred_field, grader_name)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return grader2.math_equal(pred_answer, gt_answer, timeout=True)
    except TimeoutException:
        return False
    finally:
        signal.alarm(0)


# --------------------------------------------------------------- #
# Per-dataset correctness evaluation                              #
# --------------------------------------------------------------- #

def _phase_key(data):
    """Scored-dataset key holding the question's last phase index."""
    return "q_last_phase" if "q_last_phase" in data else "last_phases"


def _search_spend(data, num_phases):
    """Per-question search spend, defined whether or not the search
    completed anything (unlike depth/ndepths, which average over
    completions). Returns (total_gens, capped).

    `capped` is 1.0 when the phase loop ran to its `num_phases`
    ceiling instead of exiting on the generation budget -- those
    runs stop early and UNDERSPEND `gen_budget` (measured 2026-08-06:
    ~45 of 80 on the AIME b=80 cov_scope=local cells), so an
    all-cells budget-parity claim needs this column to back it.
    nan when the run predates the key or the config has no phase
    loop (bon), so the nan-aware averages drop it.
    """
    total_gens = data.get("q_total_gens", np.nan)
    key = _phase_key(data)
    if num_phases is None or key not in data:
        return total_gens, np.nan
    return total_gens, float(data[key] >= num_phases - 1)


def _peak_curve(data, gt_answer, grader_name, timeout, step_budget):
    """Correctness-vs-budget curve of the running score-argmax.

    curve[b-1] = 1 iff the highest-`agg_scores` completion among
    those finished within the first b generations (`comp_gen`)
    grades correct — i.e. naive@b if the search had stopped at
    budget b. Grades only when the argmax improves, so the cost is
    O(#improvements) gradings + O(step_budget) fill. Completions
    with comp_gen > step_budget fold into the last step. Returns
    all-nan when the run has no generation axis (`comp_gen`
    absent, e.g. bon) — nan-aware pooling then drops it.
    """
    if step_budget is None or "comp_gen" not in data:
        return np.full(step_budget or 1, np.nan)
    curve = np.zeros(step_budget)
    completions = data["completions"]
    if len(completions) == 0:
        return curve
    agg_scores = data["agg_scores"]
    events = sorted(
        zip(data["comp_gen"], range(len(completions))),
        key=lambda e: e[0],
    )
    best_score = float("-inf")
    cur = 0.0
    i = 0
    for b in range(1, step_budget + 1):
        while (i < len(events)
               and min(events[i][0], step_budget) <= b):
            idx = events[i][1]
            if agg_scores[idx] > best_score:
                best_score = agg_scores[idx]
                _, ok = run_with_timeout(
                    parser.extract_answer, grader2.math_equal,
                    completions[idx], gt_answer, grader_name,
                    timeout,
                )
                cur = 1.0 if ok is True else 0.0
            i += 1
        curve[b - 1] = cur
    return curve


def _eval_question(args):
    """Grade one question; module-level so Pool can pickle it.

    Returns a dict of per-question scalars (the fields
    evaluate_correctness assembles into arrays) plus the
    `peak_curve` array (see _peak_curve).
    """
    data, grader_name, timeout, num_phases, step_budget = args
    _, gt_answer = parser.parse_ground_truth(data, grader_name)
    completions = data["completions"]
    total_gens, capped = _search_spend(data, num_phases)

    # No completions: the search produced nothing for this
    # question. Correctness is 0 (a genuine failure — it stays in
    # the denominator), but the search-cost stats are undefined
    # (mean over an empty list), so mark them nan to drop them
    # from the nan-aware averages rather than fabricate a 0.
    # total_gens/capped are exempt: they describe the search, not
    # its completions, so they stay real here.
    if len(completions) == 0:
        return {
            "pass_gb": 0.0,
            "naive_gb": 0.0,
            "weighted_gb": 0.0,
            "maj_gb": 0.0,
            "ncomps": 0,
            "depth": np.nan,
            "nphases": np.nan,
            "ndepths": np.nan,
            "total_gens": total_gens,
            "capped": capped,
            "peak_curve": _peak_curve(
                data, gt_answer, grader_name, timeout, step_budget,
            ),
        }

    row = {}
    # Aggregation-based predictions (naive / weighted / maj).
    row["naive_gb"] = _grade_pred(
        data["pred_naive@gb"], gt_answer, grader_name, timeout)
    row["weighted_gb"] = _grade_pred(
        data["pred_weighted@gb"], gt_answer, grader_name, timeout)
    row["maj_gb"] = _grade_pred(
        data["pred_maj@gb"], gt_answer, grader_name, timeout)

    # pass@gb: oracle best-of-n — correct if ANY completion is.
    pass_gb_correct = False
    for completion in completions:
        _, is_correct = run_with_timeout(
            parser.extract_answer, grader2.math_equal,
            completion, gt_answer, grader_name, timeout,
        )
        if is_correct is True:
            pass_gb_correct = True
            break
    row["pass_gb"] = pass_gb_correct

    # Search-cost stats (Scheme-C keys). Older mcts_sem_v01/v02
    # scored datasets predate this naming (pre-rename keys
    # c_depths/last_phases/ndepths_arr); fall back to those so
    # those runs remain readable without re-scoring.
    depth_key = "comp_depth" if "comp_depth" in data else "c_depths"
    phase_key = _phase_key(data)
    ndepths_key = (
        "phase_depths" if "phase_depths" in data else "ndepths_arr"
    )
    row["ncomps"] = len(completions)
    row["depth"] = np.mean(data[depth_key])
    row["nphases"] = data[phase_key]
    row["ndepths"] = np.mean(data[ndepths_key])
    row["total_gens"] = total_gens
    row["capped"] = capped
    row["peak_curve"] = _peak_curve(
        data, gt_answer, grader_name, timeout, step_budget,
    )
    return row


def evaluate_correctness(dataset, grader_name='math', timeout=2,
                         num_proc=1, num_phases=None,
                         step_budget=None):
    """Per-question metrics for one trial's scored dataset.

    `grader_name` selects the utils/parser.py `data_name` vocabulary
    ("math", "gsm8k", ...) ground-truth parsing and answer extraction
    branch on -- pass the run's `cfg.data.grader_name`.

    `num_proc` > 1 grades questions in a process pool (questions are
    independent). ProcessPoolExecutor, not multiprocessing.Pool: each
    symbolic comparison hard-kills stuck sympy in a child process of
    its own, and Pool's daemonic workers may not have children.
    <= 1 keeps the original serial path.

    `num_phases` is the run's `cfg.search.num_phases` phase ceiling,
    needed to decide `capped`; None (bon, which has no phase loop)
    leaves that column nan.

    `step_budget` (the run's `cfg.search.gen_budget`) sizes the
    per-question correctness-vs-budget curve behind `peak@gb`;
    None skips it (all-nan curves of length 1).

    Returns a dict of 1-D arrays (one entry per question):
      pass_gb, naive_gb, weighted_gb, maj_gb  (correctness in {0,1})
      ncomps, depth, nphases, ndepths     (search-cost stats)
      total_gens, capped                  (search spend)
    plus `peak_curve`, a (questions x step_budget) 2-D array.
    """
    items = [
        (data, grader_name, timeout, num_phases, step_budget)
        for data in dataset
    ]
    if num_proc > 1 and len(items) > 1:
        procs = min(num_proc, len(items))
        with ProcessPoolExecutor(max_workers=procs) as pool:
            rows = list(pool.map(_eval_question, items))
    else:
        rows = [_eval_question(item) for item in items]

    keys = (
        "pass_gb", "naive_gb", "weighted_gb", "maj_gb",
        "ncomps", "depth", "nphases", "ndepths",
        "total_gens", "capped",
    )
    out = {
        k: np.array([float(r[k]) for r in rows]) for k in keys
    }
    out["peak_curve"] = np.stack([r["peak_curve"] for r in rows])
    return out


def _load_trials(result_dir, config_name, num_trials, grader_name='math',
                 num_proc=1, num_phases=None, step_budget=None):
    """Load + evaluate each trial, returning per-metric arrays
    concatenated over all trials x questions. Trials whose scored
    .jsonl is missing are skipped (and reported), so stats can be
    computed over a partially-completed run."""
    per_trial = []
    skipped = []
    for trial_idx in range(num_trials):
        path = (
            f"{result_dir}/{config_name}"
            f"--trial-{trial_idx:03d}.jsonl"
        )
        if not os.path.exists(path):
            skipped.append(trial_idx)
            continue
        dataset_res = load_dataset(
            "json", data_files=path, split='train',
        )
        per_trial.append(
            evaluate_correctness(
                dataset_res, grader_name, num_proc=num_proc,
                num_phases=num_phases, step_budget=step_budget,
            )
        )

    if skipped:
        print(f"missing trials, skipped: {skipped}")
    if not per_trial:
        raise FileNotFoundError(
            f"no scored trials found in {result_dir} "
            f"for {config_name} (num_trials={num_trials})"
        )

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

def compute_stats_basics(
    result_dir, config_name, num_trials, grader_name='math',
    num_proc=1, num_phases=None, step_budget=None,
):
    """Aggregate correctness + search-cost stats across trials,
    save the per-question correctness arrays, and print a summary
    line. `grader_name` selects the utils/parser.py `data_name`
    vocabulary ("math", "gsm8k", ...) -- pass `cfg.data.grader_name`.
    `num_proc` > 1 parallelizes grading over questions (see
    evaluate_correctness). `num_phases` is the run's phase ceiling
    (`cfg.search.num_phases`), needed for the `capped` column; None
    leaves it nan. `step_budget` (`cfg.search.gen_budget`) enables
    `peak_gb` = max over budgets b of the pooled mean naive@b curve
    -- the best top-1 accuracy any single stopping budget <= gb
    achieves (naive_gb <= peak_gb <= pass_gb). Returns the stats
    dict (metric -> (mean, sem))."""
    stats = _load_trials(
        result_dir, config_name, num_trials, grader_name,
        num_proc=num_proc, num_phases=num_phases,
        step_budget=step_budget,
    )

    # Persist the raw per-question correctness for downstream tests.
    for name in ("pass_gb", "naive_gb", "weighted_gb", "maj_gb"):
        np.savetxt(
            f"{result_dir}/{name}_{config_name}.txt", stats[name],
        )

    order = [
        "pass_gb", "naive_gb", "weighted_gb", "maj_gb",
        "ncomps", "depth", "nphases", "ndepths",
        "total_gens", "capped",
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
    # nphases' mean is a poor summary -- the distribution is bimodal
    # (terminate in ~10 phases, or run to the ceiling), so report the
    # median too, plus how often the ceiling was hit and what the
    # generation spend actually was. `capped`>0 means gen_budget was
    # NOT fully spent on those questions (measured 2026-08-06).
    med = np.nanmedian(stats["nphases"])
    spend = (
        f"nphases_med {med:0.0f}, "
        f"gens {summary['total_gens'][0]:0.1f}, "
        f"capped {100 * summary['capped'][0]:0.1f}%"
    )

    # peak_gb: max over budgets of the pooled mean curve. The max
    # is over ONE stopping budget shared by all questions (a
    # tunable-knob claim), not a per-question oracle stop.
    curves = stats["peak_curve"]
    curve_mean = np.nanmean(curves, axis=0)
    if np.all(np.isnan(curve_mean)):
        summary["peak_gb"] = (np.nan, np.nan)
        peak = "peak_gb nan (no comp_gen axis)"
    else:
        peak_idx = int(np.nanargmax(curve_mean))
        col = curves[:, peak_idx]
        peak_sem = np.nanstd(col, ddof=1) / np.sqrt(len(col))
        summary["peak_gb"] = (curve_mean[peak_idx], peak_sem)
        summary["peak_b"] = (float(peak_idx + 1), 0.0)
        np.savetxt(
            f"{result_dir}/peak_curve_{config_name}.txt",
            curve_mean,
        )
        peak = (
            f"peak_gb {curve_mean[peak_idx]:0.4f} "
            f"(±{peak_sem:0.4f}) at b={peak_idx + 1}"
        )

    print(f"{corr}, {cost}")
    print(f"{spend}, {peak}")
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


def compute_correctness_curve_budget(
    dataset, step_budget, grader_name='math', timeout=2,
):
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

        _, gt_answer = parser.parse_ground_truth(data, grader_name)

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
                    completion, gt_answer, grader_name, timeout,
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
    result_dir, config_name, num_trials, step_budget,
    grader_name='math',
):
    """Average the budget curve over trials x questions; print and
    return the peak mean correctness with its SEM. `grader_name`
    selects the utils/parser.py `data_name` vocabulary ("math",
    "gsm8k", ...) -- pass `cfg.data.grader_name`."""
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
            dataset_res, step_budget, grader_name,
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
