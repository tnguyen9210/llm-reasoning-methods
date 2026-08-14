"""Search-cost columns straight from a scored dataset.

Reproduces the `ncomps / depth / nphases / ndepths / gens /
capped` aggregation of `utils/metrics.py::_eval_question`
without grading anything, so it runs in milliseconds per cell
instead of the ~1-2 min a full `compute_stats` pass costs (the
cost is all in sympy grading + W&B sync, neither of which these
columns need).

Validated 2026-08-13 against 87 cells whose values came from
W&B `eval/*` and from compute_stats logs: 327 checks, 0
mismatches.

NOT a compute_stats replacement: it computes no accuracy
column (pass/naive/wei/maj/peak), and it does NOT refresh the
run's W&B summary. Use compute_stats when either is wanted.
"""
import json
import glob

import numpy as np

NUM_PHASES = 1000


def cost_columns(config_hash, num_phases=NUM_PHASES,
                 results_root="results"):
    """Return the search-cost dict for one config hash.

    Mirrors _eval_question: a question with zero completions
    contributes real `gens`/`capped` but nan depth/nphases/
    ndepths, so the nan-aware means drop it. `nphases` is the
    MEDIAN (the distribution is bimodal).
    """
    pat = f"{results_root}/*/*/*cfg-{config_hash}*"
    dirs = [d for d in glob.glob(pat) if not d.endswith(".json")]
    if len(dirs) != 1:
        return None
    files = [f for f in sorted(glob.glob(f"{dirs[0]}/*trial-*.jsonl"))
             if "/generate_" not in f]
    if not files:
        return None

    ncomps, depth, nphases, ndepths, gens, capped = (
        [], [], [], [], [], [])
    for path in files:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                comps = row["completions"]
                last_phase = row.get(
                    "q_last_phase", row.get("last_phases"))
                gens.append(row.get("q_total_gens", np.nan))
                capped.append(
                    np.nan if last_phase is None
                    else float(last_phase >= num_phases - 1))
                ncomps.append(len(comps))
                if not comps:
                    depth.append(np.nan)
                    nphases.append(np.nan)
                    ndepths.append(np.nan)
                    continue
                dkey = ("comp_depth" if "comp_depth" in row
                        else "c_depths")
                nkey = ("phase_depths" if "phase_depths" in row
                        else "ndepths_arr")
                depth.append(np.mean(row[dkey]))
                nphases.append(last_phase)
                ndepths.append(np.mean(row[nkey]))

    def mean(arr):
        return float(np.nanmean(arr))

    def sem(arr):
        # Matches utils/metrics.py _mean_sem exactly: nan-aware
        # std, but divided by the FULL length (nan entries stay
        # in the denominator).
        return float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))

    return {
        "ncomps": mean(ncomps), "ncomps_sem": sem(ncomps),
        "depth": mean(depth), "depth_sem": sem(depth),
        "nphases_med": float(np.nanmedian(nphases)),
        "nphases": mean(nphases), "nphases_sem": sem(nphases),
        "ndepths": mean(ndepths), "ndepths_sem": sem(ndepths),
        "gens": mean(gens), "capped": mean(capped),
        "n_question_trials": len(ncomps),
    }
