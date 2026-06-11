"""
Check whether search-result completions are complete trajectories.

A complete trajectory ends with the final-answer phrase the system
prompt requests ("The final answer is"). Incomplete ones are
classified by tail shape, which identifies how they entered
`completed_nodes`:

  ends_step_sep  text ends with the "\n\n" step separator — the model
                 emitted an empty next step (immediate EOS), so the
                 parent text was marked completed without a final
                 answer.
  mid_sentence   text stops without a separator — the model emitted
                 EOS inside a step (premature stop), or the step hit
                 max_tokens.

Also flags completions containing "\r\n" separators, which evade the
"\n\n" vLLM stop string and merge several steps into one generation.

Usage
    python unittests/check_trajectory_completeness.py <results.jsonl> [...]
    python unittests/check_trajectory_completeness.py results/prm800k/<run>/*.jsonl
"""

import argparse
import json
from collections import Counter

PHRASE = "The final answer is"


def classify(completion: str, phrase: str) -> str:
    if phrase in completion:
        return "complete"
    if completion.rstrip(" ").endswith("\n\n"):
        return "ends_step_sep"
    return "mid_sentence"


def check_file(path: str, phrase: str, show_examples: int) -> None:
    with open(path) as f:
        results = json.load(f)

    cats: Counter = Counter()
    depth_cats: dict = {}
    crlf = 0
    examples = []
    for q_idx, comps in enumerate(results["completions"]):
        for c_idx, comp in enumerate(comps):
            cat = classify(comp, phrase)
            cats[cat] += 1
            depth = results["c_depths"][q_idx][c_idx]
            depth_cats.setdefault(depth, Counter())[cat] += 1
            if "\r\n" in comp:
                crlf += 1
            if cat != "complete" and len(examples) < show_examples:
                examples.append((q_idx, c_idx, depth, cat, comp[-200:]))

    total = sum(cats.values())
    print(f"\n=== {path}")
    print(f"total completions: {total}")
    for cat in ("complete", "ends_step_sep", "mid_sentence"):
        n = cats[cat]
        print(f"  {cat:<14} {n:4d}  ({100 * n / total:5.1f}%)")
    print(f"  contains \\r\\n  {crlf:4d}")

    print("per-depth (complete / ends_step_sep / mid_sentence):")
    for depth in sorted(depth_cats):
        c = depth_cats[depth]
        print(
            f"  d={depth:<3} "
            f"{c['complete']:3d} / {c['ends_step_sep']:3d} / "
            f"{c['mid_sentence']:3d}"
        )

    for q_idx, c_idx, depth, cat, tail in examples:
        print(f"\n--- q={q_idx} comp={c_idx} d={depth} [{cat}] tail:")
        print(repr(tail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="results .jsonl file(s)")
    parser.add_argument("--phrase", default=PHRASE)
    parser.add_argument("--examples", type=int, default=0,
                        help="show N incomplete tails per file")
    args = parser.parse_args()

    for path in args.paths:
        check_file(path, args.phrase, args.examples)


if __name__ == "__main__":
    main()
