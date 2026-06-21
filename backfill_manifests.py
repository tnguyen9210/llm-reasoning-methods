"""One-time backfill: write a manifest.json into each pre-existing
result dir that lacks one, so old runs (long-form names, created before
the prefix+hash scheme) carry the same manifest as new runs.

WHAT IT CAN AND CANNOT DO
- ✅ Records `config_name = <existing dir basename>`. This is the
  authoritative basename for the dir's trial files
  ({basename}--trial-NNN.jsonl), so a reader pointed at the dir (via
  +result_dir=...) resolves them through manifest_run_name().
- ✅ Records the knob=value pairs PARSED from the old name (the `--k-v`
  segments) under `parsed_from_name`, for human reference / manifest
  search.
- ❌ CANNOT make the dir findable by find_run_dir(cfg). That matches a
  hash over the *full* config identity; an old name encodes only the
  subset of fields the old config_name chose to show, and the rest
  aren't recoverable from the name. So the backfilled manifest gets
  `config_hash: null` and old runs stay addressable by explicit
  +result_dir=..., not by recomputed hash. (Documented limitation; the
  agreed design reaches old dirs via --result-dir anyway.)

Idempotent: skips any dir that already has a manifest.json. Dry-run by
default; pass --write to actually create files.

    python backfill_manifests.py            # dry run, list what it'd do
    python backfill_manifests.py --write     # create the manifests
"""

import argparse
import glob
import json
import os

MANIFEST_FILE = "manifest.json"


def _parse_name(name: str) -> dict:
    """Pull `--key-value` segments out of an old config_name. The
    leading `algo[--level-N]` and any value-less head segments are kept
    under `_head`. Best-effort: old names are `algo--k1-v1--k2-v2--...`,
    but some segments (the model name, `tmpl-custom`) aren't strict
    key-value, so anything not matching `k-v` is left in `_head`."""
    parts = name.split("--")
    head = [parts[0]] if parts else []
    parsed: dict = {}
    for seg in parts[1:]:
        # split on the LAST '-' so values like b-080 / dalpha-100.0 and
        # multi-dash model names are handled by falling through to head
        # when they don't look like a single k-v.
        if "-" in seg:
            k, _, v = seg.partition("-")
            # treat as k-v only when k is a short alpha tag; otherwise
            # it's part of a model name etc. -> head.
            if k.isalpha() and len(k) <= 8:
                parsed[k] = v
                continue
        head.append(seg)
    return {"_head": head, "parsed_from_name": parsed}


def backfill(root_dir: str, write: bool) -> None:
    pattern = f"{root_dir}/results/*/*/*"
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    n_skip = n_new = 0
    for d in sorted(run_dirs):
        name = os.path.basename(d)
        if name == ".ipynb_checkpoints":
            continue
        manifest_path = f"{d}/{MANIFEST_FILE}"
        if os.path.exists(manifest_path):
            n_skip += 1
            continue
        info = _parse_name(name)
        payload = {
            "config_name": name,
            "config_hash": None,   # see module docstring: unrecoverable
            "config_identity": None,
            "parsed_from_name": info["parsed_from_name"],
            "backfilled": True,
        }
        n_new += 1
        if write:
            tmp = manifest_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fout:
                json.dump(payload, fout, indent=2)
                fout.write("\n")
            os.replace(tmp, manifest_path)
            print(f"wrote  {manifest_path}")
        else:
            print(f"would write  {manifest_path}")
    verb = "wrote" if write else "would write"
    print(f"\n{verb} {n_new} manifest(s); skipped {n_skip} with one.")
    if not write and n_new:
        print("(dry run — re-run with --write to create them)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default=os.getcwd(),
        help="repo root (defaults to cwd); results/ is under it",
    )
    ap.add_argument(
        "--write", action="store_true",
        help="actually write manifests (default: dry run)",
    )
    args = ap.parse_args()
    backfill(args.root, args.write)
