"""Reconcile declared experiment intent (experiments.yaml) against
on-disk artifacts and W&B, and print a per-entry status.

This is the read-only "computed state" layer. The yaml is the
append-only ledger of what you MEANT to run; this script overlays
what HAS run on top, deriving each entry's status from the result
dir's manifest + .done markers (+ W&B run state with --wandb).

Vault design note: research-coding-practices-guides/
tracking-experiment-status.

Statuses
--------
  planned   no result dir for this config hash yet
  partial   dir exists, fewer than `trials` .done markers, no W&B
            check (or --wandb says the run is still going)
  stalled   partial AND --wandb says the run is not running
            (crash / OOM / disconnect) -> relaunch the same command
  done      all `trials` .done markers present
  orphan    a result dir whose hash matches NO yaml entry
            (printed only by --backfill / --orphans)

Usage
-----
  python status.py                      # all entries, no W&B call
  python status.py --wandb              # also classify partial->stalled
  python status.py --status stalled
  python status.py --group sem-mcts
  python status.py --done --not-recorded
  python status.py --planned --priority 1
  python status.py --backfill           # emit yaml for orphan dirs
"""
import argparse
import glob
import json
import os
import sys

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from utils.configs import (
    ExpConfig, MCTSCntConfig, MCTSSemV01Config, MCTSSemV02Config,
    config_hash, config_name, level_dir, MANIFEST_FILE,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = f"{ROOT}/conf"
QUEUE_FILE = f"{ROOT}/experiments.yaml"
RESULTS_DIR = f"{ROOT}/results"


def _is_smoketest(path):
    """True for throwaway smoke-test artifacts -- a results/smoketest/
    dataset dir or any dir tagged with a --smoketest suffix. These are
    never real experiments, so they're excluded from reconciliation
    and backfill."""
    return "/smoketest/" in path or "smoketest" in os.path.basename(path)

# Register the structured schemas once, exactly as each launcher does,
# so compose() binds onto the typed dataclasses and the hash matches
# what the launcher would have written. Search subclasses go under the
# "search" group; conf/search/<method> selects one.
_cs = ConfigStore.instance()
_cs.store(name="exp_schema", node=ExpConfig)
_cs.store(group="search", name="mcts_cnt_schema", node=MCTSCntConfig)
_cs.store(group="search", name="mcts_sem_v01_schema", node=MCTSSemV01Config)
_cs.store(group="search", name="mcts_sem_v02_schema", node=MCTSSemV02Config)


def compose_cfg(config_root, overrides):
    """Resolve a cfg from {root config name, override list} via Hydra
    compose -- no model load, no engine build. Mirrors the launcher's
    compose so config_hash()/config_name() match the recorded dir."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=config_root, overrides=overrides)


def find_dir_by_hash(cfg):
    """Result dir whose manifest records this cfg's hash, or None.
    Same contract as utils.configs.find_run_dir but rooted here."""
    target = config_hash(cfg)
    parent = f"{RESULTS_DIR}/{cfg.data.name}/{level_dir(cfg)}"
    for mpath in glob.glob(f"{parent}/*/{MANIFEST_FILE}"):
        if _is_smoketest(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fin:
                rec = json.load(fin)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("config_hash") == target:
            return os.path.dirname(mpath)
    return None


def count_done(result_dir):
    """Number of completed-trial markers in a result dir. A `.done`
    is written only after the trial's raw results are dumped, so this
    is the authoritative completed-trial count (see generate_*.py)."""
    return len(glob.glob(f"{result_dir}/*--trial-*.done"))


def wandb_state(run_id):
    """W&B run state string (running/finished/crashed/killed/...) or
    None if unknown. Lazy import so the no-network path stays fast."""
    if not run_id:
        return None
    try:
        import wandb
        return wandb.Api().run(f"tnguyen10/llm-reasoning/{run_id}").state
    except Exception as e:  # network / auth / missing run
        return f"?({type(e).__name__})"


def classify(entry, check_wandb):
    """Derive (status, detail) for one yaml entry. Pure read; never
    writes. `detail` carries the dir basename + n_done/trials etc."""
    cfg = compose_cfg(entry["config_root"], entry.get("overrides_list", []))
    h = config_hash(cfg)
    name = config_name(cfg)
    trials = entry.get("trials", 1)
    result_dir = find_dir_by_hash(cfg)
    if result_dir is None:
        return "planned", {"hash": h, "name": name, "dir": None}

    n_done = count_done(result_dir)
    base = os.path.basename(result_dir)
    detail = {"hash": h, "name": name, "dir": base,
              "n_done": n_done, "trials": trials}
    if n_done >= trials:
        return "done", detail

    # Partial. Only a W&B check can tell "still running" from "stalled".
    if check_wandb:
        run_id = None
        try:
            with open(f"{result_dir}/{MANIFEST_FILE}", encoding="utf-8") as f:
                run_id = json.load(f).get("run_id")
        except (OSError, json.JSONDecodeError):
            pass
        state = wandb_state(run_id)
        detail["wandb"] = state
        if state == "running":
            return "partial", detail
        return "stalled", detail
    return "partial", detail


def normalize_overrides(raw):
    """Accept overrides as a dict {key: val} or a list ['key=val'];
    return the Hydra ['key=val', ...] form. A dict is friendlier to
    write by hand; group selection (llm=qwen_3b) and in-group fields
    (search.ds_alpha=1000) are both just key=value to Hydra."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [f"{k}={v}" for k, v in raw.items()]
    return list(raw)


def load_queue():
    if not os.path.exists(QUEUE_FILE):
        return []
    with open(QUEUE_FILE, encoding="utf-8") as fin:
        entries = yaml.safe_load(fin) or []
    for e in entries:
        e["overrides_list"] = normalize_overrides(e.get("overrides"))
    return entries


# ------------------------------------------------------------------ #
# Backfill: disk -> yaml. Scan every result manifest; for each whose  #
# hash is NOT already claimed by a queue entry, emit a ready-made     #
# entry. This is how existing runs get seeded without hand-           #
# transcription, and is the same scan that powers orphan detection.   #
# ------------------------------------------------------------------ #
def scan_result_manifests():
    out = {}
    for mpath in glob.glob(f"{RESULTS_DIR}/*/*/*/{MANIFEST_FILE}"):
        if _is_smoketest(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fin:
                rec = json.load(fin)
        except (OSError, json.JSONDecodeError):
            continue
        h = rec.get("config_hash")
        if h:
            out[h] = (os.path.dirname(mpath), rec)
    return out


# method= -> the root config file each launcher defaults to. Backfilled
# entries need a config_root so status.py can re-compose them; map by
# the recorded method. (Only sem/cnt families are in scope here.)
_METHOD_TO_ROOT = {
    "mcts_cnt": "mcts_cnt_prm800k",
    "mcts_sem_v01": "mcts_sem_v01_prm800k",
    "mcts_sem_v02": "mcts_sem_v02_prm800k",
}


def backfill(queue):
    claimed = set()
    for e in queue:
        try:
            claimed.add(config_hash(
                compose_cfg(e["config_root"], e.get("overrides_list", []))))
        except Exception:
            pass
    found = scan_result_manifests()
    new = []
    for h, (rdir, rec) in sorted(found.items()):
        if h in claimed:
            continue
        ident = rec.get("config_identity", {})
        method = ident.get("search", {}).get("method", "")
        root = _METHOD_TO_ROOT.get(method)
        if root is None:
            continue  # out of scope (e.g. bon, bl) -- skip silently
        n_done = count_done(rdir)
        new.append({
            "launcher": f"generate_{method}.py"
            if method != "mcts_sem_v02" and method != "mcts_sem_v01"
            else "generate_mcts_sem.py",
            "config_root": root,
            "overrides": "# TODO: fill from config_identity if re-runnable",
            "trials": n_done or 1,
            "feeds": [],
            "group": "sem-mcts" if method.startswith("mcts_sem")
            else "cnt-mcts",
            "recorded": False,
            "_backfilled_from": os.path.basename(rdir),
            "_config_hash": h,
            "_run_id": rec.get("run_id"),
        })
    return new


def matches_filters(entry, status, args):
    if args.status and status != args.status:
        return False
    if args.done and status != "done":
        return False
    if args.planned and status != "planned":
        return False
    if args.not_recorded and entry.get("recorded"):
        return False
    if args.group and entry.get("group") != args.group:
        return False
    if args.priority is not None and entry.get("priority") != args.priority:
        return False
    return True


_GLYPH = {
    "planned": " ", "partial": ".", "running": ">", "stalled": "!",
    "done": "x", "orphan": "?",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb", action="store_true",
                    help="query W&B to split partial into running/stalled")
    ap.add_argument("--status", help="only this status")
    ap.add_argument("--group", help="only this group (algorithm family)")
    ap.add_argument("--done", action="store_true", help="only done entries")
    ap.add_argument("--planned", action="store_true",
                    help="only planned entries")
    ap.add_argument("--not-recorded", action="store_true",
                    dest="not_recorded",
                    help="exclude entries already written into the doc")
    ap.add_argument("--priority", type=int, help="only this priority level")
    ap.add_argument("--backfill", action="store_true",
                    help="emit yaml entries for result dirs not in the queue")
    args = ap.parse_args()

    queue = load_queue()

    if args.backfill:
        new = backfill(queue)
        if not new:
            print("# no un-queued result dirs found (nothing to backfill)")
            return
        print(f"# {len(new)} un-queued result dir(s) -- review and append "
              f"to experiments.yaml:")
        # Strip the helper keys' leading underscore note for readability.
        print(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
        return

    rows = []
    for e in queue:
        try:
            status, detail = classify(e, args.wandb)
        except Exception as ex:
            status, detail = "ERROR", {"err": f"{type(ex).__name__}: {ex}"}
        if status != "ERROR" and not matches_filters(e, status, args):
            continue
        rows.append((e, status, detail))

    if not rows:
        print("# no entries match")
        return

    for e, status, detail in rows:
        g = _GLYPH.get(status, "?")
        label = detail.get("name") or e.get("config_root")
        nd = (f"  [{detail['n_done']}/{detail['trials']}]"
              if "n_done" in detail else "")
        wb = f"  wandb={detail['wandb']}" if "wandb" in detail else ""
        grp = e.get("group", "")
        rec = "  recorded" if e.get("recorded") else ""
        print(f"[{g}] {status:8s} {grp:10s} {label}{nd}{wb}{rec}")
        if status == "ERROR":
            print(f"      {detail['err']}", file=sys.stderr)


if __name__ == "__main__":
    main()
