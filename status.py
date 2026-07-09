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
  python status.py --planned --commands # launch cmds for planned entries
  python status.py --verify             # assert all hashes still match
  python status.py --check mcts_cnt_prm800k llm=qwen_3b prm=qwen_prm \
                          search.gen_budget=320   # probe a candidate cell
"""
import argparse
import contextlib
import glob
import json
import os
import sys

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from utils.configs import (
    ExpConfig, MCTSCntConfig, MCTSSemV01Config, MCTSSemV02Config,
    BLMCTSCntConfig, BLMCTSCntV02Config, BLMCTSCntV03Config,
    BLMCTSSemConfig,
    config_hash, config_name, level_dir, results_root, MANIFEST_FILE,
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
_cs.store(
    group="search", name="mcts_bl_cnt_v01_schema", node=BLMCTSCntConfig,
)
_cs.store(
    group="search", name="mcts_bl_cnt_v02_schema", node=BLMCTSCntV02Config,
)
_cs.store(
    group="search", name="mcts_bl_cnt_v03_schema", node=BLMCTSCntV03Config,
)
_cs.store(
    group="search", name="mcts_bl_sem_v01_schema", node=BLMCTSSemConfig,
)


def compose_cfg(config_root, overrides):
    """Resolve a cfg from {root config name, override list} via Hydra
    compose -- no model load, no engine build. Mirrors the launcher's
    compose so config_hash()/config_name() match the recorded dir.

    Hydra's override parser pulls in an antlr4 runtime that prints a
    benign version-mismatch line to STDOUT on each parse; redirect that
    to stderr so --check / --commands stdout stays machine-parseable."""
    with contextlib.redirect_stdout(sys.stderr):
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            return compose(config_name=config_root, overrides=overrides)


def find_dir_by_hash(cfg):
    """Result dir whose manifest records this cfg's hash, or None.
    Same contract as utils.configs.find_run_dir but rooted here."""
    target = config_hash(cfg)
    parent = f"{RESULTS_DIR}/{results_root(cfg)}/{level_dir(cfg)}"
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
# the recorded method.
_METHOD_TO_ROOT = {
    "mcts_cnt": "mcts_cnt_prm800k",
    "mcts_cnt_v01": "mcts_cnt_prm800k",
    "mcts_sem_v01": "mcts_sem_v01_prm800k",
    "mcts_sem_v02": "mcts_sem_v02_prm800k",
    "mcts_bl_cnt_v01": "mcts_bl_cnt_v01_prm800k",
    "mcts_bl_cnt_v02": "mcts_bl_cnt_v02_prm800k",
    "mcts_bl_cnt_v03": "mcts_bl_cnt_v03_prm800k",
    "mcts_bl_sem_v01": "mcts_bl_sem_v01_prm800k",
}

_METHOD_TO_LAUNCHER = {
    "mcts_cnt": "generate_mcts_cnt.py",
    "mcts_cnt_v01": "generate_mcts_cnt.py",
    "mcts_sem_v01": "generate_mcts_sem.py",
    "mcts_sem_v02": "generate_mcts_sem.py",
    "mcts_bl_cnt_v01": "generate_mcts_bl_cnt.py",
    "mcts_bl_cnt_v02": "generate_mcts_bl_cnt.py",
    "mcts_bl_cnt_v03": "generate_mcts_bl_cnt.py",
    "mcts_bl_sem_v01": "generate_mcts_sem.py",
}

# method= -> the `group:` label used by docs/exp-comparison.md's `###`
# subsection names, so --group filtering and backfilled entries agree
# with the doc. A plain `startswith("mcts_sem")` guess (the old
# behavior) misclassifies mcts_bl_sem_v01 as "sem-mcts" -- explicit
# per-method mapping avoids that.
_METHOD_TO_GROUP = {
    "mcts_cnt": "cnt-mcts",
    "mcts_cnt_v01": "cnt-mcts",
    "mcts_sem_v01": "sem-mcts",
    "mcts_sem_v02": "sem-mcts",
    "mcts_bl_cnt_v01": "cnt-mcts-bl",
    "mcts_bl_cnt_v02": "cnt-mcts-bl",
    "mcts_bl_cnt_v03": "cnt-mcts-bl",
    "mcts_bl_sem_v01": "sem-mcts-bl",
}

# Group SELECTOR maps: a manifest records resolved *values*, not which
# group file produced them, so to emit `<group>=<file>` we map the
# group's identifying value back to its conf/<group>/<file>.yaml. Keyed
# by the field that uniquely identifies a file in that group.
#   llm  -> by llm.name      prm -> by prm.kind     data -> by data.name
# Derived from conf/<group>/*.yaml; keep in sync if files are added.
_LLM_BY_NAME = {
    "Llama3.2-1B-Instruct": "llama_1b",
    "Llama3.2-3B-Instruct": "llama_3b",
    "Llama3.2-3B-Instruct-GPTQ": "llama_3b_gptq",
    "Qwen2.5-3B-Instruct": "qwen_3b",
    "Qwen2.5-3B-Instruct-GPTQ-Int4": "qwen_3b_gptq_int4",
    "Qwen2.5-7B-Instruct-GPTQ-Int4": "qwen_7b_gptq_int4",
    "Qwen2.5-Math-1.5B-Instruct": "qwen_math_1_5b",
    "Qwen2.5-Math-7B-Instruct": "qwen_math_7b",
}
_PRM_BY_KIND = {"rlhflow": "llama_prm", "qwen": "qwen_prm"}
_DATA_BY_NAME = {"prm800k": "prm800k"}

# How to select a group file from its config_identity block: (id-field,
# value->file map). Other groups (search, gen) are plain field diffs.
_GROUP_SELECTORS = {
    "llm": ("name", _LLM_BY_NAME),
    "prm": ("kind", _PRM_BY_KIND),
    "data": ("name", _DATA_BY_NAME),
}


def derive_overrides(root, ident):
    """Reconstruct the minimal Hydra overrides that turn `root`'s
    defaults into the recorded `config_identity`. Returns (overrides
    dict, warnings list). For llm/prm/data, a changed identifying value
    becomes a group swap (`llm=qwen_3b`); any remaining in-group field
    diffs become `llm.field=val`. For search/gen, every differing field
    is a `group.field=val` override. Verified by the caller via hash
    round-trip."""
    ref = compose_cfg(root, [])
    overrides = {}
    warns = []
    # First pass: resolve any group SWAPS (llm/prm/data file selection),
    # so the per-field baseline below is taken against the SELECTED file,
    # not the root default -- otherwise a field like llm.enforce_eager
    # would be diffed against the wrong group's value.
    for group, (id_field, vmap) in _GROUP_SELECTORS.items():
        gvals = ident.get(group)
        if not isinstance(gvals, dict):
            continue
        cur_id = gvals.get(id_field)
        ref_id = getattr(getattr(ref, group, None), id_field, None)
        if cur_id != ref_id:
            fname = vmap.get(cur_id)
            if fname is None:
                warns.append(
                    f"{group}.{id_field}={cur_id!r} has no group file")
            else:
                overrides[group] = fname
    # Recompose with the swaps applied: this is the correct per-field
    # baseline for ALL groups (a swap changes its own group's defaults).
    swap_list = [f"{g}={f}" for g, f in overrides.items()]
    ref = compose_cfg(root, swap_list)
    # Second pass: every group's remaining field diffs become
    # group.field=val -- including non-selector fields of a swapped
    # group (e.g. llm.enforce_eager) and the plain groups (search, gen).
    for group, gvals in ident.items():
        if not isinstance(gvals, dict):
            continue
        ref_g = getattr(ref, group, None)
        id_field = _GROUP_SELECTORS.get(group, (None,))[0]
        for k, v in gvals.items():
            if k == id_field:
                continue  # the selector itself is handled by the swap
            ref_v = getattr(ref_g, k, None) if ref_g is not None else None
            if v != ref_v:
                overrides[f"{group}.{k}"] = v
    return overrides, warns


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

        overrides, warns = derive_overrides(root, ident)
        # Verify: re-compose with the derived overrides and check the
        # hash round-trips back to the recorded one. A mismatch means
        # an override couldn't be reconstructed (e.g. a missing group
        # file) -- flag it instead of emitting a wrong entry silently.
        ov_list = [f"{k}={v}" for k, v in overrides.items()]
        try:
            verified = config_hash(compose_cfg(root, ov_list)) == h
        except Exception as ex:
            verified = False
            warns.append(f"re-compose failed: {type(ex).__name__}: {ex}")

        entry = {
            "launcher": _METHOD_TO_LAUNCHER[method],
            "config_root": root,
            "overrides": overrides,
            "trials": n_done or 1,
            "feeds": [],
            "group": _METHOD_TO_GROUP.get(method, "cnt-mcts"),
            "recorded": False,
            "_backfilled_from": os.path.basename(rdir),
            "_config_hash": h,
            "_run_id": rec.get("run_id"),
            "_verified": verified,
        }
        if warns:
            entry["_warnings"] = warns
        new.append(entry)
    return new


# ------------------------------------------------------------------ #
# Launch-command emitter. Assemble the exact command for an entry     #
# from its own fields, so the command you run is guaranteed to match  #
# the entry status.py reconciles (no hand-transcription gap).         #
# ------------------------------------------------------------------ #
def launch_command(entry):
    parts = [
        "python", entry["launcher"],
        "--config-name", entry["config_root"],
    ]
    parts += entry.get("overrides_list", [])
    parts.append(f"run.num_trials={entry.get('trials', 1)}")
    return " ".join(parts)


# ------------------------------------------------------------------ #
# --check: compose a CANDIDATE override set (not yet in the ledger)   #
# and report its identity + whether it already exists. This is the    #
# per-cell primitive the exp-new-comparison-table skill calls, not    #
# re-deriving hashes itself -- one source of truth for compose/hash.  #
# ------------------------------------------------------------------ #
def check_candidate(config_root, overrides, queue):
    cfg = compose_cfg(config_root, overrides)
    h = config_hash(cfg)
    name = config_name(cfg)
    rdir = find_dir_by_hash(cfg)
    on_disk = rdir is not None
    n_done = count_done(rdir) if on_disk else 0
    collision = False
    for e in queue:
        try:
            if config_hash(compose_cfg(
                    e["config_root"], e.get("overrides_list", []))) == h:
                collision = True
                break
        except Exception:
            pass
    return {
        "hash": h, "name": name, "on_disk": on_disk,
        "n_done": n_done, "ledger_collision": collision,
        "dir": os.path.basename(rdir) if on_disk else None,
    }


# ------------------------------------------------------------------ #
# --verify: re-compose every ledger entry and assert its hash still   #
# matches what's recorded (the dir it was backfilled from, if any).   #
# Tripwire for launcher/config drift desyncing this script's mirrored #
# ConfigStore from the real launchers. Run after any launcher edit.   #
# ------------------------------------------------------------------ #
def verify_queue(queue):
    problems = []
    for e in queue:
        label = e.get("note") or e.get("from_dir") or str(e.get("overrides"))
        try:
            h = config_hash(compose_cfg(
                e["config_root"], e.get("overrides_list", [])))
        except Exception as ex:
            problems.append((label, f"compose failed: "
                                    f"{type(ex).__name__}: {ex}"))
            continue
        # If the entry records the dir it came from, that dir's name
        # encodes its cfg hash suffix (cfg-XXXXXXXX); check it matches.
        from_dir = e.get("from_dir")
        if from_dir and "cfg-" in from_dir:
            recorded_h = from_dir.rsplit("cfg-", 1)[1]
            if not h.startswith(recorded_h) and not recorded_h.startswith(h):
                problems.append(
                    (label, f"hash drift: composes to {h}, "
                            f"from_dir says cfg-{recorded_h}"))
    return problems


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
    ap.add_argument("--commands", action="store_true",
                    help="print launch commands for matching entries")
    ap.add_argument("--verify", action="store_true",
                    help="re-compose every entry, assert hashes still match")
    ap.add_argument("--check", nargs="+", metavar="ROOT OVERRIDE",
                    help="compose a candidate (config_root key=val ...) and "
                         "report hash/name/on-disk/collision; does not write")
    args = ap.parse_args()

    queue = load_queue()

    if args.check:
        root, overrides = args.check[0], args.check[1:]
        info = check_candidate(root, overrides, queue)
        print(yaml.safe_dump(info, sort_keys=False, allow_unicode=True),
              end="")
        return

    if args.verify:
        problems = verify_queue(queue)
        if not problems:
            print(f"# OK: all {len(queue)} entries compose and match "
                  f"their recorded hash")
            return
        print(f"# {len(problems)} problem(s):")
        for label, msg in problems:
            print(f"  ! {label}: {msg}")
        sys.exit(1)

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

    if args.commands:
        for e, status, detail in rows:
            print(f"# [{status}] {detail.get('name') or e['config_root']}")
            print(launch_command(e))
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
