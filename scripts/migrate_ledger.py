"""One-time migration: experiments.yaml + orchestration/queue.yaml
-> per-doc ledgers under experiments/ (workflow v2).

Merges the 225-entry append-only ledger with the 151-entry
orchestration queue into per-doc ledger files, one per tracking
doc (experiments/<stem>.yaml <-> docs/exp-comp-<stem>.md), seeding
each entry's lifecycle `status` (planned | inqueue | running |
scored | failed) from disk truth.

Dry-run by default: prints the full verification report and writes
nothing. --apply writes experiments/*.yaml (old files untouched --
they are retired in a later, separately reviewed step).

Usage:
    python scripts/migrate_ledger.py            # dry run + report
    python scripts/migrate_ledger.py --apply    # write experiments/
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import status as st  # noqa: E402  (reuse compose/hash machinery)
from utils.configs import config_hash  # noqa: E402

QUEUE_YAML = f"{st.ROOT}/orchestration/queue.yaml"
OUT_DIR = st.LEDGER_DIR

# Composed (data name, level) -> ledger stem. Level only matters
# for prm800k; gsm8k/aime don't use MATH levels.
def doc_stem(data_name, level):
    if data_name == "prm800k":
        if level == 5:
            return "prm800k-level5"
        if level == 4:
            return "prm800k-level4"
        return "misc"
    if data_name in ("gsm8k", "aime2025", "aime2024"):
        return "gsm8k" if data_name == "gsm8k" else "aime2025"
    return "misc"


LEDGER_HEADER = """\
# Per-doc experiment ledger (workflow v2) -- one entry per
# launchable run, cradle to grave. Companion doc:
#   docs/exp-comp-{stem}.md
#
# status: planned | inqueue | running | scored | failed
#   planned  -> table cell exists, not queued yet
#   inqueue  -> Tuan queued it (priority: lower drains first)
#   running  -> launched (launch: block records where)
#   scored   -> stats computed AND written into the doc
#   failed   -> crash/stall/missing detected by exp-check;
#               requeue (-> inqueue) only on Tuan's say-so
# `hash:` is written once at append time; --verify audits drift.
# Commands are DERIVED (launcher + config_root + overrides +
# trials); never store a command string.
# Migrated {date} by scripts/migrate_ledger.py from
# experiments.yaml + orchestration/queue.yaml.
"""


def _compose_worker(entry):
    """(hash, data_name, level, default_trials, err) per entry."""
    try:
        cfg = st.compose_cfg(
            entry["config_root"], entry.get("overrides_list", []))
        return (config_hash(cfg), str(cfg.data.name),
                int(cfg.data.level) if cfg.data.level is not None
                else None,
                int(cfg.run.num_trials), None)
    except Exception as ex:
        return None, None, None, None, f"{type(ex).__name__}: {ex}"


_LAUNCHER_TO_GROUP_HINTS = [
    ("mcts_bl_kube_v02", "kube-mcts-bl-v02"),
    ("mcts_bl_kube_v01", "kube-mcts-bl-v01"),
    ("mcts_bl_kdepth_v02", "kdepth-mcts-bl-v02"),
    ("mcts_bl_kdepth_v01", "kdepth-mcts-bl-v01"),
    ("mcts_bl_sem_v02", "sem-mcts-bl-v02"),
    ("mcts_bl_sem_v01", "sem-mcts-bl-v01"),
    ("mcts_bl_cnt_v02", "cnt-mcts-bl-v02"),
    ("mcts_bl_cnt_v01", "cnt-mcts-bl-v01"),
    ("mcts_sem", "sem-mcts"),
    ("mcts_cnt", "cnt-mcts"),
    ("bon", "bon"),
]


def group_from_root(config_root):
    for prefix, group in _LAUNCHER_TO_GROUP_HINTS:
        if config_root.startswith(prefix):
            return group
    return "misc"


_LAUNCHER_BY_ROOT_PREFIX = [
    ("mcts_bl_", "generate_mcts_bl_cnt.py"),
    ("mcts_sem", "generate_mcts_sem.py"),
    ("mcts_cnt", "generate_mcts_cnt.py"),
    ("bon", "generate_bon.py"),
]


def parse_queue_command(cmd):
    """Tokenize a queue `command:` string -> (launcher, config_root,
    overrides_dict, trials_or_None)."""
    toks = cmd.split()
    launcher = next(t for t in toks if t.endswith(".py"))
    root = toks[toks.index("--config-name") + 1]
    overrides, trials = {}, None
    for t in toks:
        if "=" not in t or t.startswith("--"):
            continue
        k, v = t.split("=", 1)
        if k == "run.num_trials":
            trials = int(v)
            continue
        if k.startswith("run."):
            continue  # run.* is hash-excluded; never store
        overrides[k] = yaml.safe_load(v)
    return launcher, root, overrides, trials


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write experiments/*.yaml (default: dry run)")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    # ---------- load both sources ----------
    with open(st.QUEUE_FILE, encoding="utf-8") as fin:
        ledger = yaml.safe_load(fin) or []
    for e in ledger:
        e["overrides_list"] = st.normalize_overrides(e.get("overrides"))
    with open(QUEUE_YAML, encoding="utf-8") as fin:
        queue = yaml.safe_load(fin) or []
    print(f"# loaded: {len(ledger)} ledger entries, "
          f"{len(queue)} queue entries")

    # ---------- manifest index: hash -> (dir, n_done) ----------
    # Disk truth located by MANIFEST hash, not recompose, so
    # drifted legacy entries still find their dirs.
    manifest_by_hash = {}
    import glob as _glob
    import json as _json
    for mpath in _glob.glob(
            f"{st.RESULTS_DIR}/*/*/*/{st.MANIFEST_FILE}"):
        if st._is_smoketest(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fin:
                m = _json.load(fin)
            h = str(m.get("config_hash"))
            d = os.path.dirname(mpath)
            manifest_by_hash[h] = (d, st.count_done(d))
        except Exception:
            continue

    # ---------- compose ledger entries (parallel) ----------
    composed = st._pmap(_compose_worker, ledger, args.jobs)
    drifted, compose_failed = [], []
    for e, (h, dname, lvl, dtrials, err) in zip(ledger, composed):
        label = e.get("note") or e.get("from_dir") or "?"
        if err:
            compose_failed.append((label, err))
            continue
        stored = str(e.get("hash") or h)
        if stored != h:
            drifted.append((label, stored, h))
        e["_hash"] = stored          # stored hash wins (links dirs)
        e["_data"], e["_level"] = dname, lvl
        if "trials" not in e:
            e["trials"] = dtrials
    if compose_failed:
        print(f"# FATAL: {len(compose_failed)} ledger entries fail "
              f"to compose:")
        for label, err in compose_failed:
            print(f"  ! {label}: {err}")
        sys.exit(1)
    if drifted:
        print(f"# WARN: {len(drifted)} drifted legacy entries "
              f"(stored hash kept -- links their real dirs):")
        for label, stored, fresh in drifted:
            print(f"  ~ {label}: stored {stored}, fresh {fresh}")

    # ---------- parse + compose queue commands ----------
    q_parsed = []
    for q in queue:
        launcher, root, ov, trials = parse_queue_command(q["command"])
        q_parsed.append({
            "queue": q, "launcher": launcher, "config_root": root,
            "overrides": ov, "trials": trials,
            "overrides_list": st.normalize_overrides(ov),
            "config_root_": root,
        })
    q_composed = st._pmap(
        _compose_worker,
        [{"config_root": p["config_root"],
          "overrides_list": p["overrides_list"]} for p in q_parsed],
        args.jobs)
    q_fail = 0
    for p, (h, dname, lvl, dtrials, err) in zip(q_parsed, q_composed):
        if err:
            print(f"  ! queue {p['queue']['id']}: compose failed: "
                  f"{err}")
            q_fail += 1
            continue
        p["hash"], p["data"], p["level"] = h, dname, lvl
        if p["trials"] is None:
            p["trials"] = dtrials
    if q_fail:
        print(f"# FATAL: {q_fail} queue commands fail to compose")
        sys.exit(1)

    dup_q = [h for h, c in Counter(
        p["hash"] for p in q_parsed).items() if c > 1]
    if dup_q:
        print(f"# WARN: {len(dup_q)} hashes queued more than once:")
        for h in dup_q:
            ids = [p["queue"]["id"] for p in q_parsed
                   if p["hash"] == h]
            print(f"  ~ {h}: {ids}")

    # ---------- merge queue into ledger by hash ----------
    by_hash = {e["_hash"]: e for e in ledger}
    n_matched = n_queue_only = 0
    new_entries = []
    for p in q_parsed:
        q = p["queue"]
        e = by_hash.get(p["hash"])
        if e is not None:
            n_matched += 1
            e["id"] = q["id"]
            if q.get("expected_hr") is not None:
                e["expected_hr"] = q["expected_hr"]
            if q.get("priority") is not None:
                e["priority"] = q["priority"]     # operational wins
            if q.get("launch"):
                e["launch"] = q["launch"]
            e["_qstatus"] = q.get("status")
        else:
            n_queue_only += 1
            ne = {
                "id": q["id"],
                "launcher": p["launcher"],
                "hash": p["hash"],
                "config_root": p["config_root"],
                "overrides": p["overrides"],
                "trials": p["trials"],
                "feeds": [],
                "group": group_from_root(p["config_root"]),
                "priority": q.get("priority"),
                "note": f"migrated from queue (id {q['id']}); "
                        f"feeds pending",
                "_hash": p["hash"], "_data": p["data"],
                "_level": p["level"], "_qstatus": q.get("status"),
            }
            if q.get("expected_hr") is not None:
                ne["expected_hr"] = q["expected_hr"]
            if q.get("launch"):
                ne["launch"] = q["launch"]
            new_entries.append(ne)
            by_hash[p["hash"]] = ne
    all_entries = ledger + new_entries

    # ---------- status seeding ----------
    for e in all_entries:
        rdir, n_done = manifest_by_hash.get(
            e["_hash"], (None, 0))
        if rdir is None and e.get("from_dir") and \
                "cfg-" in str(e.get("from_dir")):
            # Drifted legacy entry: its dir predates a re-hash;
            # the from_dir suffix still names the real artifacts.
            old_h = str(e["from_dir"]).rsplit("cfg-", 1)[1]
            rdir, n_done = manifest_by_hash.get(old_h, (None, 0))
        done = rdir is not None and n_done >= int(e.get("trials", 1))
        recorded = bool(e.pop("recorded", False))
        qstatus = e.pop("_qstatus", None)
        if done and recorded:
            e["status"] = "scored"
        elif done or (rdir is not None and n_done > 0):
            e["status"] = "running"     # exp-check settles it
        elif rdir is None and qstatus == "running":
            e["status"] = "running"     # exp-check -> missing/failed
        elif qstatus == "planned":
            e["status"] = "inqueue"
        else:
            e["status"] = "planned"

    # ---------- ids for ledger-only entries ----------
    for e in all_entries:
        if not e.get("id"):
            e["id"] = f"{e.get('group', 'misc')}-{e['_hash']}"
    dup_ids = [i for i, c in Counter(
        e["id"] for e in all_entries).items() if c > 1]
    if dup_ids:
        print(f"# WARN: {len(dup_ids)} duplicate ids -- suffixing")
        seen = Counter()
        for e in all_entries:
            if e["id"] in dup_ids:
                seen[e["id"]] += 1
                if seen[e["id"]] > 1:
                    e["id"] = f"{e['id']}-{seen[e['id']]}"

    # ---------- doc assignment + emit ----------
    files = defaultdict(list)
    for e in all_entries:
        files[doc_stem(e["_data"], e["_level"])].append(e)

    FIELD_ORDER = [
        "id", "launcher", "hash", "config_root", "overrides",
        "trials", "feeds", "group", "status", "priority",
        "expected_hr", "launch", "history", "note", "run_id",
        "from_dir",
    ]

    def clean(e):
        out = {}
        for k in FIELD_ORDER:
            if k in e and e[k] is not None:
                out[k] = e[k]
        out["hash"] = str(e["_hash"])
        return out

    # ---------- report ----------
    print("\n# ===== migration report =====")
    print(f"ledger entries:      {len(ledger)}")
    print(f"queue commands:      {len(q_parsed)} "
          f"(matched {n_matched} + queue-only {n_queue_only})")
    assert n_matched + n_queue_only == len(q_parsed)
    total = sum(len(v) for v in files.values())
    print(f"total out:           {total} "
          f"(= {len(ledger)} + {n_queue_only})")
    assert total == len(ledger) + len(new_entries)
    for stem in sorted(files):
        sc = Counter(e["status"] for e in files[stem])
        print(f"  {stem:16s} {len(files[stem]):4d}  {dict(sc)}")
    hist = Counter(e["status"] for e in all_entries)
    print(f"status histogram:    {dict(hist)}")
    n_launch = sum(1 for e in all_entries if e.get("launch"))
    print(f"launch blocks kept:  {n_launch}")
    bad_run_keys = [
        e["id"] for e in all_entries
        if any(str(k).startswith("run.")
               for k in (e.get("overrides") or {}))]
    print(f"run.* keys in overrides: {len(bad_run_keys)} "
          f"{'OK' if not bad_run_keys else bad_run_keys[:5]}")
    print(f"queue-only entries needing feeds: "
          f"{[e['id'] for e in new_entries][:10]}"
          f"{' ...' if len(new_entries) > 10 else ''}")

    if not args.apply:
        print("\n# DRY RUN -- nothing written. "
              "Re-run with --apply to write experiments/*.yaml")
        return

    import datetime
    os.makedirs(OUT_DIR, exist_ok=True)
    for stem, entries in sorted(files.items()):
        path = f"{OUT_DIR}/{stem}.yaml"
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(LEDGER_HEADER.format(
                stem=stem,
                date=datetime.date.today().isoformat()))
            fout.write("\n")
            yaml.safe_dump(
                [clean(e) for e in entries], fout,
                sort_keys=False, allow_unicode=True, width=72)
        print(f"# wrote {path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
