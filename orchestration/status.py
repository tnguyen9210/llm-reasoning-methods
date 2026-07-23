"""Reconcile declared experiment intent (orchestration/ledgers/*.yaml)
against on-disk artifacts and W&B, and print a per-entry status.

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

Usage (from the repo root)
--------------------------
  python orchestration/status.py           # all entries, no W&B call
  python orchestration/status.py --wandb   # classify partial->stalled
  python orchestration/status.py --status stalled
  python orchestration/status.py --group sem-mcts
  python orchestration/status.py --done --not-recorded
  python orchestration/status.py --planned --priority 1
  python orchestration/status.py --backfill  # yaml for orphan dirs
  python orchestration/status.py --planned --commands  # launch cmds
  python orchestration/status.py --verify  # assert hashes still match
  python orchestration/status.py --check mcts_cnt_prm800k \
      llm=qwen_3b prm=qwen_prm search.gen_budget=320  # probe a cell
  python orchestration/status.py --verify --jobs 1  # force serial

Per-entry composes run in parallel forked workers by default
(min(48, workload) -- the nodes have 96 CPU threads, 48 assumed
safely available per Tuan's standing rule). --jobs overrides.

Entries carry a `hash:` field written once at append time (see the
exp-new-comparison-table skill): membership/collision fast paths
read it instead of recomposing; --verify NEVER trusts it -- it
recomposes and flags any stored-vs-fresh mismatch as drift.
--group/--priority/--not-recorded scope which entries get composed
at all (entry-level pre-filtering), including for --verify.
"""
import argparse
import contextlib
import glob
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

# status.py lives in orchestration/; the repo root (conf/, docs/,
# results/, utils/) is one level up. Put it on sys.path so repo
# imports resolve when invoked as `python orchestration/status.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.configs import (  # noqa: E402
    ExpConfig, MCTSCntConfig, MCTSSemV01Config, MCTSSemV02Config,
    BLMCTSCntConfig, BLMCTSCntV02Config,
    BLMCTSKubeV01Config, BLMCTSKubeV02Config,
    BLMCTSKdepthV01Config, BLMCTSKdepthV02Config,
    BLMCTSSemConfig, BLMCTSSemV02Config,
    config_hash, config_name, level_dir, results_root, MANIFEST_FILE,
)

CONF_DIR = f"{ROOT}/conf"
QUEUE_FILE = f"{ROOT}/experiments.yaml"   # legacy single-file ledger
LEDGER_DIR = f"{ROOT}/orchestration/ledgers"   # per-doc ledgers (v2)
RESULTS_DIR = f"{ROOT}/results"

# Ledger stem <-> tracking doc. `misc` collects entries with no
# current doc (legacy levels, parked work); it has no doc to sync.
LEDGER_DOC = {
    "prm800k-level5": f"{ROOT}/docs/exp-comp-prm800k-level5.md",
    "prm800k-level4": f"{ROOT}/docs/exp-comp-prm800k-level4.md",
    "gsm8k": f"{ROOT}/docs/exp-comp-gsm8k.md",
    "aime2025": f"{ROOT}/docs/exp-comp-aime2025.md",
    "misc": None,
}

# Lifecycle statuses stored on v2 ledger entries (the `status:`
# field). Distinct from the COMPUTED statuses in the module
# docstring (planned/partial/stalled/done), which reconcile disk
# state and remain the default listing's vocabulary.
LIFECYCLE_STATUSES = ("planned", "inqueue", "running", "scored", "failed")

# Parallelism: the compute nodes provide 96 CPU threads; default to
# a conservative 48-worker cap (Tuan's standing rule, 2026-07-19).
# Override with --jobs N (--jobs 1 = serial, the debugging path).
# Workers are FORKED, so they inherit the already-imported Hydra +
# schema state and skip the ~10s per-process import tax that made
# serial per-entry composing slow.
DEFAULT_JOBS_CAP = 48


def _pmap(fn, items, jobs=None):
    """Order-preserving parallel map over `items` in forked worker
    processes. `fn` must be a module-level function and must not
    raise (workers return error markers instead). Serial fast path
    for singleton workloads or jobs=1."""
    items = list(items)
    n = jobs if jobs is not None else min(DEFAULT_JOBS_CAP, len(items))
    if n <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    ctx = multiprocessing.get_context("fork")
    chunk = max(1, len(items) // (n * 4))
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


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
    group="search", name="mcts_bl_kube_v01_schema", node=BLMCTSKubeV01Config,
)
_cs.store(
    group="search", name="mcts_bl_kube_v02_schema", node=BLMCTSKubeV02Config,
)
_cs.store(
    group="search", name="mcts_bl_kdepth_v01_schema",
    node=BLMCTSKdepthV01Config,
)
_cs.store(
    group="search", name="mcts_bl_kdepth_v02_schema",
    node=BLMCTSKdepthV02Config,
)
_cs.store(
    group="search", name="mcts_bl_sem_v01_schema", node=BLMCTSSemConfig,
)
_cs.store(
    group="search", name="mcts_bl_sem_v02_schema", node=BLMCTSSemV02Config,
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


def count_scored(result_dir, run_name):
    """Number of SCORED trial datasets in a result dir. The scored
    file is `{run_name}--trial-NNN.jsonl` (no `generate_` prefix --
    that's the raw dump). Scoring happens inline in the launcher or
    via prepare_scored_dataset.py."""
    return len(glob.glob(f"{result_dir}/{run_name}--trial-*.jsonl"))


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


def load_ledgers():
    """Load every per-doc ledger under orchestration/ledgers/ into
    one flat list, annotating each entry with `_ledger` (its file
    stem) and `overrides_list`. Falls back to the legacy single-file
    experiments.yaml (stem "legacy") when the ledgers dir is absent,
    so this works before, during, and after the v2 migration."""
    entries = []
    paths = sorted(glob.glob(f"{LEDGER_DIR}/*.yaml"))
    if paths:
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(path, encoding="utf-8") as fin:
                part = yaml.safe_load(fin) or []
            for e in part:
                e["_ledger"] = stem
            entries += part
    elif os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, encoding="utf-8") as fin:
            entries = yaml.safe_load(fin) or []
        for e in entries:
            e["_ledger"] = "legacy"
    for e in entries:
        e["overrides_list"] = normalize_overrides(e.get("overrides"))
    return entries


def load_queue():
    """Back-compat alias; the v2 name is load_ledgers()."""
    return load_ledgers()


def _entry_hash(entry):
    """(hash, error) for one ledger entry. Worker-safe: never
    raises, returns the error as a string instead."""
    try:
        cfg = compose_cfg(
            entry["config_root"], entry.get("overrides_list", []))
        return config_hash(cfg), None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


def entry_hashes(queue, jobs=None, use_stored=False):
    """Per-entry (hash, error) pairs, order-aligned with `queue`.
    THE shared parallel primitive: --check's collision scan,
    --verify, and --backfill's claimed-set all need every entry's
    hash -- computing them once here, in parallel, keeps each of
    those O(ledger) passes at wall-clock O(ledger / workers).

    use_stored=True trusts each entry's `hash:` field (written once
    at append time) and composes only entries that lack one -- the
    fast path for membership/collision questions. NEVER pass it for
    drift checking: a stored hash is a snapshot, and detecting that
    it no longer matches a fresh compose is --verify's entire job.
    """
    if not use_stored:
        return _pmap(_entry_hash, queue, jobs)
    out = [None] * len(queue)
    missing = []
    for i, e in enumerate(queue):
        h = e.get("hash")
        if h:
            out[i] = (str(h), None)
        else:
            missing.append(i)
    computed = _pmap(_entry_hash, [queue[i] for i in missing], jobs)
    for i, pair in zip(missing, computed):
        out[i] = pair
    return out


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
    "mcts_bl_kube_v01": "mcts_bl_kube_v01_prm800k",
    "mcts_bl_kube_v02": "mcts_bl_kube_v02_prm800k",
    "mcts_bl_kdepth_v01": "mcts_bl_kdepth_v01_prm800k",
    "mcts_bl_kdepth_v02": "mcts_bl_kdepth_v02_prm800k",
    "mcts_bl_sem_v01": "mcts_bl_sem_v01_prm800k",
    "mcts_bl_sem_v02": "mcts_bl_sem_v02_prm800k",
}

_METHOD_TO_LAUNCHER = {
    "mcts_cnt": "generate_mcts_cnt.py",
    "mcts_cnt_v01": "generate_mcts_cnt.py",
    "mcts_sem_v01": "generate_mcts_sem.py",
    "mcts_sem_v02": "generate_mcts_sem.py",
    "mcts_bl_cnt_v01": "generate_mcts_bl_cnt.py",
    "mcts_bl_cnt_v02": "generate_mcts_bl_cnt.py",
    "mcts_bl_kube_v01": "generate_mcts_bl_cnt.py",
    "mcts_bl_kube_v02": "generate_mcts_bl_cnt.py",
    "mcts_bl_kdepth_v01": "generate_mcts_bl_cnt.py",
    "mcts_bl_kdepth_v02": "generate_mcts_bl_cnt.py",
    "mcts_bl_sem_v01": "generate_mcts_sem.py",
    "mcts_bl_sem_v02": "generate_mcts_sem.py",
}

# method= -> the `group:` label used by the docs/exp-comp-*.md files'
# `###` subsection names, so --group filtering and backfilled entries
# agree with the docs. A plain `startswith("mcts_sem")` guess (the old
# behavior) misclassifies mcts_bl_sem_v01 as "sem-mcts" -- explicit
# per-method mapping avoids that.
#
# mcts_bl_kube_v01 and mcts_bl_kdepth_v01 are each their own algorithm
# family (fractional-KUBE / knapsack-depth-shaping are different
# selection criteria from bl_cnt's PUCT, not same-family sibling
# versions) -- their group labels sit off the shared "cnt-mcts-bl"
# bucket accordingly.
#
# Each family's v02 (eager terminal backprop -- see
# docs/decisions/bl-cnt-v02-eager-backprop-path-aware.md) gets its OWN
# group label rather than sharing its v01's (Tuan's call, 2026-07-18 --
# the mcts_sem_v01/v02 precedent of sharing a group was considered and
# explicitly not followed here): v02 changes the algorithm (eager vs.
# lazy backprop timing, plus a path-aware blend for cnt/kube-parent)
# enough to warrant its own docs/exp-comp-*.md subsection/table rather
# than comparing as a same-table row alongside v01.
_METHOD_TO_GROUP = {
    "mcts_cnt": "cnt-mcts",
    "mcts_cnt_v01": "cnt-mcts",
    "mcts_sem_v01": "sem-mcts",
    "mcts_sem_v02": "sem-mcts",
    "mcts_bl_cnt_v01": "cnt-mcts-bl",
    "mcts_bl_cnt_v02": "cnt-mcts-bl-v02",
    "mcts_bl_kube_v01": "kube-mcts-bl",
    "mcts_bl_kube_v02": "kube-mcts-bl-v02",
    "mcts_bl_kdepth_v01": "kdepth-mcts-bl",
    "mcts_bl_kdepth_v02": "kdepth-mcts-bl-v02",
    "mcts_bl_sem_v01": "sem-mcts-bl",
    "mcts_bl_sem_v02": "sem-mcts-bl-v02",
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


def _backfill_worker(item):
    """Build one backfill entry from a (hash, dir, manifest-record)
    tuple; None for out-of-scope methods (e.g. bon, bl). Worker-safe:
    derive_overrides' composes + the verifying re-compose are the
    per-item cost this function exists to parallelize."""
    h, rdir, rec = item
    ident = rec.get("config_identity", {})
    method = ident.get("search", {}).get("method", "")
    root = _METHOD_TO_ROOT.get(method)
    if root is None:
        return None
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
    return entry


def backfill(queue, jobs=None):
    claimed = {h for h, _ in entry_hashes(queue, jobs, use_stored=True)
               if h}
    found = scan_result_manifests()
    orphans = [(h, rdir, rec) for h, (rdir, rec) in sorted(found.items())
               if h not in claimed]
    entries = _pmap(_backfill_worker, orphans, jobs)
    return [e for e in entries if e is not None]


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
def check_candidate(config_root, overrides, queue, jobs=None,
                    with_matches=False):
    """--check / --dedup primitive. with_matches=True (--dedup)
    additionally reports every ledger entry sharing the candidate's
    hash: which ledger file, its id, lifecycle status, and feeds --
    so the exp-tables skill can REUSE the run (copy status +
    numbers from the fed doc cell) instead of double-queueing."""
    cfg = compose_cfg(config_root, overrides)
    h = config_hash(cfg)
    name = config_name(cfg)
    rdir = find_dir_by_hash(cfg)
    on_disk = rdir is not None
    n_done = count_done(rdir) if on_disk else 0
    hashes = entry_hashes(queue, jobs, use_stored=True)
    collision = any(eh == h for eh, _ in hashes)
    info = {
        "hash": h, "name": name, "on_disk": on_disk,
        "n_done": n_done, "ledger_collision": collision,
        "dir": os.path.basename(rdir) if on_disk else None,
    }
    if with_matches:
        info["matches"] = [
            {"ledger": e.get("_ledger"), "id": e.get("id"),
             "status": e.get("status"),
             "feeds": e.get("feeds") or []}
            for e, (eh, _) in zip(queue, hashes) if eh == h
        ]
        if on_disk:
            info["n_scored"] = count_scored(rdir, name)
    return info


# ------------------------------------------------------------------ #
# --check-running: verdict every `running` entry. THE exp-check      #
# skill's entire input -- it never reads raw ledgers. Verdicts:      #
#   missing        no result dir for this hash                       #
#   finished       n_done >= trials (detail: scored=k/N, wandb)      #
#   still-running  W&B says the run is alive -- ALWAYS untouched     #
#   stalled        dir exists, incomplete, W&B not running           #
# ------------------------------------------------------------------ #
def _verdict_worker(entry):
    """entry -> dict row. Worker-safe: exceptions become an ERROR
    verdict. Uses load_wandb_run_id (manifest + legacy sidecar),
    NOT a manifest-only read."""
    from utils.configs import load_wandb_run_id
    try:
        cfg = compose_cfg(
            entry["config_root"], entry.get("overrides_list", []))
        name = config_name(cfg)
        trials = int(entry.get("trials", 1))
        rdir = find_dir_by_hash(cfg)
        row = {
            "id": entry.get("id") or entry.get("note") or "?",
            "ledger": entry.get("_ledger"),
            "trials": trials,
        }
        if rdir is None:
            row.update(verdict="missing", n_done=0, scored=0,
                       wandb=None, dir=None)
            return row
        n_done = count_done(rdir)
        n_scored = count_scored(rdir, name)
        run_id = load_wandb_run_id(rdir)
        state = wandb_state(run_id)
        row.update(n_done=n_done, scored=n_scored, wandb=state,
                   dir=os.path.basename(rdir))
        if n_done >= trials:
            row["verdict"] = "finished"
        elif state == "running":
            row["verdict"] = "still-running"
        else:
            row["verdict"] = "stalled"
        return row
    except Exception as ex:
        return {"id": entry.get("id") or "?", "verdict": "ERROR",
                "err": f"{type(ex).__name__}: {ex}"}


def check_running(entries, jobs=None):
    """Verdict rows for every entry with lifecycle status
    `running`, in ledger order."""
    running = [e for e in entries if e.get("status") == "running"]
    return _pmap(_verdict_worker, running, jobs)


# ------------------------------------------------------------------ #
# --sync-doc: rewrite table STATUS CELLS from ledger truth.           #
# Conservative v1 (the one status.py writer, and it writes docs      #
# only): syncs ONLY tables identified by a stable table-id and/or a   #
# feeds key, and ONLY rows that match exactly one entry (bijective).  #
# Never touches numbers; never downgrades a scored cell (reported as  #
# a mismatch instead). Everything unmatched is listed, never guessed. #
# ------------------------------------------------------------------ #
_DOC_LLM_ALIASES = {
    "llama-1b": "llama_1b", "llama-3b": "llama_3b",
    "llama-3b-gptq": "llama_3b_gptq",
    "llama-3b gptq": "llama_3b_gptq",
    "qwen-3b": "qwen_3b",
    "qwen-3b gptq-int4": "qwen_3b_gptq_int4",
    "qwen-7b gptq-int4": "qwen_7b_gptq_int4",
    "qwen-math-1.5b": "qwen_math_1_5b",
    "qwen-math-7b": "qwen_math_7b",
    # prm column shorthands (doc cells name the PRM bare; rlhflow
    # is the llama_prm group's display nickname)
    "qwen": "qwen_prm",
    "rlhflow": "llama_prm",
}

_FEEDS_RE = None  # compiled lazily in _section_feeds_key


def _section_feeds_key(section_text):
    """Extract the feeds key a table's blockquote names, or None.
    Recognized shapes: 'feeds `key`', 'feeds\\n> `key`',
    'experiments.yaml group `g`, feeds `key`'."""
    import re
    global _FEEDS_RE
    if _FEEDS_RE is None:
        _FEEDS_RE = re.compile(
            r"feeds\s*(?:>\s*)?\n?\s*(?:>\s*)?`([\w./-]+)`")
    m = _FEEDS_RE.search(section_text)
    return m.group(1) if m else None


# Stable table IDs (2026-07-23): every #### table carries an opaque
# `<!-- table-id: tbl-xxxxxx -->` line directly under its heading,
# minted once (--mint-table-ids) and never renamed. Ledger `feeds`
# may name either the tbl-id or the legacy human feeds key; the
# matcher accepts both, so human keys can be relabeled or dropped
# without breaking the ledger<->doc join. Design:
# docs/decisions/stable-table-ids.md.
_TABLE_ID_RE = None  # compiled lazily in _section_table_id


def _section_table_id(section_text):
    """Extract the stable `<!-- table-id: tbl-xxxxxx -->` a table
    carries, or None."""
    import re
    global _TABLE_ID_RE
    if _TABLE_ID_RE is None:
        _TABLE_ID_RE = re.compile(
            r"<!--\s*table-id:\s*(tbl-[0-9a-f]{6,8})\s*-->")
    m = _TABLE_ID_RE.search(section_text)
    return m.group(1) if m else None


def _norm_cell(cell):
    """Normalize a table cell for matching: strip bold/backticks/
    trailing qualifiers like 'fp16'."""
    c = cell.strip().strip("*`").strip()
    c = c.replace(" fp16", "")
    return c.lower()


def _cell_matches(cell, value):
    """Does a (normalized) row cell denote this override value?"""
    c = _norm_cell(cell)
    if c in ("—", "-", ""):
        return None  # not applicable in this row -- neutral
    v = str(value)
    if c == v.lower():
        return True
    if c in _DOC_LLM_ALIASES and _DOC_LLM_ALIASES[c] == v:
        return True
    try:
        return abs(float(c) - float(v)) < 1e-9
    except (TypeError, ValueError):
        return False


def _entry_value(entry, key):
    """The entry's effective value for an override key, or None if
    the entry doesn't set it (conf default -- v1 does not compose
    to resolve defaults; unmatched rows surface in the report)."""
    ov = entry.get("overrides") or {}
    if isinstance(ov, dict):
        return ov.get(key)
    for item in entry.get("overrides_list", []):
        k, _, v = item.partition("=")
        if k == key:
            return v
    return None


def _map_columns(headers, varied_keys):
    """header index -> override key, for headers that name a varied
    key ('alpha' -> search.alpha; 'llm' -> llm)."""
    out = {}
    for i, htxt in enumerate(headers):
        h = htxt.strip().lower()
        for k in varied_keys:
            tail = k.rsplit(".", 1)[-1].lower()
            if h == k.lower() or h == tail:
                out[i] = k
                break
    return out


def sync_doc(doc_path, entries, apply=False):
    """Patch status cells in `doc_path` from ledger truth. Returns
    a report dict; writes only when apply=True."""
    with open(doc_path, encoding="utf-8") as fin:
        lines = fin.read().splitlines(keepends=True)

    report = {"patched": [], "mismatches": [], "skipped": [],
              "unsynced_tables": [], "orphan_feeds": []}

    # Split into sections at #### headings.
    doc_idents = set()
    sec_starts = [i for i, ln in enumerate(lines)
                  if ln.startswith("#### ")]
    for si, start in enumerate(sec_starts):
        end = (sec_starts[si + 1] if si + 1 < len(sec_starts)
               else len(lines))
        text = "".join(lines[start:end])
        key = _section_feeds_key(text)
        tid = _section_table_id(text)
        idents = {k for k in (tid, key) if k}
        doc_idents |= idents
        title = lines[start].strip("# \n")
        if not idents:
            report["unsynced_tables"].append(
                (title, "no table-id / feeds key"))
            continue
        fed = [e for e in entries
               if idents & set(e.get("feeds") or [])]
        if not fed:
            label = " / ".join(f"`{k}`" for k in sorted(idents))
            report["unsynced_tables"].append(
                (title, f"no entries feed {label}"))
            continue

        # Varied keys: override keys whose values differ (or are
        # not universally present) across the fed entries.
        all_keys = set()
        for e in fed:
            ov = e.get("overrides") or {}
            all_keys |= set(ov if isinstance(ov, dict) else [])
        varied = [k for k in sorted(all_keys)
                  if len({str(_entry_value(e, k)) for e in fed}) > 1]
        if not varied and len(fed) > 1:
            report["unsynced_tables"].append(
                (title, "entries indistinguishable (no varied keys)"))
            continue
        # Match on EVERY override key the fed entries carry, not
        # just the fed-varied ones: a key constant across the fed
        # set (e.g. lam=0.01) can still differ from other rows of
        # the table, and matching on the varied subset alone let
        # the wrong row grab an entry (aime sweep tables,
        # 2026-07-22). Keys an entry leaves unset stay
        # conservative: _cell_matches(cell, None) is False for any
        # explicit cell value, so such entries surface as skips.
        varied = sorted(all_keys)

        # Locate the table: first run of | lines with a status col.
        trows = [i for i in range(start, end)
                 if lines[i].lstrip().startswith("|")]
        if len(trows) < 3:
            report["unsynced_tables"].append((title, "no table"))
            continue
        headers = [h for h in lines[trows[0]].strip().split("|")]
        try:
            status_col = [h.strip().lower()
                          for h in headers].index("status")
        except ValueError:
            report["unsynced_tables"].append(
                (title, "no status column"))
            continue
        colmap = _map_columns(headers, varied)

        matched_entries = set()
        for ri in trows[2:]:  # skip header + separator
            cells = lines[ri].split("|")
            if len(cells) <= status_col:
                continue
            cands = []
            for e in fed:
                ok = True
                for ci, k in colmap.items():
                    if ci >= len(cells):
                        continue
                    hit = _cell_matches(cells[ci], _entry_value(e, k))
                    if hit is False:
                        ok = False
                        break
                if ok:
                    cands.append(e)
            if len(cands) != 1:
                report["skipped"].append(
                    (title, lines[ri].strip()[:60],
                     f"{len(cands)} candidate entries"))
                continue
            e = cands[0]
            eid = id(e)
            if eid in matched_entries:
                report["skipped"].append(
                    (title, lines[ri].strip()[:60],
                     "entry already matched another row"))
                continue
            matched_entries.add(eid)
            cur = cells[status_col].strip()
            new = e.get("status") or "planned"
            if _norm_cell(cur) == new:
                continue
            if cur.startswith("scored") and new != "scored":
                report["mismatches"].append(
                    (title, lines[ri].strip()[:60],
                     f"doc says {cur!r}, ledger says {new!r} -- "
                     f"NOT downgrading"))
                continue
            cells[status_col] = f" {new} "
            report["patched"].append(
                (title, e.get("id") or "?", f"{cur} -> {new}", ri))
            lines[ri] = "|".join(cells)

    # Lint: entry feeds that resolve to nothing in this doc --
    # catches renamed keys, typos, and stale wiring.
    for e in entries:
        for f in (e.get("feeds") or []):
            if f not in doc_idents:
                report["orphan_feeds"].append((e.get("id") or "?", f))

    if apply and report["patched"]:
        with open(doc_path, "w", encoding="utf-8") as fout:
            fout.write("".join(lines))
    return report


def mint_table_ids(apply=False):
    """Ensure every #### table in every tracked doc carries a stable
    `<!-- table-id: tbl-xxxxxx -->` line (inserted directly under the
    heading). Existing IDs are never rewritten; duplicate IDs are
    reported. IDs are unique across ALL docs (6 hex chars, collision-
    checked at mint time). Returns
    {doc_path: {"existing": n, "minted": [(line0, title, id)]}},
    plus a "dups" list of (id, locations)."""
    import re
    import secrets
    id_re = re.compile(r"<!--\s*table-id:\s*(tbl-[0-9a-f]{6,8})\s*-->")
    docs = [(stem, p) for stem, p in LEDGER_DOC.items() if p]

    # Pass 1: read everything, collect existing IDs globally.
    texts, seen, dups = {}, {}, []
    for _, p in docs:
        with open(p, encoding="utf-8") as fin:
            texts[p] = fin.read().splitlines(keepends=True)
        for i, ln in enumerate(texts[p]):
            m = id_re.search(ln)
            if m:
                tid = m.group(1)
                loc = f"{os.path.basename(p)}:L{i + 1}"
                if tid in seen:
                    dups.append((tid, seen[tid], loc))
                else:
                    seen[tid] = loc

    # Pass 2: mint per doc, insert bottom-up (keeps line numbers).
    out = {"dups": dups}
    for _, p in docs:
        lines = texts[p]
        sec_starts = [i for i, ln in enumerate(lines)
                      if ln.startswith("#### ")]
        minted, existing = [], 0
        for si, start in enumerate(sec_starts):
            end = (sec_starts[si + 1] if si + 1 < len(sec_starts)
                   else len(lines))
            if id_re.search("".join(lines[start:end])):
                existing += 1
                continue
            while True:
                tid = "tbl-" + secrets.token_hex(3)
                if tid not in seen:
                    break
            seen[tid] = f"{os.path.basename(p)}:L{start + 1}"
            minted.append((start, lines[start].strip("# \n"), tid))
        for start, _, tid in sorted(minted, reverse=True):
            lines.insert(start + 1, f"<!-- table-id: {tid} -->\n")
        if apply and minted:
            with open(p, "w", encoding="utf-8") as fout:
                fout.write("".join(lines))
        out[p] = {"existing": existing, "minted": minted}
    return out


# ------------------------------------------------------------------ #
# --queue: the run-cycle worklist. Entries whose lifecycle status is  #
# `inqueue`, sorted by (priority asc, ledger, file order) -- the      #
# exp-run skill consumes exactly this, never the raw ledgers.         #
# ------------------------------------------------------------------ #
def list_queue(entries):
    """inqueue entries in drain order. Missing priority sorts last
    (matches the old orchestrator's rule)."""
    work = [
        (i, e) for i, e in enumerate(entries)
        if e.get("status") == "inqueue"
    ]
    work.sort(key=lambda ie: (
        ie[1].get("priority") is None,
        ie[1].get("priority") or 0,
        ie[1].get("_ledger", ""),
        ie[0],
    ))
    return [e for _, e in work]


def uniqueness_problems(entries):
    """Global duplicate-id / duplicate-hash detection across all
    ledgers. Duplicate hashes mean two entries compose to the same
    result dir (double-queue / re-run collision); duplicate ids
    break targeted edits."""
    problems = []
    seen_id, seen_hash = {}, {}
    for e in entries:
        eid = e.get("id")
        if eid:
            if eid in seen_id:
                problems.append((
                    eid,
                    f"duplicate id (also in {seen_id[eid]}, "
                    f"this one in {e.get('_ledger')})"))
            else:
                seen_id[eid] = e.get("_ledger")
        h = e.get("hash")
        if h:
            h = str(h)
            if h in seen_hash:
                problems.append((
                    e.get("id") or e.get("note") or h,
                    f"duplicate hash {h} (also entry "
                    f"'{seen_hash[h]}')"))
            else:
                seen_hash[h] = e.get("id") or e.get("note") or "?"
    return problems


# ------------------------------------------------------------------ #
# --verify: re-compose every ledger entry and assert its hash still   #
# matches what's recorded (the dir it was backfilled from, if any).   #
# Tripwire for launcher/config drift desyncing this script's mirrored #
# ConfigStore from the real launchers. Run after any launcher edit.   #
# ------------------------------------------------------------------ #
def verify_queue(queue, jobs=None):
    """ALWAYS recomposes -- never trusts stored `hash:` fields.
    Stored hashes are what --verify audits, not what it reads."""
    problems = []
    for e, (h, err) in zip(queue, entry_hashes(queue, jobs)):
        label = e.get("note") or e.get("from_dir") or str(e.get("overrides"))
        if err is not None:
            problems.append((label, f"compose failed: {err}"))
            continue
        # Stored hash (written once at append time): a mismatch means
        # the config groups/defaults changed since the entry was
        # appended -- drift protection for EVERY entry, not just
        # backfilled ones.
        stored = e.get("hash")
        if stored and str(stored) != h:
            problems.append(
                (label, f"hash drift: composes to {h}, "
                        f"stored hash: says {stored}"))
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


def _classify_worker(item):
    """(entry, check_wandb) -> (status, detail). Worker-safe: the
    per-entry compose + dir scan (+ optional W&B call) runs in a
    forked worker; exceptions become an ERROR row, never a raise."""
    entry, check_wandb = item
    try:
        return classify(entry, check_wandb)
    except Exception as ex:
        return "ERROR", {"err": f"{type(ex).__name__}: {ex}"}


def matches_entry_filters(entry, args):
    """Filters decidable from the entry ALONE (no compose needed):
    group, priority, recorded. Applied BEFORE the per-entry compose
    pass, so a scoped query (--group X) composes only X's entries
    instead of the whole ledger."""
    if args.not_recorded and entry.get("recorded"):
        return False
    if args.group and entry.get("group") != args.group:
        return False
    if args.priority is not None and entry.get("priority") != args.priority:
        return False
    if getattr(args, "ledger", None) and \
            entry.get("_ledger") != args.ledger:
        return False
    return True


def matches_status_filters(status, args):
    """Filters that need the computed status -- applied after."""
    if args.status and status != args.status:
        return False
    if args.done and status != "done":
        return False
    if args.planned and status != "planned":
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
    ap.add_argument("--ledger",
                    help="only entries from this per-doc ledger stem "
                         "(e.g. prm800k-level5)")
    ap.add_argument("--queue", action="store_true",
                    help="list inqueue entries in drain order "
                         "(priority asc) with launch commands")
    ap.add_argument("--check-running", action="store_true",
                    dest="check_running",
                    help="verdict every running entry: finished | "
                         "still-running | stalled | missing")
    ap.add_argument("--dedup", nargs="+", metavar="ROOT OVERRIDE",
                    help="like --check, plus every ledger entry "
                         "sharing the candidate's hash (file, id, "
                         "status, feeds)")
    ap.add_argument("--sync-doc", dest="sync_doc", metavar="STEM",
                    help="patch status cells in the doc for this "
                         "ledger stem from ledger truth (dry-run "
                         "unless --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="with --sync-doc / --mint-table-ids: "
                         "actually write the doc(s)")
    ap.add_argument("--mint-table-ids", action="store_true",
                    dest="mint_table_ids",
                    help="ensure every #### table in every tracked "
                         "doc carries a stable <!-- table-id --> "
                         "(dry-run unless --apply)")
    ap.add_argument("--backfill", action="store_true",
                    help="emit yaml entries for result dirs not in the queue")
    ap.add_argument("--commands", action="store_true",
                    help="print launch commands for matching entries")
    ap.add_argument("--verify", action="store_true",
                    help="re-compose every entry, assert hashes still match")
    ap.add_argument("--check", nargs="+", metavar="ROOT OVERRIDE",
                    help="compose a candidate (config_root key=val ...) and "
                         "report hash/name/on-disk/collision; does not write")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes for per-entry composes "
                         f"(default min({DEFAULT_JOBS_CAP}, workload); "
                         "1 = serial)")
    args = ap.parse_args()

    queue = load_ledgers()

    if args.mint_table_ids:
        out = mint_table_ids(apply=args.apply)
        mode = "APPLIED" if args.apply else "DRY RUN"
        print(f"# --mint-table-ids [{mode}]")
        for p, rep in out.items():
            if p == "dups":
                continue
            print(f"  {os.path.basename(p)}: "
                  f"existing={rep['existing']} "
                  f"minted={len(rep['minted'])}")
            for start, title, tid in rep["minted"]:
                print(f"    + L{start + 1} {tid}  {title}")
        for tid, a, b in out["dups"]:
            print(f"  !! DUPLICATE table-id {tid}: {a} AND {b}")
        return

    if args.check or args.dedup:
        spec = args.check or args.dedup
        root, overrides = spec[0], spec[1:]
        info = check_candidate(root, overrides, queue, args.jobs,
                               with_matches=bool(args.dedup))
        print(yaml.safe_dump(info, sort_keys=False, allow_unicode=True),
              end="")
        return

    if args.sync_doc:
        doc = LEDGER_DOC.get(args.sync_doc)
        if doc is None:
            print(f"# no doc for ledger stem {args.sync_doc!r} "
                  f"(known: {sorted(k for k, v in LEDGER_DOC.items() if v)})")
            sys.exit(1)
        scope = [e for e in queue
                 if e.get("_ledger") == args.sync_doc]
        rep = sync_doc(doc, scope, apply=args.apply)
        mode = "APPLIED" if args.apply else "DRY RUN"
        print(f"# --sync-doc {args.sync_doc} [{mode}]")
        for title, eid, change, ri in rep["patched"]:
            print(f"  patch L{ri + 1}: {eid}: {change}   ({title})")
        for title, row, msg in rep["mismatches"]:
            print(f"  ! MISMATCH {title}: {row}: {msg}")
        for title, row, msg in rep["skipped"]:
            print(f"  ~ skip {title}: {row}: {msg}")
        for title, msg in rep["unsynced_tables"]:
            print(f"  - unsynced: {title}: {msg}")
        for eid, f in rep["orphan_feeds"]:
            print(f"  ? orphan feeds: {eid}: `{f}` matches no table")
        print(f"# patched={len(rep['patched'])} "
              f"mismatches={len(rep['mismatches'])} "
              f"skipped={len(rep['skipped'])} "
              f"unsynced={len(rep['unsynced_tables'])} "
              f"orphans={len(rep['orphan_feeds'])}")
        return

    if args.check_running:
        scope = [e for e in queue if matches_entry_filters(e, args)]
        rows = check_running(scope, args.jobs)
        if not rows:
            print("# no running entries")
            return
        for r in rows:
            if r["verdict"] == "ERROR":
                print(f"{r['id']}  ERROR  {r['err']}")
                continue
            print(f"{r['id']}  {r['verdict']}  "
                  f"{r['n_done']}/{r['trials']}  "
                  f"scored={r['scored']}/{r['trials']}  "
                  f"wandb={r['wandb']}  dir={r['dir']}")
        return

    if args.queue:
        scope = [e for e in queue if matches_entry_filters(e, args)]
        work = list_queue(scope)
        if not work:
            print("# queue empty (no inqueue entries)")
            return
        for e in work:
            eid = e.get("id") or e.get("note") or "?"
            prio = e.get("priority", "-")
            hr = e.get("expected_hr", "-")
            print(f"# {eid}  ledger={e.get('_ledger')}  prio={prio}"
                  f"  expected_hr={hr}  hash={e.get('hash')}")
            print(launch_command(e))
        return

    if args.verify:
        scope = [e for e in queue if matches_entry_filters(e, args)]
        scope_note = (f" (scoped: {len(scope)}/{len(queue)})"
                      if len(scope) != len(queue) else "")
        problems = uniqueness_problems(queue)  # always global
        problems += verify_queue(scope, args.jobs)
        if not problems:
            print(f"# OK: all {len(scope)} entries compose and match "
                  f"their recorded hash{scope_note}; ids/hashes "
                  f"globally unique")
            return
        print(f"# {len(problems)} problem(s){scope_note}:")
        for label, msg in problems:
            print(f"  ! {label}: {msg}")
        sys.exit(1)

    if args.backfill:
        new = backfill(queue, args.jobs)
        if not new:
            print("# no un-queued result dirs found (nothing to backfill)")
            return
        print(f"# {len(new)} un-queued result dir(s) -- review and append "
              f"to the matching orchestration/ledgers/*.yaml:")
        # Strip the helper keys' leading underscore note for readability.
        print(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
        return

    # Pre-filter on entry-level criteria BEFORE composing anything:
    # a scoped listing (--group X) then composes only X's entries.
    scoped = [e for e in queue if matches_entry_filters(e, args)]
    results = _pmap(_classify_worker,
                    [(e, args.wandb) for e in scoped], args.jobs)
    rows = []
    for e, (status, detail) in zip(scoped, results):
        if status != "ERROR" and not matches_status_filters(status, args):
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
