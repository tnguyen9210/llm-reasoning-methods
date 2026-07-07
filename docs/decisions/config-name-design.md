# `config_name` and result-dir identity: design lineage

Four log entries trace one continuous thread — how a run's result-dir
name went from "not self-describing" to "encodes everything, fragile
under schema growth" to "readable prefix + collision-safe hash,
located by recorded fact." This doc covers the lineage in order;
[decisions-log.md](../decisions-log.md) has each entry's full original
text (2026-06-17, 2026-06-18, 2026-06-20, 2026-06-21).

## Current mechanism (verified 2026-07-07, matches 2026-06-21)

`utils/configs.py`:

- **`config_name(cfg)`** — `{algo}{--level-N if set}--{llm}{--prm}
  --d-{depth}--bs-{batch}--b-{budget:03d}--cfg-{hash8}`. `algo` is
  `cfg.search.method` (not `cfg.algo` — that field is purely an
  `algo_dict` dispatch key in each launcher, invisible in the name).
  The prefix is a **curated, cosmetic** subset for skimming; everything
  else (cpuct, lam, proj, cov, tmpl, prm_batch_size, …) is *not* in the
  name — only in the hash and the manifest.
- **`config_identity(cfg)` / `config_hash(cfg)`** — the collision-safe
  identity. `_HASH_GROUPS = ("search", "gen", "llm", "prm", "data")`
  defines which config groups are hashed; `_HASH_EXCLUDE` strips
  cosmetic/environment-only fields per group (e.g. `llm.
  gpu_memory_utilization`, `prm.device_map`, `search.embeds_mean`) so
  they can vary without changing the hash. `run` is not a hash group at
  all — `run.num_trials`, `run.num_questions`, `run.results_subdir` are
  excluded by omission, which is what makes `run.results_subdir`
  hash-neutral output-rerouting possible (used by smoke tests). Hashed
  as sorted-key canonical JSON via sha1, truncated to 8 hex chars.
- **`write_manifest(result_dir, cfg, ...)`** — records `config_name`,
  `config_hash`, `config_identity` (and optionally `run_id`) into
  `{result_dir}/manifest.json` at run creation. This is the recorded
  fact readers trust.
- **`find_run_dir(root_dir, cfg)`** — locates an existing run by
  globbing `results/{results_root}/{level_dir}/*/manifest.json` and
  matching the **recorded** `config_hash` against the target — never by
  recomputing and trusting a name.
- **`resolve_result_dir(root_dir, cfg, override=None)`** — the reader
  entry point (`compute_stats`, `prepare_scored_dataset`), in order:
  (1) an explicit `+result_dir=` override, (2) `find_run_dir`'s
  manifest match, (3) fall back to the freshly-computed `config_name`
  path for a brand-new run. `manifest_run_name(result_dir)` supplies
  the authoritative basename for trial files, so they resolve even if
  `config_name`'s format changes later.
- **The launcher is the one allowed recompute site** — resume (`.done`
  marker) needs a deterministic config→dir mapping to decide
  resume-vs-fresh, so only the launcher recomputes `config_name`;
  readers never do.

## How the design got here

### 2026-06-17 — make names self-describing

**Problem:** result dirs and W&B run names didn't carry level, model,
or template — not interpretable on their own, and needed redundant
side tags. **Fix:** bake `--level-{N}`, the model name, and
`--tmpl-{custom|native}` into `config_name`, applied consistently
across `mcts_cnt` and `bon`. This established the precedent the next
entry generalizes: the name is the experiment's identity, so it should
be self-describing.

### 2026-06-18 — the principle, and the bug that motivated it

**Problem:** a v02 smoke test with `embeds_proj=sparse` silently
*resumed* and skipped a trial from an earlier non-projection run —
`config_name` didn't encode the projection knobs, so both configs
mapped to the same dir, and the resume/`.done` mechanism conflated two
different experiments. **Principle decided:** any config knob that
changes results must appear in `config_name`, so distinct configs get
distinct dirs and resume can't conflate them. **As implemented then:**
a `--proj-{mode}{dim}` tag, appended *only when projection was on*, to
avoid renaming/orphaning existing no-projection dirs. The principle
(encode every result-affecting knob) is the one that survives to
today; the specific *conditional* tagging choice does not (see next).

### 2026-06-20 — the conditional tagging gets reversed

**Problem:** the `embeds_proj × cov_update` sweep needed `proj=none`
as a first-class cell, but with the tag suppressed for `none`, that
cell's dir had no projection marker at all — not self-describing next
to its `--proj-sparse512` sibling, and inconsistent with the
always-shown `--cov-` tag from the same 2026-06-18 batch. **Fix:**
make the projection tag unconditional, including `--proj-none{dim}`.
This deliberately reversed the *specific* "append only when on"
sub-choice from 2026-06-18 while *strengthening* the general
encode-every-knob principle (a swept knob, including its `none` value,
belongs in the name). The entry's own "Caveat/open" section flagged
this as symptomatic of a bigger problem: schema growth kept forcing
either an old-dir rename or an asymmetric tagging rule — the
structural fix was still pending.

### 2026-06-21 — split identity from addressing (the actual fix)

**Root cause identified** (vault note
`question-config-name-experiment-naming`): `config_name` had been
doing two jobs with opposite stability requirements — *identity*
(should change as the schema grows, since a new knob is a new
experiment) and *addressing* (needs to stay stable, since something
has to keep finding old dirs). Recomputing an addressing key against a
live, evolving schema is inherently fragile — this is why growing the
schema kept forcing dir renames (hit ~3× in one session before this
fix).

**Fix:** split the two jobs, as described in "Current mechanism"
above — a cosmetic, curated *prefix* for identity-as-display, and a
collision-safe *hash* over the complete config for identity-as-fact,
with the hash **recorded once** into a manifest at creation time so
readers locate runs by matching that recorded fact rather than
re-deriving a name against a schema that may have grown since. This is
the actual structural fix the 2026-06-20 entry's "Caveat/open" flagged
as still pending.

**Migration:** new runs get the short prefix+hash names; the 45
existing dirs at the time were backfilled via `backfill_manifests.py
--write`, which records the *old* long-form name as `config_name` with
`config_hash: null` (the full identity isn't recoverable from an old
name alone) — old dirs stay addressable by path, not by recomputed
hash, which is the agreed design rather than a gap.

## Why the whole lineage is worth keeping, not just the final state

Each earlier entry solved a real problem correctly *for its scope*,
and each fix's limits are exactly what motivated the next one:
self-describing names (2026-06-17) didn't anticipate that *not every*
result-affecting knob was in the name, which caused a real silent
resume bug (2026-06-18); the encode-everything principle was sound but
its conditional-tagging implementation created asymmetric, hard-to-read
sweep cells (2026-06-20); and the recurring tension between wanting
short names and never breaking old addressing was the real structural
problem underneath all three (2026-06-21). Reading the lineage explains
*why* the current split (cosmetic prefix vs. collision-safe hash,
recorded once) is the right shape — it's not an arbitrary design, it's
the resolution of three earlier attempts' specific failure modes.

## Revisit if

`results/` grows enough that `find_run_dir`'s `O(N)` glob-and-match
becomes slow (add an index file), or run-affecting state starts living
outside `cfg` entirely (an env var or hardcoded constant) — then the
manifest is incomplete and the hash under-identifies a run. Currently
only the hardcoded sparse-projection seed
([decisions/sparse-random-projection.md](sparse-random-projection.md))
is in this category, and it's fixed by design, so this isn't yet a
live risk.
