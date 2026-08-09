# llm-reasoning-methods

LLM reasoning via search (MCTS / BoN / beam) + reward
models (PRMs/ORMs). Experiments are Hydra-configured,
tracked in W&B (`tnguyen10/llm-reasoning`) and in
per-doc ledgers under `orchestration/`.

## Environment
- **py311 env pinned for Volta**: dev node is V100S
  (sm_70) -> vllm <=0.18.x + torch cu126 only, no
  bf16. Never upgrade past those.
- U-Arizona HPC (SLURM). GPU work runs via
  `srun --overlap` inside held allocations — never on
  the login shell.
- Home quota is tight: big artifacts go to `results/`,
  never `~`.
- Nodes have 96 threads; parallelize with
  `min(48, workload)` workers and expose a `--jobs`
  override.

## Commands
- `python orchestration/status.py` — the single entry
  point for ledger state: `--queue`, `--running`,
  `--check-running`, `--dedup`, `--sync-doc`,
  `--verify`, `--mint-table-ids`.
- `python compute_stats.py --config-name <root>
  <overrides>` — the only legitimate source of result
  numbers (CPU-only is fine).
- `prepare_scored_dataset.py` — requires CUDA; never
  run it on the login shell.

## Conventions
- Result dirs: naming v2 — `cfg-<hash>` short names +
  `config_hash` in the manifest. A legacy long-string
  scheme still exists for old runs; check both.
- W&B: `wandb.log({...}, step=trial_idx)`; never log
  the trial index as a metric column.
- Code style: strict PEP 8 — 79-char code lines,
  72-char comments/docstrings. Paths are strings /
  f-strings, not pathlib.
- Doc tables: scored cells `.NNNN<br>±.NNNN` (4 dp);
  status cells are bare words
  (`planned|inqueue|running|scored|failed`). Outside
  the dedicated enforce_eager table, comparison cells
  use `enforce_eager=False` runs.
- Commit messages: no Co-Authored-By trailers.

## Workflow
- Experiments flow through
  `orchestration/ledgers/*.yaml` + the
  `exp-{tables,run,check,cron,smoke-test}` skills —
  never launch or record results by hand.
- Doc status cells are DERIVED from the ledger
  (`status.py --sync-doc`).
- Technical decisions -> `docs/decisions-log.md`
  (+ `docs/decisions/`); algorithm registry ->
  `docs/algorithms.md`; planning lives in the vault,
  not here.

## Do not
- Hand-edit exp-comp status cells or anything between
  generated markers.
- Run smoke/debug launches without
  `WANDB_MODE=offline`.
- Re-add `unittests/results/` to git (ignored
  deliberately, `902b13b`).
- Trust W&B run state as liveness: `crashed` can be a
  dropped heartbeat; the nvidia-smi probe is ground
  truth.
- Append a ledger entry without `--dedup`: one entry
  per config hash, ever.
