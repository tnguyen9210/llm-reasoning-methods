# Stable table IDs for exp-comp docs

**Moved to the Second Brain vault (2026-07-23, Tuan's call):**
`second-brain/03_projects/llm-reasoning-workflow-routines/`
`discussions/stable-table-ids.md`

Full content in git history (last in-repo revision: ea8b0fa).
Mechanism summary: every `####` table carries an immutable
`<!-- table-id: tbl-xxxxxx -->`; mint via
`python orchestration/status.py --mint-table-ids --apply`;
ledger `feeds` accepts tbl-ids or legacy keys.
