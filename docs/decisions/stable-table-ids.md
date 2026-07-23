# Stable table IDs for exp-comp docs

Date: 2026-07-23. Status: implemented (proposed by Tuan mid-
reconcile; approved verbally same day).

## Problem

The ledger<->doc join keyed on a human-authored `feeds` string in
each table's blockquote. Two failure modes bit on 2026-07-23:

1. **Absence** — 34 of 34 level-4 tables (and 24 level-5, all
   gsm8k/aime2025) carried no key at all, so `--sync-doc` could
   not route 13 finished cnt-mcts entries.
2. **Fragility** — the string doubles as identifier and
   description (model, method, budget baked in), so retitling or
   splitting a table silently orphans its entries. Duplicate
   titles across sections (`#### model family, size, quantization
   comparison (qwen PRM)` appears in both cnt and sem sections)
   make title-based reference ambiguous.

## Decision

Identity and description become separate fields.

- Every `####` table carries an opaque, immutable line directly
  under its heading:

      <!-- table-id: tbl-xxxxxx -->

  6 hex chars, minted once, globally unique across all tracked
  docs, invisible in rendered markdown. Never renamed, even if
  the table's title, section, or file changes.
- Ledger `feeds` lists may name **either** a tbl-id or a legacy
  feeds string; the sync matcher accepts both (union), so
  existing string wiring keeps working and keys can be relabeled
  without breaking the join. New entries should prefer the
  tbl-id (add a `# comment` naming the table for readability).
- The human `feeds \`key\`` line in blockquotes is now a label,
  not load-bearing.

## Mechanism (status.py)

- `--mint-table-ids [--apply]` — inserts IDs for any table
  lacking one; idempotent; reports duplicates. Run it after
  adding a new table.
- `_section_table_id()` + union matching in `sync_doc()`.
- Lint, two halves: mint reports tables without IDs and
  duplicate IDs; `--sync-doc` reports **orphan feeds** (entry
  feeds values matching no table identifier in the doc).

## Migration record

- 2026-07-23: minted 112 IDs (level-5 35, level-4 34, gsm8k 21,
  aime2025 22). Zero duplicates.
- First consumers: 11 level-4 cnt-mcts entries wired by tbl-id
  (b320 model-family `tbl-4e21d6`, b80 model-family
  `tbl-6fe5a2`, agg_strategy `tbl-3ea294`, enforce_eager
  `tbl-adf2f8`).
- Known debt, parked: ~58 (level-4) + ~83 (level-5) orphan
  feeds strings on legacy entries — historical wiring that
  predates per-table keys. Triage separately; the lint keeps
  the count visible.

## Rejected alternative

Lint-hardened human keys only (uniqueness check + immutable-
once-minted convention) — ~80% of the value, but keeps identity
coupled to description and does nothing about the 58 duplicate-
title ambiguities. Opaque IDs chosen since the matcher change
is the same either way.
