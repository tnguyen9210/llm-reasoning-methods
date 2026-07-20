# exp-new-comparison-table

Loaded when: Tuan asks to add a new comparison table to one
of the `docs/exp-comp-*.md` tracking docs (e.g.
`exp-comp-prm800k-level4.md`, `exp-comp-prm800k-level5.md`,
`exp-comp-gsm8k.md`) and queue its experiments — e.g. "add a
table comparing X across Y with prm=Z", "make a ds_alpha
sweep table for qwen PRM", "add a b=320 model-family table".

This skill creates two artifacts from one intent statement:
1. a `####` comparison table in the relevant `docs/exp-comp-*.md`
   doc (all cells `planned`), and
2. the matching planned entries in `experiments.yaml`.

It does **not** launch runs, and **does not commit** — it
reports what it made and waits for Tuan's go-ahead (he has a
standing rule: show changes, wait for explicit confirmation
before any git commit/push).

---

## 0. The invariants this skill exists to protect

These are the judgment steps a slash command would skip.
Skipping them is what causes drift. Do not skip them.

- **Compose-verify every cell offline** before writing it —
  via `python status.py --check <root> <overrides>`. Never
  hand-derive a hash or assume a config composes.
- **Hash-collision check against the ledger.** If a cell's
  `ledger_collision` is true, an entry already exists — do
  NOT append a duplicate (that breaks the append-only
  invariant). Reuse / point `feeds` at it instead.
- **Detect the mixed case.** A "new table" can have cells
  that are already on disk (`on_disk: true`) — runs launched
  before the table existed. Mark those rows accurately
  (`partial`/`done`, not `planned`) and do not re-queue
  them.
- **Match per-model template conventions** (§4).
- **Report, don't auto-commit.**

---

## 1. The system this fits into

Three layers (full design: vault guide
`research-coding-practices-guides/tracking-experiment-status`):

- `experiments.yaml` — append-only intent ledger. One entry
  per launchable run. Never delete or reorder; status is
  computed, not stored.
- `status.py` — the reconciler. **One source of truth** for
  compose/hash/collision logic. This skill calls it; it does
  NOT reimplement composing or hashing.
- `docs/exp-comp-*.md` — the human report (one file per
  dataset/level). Tables are a *view*; numbers only land here
  from `done` runs (that's the separate `exp-record-results`
  skill's job).

The link between a ledger entry and a doc table is the
`feeds` key — a loose string roughly tracking the doc
heading (e.g. `sem-mcts/ds_alpha-sweep-qwen`,
`cnt-mcts/model-family-b320-qwen`).

---

## 2. Inputs to gather (infer, then confirm)

Lead by inferring from what Tuan said; ask only what's
genuinely ambiguous. Needed:

- **What it compares** (the table's research question — goes
  in the `Compares:` blurb).
- **LLM model(s)** → `llm=` group(s). Available in
  `conf/llm/`: `llama_1b`, `llama_3b`, `llama_3b_gptq`,
  `qwen_3b`, `qwen_3b_gptq_int4`, `qwen_7b_gptq_int4`,
  `qwen_math_1_5b`, `qwen_math_7b`.
- **PRM** → `prm=` group. `conf/prm/`: `llama_prm`
  (default), `qwen_prm`.
- **Method / config_root** → which launcher + root config.
  Map below (§3).
- **Fixed vs. varied knobs.** The varied knob(s) become the
  table's row axis; everything else is "Fixed:".
- **Budget** if non-default (`search.gen_budget`, default
  80). A different budget usually means a different doc
  section (see §5).
- **trials** — default 2 (matches recent convention; ask if
  unsure).

Confirm the resolved grid (models × varied-knob values, the
fixed set, the feeds key) with Tuan before writing, via the
resolved-cell table from §6. Ask a question only when a
choice materially changes the output (which models? new
budget → new section?).

---

## 3. Method → launcher / config_root

| method | launcher | config_root |
|---|---|---|
| cnt-mcts | `generate_mcts_cnt.py` | `mcts_cnt_prm800k` |
| sem-mcts v01 (policy embeds) | `generate_mcts_sem.py` | `mcts_sem_v01_prm800k` |
| sem-mcts v02 (PRM embeds) | `generate_mcts_sem.py` | `mcts_sem_v02_prm800k` |

`group:` in the ledger is `cnt-mcts` or `sem-mcts`.

---

## 4. Per-model template convention (easy to get wrong)

`use_custom_template` is **baked into each `conf/llm/*.yaml`**
already: qwen configs set `use_custom_template: false`
(native chat template), llama configs default to custom
(true). So:

- **For cnt-mcts and any case where you select a whole `llm=`
  group**, you do NOT pass `use_custom_template` — selecting
  the group is enough.
- **Only pass `llm.use_custom_template=false` explicitly**
  when overriding a field on an *already-selected* group in a
  context that needs it (this came up for sem-v01/v02 qwen
  rows in earlier tables). When in doubt, `--check` both ways
  and confirm the hash matches the convention you intend.

---

## 5. Where the table goes in the doc

- Default budget (b=80): under `## Tuning tables
  [gen_budget=80]`, in the right `### <algorithm>`
  subsection.
- Non-default budget (160/320/…): under `## Tuning tables
  [gen_budget=160, 320, …]`, creating the `### <algorithm>`
  subsection there if absent. (Precedent: the b=320 qwen-PRM
  cnt-mcts table lives here.)

Heading hierarchy: `## Tuning tables [...]` → `###
<algorithm>` → `#### <this comparison>`.

---

## 6. Procedure

1. **Resolve cells.** Build the override list for each cell
   (model × varied value, plus fixed overrides).
2. **`--check` every cell.** For each:
   ```
   python status.py --check <config_root> <key=val> <key=val> ...
   ```
   Record `hash`, `on_disk`, `n_done`, `ledger_collision`.
   Confirm all hashes are distinct (a collision *within* the
   new set means two cells resolved identically — a spec
   bug; stop and re-examine).
3. **Show Tuan the resolved-cell table** and get go-ahead:
   ```
   | cell | hash | on_disk | in_ledger | -> action |
   ```
   action = `queue` (new), `skip-queue (already in ledger)`,
   or `already on disk: <n_done> trials`.
4. **Write the doc `####` section** in house style (§7), in
   the location from §5. Cell `status` column reflects the
   `--check` reality: `planned` if not on disk, else
   `partial`/`scored` with the trial count.
5. **Append ledger entries** — only for cells with
   `ledger_collision: false`. Each:
   ```yaml
   - launcher: <launcher>
     hash: "<hash from step 2's --check>"
     config_root: <config_root>
     overrides: {<dict form>}
     trials: <n>
     feeds: [<feeds-key>]
     group: <cnt-mcts|sem-mcts>
     priority: <1|2>
     recorded: false
     note: <short human label>
   ```
   Precede the block with a `# --- planned: <description>`
   comment. Append at the bottom (never reorder). The `hash:`
   field is written ONCE here, quoted, verbatim from `--check`
   (never hand-derived): status.py's fast paths (collision
   scans) read it instead of recomposing, and `--verify`
   audits it against a fresh compose — a mismatch there means
   the config groups drifted since this entry was appended.
6. **Re-verify the file parses + nothing collided:**
   ```
   python status.py --verify
   python status.py --group <group>   # eyeball the new rows
   ```
7. **Report**: N cells, how many queued vs. already on
   disk/in ledger, the feeds key, and the launch commands
   (`python status.py --planned --group <group> --commands`,
   filtered to the new ones). Then stop — ask whether to
   commit.

---

## 7. Doc house style (match existing tables exactly)

Each `####` section is: a blockquote preamble, then the
table, then an Analysis blockquote.

```markdown
#### <title>
> **Compares:** <research question — what varies, what it
> isolates, why it's worth reading>.
>
> **Fixed:** <the held-constant knobs, terse: tmpl, bs-4,
> d-20, b=N, proj, cov_update, prm, ds_beta, …>.
>
> ⚠️ <caveats — e.g. "Entirely planned, no runs yet";
> prm_batch_size mismatches; budget cost warnings>.
>
> **W&B:** <run ids per cell, or "none yet (no runs exist)">.

| <axis cols> | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|...|
| <row> | — | planned | — | — | — | — | — |

> **Analysis.** <what the table will show once filled; for an
> all-planned table: "No data yet — nothing to take away.
> Once filled, the key read is …">.
> **Limitations / follow-up:** <planned cells pointer to
> experiments.yaml group + feeds key; deferred variants>.
```

Conventions:
- Wrap doc prose to ~72 chars.
- Scored numeric cells use `.NNNN<br>±.NNNN`.
- `status` values seen in the doc: `planned`, `scored`,
  `scored ⚠` (caveat). Planned rows use `—` for every
  numeric column and `—` for trials.
- If you add a column the analogous table lacks (e.g. a
  `prm` column), add it as a real column AND keep it in
  `Fixed:` if every row shares the value — both is fine and
  not contradictory.

---

## 8. After the table exists

- Launching: `python status.py --planned --group <group>
  --commands` emits the exact commands. (Tuan launches; or,
  if he asks, run them here via Bash.)
- Recording results once runs finish: that's the
  `exp-record-results` skill — `status.py --done
  --not-recorded` → compute stats → audit the cell → write →
  flip `recorded: true`.

---

## 9. Failure modes specific to this skill

- **Duplicate-append** (the worst): appending an entry for a
  cell already in the ledger. The `ledger_collision` check
  in step 2 prevents this — honor it.
- **Stale-doc cell**: writing a cell as `planned` when it's
  already on disk. The `on_disk` check prevents this.
- **Wrong template**: forgetting/over-applying
  `use_custom_template`. §4 + a confirming `--check`.
- **Budget in wrong section**: a b≠80 table under the b=80
  heading. §5.
- **feeds key drift**: a key that doesn't match the doc
  heading text. Keys are loose by design, but keep them
  readable and consistent with siblings; renaming later is a
  cheap find-and-replace across the ledger.
