"""Regenerate the Contents block in every docs/exp-comp-*.md.

The TOC lists `##` sections, `###` algorithm groups and `####`
tables, each linked by GitHub anchor, with the table's stable
tbl-id alongside so a ledger `feeds` value can be traced to a
table by eye.

The block lives between two sentinel comments and is rewritten
in place, so this is safe to re-run after adding tables. Run
from the repo root:

    python scripts/gen_toc.py
"""
import os
import re

DOCS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs")
BEGIN = "<!-- toc:begin -- generated, do not hand-edit -->"
END = "<!-- toc:end -->"


def slug(text):
    """GitHub-flavoured heading anchor."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)     # drop punctuation, keep _
    s = re.sub(r"\s+", "-", s)
    return s


def esc(text):
    """Escape brackets so they survive as markdown link text."""
    return text.replace("[", r"\[").replace("]", r"\]")


def build_toc(lines):
    seen = {}
    out = [BEGIN, "## Contents", ""]
    n_tbl = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,4}) (.+?)\s*$", line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2)
        if title == "Contents":
            continue
        a = slug(title)
        seen[a] = seen.get(a, -1) + 1
        anchor = a if not seen[a] else f"{a}-{seen[a]}"
        indent = "  " * (level - 2)
        label = esc(title)
        if level == 2:
            out.append(f"{indent}- [**{label}**](#{anchor})")
        elif level == 3:
            out.append(f"{indent}- [{label}](#{anchor})")
        else:                           # a table -- find its id
            tid = ""
            for j in range(i + 1, min(i + 4, len(lines))):
                t = re.search(r"table-id:\s*(tbl-\w+)", lines[j])
                if t:
                    tid = f" · `{t.group(1)}`"
                    break
            n_tbl += 1
            out.append(f"{indent}- [{label}](#{anchor}){tid}")
    out += ["", f"*{n_tbl} tables. Regenerate with "
                f"`python scripts/gen_toc.py`.*", END, ""]
    return out


def process(path):
    text = open(path).read()
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?",
                  "", text, flags=re.S)
    lines = text.split("\n")
    toc = build_toc(lines)
    idx = next(i for i, l in enumerate(lines) if l.startswith("## "))
    lines[idx:idx] = toc
    open(path, "w").write("\n".join(lines))
    n_tbl = sum(1 for l in toc if "`tbl-" in l)
    print(f"{os.path.basename(path):32s} {n_tbl} tables, "
          f"{len(toc)} toc lines")


def main():
    for f in sorted(os.listdir(DOCS)):
        if f.startswith("exp-comp-") and f.endswith(".md"):
            process(os.path.join(DOCS, f))


if __name__ == "__main__":
    main()
