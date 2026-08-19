"""CPU-only checks for mcts_sem_v02's select_child tie band knob.

Mirror of check_cnt_tie_tol.py on the sem side. The knob makes the
historically hardcoded 1e-4 band configurable (`search.tie_tol`),
neutral default 1e-4. Pins:

  A  tie_tol=1e-4 (default) reproduces the historical band on both
     selection paths — the pure `_diverse_select` decision and the
     first-visit `_select_by_q_value` path.
  B  tie_tol=0.0 gives exact-equality ties (cnt-mcts-aligned):
     sub-band gaps become a deterministic argmax, exact ties still
     randomize.

Run:  python unittests/check_sem_tie_tol.py
"""
import logging
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.mcts_sem_search_v02_00_00 import (  # noqa: E402
    MCTS, _diverse_select,
)
from utils.configs import MCTSSemV02Config    # noqa: E402

logging.disable(logging.CRITICAL)

D = 4
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


class Cfg:
    """Minimal stand-in for the composed ExpConfig."""
    def __init__(self, **kw):
        self.search = MCTSSemV02Config(**kw)


def diverse_picks(scores, tie_tol, n=400, seed=7):
    """Drive the pure decision function with ds_alpha=0 so only the
    q term and the tie band matter."""
    V_inv = np.eye(D)
    embeds = [np.eye(D)[i % D] for i in range(len(scores))]
    random.seed(seed)
    return [
        _diverse_select(V_inv, embeds, scores, ds_alpha=0.0,
                        ds_beta=1.0, tie_tol=tie_tol)
        for _ in range(n)
    ]


def qpath_picks(qs, tie_tol, n=400, seed=7):
    """Drive select_child on a fresh root (visit_count == 1), i.e.
    the `_select_by_q_value` path. Children carry zero embeddings so
    the post-selection covariance fold is exercised but inert."""
    agent = MCTS(config=Cfg(embeds_dim=D, tie_tol=tie_tol),
                 question="q")
    root = agent.root
    for i, q in enumerate(qs):
        ch = agent.create_node(parent=root)
        ch.tag = f"0.{i + 1}"
        ch.embeds = np.zeros(D)
        ch.update(q)
        root.children.append(ch)
    random.seed(seed)
    return [
        int(agent.select_child(root).tag.split(".")[1]) - 1
        for _ in range(n)
    ]


# --- A. default 1e-4 band == the historical behavior ---------------

seq = diverse_picks([0.5, 0.5 - 1e-6, 0.4], tie_tol=1e-4)
check("A1 _diverse_select band: 1e-6 gap randomizes, 0.1 gap out",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

check("A2 schema default is the historical 1e-4",
      MCTSSemV02Config().tie_tol == 1e-4,
      f"default={MCTSSemV02Config().tie_tol}")

seq = qpath_picks([0.5, 0.5 - 1e-6, 0.4], tie_tol=1e-4)
check("A3 first-visit q path: same band via search.tie_tol",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

# --- B. tie_tol=0.0 == exact-equality ties (cnt-aligned) -----------

seq = diverse_picks([0.5, 0.5 - 1e-6], tie_tol=0.0)
check("B1 _diverse_select tol=0: 1e-6 gap is deterministic",
      set(seq) == {0}, f"set={sorted(set(seq))}")

seq = diverse_picks([0.3, 0.5, 0.5], tie_tol=0.0)
check("B2 _diverse_select tol=0: exact ties still randomize",
      sorted(set(seq)) == [1, 2], f"set={sorted(set(seq))}")

seq = qpath_picks([0.5, 0.5 - 1e-6], tie_tol=0.0)
check("B3 first-visit q path tol=0: deterministic argmax",
      set(seq) == {0}, f"set={sorted(set(seq))}")


print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
