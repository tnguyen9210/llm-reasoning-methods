"""CPU-only checks for mcts_cnt_v01's q_beta weight + first-visit
q-value branch.

Selection is pure python over node q/visit state, so no GPU and no
vLLM engine are needed. Pins four contracts:

  A  q_beta=1.0 (default) reproduces the OLD single-formula
     algorithm bit-for-bit on BOTH branches — the first-visit
     q-argmax and the later PUCT picks — RNG stream included.
  B  the first-visit branch is a RAW q argmax, independent of
     q_beta, mirroring sem-mcts-v02's `_select_by_q_value`
     dispatch at visit_count()==1. This is what stops q_beta=0
     from degenerating into a coin flip on the first descent.
  C  q_beta=0.0 after the first visit is pure exploration: q is
     ignored entirely and the least-visited child wins, for any
     cpuct > 0 — i.e. the exact cpuct->inf limit WITHOUT the
     cpuct=1e18 float-spacing stand-in, which it reproduces
     pick-for-pick.
  D  q_beta still scales exploitation: raising it flips a
     decision the exploration term would otherwise win.

Run:  python unittests/check_cnt_q_beta.py
"""
import logging
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.mcts_cnt_search_v01_00_00 import MCTS  # noqa: E402
from utils.configs import MCTSCntConfig          # noqa: E402

# select_child narrates every child at CRITICAL; silence it so the
# check output stays readable.
logging.disable(logging.CRITICAL)

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


class Cfg:
    """Minimal stand-in for the composed ExpConfig."""
    def __init__(self, **kw):
        self.search = MCTSCntConfig(**kw)


def make_parent(qs, cpuct=2.0, q_beta=1.0, tie_tol=0.0, visits=None,
                parent_visits=2):
    """An agent + a root carrying one child per q in `qs`.

    Child i gets visits[i] updates of value qs[i] (so q_value ==
    qs[i] exactly). `parent_visits=1` leaves the root on its seed
    visit, which is the first-visit branch; >1 exercises PUCT.
    """
    agent = MCTS(config=Cfg(cpuct=cpuct, q_beta=q_beta,
                            tie_tol=tie_tol), question="q")
    root = agent.root
    for _ in range(parent_visits - 1):
        root.update(0)
    if visits is None:
        visits = [1] * len(qs)
    for i, (q, n) in enumerate(zip(qs, visits)):
        ch = agent.create_node(parent=root)
        ch.tag = f"0.{i + 1}"
        for _ in range(n):
            ch.update(q)
        root.children.append(ch)
    return agent, root


def picks(agent, root, n=400, seed=7):
    random.seed(seed)
    return [
        int(agent.select_child(root).tag.split(".")[1]) - 1
        for _ in range(n)
    ]


def old_select(node, cpuct):
    """The pre-knob algorithm: ONE formula for every visit
    (q + cpuct*u), exact-equality ties, one random.choice."""
    best_value = -float("inf")
    best_childs = []
    for ch in node.children:
        if node.visit_count() == 0 or ch.visit_count() == 0:
            u = 0.0
        else:
            u = cpuct * np.sqrt(
                np.log(node.visit_count()) / ch.visit_count()
            )
        v = ch.q_value() + u
        if v == best_value:
            best_childs.append(ch)
        elif v > best_value:
            best_value = v
            best_childs = [ch]
    return random.choice(best_childs)


def old_picks(root, cpuct, n=400, seed=7):
    random.seed(seed)
    return [
        int(old_select(root, cpuct).tag.split(".")[1]) - 1
        for _ in range(n)
    ]


# --- A. q_beta=1.0 == the old single-formula algorithm -------------

check("A1 schema default is 1.0",
      MCTSCntConfig().q_beta == 1.0,
      f"default={MCTSCntConfig().q_beta}")

agent, root = make_parent([0.9, 0.1], cpuct=2.0, visits=[3, 1],
                          parent_visits=4)
check("A2 q_beta=1, PUCT branch: picks == old algorithm",
      picks(agent, root) == old_picks(root, cpuct=2.0))

agent, root = make_parent([0.3, 0.5, 0.5, 0.2], cpuct=2.0,
                          parent_visits=1)
new_seq = picks(agent, root)
check("A3 q_beta=1, first-visit branch: picks == old algorithm",
      new_seq == old_picks(root, cpuct=2.0))
check("A4 first visit randomizes over exactly the tied max pair",
      sorted(set(new_seq)) == [1, 2], f"set={sorted(set(new_seq))}")

# --- B. the first-visit branch ignores q_beta ----------------------

agent, root = make_parent([0.3, 0.9, 0.2], cpuct=2.0, q_beta=0.0,
                          parent_visits=1)
seq = picks(agent, root)
check("B1 q_beta=0 on first visit: still the raw q argmax",
      set(seq) == {1}, f"set={sorted(set(seq))}")

agent, root = make_parent([0.5, 0.5 - 9e-5, 0.4], cpuct=2.0,
                          q_beta=0.0, tie_tol=1e-4, parent_visits=1)
seq = picks(agent, root)
check("B2 q_beta=0 first visit honors the tie band (sem parity)",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

agent, root = make_parent([0.3, 0.9, 0.2], cpuct=2.0, q_beta=5.0,
                          parent_visits=1)
seq = picks(agent, root)
check("B3 first visit is scale-free in q_beta (5.0 -> same pick)",
      set(seq) == {1}, f"set={sorted(set(seq))}")

# --- C. q_beta=0 after the first visit == pure exploration ---------

agent, root = make_parent([0.9, 0.1], cpuct=2.0, q_beta=0.0,
                          visits=[2, 1], parent_visits=4)
seq = picks(agent, root)
check("C1 q_beta=0: least-visited child wins, q ignored",
      set(seq) == {1}, f"set={sorted(set(seq))}")

agent, root = make_parent([0.9, 0.1], cpuct=2.0, q_beta=0.0,
                          parent_visits=3)
seq = picks(agent, root)
check("C2 q_beta=0: equal visits -> exact tie, uniform random",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

agent, root = make_parent([0.9, 0.1, 0.5], cpuct=2.0, q_beta=0.0,
                          visits=[2, 1, 1], parent_visits=5)
beta0_seq = picks(agent, root)
agent, root = make_parent([0.9, 0.1, 0.5], cpuct=1e18, q_beta=1.0,
                          visits=[2, 1, 1], parent_visits=5)
inf_seq = picks(agent, root)
check("C3 q_beta=0 == the cpuct=1e18 stand-in, pick-for-pick",
      beta0_seq == inf_seq and sorted(set(beta0_seq)) == [1, 2],
      f"set={sorted(set(beta0_seq))}")

# --- D. q_beta still weights exploitation --------------------------

# visits [2, 1] at parent_visits=4 give u = [1.665, 2.355], a 0.69
# exploration edge for child 1. A 0.4 q-gap loses to it at
# q_beta=1 and beats it at q_beta=10.
agent, root = make_parent([0.5, 0.1], cpuct=2.0, q_beta=1.0,
                          visits=[2, 1], parent_visits=4)
lo_seq = picks(agent, root)
agent, root = make_parent([0.5, 0.1], cpuct=2.0, q_beta=10.0,
                          visits=[2, 1], parent_visits=4)
hi_seq = picks(agent, root)
check("D1 raising q_beta flips exploration's pick to exploitation",
      set(lo_seq) == {1} and set(hi_seq) == {0},
      f"q_beta=1 -> {sorted(set(lo_seq))}, "
      f"q_beta=10 -> {sorted(set(hi_seq))}")

# --- E. degenerate input -------------------------------------------

agent, root = make_parent([], cpuct=2.0, parent_visits=1)
check("E1 no children on the first-visit branch -> None",
      agent.select_child(root) is None)


print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
