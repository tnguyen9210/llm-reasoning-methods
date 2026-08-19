"""CPU-only checks for mcts_cnt_v01's select_child tie band.

No GPU, no vLLM engine: selection is pure python over node q/visit
state, so it can be exercised directly. Pins three contracts:

  A  tie_tol=0.0 reproduces the OLD exact-equality tie-break
     bit-for-bit — same picks AND the same RNG stream (one
     random.choice per call), so pre-knob trajectories replay
     unchanged under the same seed.
  B  tie_tol=1e-4 matches sem-mcts-v02's band semantics
     (_select_by_q_value / _diverse_select): gaps inside the band
     randomize uniformly, gaps outside never get picked.
  C  the cpuct=1e18 "infinity" stand-in behaves as documented: q
     vanishes below float64 resolution of the sum, so equal-visit
     children tie exactly (uniform random even at tie_tol=0.0)
     and a lower-visit child always beats a higher-visit one.

Run:  python unittests/check_cnt_tie_tol.py
"""
import logging
import os
import random
import sys

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


def make_parent(qs, cpuct, tie_tol, visits=None, parent_visits=2):
    """An agent + a root carrying one child per q in `qs`.

    Child i gets visits[i] updates of value qs[i] (so q_value ==
    qs[i] exactly). The root's visit count is raised to
    `parent_visits` so the PUCT u-term is exercised
    (log(parent_visits) > 0 once parent_visits > 1).
    """
    agent = MCTS(config=Cfg(cpuct=cpuct, tie_tol=tie_tol),
                 question="q")
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
    """The pre-knob algorithm, verbatim semantics: scan children,
    collect exact-equality ties against the running max, one
    random.choice at the end."""
    best_value = -float("inf")
    best_childs = []
    for ch in node.children:
        v = ch.puct(cpuct=cpuct)
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


# --- A. tie_tol=0.0 == the old algorithm, picks AND stream ---------

agent, root = make_parent([0.3, 0.5, 0.5, 0.2], cpuct=0.0,
                          tie_tol=0.0)
new_seq = picks(agent, root)
old_seq = old_picks(root, cpuct=0.0)
check("A1 tol=0: pick sequence == old algorithm (same seed)",
      new_seq == old_seq)
check("A2 tol=0: exact tie randomizes over exactly the tied pair",
      sorted(set(new_seq)) == [1, 2], f"set={sorted(set(new_seq))}")

agent, root = make_parent([0.5, 0.5 - 1e-6], cpuct=0.0, tie_tol=0.0)
seq = picks(agent, root)
check("A3 tol=0: 1e-6 gap stays a deterministic argmax",
      set(seq) == {0}, f"set={sorted(set(seq))}")

agent, root = make_parent([0.9, 0.1], cpuct=2.0, tie_tol=0.0,
                          visits=[3, 1], parent_visits=4)
new_seq = picks(agent, root)
old_seq = old_picks(root, cpuct=2.0)
check("A4 tol=0, cpuct=2: u-term case still == old algorithm",
      new_seq == old_seq)

# --- B. tie_tol=1e-4 == sem-mcts-v02 band semantics ----------------

agent, root = make_parent([0.5, 0.5 - 1e-6, 0.4], cpuct=0.0,
                          tie_tol=1e-4)
seq = picks(agent, root)
check("B1 band: 1e-6 gap randomizes over the near-tied pair",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

agent, root = make_parent([0.5, 0.5 - 9e-5, 0.5 - 2e-4],
                          cpuct=0.0, tie_tol=1e-4)
seq = picks(agent, root)
check("B2 band: 9e-5 inside, 2e-4 outside",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

# --- C. the cpuct=1e18 infinity stand-in ---------------------------

agent, root = make_parent([0.9, 0.1], cpuct=1e18, tie_tol=0.0,
                          parent_visits=3)
seq = picks(agent, root)
check("C1 1e18: equal visits -> q vanishes, exact tie, uniform",
      sorted(set(seq)) == [0, 1], f"set={sorted(set(seq))}")

agent, root = make_parent([0.9, 0.1], cpuct=1e18, tie_tol=0.0,
                          visits=[2, 1], parent_visits=4)
seq = picks(agent, root)
check("C2 1e18: lower-visit child always wins across visit gap",
      set(seq) == {1}, f"set={sorted(set(seq))}")

# --- D. degenerate input -------------------------------------------

agent, root = make_parent([], cpuct=2.0, tie_tol=0.0)
check("D1 no children -> None", agent.select_child(root) is None)


print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
