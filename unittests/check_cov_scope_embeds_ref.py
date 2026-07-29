"""CPU-only checks for mcts_sem_v02's cov_scope / embeds_ref knobs.

No GPU, no vLLM, no PRM: the covariance plumbing and the selection
logic are pure numpy, so both can be exercised directly. Two halves:

  Part A  the covariance algebra -- per-node V equals the brute-force
          ridge inverse, folding at one node leaves its siblings
          alone, sm agrees with exact, fp32 tracks fp64.

  Part B  END-TO-END EQUIVALENCE. A scripted generator/PRM stub drives
          the real mcts_search loop, so generation is deterministic by
          construction and the ONLY thing that can differ between two
          runs is the selection logic. This is what licenses reading
          cov_scope="global" as "the pre-merge behavior": it pins not
          just the arithmetic but the RNG consumption, which code
          review is bad at checking (one extra random.choice draw and
          two runs diverge with correct math on both sides).

Run:  python unittests/check_cov_scope_embeds_ref.py
"""
import contextlib
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.mcts_sem_search_v02_00_00 as sem                   # noqa: E402
from core.mcts_sem_search_v02_00_00 import MCTS, mcts_search  # noqa: E402
from utils.configs import MCTSSemV02Config                    # noqa: E402


D = 16
LAM = 0.01
RNG = np.random.default_rng(0)

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


class Cfg:
    """Minimal stand-in for the composed ExpConfig."""
    def __init__(self, **kw):
        self.search = MCTSSemV02Config(**kw)


def make_agent(cov_update="sm", cov_scope="local", cov_dtype="fp64",
               embeds_ref="absolute", **kw):
    cfg = Cfg(embeds_dim=D, lam=LAM, cov_update=cov_update,
              cov_scope=cov_scope, cov_dtype=cov_dtype,
              embeds_ref=embeds_ref, **kw)
    return MCTS(config=cfg, question="q")


def fold(agent, node, u):
    agent._cov_fold(
        node, np.asarray(u, dtype=agent.cov_dtype).reshape(-1, 1)
    )


def unit(v):
    return v / np.linalg.norm(v)


def brute_force_inv(vectors, lam=LAM, d=D):
    V = lam * np.eye(d)
    for u in vectors:
        V = V + np.outer(u, u)
    return np.linalg.inv(V)


print("=" * 62)
print("Part A -- covariance algebra")
print("=" * 62)

# --- 1. ridge init: an unfolded node reads (1/lam) I
for upd in ("sm", "exact"):
    a = make_agent(upd)
    check(f"[{upd}] unfolded node reads (1/lam)I",
          np.allclose(a._cov_read(a.root), (1.0 / LAM) * np.eye(D)),
          f"cnt_cov_nodes={a.cnt_cov_nodes}")
    check(f"[{upd}] ridge read allocates no per-node state",
          a.cnt_cov_nodes == 0 and a.root.V_inv_local is None)

# --- 2. algebra: V_n^-1 == inv(lam I + sum u u^T) over its own folds
for upd in ("sm", "exact"):
    a = make_agent(upd)
    us = [unit(RNG.normal(size=D)) for _ in range(7)]
    for u in us:
        fold(a, a.root, u)
    err = np.abs(a._cov_read(a.root) - brute_force_inv(us)).max()
    check(f"[{upd}] local V_inv == brute-force inverse", err < 1e-9,
          f"max|diff|={err:.3e} over {len(us)} folds")
    check(f"[{upd}] fold count tracked", a.root.cov_n_folds == len(us))
    check(f"[{upd}] one allocation for one node", a.cnt_cov_nodes == 1)

# --- 3. isolation: folding at A must not move B (the whole point)
a = make_agent("sm")
n_a, n_b = a.create_node(parent=a.root), a.create_node(parent=a.root)
before_b = a._cov_read(n_b).copy()
for _ in range(5):
    fold(a, n_a, unit(RNG.normal(size=D)))
check("local: folding at A leaves B at the ridge init",
      np.allclose(a._cov_read(n_b), before_b))
check("local: A actually moved",
      not np.allclose(a._cov_read(n_a), before_b))
check("local: two nodes -> one allocation (only A folded)",
      a.cnt_cov_nodes == 1)

# --- 4. global scope: the same folds are shared by every node
g = make_agent("sm", cov_scope="global")
g_a, g_b = g.create_node(parent=g.root), g.create_node(parent=g.root)
for _ in range(5):
    fold(g, g_a, unit(RNG.normal(size=D)))
check("global: folding at A also moves B (pre-merge semantics)",
      np.allclose(g._cov_read(g_b), g._cov_read(g_a)))
check("global: no per-node allocation",
      g.cnt_cov_nodes == 0 and g_a.V_inv_local is None)

# --- 5. sm vs exact agree on the same fold sequence
us = [unit(RNG.normal(size=D)) for _ in range(20)]
outs = {}
for upd in ("sm", "exact"):
    a = make_agent(upd)
    for u in us:
        fold(a, a.root, u)
    outs[upd] = a._cov_read(a.root)
err = np.abs(outs["sm"] - outs["exact"]).max()
check("sm == exact after 20 local folds", err < 1e-9,
      f"max|diff|={err:.3e}")

# --- 6. fp32 path runs and stays close to fp64
a32 = make_agent("sm", cov_dtype="fp32")
for u in us:
    fold(a32, a32.root, u)
err32 = np.abs(a32._cov_read(a32.root) - outs["exact"]).max()
check("fp32 local path within 1e-3 of fp64 exact", err32 < 1e-3,
      f"max|diff|={err32:.3e}")


print()
print("=" * 62)
print("Part A2 -- embeds_ref (_cov_vec)")
print("=" * 62)


def node_with(agent, parent, vec):
    n = agent.create_node(parent=parent)
    n.embeds = np.asarray(vec, dtype=np.float64)
    return n


# --- 7. absolute mode returns the child's own embedding untouched
a = make_agent(embeds_ref="absolute")
p = node_with(a, a.root, unit(RNG.normal(size=D)))
c = node_with(a, p, unit(RNG.normal(size=D)))
check("absolute: _cov_vec is the child's embedding",
      np.array_equal(a._cov_vec(p, c), c.embeds))

# --- 8. relative mode returns the normalized displacement
a = make_agent(embeds_ref="relative")
p = node_with(a, a.root, unit(RNG.normal(size=D)))
c = node_with(a, p, unit(RNG.normal(size=D)))
got = a._cov_vec(p, c)
want = unit(c.embeds - p.embeds)
check("relative: _cov_vec == unit(child - parent)",
      np.allclose(got, want), f"max|diff|={np.abs(got-want).max():.3e}")
check("relative: result is unit-norm (ds_alpha stays comparable)",
      abs(np.linalg.norm(got) - 1.0) < 1e-12,
      f"||x||={np.linalg.norm(got):.12f}")

# --- 9. the shared component really is what gets removed
#     Siblings that share a long text prefix cluster near the parent;
#     absolute vectors are then nearly collinear, differences are not.
base = unit(RNG.normal(size=D))
sibs = [unit(base + 0.05 * RNG.normal(size=D)) for _ in range(4)]
par = node_with(a, a.root, base)
kids = [node_with(a, par, s) for s in sibs]
abs_cos = np.mean([
    abs(float(sibs[i] @ sibs[j]))
    for i in range(4) for j in range(i + 1, 4)
])
rel = [a._cov_vec(par, k) for k in kids]
rel_cos = np.mean([
    abs(float(rel[i] @ rel[j]))
    for i in range(4) for j in range(i + 1, 4)
])
check("relative: decorrelates clustered siblings",
      rel_cos < abs_cos - 0.5,
      f"mean |cos| absolute={abs_cos:.3f} -> relative={rel_cos:.3f}")

# --- 10. root fallback: no parent embedding -> absolute
a = make_agent(embeds_ref="relative")
c = node_with(a, a.root, unit(RNG.normal(size=D)))
check("relative: root (embeds=None) falls back to absolute",
      a.root.embeds is None
      and np.array_equal(a._cov_vec(a.root, c), c.embeds))

# --- 10b. root WITH an embedding uses relative like any other node
#      (mcts_search calls _embed_root under embeds_ref="relative"; the
#      None branch above is the defensive path, not the depth-0 one)
a = make_agent(embeds_ref="relative")
a.root.embeds = unit(RNG.normal(size=D))
c = node_with(a, a.root, unit(RNG.normal(size=D)))
check("relative: embedded root uses the displacement, not absolute",
      np.allclose(a._cov_vec(a.root, c),
                  unit(c.embeds - a.root.embeds))
      and not np.array_equal(a._cov_vec(a.root, c), c.embeds))

# --- 10c. with the root embedded, NO fold anywhere is absolute --
#      This is what makes global+relative coherent: one V must not
#      mix absolute positions with displacements.
a = make_agent(embeds_ref="relative")
a.root.embeds = unit(RNG.normal(size=D))
kid = node_with(a, a.root, unit(RNG.normal(size=D)))
gkid = node_with(a, kid, unit(RNG.normal(size=D)))
absolute_folds = [
    n for n in (a.root, kid) if n.embeds is None
]
check("relative: embedded root leaves zero absolute-fallback nodes",
      not absolute_folds
      and not np.array_equal(a._cov_vec(a.root, kid), kid.embeds)
      and not np.array_equal(a._cov_vec(kid, gkid), gkid.embeds))

# --- 11. identical child/parent -> zero vector, and folding is a no-op
a = make_agent("sm", embeds_ref="relative")
v = unit(RNG.normal(size=D))
p = node_with(a, a.root, v)
c = node_with(a, p, v.copy())
z = a._cov_vec(p, c)
check("relative: identical embeddings -> zero vector",
      np.allclose(z, 0.0))
before = a._cov_read(p).copy()
fold(a, p, z)
check("relative: folding a zero vector is a no-op",
      np.allclose(a._cov_read(p), before))


print()
print("=" * 62)
print("Part A3 -- guards")
print("=" * 62)


def raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        return needle in str(e)
    return False


check("unknown cov_scope rejected",
      raises(lambda: make_agent(cov_scope="tree"), "unknown cov_scope"))
check("unknown embeds_ref rejected",
      raises(lambda: make_agent(embeds_ref="sibling"),
             "unknown embeds_ref"))
check("relative + center_mode=local rejected (double-centering)",
      raises(lambda: make_agent(embeds_ref="relative", embeds_center=True,
                                embeds_center_mode="local"),
             "double-centers"))
check("relative + center_mode=fixed allowed (different offset)",
      make_agent(embeds_ref="relative", embeds_center=True,
                 embeds_center_mode="fixed") is not None)

# The memory guard: d=4096 + gen_budget=320 is a legal pre-merge
# config (embeds_proj=none forces embeds_dim to the raw PRM hidden
# size) that would need ~40 GiB per question under local scope.
big = lambda d, b, scope: (lambda: MCTS(   # noqa: E731
    config=Cfg(embeds_dim=d, lam=LAM, cov_update="sm", cov_scope=scope,
               gen_budget=b),
    question="q"))
check("local + d=4096/b=320 refused (~40 GiB)",
      raises(big(4096, 320, "local"), "over the"))
check("local + d=2048/b=320 refused (~10 GiB)",
      raises(big(2048, 320, "local"), "over the"))
check("local + d=512/b=320 allowed (0.6 GiB)",
      big(512, 320, "local")() is not None)
check("GLOBAL + d=4096/b=320 still allowed (one matrix, 128 MiB)",
      big(4096, 320, "global")() is not None)


print()
print("=" * 62)
print("Part B -- end-to-end equivalence on a scripted generator")
print("=" * 62)


class _Info:
    """Stands in for a vLLM output row (see _generate_candidates)."""
    def __init__(self, text, stop):
        self.next_texts = [text]
        self.stop_reasons = [stop]
        self.lookahead_texts = [text]


class ScriptedSearch:
    """Deterministic generator+PRM: candidate text, embedding and
    score are all pure functions of (node text, index), so two runs
    differ only if the SELECTION differs."""

    def __init__(self, branch=3, dim=D, depth_eos=4):
        self.branch, self.dim, self.depth_eos = branch, dim, depth_eos

    def _vec(self, key):
        r = np.random.default_rng(abs(hash(key)) % (2**31))
        return unit(r.normal(size=self.dim))

    def candidates(self, node):
        infos, embeds, scores = [], [], []
        for i in range(self.branch):
            key = f"{node.state['text']}|{i}"
            eos = node.depth + 1 >= self.depth_eos
            infos.append(_Info(f"s{node.depth}_{i}\n\n",
                               "EOS" if eos else "length_step"))
            embeds.append(self._vec(key))
            scores.append(float(
                np.random.default_rng(abs(hash(key)) % (2**31)).random()
            ))
        return infos, embeds, scores


def run_scripted(cov_scope, embeds_ref, seed=7, phases=25, budget=40,
                 cov_update="sm", ds_alpha=10.0):
    """Drive the real MCTS through a scripted expansion loop.

    Mirrors mcts_search's descent exactly (terminal check, expand-if-
    childless, select, budget break) without touching vLLM/PRM.
    Returns the sequence of selected node tags -- the finest-grained
    observable of "did the two configs make the same choices".
    """
    import random as _random
    cfg = Cfg(embeds_dim=D, lam=LAM, cov_update=cov_update,
              cov_scope=cov_scope, embeds_ref=embeds_ref,
              ds_alpha=ds_alpha, ds_beta=1.0, max_depth=6,
              gen_budget=budget, num_phases=phases, batch_size=3)
    np.random.seed(seed)
    _random.seed(seed)
    agent = MCTS(config=cfg, question="q")
    script = ScriptedSearch()

    trace, gen_cnt = [], 0
    for p in range(phases):
        node = agent.root
        for d in range(cfg.search.max_depth + 1):
            if node.is_terminal:
                agent.backprop(node)
                break
            if not node.has_children():
                gen_cnt += 1
                infos, embeds, scores = script.candidates(node)
                agent.expand_node(node, infos, embeds, scores, p, gen_cnt)
            nxt = agent.select_child(node)
            if nxt is None:
                break
            trace.append(nxt.tag)
            node = nxt
            if gen_cnt >= cfg.search.gen_budget:
                break
        if gen_cnt >= cfg.search.gen_budget:
            break
    return trace, gen_cnt, agent


# --- 12. the loop actually exercises the code (not a trivial trace)
t_gg, g_gg, a_gg = run_scripted("global", "absolute")
check("scripted loop produces a non-trivial trace",
      len(t_gg) > 30 and g_gg > 5,
      f"{len(t_gg)} selections, gen_cnt={g_gg}")

# --- 13. determinism: same config twice -> identical trace
t_again, _, _ = run_scripted("global", "absolute")
check("same config is reproducible", t_gg == t_again,
      f"{len(t_gg)} selections")

# --- 14. THE REGRESSION LOCK. cov_scope="global" + embeds_ref=
#     "absolute" is what every pre-merge run did. It must not have
#     moved -- and unlike the algebra checks, this pins RNG
#     consumption through the real select_child dispatcher.
t_local, _, a_local = run_scripted("local", "absolute")
_first = next(
    (i for i, (x, y) in enumerate(zip(t_gg, t_local)) if x != y), None
)
check("local vs global DIVERGE (the knob does something)",
      t_gg != t_local, f"first differs at index {_first}")
check("local scope allocated per-node covariances",
      a_local.cnt_cov_nodes > 0 and a_gg.cnt_cov_nodes == 0,
      f"local={a_local.cnt_cov_nodes} global={a_gg.cnt_cov_nodes}")

# --- 15. ds_alpha=0 kills the bonus -> scope cannot matter
t0_g, _, _ = run_scripted("global", "absolute", ds_alpha=0.0)
t0_l, _, _ = run_scripted("local", "absolute", ds_alpha=0.0)
check("ds_alpha=0: local == global (bonus multiplied by zero)",
      t0_g == t0_l, f"{len(t0_g)} selections")

# --- 16. embeds_ref changes selections, under both scopes
t_gp, _, _ = run_scripted("global", "relative")
t_lp, _, _ = run_scripted("local", "relative")
check("embeds_ref=relative changes selections (global scope)",
      t_gp != t_gg)
check("embeds_ref=relative changes selections (local scope)",
      t_lp != t_local)
check("all four cov_scope x embeds_ref cells are distinct",
      len({tuple(t_gg), tuple(t_local), tuple(t_gp), tuple(t_lp)}) == 4)

# --- 17. ds_alpha=0 also collapses embeds_ref (it only feeds V)
t0_gp, _, _ = run_scripted("global", "relative", ds_alpha=0.0)
check("ds_alpha=0: embeds_ref is inert too", t0_gp == t0_g)

# --- 18. exact and sm agree end-to-end under local scope
t_sm, _, _ = run_scripted("local", "relative", cov_update="sm")
t_ex, _, _ = run_scripted("local", "relative", cov_update="exact")
check("local+parent: cov_update sm == exact end-to-end",
      t_sm == t_ex, f"{len(t_sm)} selections")


print()
print("=" * 62)
print("Part C -- the REAL mcts_search loop + the results schema")
print("=" * 62)

# Part B mirrors the descent loop; Part C runs the actual one. The
# seam is _generate_candidates: patching that single module-level
# function leaves everything above it -- the descent, the None
# guard, the results dict -- as the code under test.


class _Ns:
    """Attribute bag standing in for a config subtree."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FullCfg(Cfg):
    """Cfg plus the llm/gen subtrees mcts_search itself reads."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.llm = _Ns(use_custom_template=False)
        self.gen = _Ns(temperature=0.8, max_tokens=64, top_p=1.0,
                       system_prompt="", date_string="",
                       agg_strategy="last", custom_chat_template=None)


class _FakeLLM:
    """mcts_search calls only get_tokenizer(); the tokenizer itself
    is never touched because use_custom_template is False and
    _compute_response_start_idx is patched out."""
    def get_tokenizer(self):
        return _Ns(chat_template=None)


@contextlib.contextmanager
def patched_generation(starve=False, branch=3):
    """Swap generation for the scripted stub. starve=True returns
    ZERO candidates, the only way to reach select_child -> None."""
    script = ScriptedSearch(branch=branch)

    def fake_gen(question, current_node, d, p, config, tokenizer,
                 llm_vllm, llm_vllm_embeds, prm, response_start_idx,
                 sampling_params):
        if starve:
            return [], [], []
        return script.candidates(current_node)

    orig_gen = sem._generate_candidates
    orig_idx = sem._compute_response_start_idx
    sem._generate_candidates = fake_gen
    sem._compute_response_start_idx = lambda *a, **kw: 0
    try:
        yield
    finally:
        sem._generate_candidates = orig_gen
        sem._compute_response_start_idx = orig_idx


def full_cfg(cov_scope, budget=40, phases=25, branch=3):
    return FullCfg(embeds_dim=D, lam=LAM, cov_scope=cov_scope,
                   embeds_ref="absolute", ds_alpha=10.0, ds_beta=1.0,
                   max_depth=6, gen_budget=budget, num_phases=phases,
                   batch_size=branch)


def drive_one(cov_scope, starve=False, **kw):
    """Real mcts_search on one question; returns (tuple, agent) so
    the tree can be inspected after the call."""
    import random as _random
    cfg = full_cfg(cov_scope, **kw)
    np.random.seed(7)
    _random.seed(7)
    agent = MCTS(config=cfg, question="q")
    with patched_generation(starve=starve, branch=cfg.search.batch_size):
        out = mcts_search("q", agent, cfg, _FakeLLM(), None, None)
    return out, agent


def drive_batch(cov_scope, questions=("q0", "q1"), **kw):
    """Real _search over a batch; returns the results dict."""
    cfg = full_cfg(cov_scope, **kw)
    with patched_generation(branch=cfg.search.batch_size):
        return sem._search(list(questions), cfg, 0,
                           _FakeLLM(), None, None)


# --- 19. the real loop runs, and now returns cnt_cov_nodes
out_l, agent_l = drive_one("local")
check("real mcts_search runs end-to-end", len(out_l) == 9,
      f"{len(out_l)} fields, {len(out_l[0])} completions")
check("9th return field is the agent's cnt_cov_nodes",
      out_l[8] == agent_l.cnt_cov_nodes and out_l[8] > 0,
      f"cnt_cov_nodes={out_l[8]}")

# --- 20. ISSUE 6b. A zero-candidate expansion leaves the node
#     childless, so select_child returns None. Before the guard this
#     wrote None into current_node and died one iteration later on
#     `current_node.is_terminal`. Only reachable via this stub today
#     (batch_size >= 1 always yields >= 1 candidate), which is
#     exactly why it needs a test rather than a live run.
_starved_err = None
try:
    out_s, agent_s = drive_one("local", starve=True)
except Exception as exc:                       # noqa: BLE001
    _starved_err = f"{type(exc).__name__}: {exc}"
check("zero-candidate expansion does not crash the descent",
      _starved_err is None, _starved_err or "returned cleanly")
if _starved_err is None:
    check("starved run returns an empty but well-formed result",
          len(out_s) == 9 and out_s[0] == [],
          f"{len(out_s[6])} phases recorded")
    # The guard backprops BEFORE breaking, so each starved phase
    # still credits the node it was standing on -- here the root,
    # 25 times. The +1 is BaseTree.__init__ seeding the root with
    # one visit. A bare `break` would leave it at that seed alone.
    check("the aborted phase is still backpropped (work credited)",
          agent_s.root.visit_count() == 1 + 25,
          f"root visits={agent_s.root.visit_count()} (1 seed + 25)")

# --- 21. ISSUE 5. The results schema. Global runs keep the EXACT
#     key set every earlier v02 run wrote, so a mixed-vintage set of
#     trial files still unions cleanly -- and because cnt_cov_nodes
#     is structurally 0 under global, the column would be all zeros
#     anyway. (The stats path is indifferent: evaluate_correctness
#     returns a fixed key tuple, so extra columns never reach
#     _load_trials. This check guards the artifact, not the stats.)
_PRE_CHANGE_KEYS = {
    "completions", "comp_depth", "comp_phase", "comp_gen",
    "q_total_gens", "q_last_phase", "phase_depths",
    "q_nodes_max_depth",
}
res_g = drive_batch("global")
check("cov_scope=global keeps the pre-change key set exactly",
      set(res_g) == _PRE_CHANGE_KEYS,
      f"extra={sorted(set(res_g) - _PRE_CHANGE_KEYS)}")

res_l = drive_batch("local")
check("cov_scope=local adds exactly q_cov_nodes",
      set(res_l) - _PRE_CHANGE_KEYS == {"q_cov_nodes"},
      f"keys={sorted(set(res_l) - _PRE_CHANGE_KEYS)}")

# --- 22. the auto-attach contract: core.scoring.build_scored_dataset
#     attaches a results value iff it is a list whose length equals
#     the question count. Anything else is silently dropped.
_qcn = res_l["q_cov_nodes"]
check("q_cov_nodes satisfies build_scored_dataset's attach rule",
      isinstance(_qcn, list) and len(_qcn) == len(res_l["completions"]),
      f"len={len(_qcn)} questions={len(res_l['completions'])}")
check("q_cov_nodes is a positive int per question",
      all(isinstance(v, int) and v > 0 for v in _qcn), f"{_qcn}")

# --- 23. the memory guard's premise: cnt_cov_nodes is bounded by
#     gen_budget, so peak bytes <= gen_budget * d^2 * itemsize.
check("q_cov_nodes stays within gen_budget (guard's bound holds)",
      all(v <= 40 for v in _qcn), f"max={max(_qcn)} budget=40")


print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
