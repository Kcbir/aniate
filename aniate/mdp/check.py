"""Does a simulator behave like the model says it should?

``check`` runs episodes through a simulator and tests every observed step
against the model:

* **impossible transitions** -- a successor the model gives probability 0
* **transition frequencies** -- a chi-square test per state-action pair
* **rewards** -- each ``(s, a, s')`` reward against the model's
* **termination** -- the simulator ends episodes exactly where the model does
* **start states** -- against the initial distribution
* **returns** -- the mean sampled return against the exact value of the policy

The simulator is the model's own :class:`Env` by default.  Pass ``env=`` to
test *your* simulator -- anything with Gym-style ``reset()`` and ``step()``.
Statistical tests are Bonferroni-corrected, so ``alpha`` is the chance of
any false alarm within a test family.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import chi2, norm

from aniate import log as _log
from aniate.mdp.env import Env, _is_product, _unwrap, default_max_steps, truncation_bias
from aniate.mdp.policies import as_policy, policy_matrix
from aniate.mdp.validate import Issue, reachable

__all__ = ["check", "CheckReport"]

MIN_EXPECTED = 5.0


def _pooled_chi2(observed, probs):
    """Pearson chi-square with cells of expected count < 5 pooled.  None if untestable."""
    n = observed.sum()
    if n == 0:
        return None
    expected = probs * n
    order = np.argsort(expected)
    bins_o, bins_e, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(observed[order], expected[order]):
        acc_o += o
        acc_e += e
        if acc_e >= MIN_EXPECTED:
            bins_o.append(acc_o)
            bins_e.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        if not bins_e:
            return None
        bins_o[-1] += acc_o
        bins_e[-1] += acc_e
    if len(bins_e) < 2:
        return None
    o, e = np.array(bins_o), np.array(bins_e)
    stat = float(((o - e) ** 2 / e).sum())
    return stat, len(e) - 1, float(chi2.sf(stat, len(e) - 1))


def _unpack_reset(out):
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out
    return out, {}


def _unpack_step(out):
    if len(out) == 5:
        obs, r, term, trunc, _ = out
    elif len(out) == 4:                     # old gym: (obs, reward, done, info)
        obs, r, done, info = out
        trunc = bool(info.get("TimeLimit.truncated", False))
        term = bool(done) and not trunc
    else:
        raise TypeError(f"env.step() must return 4 or 5 values; got {len(out)}")
    return obs, float(r), bool(term), bool(trunc)


class CheckReport:
    """What ``check`` found: ``ok``, ``issues``, and the numbers in ``stats``."""

    def __init__(self, issues, stats):
        self.issues = list(issues)
        self.stats = dict(stats)

    @property
    def ok(self):
        return not self.issues

    def describe(self):
        """The report as a JSON-serialisable dict."""
        stats = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                 for k, v in self.stats.items()}
        return {"kind": "CheckReport", "ok": self.ok, "stats": stats,
                "issues": [i.describe() for i in self.issues]}

    def _repr_html_(self):
        from aniate.notebook import to_html

        return to_html(self)

    def __repr__(self):
        st = self.stats
        head = "PASSED" if self.ok else f"FAILED ({len(self.issues)} issue(s))"
        lines = [f"CheckReport {head} -- {st['episodes']:,} episodes, {st['steps']:,} steps"]
        lines += [str(i) for i in self.issues]
        lines.append(f"  coverage    : {st['pairs_visited']} of {st['pairs_reachable']} reachable "
                     "state-action pairs visited")
        lines.append(f"  transitions : {st['pairs_tested']} pairs tested at alpha={st['alpha']:g}; "
                     f"{st['pairs_untested']} deterministic or too rarely visited to test")
        lines.append(f"  rewards     : {st['rewards_compared']} distinct transitions compared")
        if st.get("model_return") is not None:
            lines.append(f"  returns     : sampled {st['sampled_return']:.6g} +/- {st['sampled_stderr']:.2g} "
                         f"(s.e.), model {st['model_return']:.6g}")
        else:
            lines.append(f"  returns     : not compared ({st.get('return_note', 'n/a')})")
        return "\n".join(lines)


def check(model, env=None, policy=None, *, episodes=300, max_steps=None, seed=0,
          state=None, action=None, alpha=1e-3):
    """Simulate and test behaviour against ``model``.  Returns :class:`CheckReport`.

    Parameters
    ----------
    model : MDP, or an NMDP / non-Markovian model
    env : simulator, optional
        Gym-style: ``reset()`` returns ``obs`` or ``(obs, info)``; ``step(action)``
        returns ``(obs, reward, terminated, truncated, info)`` or the old 4-tuple.
        For a non-Markovian model it observes *world* states; memory is tracked
        here, from the automaton, independently of the product construction.
    policy : optional
        Any policy form; uniformly random (which explores) by default.
    state : callable ``obs -> state label``, optional
        How to read the simulator's observations.  Default: label, or position.
    action : callable ``action position -> what env.step expects``, optional
        Default: the action label for the built-in Env, the position otherwise.
    alpha : float
        Family-wise false-alarm rate for each statistical test.
    """
    from aniate.mdp.text import state_str

    m = _unwrap(model)
    product = _is_product(m)
    world = m.base if product else m
    own = env is None
    rng = np.random.default_rng(seed)
    T = default_max_steps(m) if max_steps is None else int(max_steps)
    if own:
        env = Env(m, seed=seed, labels=True, hide_memory=product, max_steps=T, log=False)
    encode = state or world.states.index
    decode = action or ((lambda a: m.actions[a]) if own else (lambda a: a))
    pol = as_policy(m, policy, rng)

    counts = defaultdict(lambda: defaultdict(int))            # (s, a) -> {s2: n}
    rstats = {}                                                # (s, a, s2) -> [n, sum, sumsq, min, max]
    starts = np.zeros(world.S)
    term_bad, unknown = [], []
    returns, lengths, truncs = [], [], []
    steps = 0

    for ep in range(int(episodes)):
        if ep == 0 and not own:
            try:
                out = env.reset(seed=seed)
            except TypeError:
                out = env.reset()
        else:
            out = env.reset()
        obs, _ = _unpack_reset(out)
        try:
            w = encode(obs)
        except (KeyError, IndexError, TypeError) as exc:
            unknown.append(f"reset() observation {obs!r}: {exc}")
            break
        starts[w] += 1
        q = m.machine.initial if product else None
        s = m.flat(world.states[w], q) if product else w
        pol.reset()
        G, disc, L, ended = 0.0, 1.0, 0, bool(m.terminal[s])
        while not ended and L < T:
            a = pol.step(s)
            obs, r, term, trunc = _unpack_step(env.step(decode(a)))
            try:
                w2 = encode(obs)
            except (KeyError, IndexError, TypeError) as exc:
                unknown.append(f"step() observation {obs!r}: {exc}")
                ended = True
                break
            if product:
                q = m.machine.step(q, m.labels.step_events(w, a, w2))[0]
                s2 = m.flat(world.states[w2], q)
                model_term = bool(m.terminal[s2]) if own else bool(world.terminal[w2])
            else:
                s2 = w2
                model_term = bool(m.terminal[s2])
            counts[(s, a)][s2] += 1
            st = rstats.get((s, a, s2))
            if st is None:
                rstats[(s, a, s2)] = [1, r, r * r, r, r]
            else:
                st[0] += 1
                st[1] += r
                st[2] += r * r
                st[3] = min(st[3], r)
                st[4] = max(st[4], r)
            if term != model_term and len(term_bad) < 50:
                term_bad.append((s, a, s2, term))
            G += disc * r
            disc *= m.gamma
            L += 1
            s, w = s2, w2
            ended = term or model_term or bool(m.terminal[s])
            if trunc and not ended:
                break
        steps += L
        returns.append(G)
        lengths.append(L)
        truncs.append(not ended)

    def sa(s, a):
        return f"{state_str(m, s)} / {m.actions[a]}"

    issues = []
    if unknown:
        issues.append(Issue("unknown_observation", "error",
                            "the simulator produced observations that are not model states "
                            "(pass state= to translate them)", unknown, len(unknown)))

    impossible, tests, untested = [], [], 0
    for (s, a), nexts in counts.items():
        idx, prob, _ = m._row(s, a)
        support = {int(j): float(p) for j, p in zip(idx, prob) if p > 0}
        for j, n in nexts.items():
            if j not in support:
                impossible.append(f"{sa(s, a)} -> {state_str(m, j)} (seen {n}x)")
        cols = list(support)
        observed = np.array([nexts.get(j, 0) for j in cols], dtype=float)
        p = np.array([support[j] for j in cols])
        res = _pooled_chi2(observed, p / p.sum())
        if res is None:
            untested += 1
        else:
            tests.append((s, a, cols, observed, p, res))
    if impossible:
        issues.append(Issue("impossible_transition", "error",
                            "the simulator made transitions the model says cannot happen",
                            impossible, len(impossible)))
    if tests:
        cut = alpha / len(tests)
        rejected = sorted((t for t in tests if t[5][2] < cut), key=lambda t: t[5][2])
        if rejected:
            ex = []
            for s, a, cols, observed, p, (_, _, pval) in rejected:
                n = observed.sum()
                worst = int(np.argmax(np.abs(observed / n - p)))
                ex.append(f"{sa(s, a)} -> {state_str(m, cols[worst])}: seen "
                          f"{observed[worst] / n:.3f}, model {p[worst]:.3f} (n={int(n)}, p={pval:.1e})")
            issues.append(Issue("transition_frequency", "error",
                                "transition frequencies disagree with the model's probabilities",
                                ex, len(rejected)))

    reward_bad = []
    keys = [k for k in rstats if np.any(m._row(k[0], k[1])[0] == k[2])]
    crit = norm.isf(alpha / (2 * max(len(keys), 1)))
    for (s, a, s2) in keys:
        n, tot, sq, lo, hi = rstats[(s, a, s2)]
        idx, _, rew = m._row(s, a)
        expected = float(rew[np.flatnonzero(idx == s2)[0]])
        mean = tot / n
        if hi - lo <= 1e-9 * (1.0 + abs(expected) + abs(mean)):
            bad = abs(mean - expected) > 1e-6 * (1.0 + abs(expected))
        else:
            var = max((sq - n * mean * mean) / max(n - 1, 1), 0.0)
            bad = n > 1 and abs(mean - expected) > crit * math.sqrt(var / n)
        if bad:
            reward_bad.append(f"{sa(s, a)} -> {state_str(m, s2)}: simulator {mean:+.6g}"
                              f"{' (mean)' if hi > lo else ''}, model {expected:+.6g}")
    if reward_bad:
        issues.append(Issue("reward_mismatch", "error",
                            "simulator rewards disagree with the model's transition rewards",
                            reward_bad, len(reward_bad)))

    if term_bad:
        ex = [f"{sa(s, a)} -> {state_str(m, s2)}: simulator "
              f"{'ended' if said else 'continued'}, model {'continues' if said else 'ends'}"
              for s, a, s2, said in term_bad]
        issues.append(Issue("termination_mismatch", "error",
                            "the simulator ends episodes in different places than the model",
                            ex, len(term_bad)))

    if starts.sum() > 0:
        bad_starts = [f"{world.states[i]} (seen {int(starts[i])}x)"
                      for i in np.flatnonzero((starts > 0) & (world.initial == 0))]
        if bad_starts:
            issues.append(Issue("impossible_start", "error",
                                "episodes started in states the initial distribution excludes",
                                bad_starts, len(bad_starts)))
        else:
            sup = np.flatnonzero(world.initial > 0)
            res = _pooled_chi2(starts[sup], world.initial[sup] / world.initial[sup].sum())
            if res is not None and res[2] < alpha:
                issues.append(Issue("start_frequency", "error",
                                    f"start-state frequencies disagree with the initial "
                                    f"distribution (p={res[2]:.1e})"))

    stats = {
        "episodes": len(returns), "steps": steps, "alpha": alpha,
        "pairs_visited": len(counts),
        "pairs_reachable": int((reachable(m) & ~m.terminal).sum() * m.A),
        "pairs_tested": len(tests), "pairs_untested": untested,
        "rewards_compared": len(keys),
    }
    returns = np.array(returns)
    if len(returns) > 1 and not unknown:
        try:
            v0 = float(m.initial @ _value(m, policy))
        except (TypeError, ValueError) as exc:
            stats["return_note"] = str(exc).split(";")[0]
        else:
            bias = truncation_bias(m, np.array(lengths), np.array(truncs))
            mean = float(returns.mean())
            se = float(returns.std(ddof=1) / math.sqrt(len(returns)))
            stats.update(sampled_return=mean, sampled_stderr=se, model_return=v0,
                         truncation_bias_bound=bias)
            if math.isfinite(bias):
                tol = norm.isf(alpha / 2) * se + bias + 1e-9 * (1.0 + abs(v0))
                if abs(mean - v0) > tol:
                    issues.append(Issue(
                        "return_mismatch", "error",
                        f"mean sampled return {mean:.6g} is {abs(mean - v0) / max(se, 1e-300):.1f} "
                        f"standard errors from the model's value {v0:.6g}"))
            else:
                stats["return_note"] = "episodes were truncated with gamma=1"

    report = CheckReport(issues, stats)
    _log.info("check", "PASSED" if report.ok else f"FAILED with {len(issues)} issue(s)",
              f"{stats['episodes']:,} episodes", f"{steps:,} steps",
              f"{stats['pairs_tested']} pairs tested",
              f"return {stats['sampled_return']:.5g} vs model {stats['model_return']:.5g}"
              if "model_return" in stats else None)
    for issue in issues:
        _log.warning("check", issue.line())
    return report


def _value(m, policy):
    from aniate.mdp.solve import evaluate

    policy_matrix(m, policy)          # raises TypeError for step()-only policies
    return evaluate(m, policy)
