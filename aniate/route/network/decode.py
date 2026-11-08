"""Simulation-guided beam search over the network: propose moves, finish each with a greedy rollout, keep the best."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


from .beam import _clone_env, _head_logps


def _refresh(ctx, obs, env):
    fn = getattr(ctx, "refresh", None)
    if fn is not None:
        fn(obs, env)
    return ctx


@dataclass
class _Node:
    env: object
    obs: dict
    ret: float
    logp: float
    acts: tuple = ()
    terminal: bool = False
    score: float = -np.inf


def _top_children(policy, ctx, node, device, gamma, collapse_zero_items=True):
    _refresh(ctx, node.obs, node.env)
    H = _head_logps(policy, ctx, node.obs, node.env, device)
    cands = []
    for pi, (rej, k_slot, k_action, m_j) in enumerate(H["pairs"]):
        base = H["rej_lp"][rej] + H["k_lp"][rej][k_slot]
        zd_lp_arr, m_zd = H["zd"][rej]
        for zd in np.flatnonzero(m_zd):
            m_zc_now = H["zc_feasible"][(rej, int(zd))]
            for zc in np.flatnonzero(m_zc_now):
                partial = base + zd_lp_arr[zd] + H["zc_lp"][zc]
                for j in np.flatnonzero(m_j):
                    cands.append((
                        float(partial + H["j_lp"][pi][j]),
                        {"j": int(j), "z_curr": int(zc), "rejoin": int(rej),
                         "z_drone": int(zd), "k": int(k_action)},
                    ))
    cands.sort(key=lambda c: c[0], reverse=True)
    # With zero profit AND zero weight, the two collection bits lead to exactly
    # the same simulator state. They must not consume separate beam slots.
    inst = node.env.inst
    if collapse_zero_items and not np.any(inst.profits) and not np.any(inst.weights):
        distinct, seen = [], set()
        for candidate in cands:
            action = candidate[1]
            key = (action["rejoin"], action["k"], action["j"])
            if key not in seen:
                seen.add(key)
                distinct.append(candidate)
                if len(distinct) == gamma:
                    break
        return distinct
    return cands[:gamma]


@torch.no_grad()
def _greedy_complete(policy, ctx, node, max_steps):
    env = _clone_env(node.env)
    obs = node.obs
    ret = node.ret
    acts = []
    for _ in range(max_steps):
        a = policy.act(obs, env, deterministic=True, ctx=ctx).action
        obs, r, term, trunc, _ = env.step(a)
        ret += float(r)
        acts.append(a)
        if term:
            return ret, tuple(acts)
        if trunc:
            return None, ()
    return None, ()


@torch.no_grad()
def sgbs_search(policy, env, inst, *, beta: int = 4, gamma: int = 4,
                n_aug: int = 8, ctx_override=None, return_actions: bool = False,
                collapse_zero_items: bool = True):
    device = policy.device
    max_depth = 2 * (inst.n + 2) + 2
    best_ret, best_acts = None, None

    n_views = 1 if ctx_override is not None else max(1, n_aug)
    for aug in range(n_views):
        ctx = (ctx_override if ctx_override is not None
               else policy.encode(inst, aug_idx=aug))
        e0 = _clone_env(env)
        obs0, _ = e0.reset(options={"instance": inst})
        beam = [_Node(env=e0, obs=obs0, ret=0.0, logp=0.0)]

        for _depth in range(max_depth):
            if all(nd.terminal for nd in beam):
                break
            cand = []
            for nd in beam:
                if nd.terminal:
                    cand.append(nd)
                    continue
                for lp, action in _top_children(policy, ctx, nd, device, gamma, collapse_zero_items):
                    child_env = _clone_env(nd.env)
                    obs, r, term, trunc, _ = child_env.step(action)
                    if trunc and not term:
                        continue
                    cand.append(_Node(
                        env=child_env, obs=obs, ret=nd.ret + float(r),
                        logp=nd.logp + lp,
                        acts=(nd.acts + (action,)) if return_actions else (),
                        terminal=bool(term)))
            if not cand:
                break

            for nd in cand:
                if nd.terminal:
                    nd.score = nd.ret
                    if best_ret is None or nd.ret > best_ret:
                        best_ret, best_acts = nd.ret, nd.acts
                    continue
                ret, tail = _greedy_complete(policy, ctx, nd, max_depth)
                if ret is None:
                    nd.score = -np.inf
                    continue
                nd.score = ret
                if best_ret is None or ret > best_ret:
                    best_ret = ret
                    best_acts = (nd.acts + tail) if return_actions else ()

            cand.sort(key=lambda nd: (nd.score, nd.logp), reverse=True)
            beam = cand[:beta]

    if return_actions:
        return best_ret, best_acts
    return best_ret
