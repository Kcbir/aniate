"""Train the neural accelerator by imitating the metaheuristic, as the paper does.

    net = ncube.train(30)                       # random problems with 30 customers
    net = ncube.train(my_problems)              # or a list of (nodes, profits, weights)
    ncube(nodes, profits, weights, drone=True, network=net)
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from .. import _problem
from ..metaheuristic import run
from . import instance_of
from .attention import candidate_features
from .masking import NO_LAUNCH
from .model import Network, NetworkConfig, build_dyn_feats
from .simulator import TTPDEnv, actions_of

BOX, PROFIT, WEIGHT = 300, (1, 1000), (1000, 1009)     # the paper's uniform-box recipe


def _random_problems(n, count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        nodes = [(rng.randint(0, BOX), rng.randint(0, BOX)) for _ in range(n + 1)]
        yield nodes, [rng.randint(*PROFIT) for _ in range(n)], [rng.randint(*WEIGHT) for _ in range(n)]


def _steps(inst, actions):
    """Walk a plan through the environment, keeping the state and masks the network will see at each step."""
    slot = inst.n + 2
    env = TTPDEnv(n=inst.n)
    obs, _ = env.reset(options={"instance": inst})
    steps = []
    for a in actions:
        rejoin, z_drone = int(a.get("rejoin", 0)), int(a.get("z_drone", 0))
        k, j = int(a.get("k", NO_LAUNCH)), int(a.get("j", inst.sink))
        feat_j, feat_k = candidate_features(obs, inst)
        steps.append(dict(
            c_t=int(obs["c_t"]), U_mask=np.asarray(obs["U_mask"], dtype=bool).copy(),
            scalars=np.array([obs["W_norm"], obs["tau_norm"], obs["U_frac"], obs["R_norm"], obs["drone_norm"],
                              obs["drone_wait_norm"], obs["drone_elapsed_norm"], obs["ED_norm"],
                              obs["endurance_slack_norm"]], dtype=np.float32),
            dyn=build_dyn_feats(obs, env, inst), feat_j=feat_j, feat_k=feat_k,
            drone_D_slot=int(obs["drone_D"]) if bool(obs["in_flight"]) else slot,
            a_rej=rejoin, a_zd=z_drone, a_zc=int(a.get("z_curr", 0)), a_k=slot if k == NO_LAUNCH else k, a_j=j,
            m_rej=np.asarray(env.current_masks()["rejoin"], dtype=bool),
            m_zd=np.asarray(env.mask_z_drone_now(), dtype=bool) if rejoin else np.array([True, False]),
            m_zc=np.asarray(env.mask_z_curr_after(rejoin, z_drone), dtype=bool),
            m_k_ext=np.asarray(env.masks_for_launch(rejoined=rejoin == 1)["k_ext"], dtype=bool),
            m_j=np.asarray(env.mask_j_after_launch(k, rejoined=rejoin == 1), dtype=bool)))
        obs, _, done, cut, _ = env.step(a)
        if done or cut:
            break
    return steps


def _buffer(examples, network, views):
    rows = []
    for inst, steps in examples:
        for view in range(views):
            feats = network.build_encoder_inputs(inst, network.device, view).squeeze(0).detach()
            rows.extend((feats, step) for step in steps)
    stack = lambda key, dtype: torch.as_tensor(np.stack([s[key] for _, s in rows]), dtype=dtype)
    number = lambda key: torch.tensor([s[key] for _, s in rows], dtype=torch.long)
    buffer = {"feats": torch.stack([f for f, _ in rows]), "dyn": stack("dyn", torch.float32),
              "feat_j": stack("feat_j", torch.float32), "feat_k": stack("feat_k", torch.float32),
              "scalars": stack("scalars", torch.float32), "U_mask": stack("U_mask", torch.bool)}
    buffer.update({key: number(key) for key in ("c_t", "drone_D_slot", "a_rej", "a_zd", "a_zc", "a_k", "a_j")})
    buffer.update({key: stack(key, torch.bool) for key in ("m_rej", "m_zd", "m_zc", "m_k_ext", "m_j")})
    return buffer


def _loss(network, data, rows):
    out = network.forward_heads(
        feats=data["feats"][rows], dyn=data["dyn"][rows], c_t=data["c_t"][rows], scalars=data["scalars"][rows],
        U_mask=data["U_mask"][rows], drone_D_slot=data["drone_D_slot"][rows], mask_rejoin=data["m_rej"][rows],
        mask_zd=data["m_zd"][rows], mask_zc=data["m_zc"][rows], mask_k_ext=data["m_k_ext"][rows],
        mask_j=data["m_j"][rows], a_rejoin=data["a_rej"][rows], a_zd=data["a_zd"][rows], a_zc=data["a_zc"][rows],
        a_k_slot=data["a_k"][rows], a_j=data["a_j"][rows], feat_j=data["feat_j"][rows], feat_k=data["feat_k"][rows])
    return -out["log_prob"].mean()


def train(problems, *, examples=100, seconds=None, epochs=60, save=None, seed=0, log=True,
          batch=512, rate=5e-4, views=4, endurance=None):
    """Solve `problems` with the metaheuristic, then teach the network to imitate those plans.

    problems   a number of customers for random problems, or a list of (nodes, profits, weights), all one size.
    examples   how many random problems to make.
    seconds    search time for each one. Half a second per customer by default.
    epochs     passes over the plans, stopping early once held-out plans stop improving.
    save       a path to write the trained weights to.
    endurance  the longest drone flight in your problems. Unlimited by default.
    """
    talk = print if log else (lambda *_: None)
    torch.manual_seed(seed)
    if isinstance(problems, int):
        problems = list(_random_problems(problems, examples, seed))
    if len({len(p[0]) for p in problems}) != 1:
        raise ValueError("train one size at a time: every problem needs the same number of customers")

    solved, started = [], time.perf_counter()
    for i, (nodes, profits, weights) in enumerate(problems):
        P, _, order = _problem(nodes, profits, weights, True, endurance, None, None, None, None)
        truck, sorties, z, _ = run(P, max(1.0, 0.5 * P.n) if seconds is None else seconds, seed=seed + i)
        inst = instance_of(P, order)
        solved.append((inst, _steps(inst, actions_of(P.n, truck, sorties, z))))
        talk(f"\rsolved {i + 1} of {len(problems)}", end="", flush=True) if log else None
    talk(f"\rsolved {len(solved)} problems in {time.perf_counter() - started:.0f} s")

    network = Network(NetworkConfig())
    held = max(1, len(solved) // 10) if len(solved) > 1 else 0
    data = _buffer(solved[held:], network, views)
    check = _buffer(solved[:held], network, 1) if held else None
    optimiser = torch.optim.AdamW(network.parameters(), lr=rate, weight_decay=1e-4)

    rows_total = data["c_t"].shape[0]
    best, best_state, patience = math.inf, None, 0
    for epoch in range(epochs):
        for group in optimiser.param_groups:
            group["lr"] = rate * min(1.0, (epoch + 1) / 5) * (0.5 + 0.5 * math.cos(math.pi * epoch / max(epochs, 1)))
        network.train()
        order = torch.randperm(rows_total)
        for at in range(0, rows_total, batch):
            loss = _loss(network, data, order[at:at + batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
        network.eval()
        with torch.no_grad():
            score = float(_loss(network, check, torch.arange(check["c_t"].shape[0]))) if check else float(loss)
        if score < best - 1e-4:
            best, patience = score, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            patience += 1
        talk(f"epoch {epoch + 1}: held-out loss {score:.4f}")
        if patience >= 20:
            break

    network.load_state_dict(best_state)
    network.eval()
    if save is not None:
        torch.save({"policy": network.state_dict(), "cfg": vars(network.cfg)}, save)
        talk(f"saved {save}")
    return network
