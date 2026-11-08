"""The neural accelerator: a trained network that builds the plan instead of searching for it.

Use our weights, or train your own. Both need PyTorch:  pip install "aniate[neural]"

    ncube(nodes, profits, weights, drone=True)                # our weights

    net = ncube.train(30)                                     # your own, on 30 customers
    ncube(nodes, profits, weights, drone=True, network=net)
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS = "https://huggingface.co/Murjani/ttpd-weights/resolve/main/neural"
SIZES = (5, 10, 15, 20, 30, 40, 50)
BEAM, CHILDREN, VIEWS = 4, 4, 8


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError('ncube needs PyTorch: pip install "aniate[neural]"') from error
    return torch


def _cache():
    return Path(os.environ.get("ANIATE_CACHE", Path.home() / ".cache" / "aniate"))


def _ours(n):
    """The closest size we trained, downloaded once."""
    size = min(SIZES, key=lambda s: (abs(s - n), s))
    path = _cache() / f"a280_n{size}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(f"{WEIGHTS}/a280_n{size}.pt", part)
        except urllib.error.URLError as error:
            raise RuntimeError(f"could not download our weights from {WEIGHTS}: {error}. "
                               "Pass network= a path to your own, or train them with ncube.train") from error
        part.replace(path)
    return path


def load(source, n=None):
    """Load a network: True for our weights at size n, a path to a saved file, or a network already in memory."""
    torch = _torch()
    from .model import Network, NetworkConfig
    if isinstance(source, Network):
        return source.eval()
    payload = torch.load(_ours(n) if source is True else source, map_location="cpu", weights_only=True)
    network = Network(NetworkConfig(**payload["cfg"]))
    network.load_state_dict(payload["policy"])
    return network.eval()


def instance_of(P, order):
    """The solver's Problem as the network's instance. No flight limit reaches the network as unlimited."""
    import numpy as np
    from .instance import TTPDInstance
    return TTPDInstance(n=P.n, coords=np.array(order, dtype=float), profits=np.array(P.p, dtype=float),
                        weights=np.array(P.w, dtype=float), dist=np.array(P.dist, dtype=float), W=P.Wcap, v_max=P.vmax,
                        v_min=P.vmin, v_D=P.vD, R=P.R, node_ids=list(range(P.n + 2)), ED=P.ED)


def plan_of(n, actions):
    """The network's action sequence as (truck, sorties, z)."""
    sink = n + 1
    truck, sorties, z = [], [], [0] * (n + 2)
    here, launch, target = 0, -1, -1
    for action in actions:
        if action.get("rejoin") and target >= 0:
            sorties.append((launch, target, here))
            z[target] = int(action.get("z_drone", 0))
            launch, target = -1, -1
        if 0 < here <= n:
            z[here] = int(action.get("z_curr", 0))
        if int(action.get("k", -1)) >= 0:
            launch, target = here, int(action["k"])
        if here == sink:
            break
        here = int(action.get("j", sink))
        if 0 < here <= n:
            truck.append(here)
    return tuple(truck), tuple(sorties), tuple(z)


def accelerate(P, order, source):
    """Plan with the network alone. The paper found no gain from handing its plan to the search."""
    _torch()
    from .decode import sgbs_search
    from .simulator import TTPDEnv
    network = load(source, P.n)
    inst = instance_of(P, order)
    value, actions = sgbs_search(network, TTPDEnv(n=P.n), inst, beta=BEAM, gamma=CHILDREN, n_aug=VIEWS, return_actions=True)
    if value is None:
        raise RuntimeError("the network found no feasible plan; use cube instead")
    return plan_of(P.n, actions)


def train(problems, **options):
    _torch()
    from .train import train as run
    return run(problems, **options)
