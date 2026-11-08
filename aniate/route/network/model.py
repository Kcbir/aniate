"""The neural accelerator's network: a light encoder with a heavy, state-conditioned decoder.

The architecture follows Luo et al., "Neural Combinatorial Optimization with Heavy Decoder: Toward Large
Scale Generalization", NeurIPS 2023, with the TTP-D's five action heads. The decoder re-reads every node
at every step, so it always knows where the truck is, what is left, and how far the drone can still reach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


from .attention import AttentionPolicy, candidate_features
from .decoder import DecoderConfig, StructuredDecoder
from .encoder import EncoderConfig, GATLayer

# Per-node dynamic features, rebuilt at every decision epoch. Order is fixed and
# is relied on by the saved weights.
DYN_FEATS = (
    "is_current",        # the truck is standing here
    "is_available",      # unserved and uncommitted
    "is_drone_target",   # the in-flight sortie is headed here
    "is_sink",           # the return depot copy
    "d_from_c",          # distance from the truck, over d_max
    "d_to_sink",         # distance on to the sink, over d_max
    "feas_j",            # admissible as the next truck move
    "feas_k",            # admissible as a launch target
    "w_over_slack",      # item weight over the truck's remaining capacity
    "sortie_cost",       # cheapest round trip c_t -> i -> rejoin, over ED
)
DYN_DIM = len(DYN_FEATS)


@dataclass
class NetworkConfig:
    node_feat_dim: int = 4
    dyn_feat_dim: int = DYN_DIM
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6        # the heavy decoder's depth
    d_ff: int = 512
    scalar_dim: int = 9      # matches DecoderConfig


def build_dyn_feats(obs: dict, env, inst) -> np.ndarray:
    """(M, DYN_DIM) state-conditioned node features, M = n + 2.

    Distances are dihedral-invariant, so this block does not depend on the
    augmentation view and is shared across the eight views of one state.
    """
    M = inst.n + 2
    c = int(obs["c_t"])
    sink = inst.sink
    d = inst.dist
    d_max = float(inst.d_max) if inst.d_max > 0 else 1.0

    avail = np.asarray(obs["U_mask"], dtype=bool)
    in_flight = bool(obs["in_flight"])
    drone_D = int(obs["drone_D"])

    out = np.zeros((M, DYN_DIM), dtype=np.float32)
    out[c, 0] = 1.0
    out[:, 1] = avail[:M]
    if in_flight and 0 <= drone_D < M:
        out[drone_D, 2] = 1.0
    out[sink, 3] = 1.0
    out[:, 4] = d[c, :M] / d_max
    out[:, 5] = d[:M, sink] / d_max

    masks = env.current_masks()
    out[:, 6] = np.asarray(masks["j"], dtype=bool)[:M]
    k_ext = np.asarray(env.masks_for_launch(rejoined=False)["k_ext"], dtype=bool)
    out[:, 7] = k_ext[:M]

    slack = max(float(inst.W) - float(obs["W_t"]), 1e-9)
    out[:, 8] = np.clip(np.asarray(inst.weights[:M], dtype=float) / slack, 0.0, 2.0)

    # Cheapest sortie that collects i from here: fly c -> i, then rejoin at the
    # nearest node the truck could still be at. Normalised by the endurance
    # budget, so >1 means no admissible rejoin exists for that launch.
    rejoin_pool = np.flatnonzero(avail[:M])
    ED = inst.ED if inst.ED is not None else (2.0 * d_max)
    ED = max(float(ED), 1e-9)
    if rejoin_pool.size:
        # d[i, j] over admissible rejoin nodes j, excluding j == i
        back = d[:M][:, rejoin_pool].copy()
        for col, j in enumerate(rejoin_pool):
            back[j, col] = np.inf
        best_back = back.min(axis=1)
    else:
        best_back = d[:M, sink].astype(float)
    best_back = np.minimum(best_back, d[:M, sink])
    out[:, 9] = np.clip((d[c, :M] + best_back) / ED, 0.0, 4.0)
    return out


class NetworkContext:
    """Holds the light embedding for one instance/view and the heavy embedding
    for the most recently seen state.

    `refresh` must be called before any head reads `h_nodes`. The decode drivers
    in `decode.py` do that once per (beam node, depth); `Network.act`
    does it for greedy rollouts.
    """

    __slots__ = ("policy", "inst", "light_emb", "n_nodes", "dummy_slot",
                 "source_idx", "h_nodes", "h_with_dummy")

    def __init__(self, policy, inst, light_emb):
        self.policy = policy
        self.inst = inst
        self.light_emb = light_emb
        self.n_nodes = inst.n + 2
        self.dummy_slot = inst.n + 2
        self.source_idx = inst.source
        self.h_nodes = None
        self.h_with_dummy = None

    def refresh(self, obs, env):
        dyn = build_dyn_feats(obs, env, self.inst)
        dyn_t = torch.as_tensor(dyn, dtype=torch.float32,
                                device=self.policy.device).unsqueeze(0)
        self.h_nodes = self.policy.heavy(self.light_emb, dyn_t)
        self.h_with_dummy = self.policy.decoder.build_h_with_dummy(self.h_nodes)
        return self


class Network(nn.Module):
    """Same five-head factored decoder as `AttentionPolicy`, different split of
    capacity between encoding and decoding.

    The decoding semantics are literally shared: `_decode`, `_scalars_from_obs`,
    `_drone_target_emb`, `build_encoder_inputs` and `evaluate_batch` are bound
    from `AttentionPolicy`, so the two differ only in the embeddings the heads read.
    """

    build_encoder_inputs = staticmethod(AttentionPolicy.build_encoder_inputs)
    _augment_coords = staticmethod(AttentionPolicy._augment_coords)
    _scalars_from_obs = AttentionPolicy._scalars_from_obs
    _drone_target_emb = AttentionPolicy._drone_target_emb
    _decode = AttentionPolicy._decode
    evaluate_batch = AttentionPolicy.evaluate_batch

    def __init__(self, cfg: NetworkConfig | None = None,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        D = self.cfg.d_model

        # light encoder: one linear map, no attention
        self.node_in = nn.Linear(self.cfg.node_feat_dim, D)

        # heavy decoder: dynamic-state injection, then n_layers of attention
        self.dyn_in = nn.Linear(self.cfg.dyn_feat_dim, D)
        layer_cfg = EncoderConfig(node_feat_dim=self.cfg.node_feat_dim,
                                  d_model=D, n_heads=self.cfg.n_heads,
                                  n_layers=self.cfg.n_layers, d_ff=self.cfg.d_ff,
                                  dropout=0.0)
        self.layers = nn.ModuleList([GATLayer(layer_cfg)
                                     for _ in range(self.cfg.n_layers)])
        self.ln_out = nn.LayerNorm(D)

        self.decoder = StructuredDecoder(
            DecoderConfig(d_model=D, n_heads=self.cfg.n_heads,
                          scalar_dim=self.cfg.scalar_dim))
        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def light(self, node_feats: torch.Tensor) -> torch.Tensor:
        return self.node_in(node_feats)

    def heavy(self, light_emb: torch.Tensor, dyn: torch.Tensor,
              node_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = light_emb + self.dyn_in(dyn)
        kpm = None if node_mask is None else ~node_mask
        for layer in self.layers:
            h = layer(h, kpm)
        h = self.ln_out(h)
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)
        return h

    def encode(self, inst, aug_idx: int = 0) -> NetworkContext:
        feats = self.build_encoder_inputs(inst, self.device, aug_idx)
        return NetworkContext(self, inst, self.light(feats))

    def act(self, obs, env, *, deterministic: bool = False,
            ctx: NetworkContext | None = None, temperature: float = 1.0):
        if ctx is None:
            ctx = self.encode(env.inst)
        ctx.refresh(obs, env)
        with torch.no_grad():
            return self._decode(obs, env, ctx, deterministic=deterministic,
                                forced=None, temperature=temperature)

    def act_force(self, obs, env, force: dict, *, ctx: NetworkContext,
                  deterministic: bool = False):
        ctx.refresh(obs, env)
        with torch.no_grad():
            return self._decode(obs, env, ctx, deterministic=deterministic,
                                forced=force)

    def evaluate(self, obs, env, action):
        ctx = self.encode(env.inst)
        ctx.refresh(obs, env)
        with torch.no_grad():
            return self._decode(obs, env, ctx, deterministic=True, forced=action)

    def forward_heads(self, feats, dyn, node_mask=None, **kw):
        """Batched training forward: light encode, heavy re-embed, then the same
        five-head evaluation the BC pipeline uses.

        `feat_j` / `feat_k` (the pointer heads' per-candidate geometry bias) are
        passed through unchanged, so both models see identical candidate
        features and the only difference stays the node embeddings.
        """
        h = self.heavy(self.light(feats), dyn, node_mask=node_mask)
        return self.evaluate_batch(h_nodes=h, node_mask=node_mask, **kw)
