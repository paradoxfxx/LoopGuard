# -*- coding: utf-8 -*-
"""LoopGuard streaming nets — causal, O(1)-state, per-step classification head.

Two backbones under identical protocol (技术方案 §2.2 诚实条款):
  CausalGRU  — nn.GRU unidirectional (paper-1 champion's streaming form)
  CausalSSM  — selective state-space (Mamba-style) single direction, causal conv
Both: DiffStem(3m) -> in_proj -> encoder -> per-step head (5 classes).
step(x_t, state) gives the O(1) streaming form used on DSP.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12-channel "local substation view" subset of paper-1's 131 channels:
# grid39 layout: [0..45]=P_flow(f->t) on 46 branches, [46..91]=P_flow(t->f),
# [92..130]=P_injection at 39 buses (bus order = case39 bus list)
SUBSET12 = [1, 5, 10, 15, 21, 26, 32, 38, 92 + 5, 92 + 12, 92 + 20, 92 + 30]


class DiffStem(nn.Module):
    """paper-1 temporal-difference stem: [x, dx, x - baseline] (3m features)."""
    def forward(self, x):                       # x: (B, T, m) or (B, m) handled outside
        d = torch.cat([x[:, :1], x[:, 1:] - x[:, :-1]], dim=1)
        q = max(1, x.shape[1] // 4)
        b = x - x[:, :q].mean(dim=1, keepdim=True)
        return torch.cat([x, d, b], dim=-1)


class CausalGRU(nn.Module):
    def __init__(self, m: int, d: int = 48, layers: int = 2, n_cls: int = 5,
                 diff_stem: bool = True):
        super().__init__()
        self.diff_stem = diff_stem
        cin = 3 * m if diff_stem else m
        self.in_proj = nn.Linear(cin, d)
        self.rnn = nn.GRU(d, d, num_layers=layers, batch_first=True)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):                       # (B, T, m) -> logits (B, T, 5)
        if self.diff_stem:
            x = DiffStem()(x)
        h = self.in_proj(x)
        y, _ = self.rnn(h)
        return self.head(y)


class CausalSSM(nn.Module):
    """Single-direction selective SSM. Causal conv (left padding), ZOH scan."""
    def __init__(self, m: int, d: int = 48, layers: int = 2, n_cls: int = 5,
                 diff_stem: bool = True, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.diff_stem = diff_stem
        cin = 3 * m if diff_stem else m
        self.in_proj = nn.Linear(cin, d)
        self.d, self.d_state = d, d_state
        d_inner = expand * d
        self.d_inner = d_inner
        dt_rank = max(8, d // 16)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            L = nn.Module()
            L.in_proj = nn.Linear(d, 2 * d_inner, bias=False)
            L.conv = nn.Conv1d(d_inner, d_inner, 4, padding=0, groups=d_inner)  # causal: crop right
            L.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
            L.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                           + math.log(0.001))
            with torch.no_grad():
                L.dt_proj.weight.zero_()
                L.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            L.register_buffer("A", (-torch.arange(1, d_state + 1,
                             dtype=torch.float32)).repeat(d_inner, 1), persistent=False)
            L.D = nn.Parameter(torch.ones(d_inner))
            L.out_proj = nn.Linear(d_inner, d, bias=False)
            L.norm = nn.LayerNorm(d)
            self.layers.append(L)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):                       # (B, T, m) -> (B, T, 5)
        if self.diff_stem:
            x = DiffStem()(x)
        h = self.in_proj(x)
        for L in self.layers:
            h = h + self._block(L, self._norm_apply(L, h))
        return self.head(h)

    def _norm_apply(self, L, h):
        return L.norm(h)

    def _block(self, L, h):
        B, T, _ = h.shape
        xz = L.in_proj(h)
        u, z = xz.chunk(2, dim=-1)
        u = F.pad(u.transpose(1, 2), (3, 0))[:, :, :T].transpose(1, 2)  # causal conv
        x_dbl = L.x_proj(u)
        dt_rank = L.dt_proj.in_features
        dt, Bp, Cp = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        u = F.silu(u)
        dt = F.softplus(L.dt_proj(dt)).clamp(max=0.1)
        A = L.A
        st = u.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBu = (dA - 1.0) / A * Bp[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            st = dA * st + dBu
            ys.append(torch.einsum("bdn,bn->bd", st, Cp[:, t]))
        y = torch.stack(ys, dim=1)
        y = y + u * L.D
        y = y * F.silu(z)
        return L.out_proj(y)


def build(name: str, m: int, d: int = 48, layers: int = 2) -> nn.Module:
    if name == "gru":
        return CausalGRU(m, d, layers)
    if name == "ssm":
        return CausalSSM(m, d, layers)
    raise ValueError(name)
