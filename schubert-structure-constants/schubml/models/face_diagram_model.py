"""Width-equivariant face-diagram model (K2; EXPERT_RESPONSE_5 SS9).

Input: the face lattice of a cut set D* of size k -- the Boolean lattice
B_k with 2^k nodes (faces E subseteq D*) and k * 2^(k-1) covering edges --
with per-face certificate features (node) and per-inclusion twist /
kernel-overlap features (edge).  NO raw permutations, no descent-set
identities anywhere.

Equivariance: relabeling the k descent positions acts on the lattice by
permuting bit positions; every learned map here is a shared per-node /
per-edge MLP composed with MEAN aggregation over covering neighbors, so
the network commutes with that action exactly: outputs at the top node
are invariant, per-coatom outputs are equivariant (verified numerically
in scripts/k2_face_model.py eval --equivariance).

The same weights run on any width k: nothing in the parameterization
depends on k (no positional embeddings, no per-level parameters); levels
enter only through scalar node features (|E|, k - |E|, |E|/k).

Targets (multi-task heads):
  top node   -> corank logits (dim O_T, classes 0..MAX_CORANK) + c>0 aux
  coatom j   -> crit-incidence logit + infeasible-coatom logit
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MAX_CORANK = 10          # corank clipped to 0..10 (11 classes)


def lattice_edges(k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical covering-edge lists of B_k: for every mask m and free bit
    j, edge (child=m, parent=m | 1<<j), in (m, j) lex order.  This order
    matches FaceEngine.edges in scripts/k2_face_model.py."""
    ch, pa = [], []
    for m in range(1 << k):
        for j in range(k):
            if not m >> j & 1:
                ch.append(m)
                pa.append(m | 1 << j)
    return torch.tensor(ch, dtype=torch.long), torch.tensor(pa, dtype=torch.long)


class MLP(nn.Sequential):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class FaceDiagramModel(nn.Module):
    def __init__(self, n_node_feat: int, n_edge_feat: int,
                 d_model: int = 96, n_rounds: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d = d_model
        self.n_rounds = n_rounds
        self.node_in = nn.Linear(n_node_feat, d_model)
        self.edge_in = nn.Linear(n_edge_feat, d_model)
        self.msg_up = nn.ModuleList(
            [MLP(2 * d_model, 2 * d_model, d_model) for _ in range(n_rounds)])
        self.msg_dn = nn.ModuleList(
            [MLP(2 * d_model, 2 * d_model, d_model) for _ in range(n_rounds)])
        self.upd = nn.ModuleList(
            [MLP(3 * d_model, 2 * d_model, d_model) for _ in range(n_rounds)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_rounds)])
        self.drop = nn.Dropout(dropout)
        self.head_corank = MLP(d_model, d_model, MAX_CORANK + 1)
        self.head_pos = MLP(d_model, d_model, 1)
        self.head_coatom = MLP(d_model, d_model, 2)   # (crit, infeas)

    def forward(self, node: torch.Tensor, edge: torch.Tensor,
                e_child: torch.Tensor, e_parent: torch.Tensor,
                k: int) -> Dict[str, torch.Tensor]:
        """node [B, M, F], edge [B, nE, Fe]; e_child/e_parent [nE] shared
        across the batch (all instances have the same width k)."""
        B, M, _ = node.shape
        h = self.node_in(node)                        # [B, M, d]
        e = self.edge_in(edge)                        # [B, nE, d]
        # per-node degrees (constant over batch)
        deg_up = torch.zeros(M, device=node.device).index_add_(
            0, e_child, torch.ones_like(e_child, dtype=torch.float))
        deg_dn = torch.zeros(M, device=node.device).index_add_(
            0, e_parent, torch.ones_like(e_parent, dtype=torch.float))
        deg_up = deg_up.clamp(min=1).view(1, M, 1)
        deg_dn = deg_dn.clamp(min=1).view(1, M, 1)
        for r in range(self.n_rounds):
            h_child = h[:, e_child]                   # [B, nE, d]
            h_parent = h[:, e_parent]
            # "up" messages flow from parent to child (information about
            # larger faces), "down" messages from child to parent.
            m_up = self.msg_up[r](torch.cat([h_parent, e], dim=-1))
            m_dn = self.msg_dn[r](torch.cat([h_child, e], dim=-1))
            agg_up = torch.zeros_like(h).index_add_(1, e_child, m_up) \
                / deg_up
            agg_dn = torch.zeros_like(h).index_add_(1, e_parent, m_dn) \
                / deg_dn
            delta = self.upd[r](torch.cat([h, agg_up, agg_dn], dim=-1))
            h = self.norms[r](h + self.drop(delta))
        top = h[:, M - 1]                             # [B, d]
        coatom_idx = torch.tensor(
            [(M - 1) ^ (1 << j) for j in range(k)], device=node.device)
        hco = h[:, coatom_idx]                        # [B, k, d]
        co = self.head_coatom(hco)                    # [B, k, 2]
        return dict(
            corank=self.head_corank(top),             # [B, C]
            pos=self.head_pos(top).squeeze(-1),       # [B]
            crit=co[..., 0],                          # [B, k]
            infeas=co[..., 1],                        # [B, k]
        )
