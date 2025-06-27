# https://github.com/RobertCsordas/moeut/tree/master
import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def log_mean(x: torch.Tensor, dim: int = 0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return -(l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


class SigmaMoE(torch.nn.Module):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        k: int,
        activation=F.relu,
        v_dim: Optional[int] = None,
        expert_dropout: float = 0.0,
    ):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation = activation
        self.expert_dropout = expert_dropout

        self.sel_hist = []

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.values, 0, std_scale / math.sqrt(self.n_experts * self.expert_size))

        self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def get_reg_loss(self) -> torch.Tensor:
        if not self.sel_hist:
            return 0

        # Average over time and layers.
        loss = entropy_reg(torch.stack(self.sel_hist, dim=-2).flatten(-3, -2), -2)
        self.sel_hist = []
        return loss

    def forward(
        self, input: torch.Tensor, sel_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        sel = F.linear(sel_input if sel_input is not None else input, self.expert_sel, None)
        if self.training:
            self.sel_hist.append(sel)

        # Selection activation and topk
        sel = F.sigmoid(sel)

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))

        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = cvmm_prepare_sel2(sel_index.int())

        # "Up-projection" layer for each head
        scores = cvmm(input, sel_indices, self.keys)
        scores = self.activation(scores)

        # Down projection layer for each head
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        out = cvmm(scores, sel_indices, self.values)

        res = out.view(*input.shape[:-1], self.v_dim)
        return res
