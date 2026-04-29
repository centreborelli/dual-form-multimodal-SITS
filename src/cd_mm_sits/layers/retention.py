"""Implementation of retention inspired by
https://github.com/Jamie-Stirling/RetNet/blob/main/src/retention.py"""

import math
from typing import Literal

import torch
import torch.nn as nn

from cd_mm_sits.layers.activation import ModifiedELU
from cd_mm_sits.layers.linear_attention import FlexibleAttnInput
from cd_mm_sits.layers.rope import MyROPE, MyTimeROPE


class SimpleRetention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size,
        xpos: Literal["xpos", "timexpos"] = "xpos",
        double_v_dim: bool = False,
        T_base: float = 1e4,
        gamma: float = 0.4,
        scale_base: float = 1000,
    ):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.head_size = head_size
        self.scale_base = 1000
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma
        self.W_Q = nn.Linear(embed_dim, head_size, bias=False)
        self.W_K = nn.Linear(embed_dim, head_size, bias=False)
        self.W_V = nn.Linear(embed_dim, self.v_dim, bias=False)
        # self.W_Q = nn.Parameter(torch.randn(embed_dim, head_size) / embed_dim)
        # self.W_K = nn.Parameter(torch.randn(embed_dim, head_size) / embed_dim)
        # self.W_V = nn.Parameter(torch.randn(embed_dim, self.v_dim) / embed_dim)
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.embed_dim)
        self.act_fun = ModifiedELU()
        if xpos == "xpos":
            self.xpos = MyROPE(
                embed_dim=head_size,
                T_base=T_base,
            )
        elif xpos == "timexpos":
            self.xpos = MyTimeROPE(
                embed_dim=head_size,
                T_base=T_base,
            )
        else:
            raise NotImplementedError
        assert isinstance(self.xpos, nn.Module)
        assert self.xpos.head_dim == head_size, (
            f"Incorrect instantatio, of xpos {self.xpos.head_dim} should be {head_size}"
        )

    def forward(
        self,
        query: FlexibleAttnInput,
        key: FlexibleAttnInput,
        value: FlexibleAttnInput,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        average_attn_weights: bool = False,
        need_weights: bool = False,
        eps: float = 1e-10,
        is_causal: bool = False,
    ):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, embed_dim)
        """
        Q = self.act_fun(self.W_Q(query.content)) * self.inv_sqrt_head_dim
        K = self.act_fun(self.W_K(key.content)) * self.inv_sqrt_head_dim
        if isinstance(self.xpos, MyTimeROPE):
            Q = self.xpos(x=Q, time=query.time)
            K = self.xpos(x=K, time=key.time)
            D = self._get_D(query.time, key.time)
        else:
            Q = self.xpos(x=Q)
            K = self.xpos(x=K)
            pos_query = torch.arange(Q.shape[1], device=Q.device).expand(Q.shape[0], -1)
            pos_key = torch.arange(K.shape[1], device=Q.device).expand(Q.shape[0], -1)
            D = self._get_D(pos_query, pos_key)
        V = self.W_V(value.content)
        ret = torch.bmm(Q, K.transpose(1, 2))
        # ret = Q @ K.permute(0, 2, 1)
        ret = ret * D
        if key_padding_mask is not None:
            ret = ret.masked_fill(key_padding_mask[:, None, :], 0)

        if need_weights:
            return ret @ V, ret.detach()
        return ret @ V, None

    def _get_D(self, pos_query: torch.Tensor, pos_key: torch.Tensor):
        """pos is dim B,T"""
        n = pos_query.unsqueeze(2)
        m = pos_key.unsqueeze(1)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        dis = (n - m).div(self.scale_base)
        assert self.gamma < 1
        D = (self.gamma**dis) * (
            n >= m
        ).float()  # this results in some NaN when n is much larger than m
        assert torch.max(D) <= 1
        D = torch.nan_to_num(D)
        return D


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        double_v_dim=False,
        xpos: Literal["xpos", "timexpos"] = "xpos",
        T_base: float = 1e4,
        scale_base: float = 1000,
        dk: int | None = None,
    ):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for
        Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.v_dim = embed_dim * 2 if double_v_dim else embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by heads"
        self.dk = dk if dk is not None else embed_dim
        self.head_size = embed_dim // num_heads
        self.head_v_dim = embed_dim * 2 if double_v_dim else embed_dim
        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Linear(embed_dim, self.v_dim)
        self.W_O = nn.Linear(self.v_dim, embed_dim)
        # self.W_G = nn.Parameter(torch.randn(embed_dim, self.v_dim) / embed_dim)
        # self.W_O = nn.Parameter(torch.randn(self.v_dim, embed_dim) / embed_dim)
        self.group_norm = nn.GroupNorm(num_heads, self.v_dim)
        self.gammas = [1 - 2 ** (-5 - h_index) for h_index in range(num_heads)]
        self.retentions = nn.ModuleList(
            [
                SimpleRetention(
                    embed_dim=self.embed_dim,
                    head_size=self.head_size,
                    xpos=xpos,
                    double_v_dim=double_v_dim,
                    T_base=T_base,
                    gamma=self.gammas[h],
                    scale_base=scale_base,
                )
                for h in range(num_heads)
            ]
        )

    def forward(
        self,
        query: FlexibleAttnInput,
        key: FlexibleAttnInput,
        value: FlexibleAttnInput,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        average_attn_weights: bool = False,
        need_weights: bool = False,
        eps: float = 1e-10,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = [
            self.retentions[i](
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )
            for i in range(self.num_heads)
        ]
        Y, weights = zip(*Y, strict=False)
        Y = torch.cat(Y, dim=2)
        if need_weights:
            weights = torch.stack(weights, dim=1)
        else:
            weights = None
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return self.W_O(self.swish(self.W_G(query.content)) * Y), weights
