from typing import Literal, override

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn

from cd_mm_sits.constant.model import PAD_MOD_IGNORE
from cd_mm_sits.layers.linear_attention import FlexibleAttnInput, LinearAttention


class CosformerAttention(LinearAttention):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int | None = None,
        vdim: int | None = None,
        causal: bool = False,
        has_outproj: bool = True,
        act_fun: Literal["relu", "elu"] = "relu",
        dk: int | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            causal=causal,
            has_outproj=has_outproj,
            act_fun=act_fun,
            dk=dk,
        )
        self.reweighting = True
        self.register_buffer("weight_index", None, persistent=False)
        self.register_buffer("r_weight_index", None, persistent=False)

    def get_index(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """return the weight matrix to reweight
        the attention score
        """
        if (
            self.weight_index is None
            or self.weight_index.shape[1] < seq_len
            or self.weight_index.device != device
        ):
            self.weight_index = (np.pi / 2) * torch.arange(
                1, seq_len + 1, device=device, dtype=dtype
            ).view(1, -1, 1)
        return self.weight_index[:, :seq_len, :]
        # return index

    @override
    def get_recurrent_weight_index(
        self,
        x_index: Tensor,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int = 32,
    ):
        """this method output efficiently in recurrence mode the weight index matrix
        by default if nothing has been previously done before it create
        an matrix of shape seq_len
        """
        if self.r_weight_index is None:
            self.r_weight_index = (np.pi / 2) * torch.arange(
                1, seq_len + 1, device=device, dtype=dtype
            )
            return self.r_weight_index[x_index.squeeze(1)]
        elif torch.max(x_index) > self.r_weight_index.shape[0]:
            weight_index = x_index.view(-1, -1, 1).to(dtype=dtype)
            weight_index = weight_index.mul(torch.pi / 2)
            return weight_index
        else:
            return self.r_weight_index[x_index.squeeze(1)]

    @override
    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput
    ) -> tuple[Tensor, Tensor]:
        q_ = q.content
        k_ = k.content
        tgt_len = q_.shape[1]
        src_len = k_.shape[1]
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m, device=q_.device, dtype=q_.dtype)
        q_sin = torch.sin(weight_index[:, :tgt_len, :] / m)
        q_cos = torch.cos(weight_index[:, :tgt_len, :] / m)
        k_sin = torch.sin(weight_index[:, :src_len, :] / m)
        k_cos = torch.cos(weight_index[:, :src_len, :] / m)
        q_ = torch.cat([q_ * q_sin, q_ * q_cos], dim=-1)
        k_ = torch.cat([k_ * k_sin, k_ * k_cos], dim=-1)
        # TODO ensure that reweighting is definite +

        return q_, k_

    @override
    def qk_reccurent_reweighting(
        self,
        x_index: Tensor,
        m: int,
        seq_len: int,
        q: FlexibleAttnInput,
        k: FlexibleAttnInput,
    ):
        """Use cos reweigting"""
        weight_index = self.get_recurrent_weight_index(
            x_index=x_index,
            seq_len=seq_len,
            dtype=q.content.dtype,
            device=q.content.device,
        )
        weight_index = weight_index.view(-1, 1, 1, 1)
        q_ = torch.cat(
            [
                torch.mul(q.content, torch.sin(weight_index / m)),
                torch.mul(q.content, torch.cos(weight_index / m)),
            ],
            dim=-1,
        )
        # (N , h, S, 2 * d)
        k_ = torch.cat(
            [
                torch.mul(k.content, torch.sin(weight_index / m)),
                torch.mul(k.content, torch.cos(weight_index / m)),
            ],
            dim=-1,
        )
        # (N , h, S, 2 * d)
        q_ = rearrange(q_, "B H T C -> (B H) T C")
        k_ = rearrange(k_, "B H T C -> (B H) T C ")
        return q_, k_


class TimeCosFormerAttention(LinearAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int | None = None,
        vdim: int | None = None,
        causal: bool = False,
        has_outproj: bool = True,
        act_fun: Literal["relu", "elu"] = "relu",
        max_effect: int = 700,
        dk: int | None = None,
    ):
        self.max_effect = max_effect
        self.max_time = 700
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            causal=causal,
            has_outproj=has_outproj,
            act_fun=act_fun,
            dk=dk,
        )
        self.reweighting = True
        self.register_buffer("weight_index", None, persistent=False)
        self.register_buffer("r_weight_index", None, persistent=False)

    def get_time_encoding(self, times: Tensor):
        """return a cosformer inspired rewieghting based on acquisition date
        :param times: a Tensor of shape (B, T)
        :returns:
        a weight matrix of shape (B,T,1)
        """
        # times = repeat(times, "B T -> (B H ) T", H=self.num_heads)
        times = times.unsqueeze(-1).expand(-1, -1, self.num_heads)
        times = rearrange(times, "B T H -> (B H ) T")
        index = np.pi / 2 * times
        return index.unsqueeze(-1)

    @override
    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput
    ) -> tuple[Tensor, Tensor]:
        q_ = q.content
        k_ = k.content
        m = self.max_effect

        # get index and send to cuda
        assert q.time is not None, "TimeCosFormer require temporal info"
        assert k.time is not None
        weight_index_q = self.get_time_encoding(times=q.time)
        weight_index_k = self.get_time_encoding(times=k.time)

        # (N * h, L, 2 * d)
        q_sin = torch.sin(weight_index_q / m)
        q_cos = torch.cos(weight_index_q / m)
        k_sin = torch.sin(weight_index_k / m)
        k_cos = torch.cos(weight_index_k / m)

        # Allocate output tensors
        q_ = torch.cat([q_ * q_sin, q_ * q_cos], dim=-1)
        k_ = torch.cat([k_ * k_sin, k_ * k_cos], dim=-1)

        return q_, k_


def expand_mod_weights(mod_w: Tensor, num_heads) -> Tensor:
    return rearrange(mod_w, "b t (d h)-> (b h) t d", h=num_heads)
