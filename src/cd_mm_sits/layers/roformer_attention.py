import math
from typing import Literal, override

import torch
from einops import rearrange
from torch import Tensor

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput, LinearAttention
from cd_mm_sits.model.temporal_positional_encoder import TimeDeltaRotaryPE


class RoFormerAttention(LinearAttention):
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
        base: int = 10000,
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
        self.rope = TimeDeltaRotaryPE(base=base, d=self.dk // self.num_heads)
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.dk // self.num_heads)

    def get_index(self, input_shape: tuple, device):
        """
        :returns:
        a weight matrix of shape (B,T,1)
        """

        return rearrange(
            torch.arange(input_shape[1], device=device)[None, :].expand(
                input_shape[0], -1
            ),
            "BH T -> (BH  T) ",
        )

    @override
    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput
    ) -> tuple[Tensor, Tensor]:
        q_ = q.content * self.inv_sqrt_head_dim
        k_ = k.content * self.inv_sqrt_head_dim
        B = q_.shape[0]
        # get index and send to cuda

        weight_index_q = self.get_index(q_.shape, device=q_.device)
        weight_index_k = self.get_index(k_.shape, device=k_.device)
        assert q_.shape[0] * q_.shape[1] == weight_index_q.shape[0], (
            f"q {q_.shape} time {weight_index_q.shape}"
        )
        assert q_.shape[-1] == self.dk // self.num_heads, f"dk {self.dk},q {q_.shape}"
        q_ = self.rope(x=rearrange(q_, "B T C -> (B T ) C "), delta=weight_index_q)
        k_ = self.rope(x=rearrange(k_, "B T C -> (B T )  C "), delta=weight_index_k)

        return rearrange(q_, "(B T ) C -> B T C", B=B), rearrange(
            k_, "(B T) C -> B T C ", B=B
        )


class TimeRoFormerAttention(LinearAttention):
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
        base: int = 10000,
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
        self.rope = TimeDeltaRotaryPE(base=base, d=self.dk // self.num_heads)
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.dk // self.num_heads)

    def get_time(self, times: Tensor):
        """
        :returns:
        a weight matrix of shape (B,T,1)
        """
        # times = repeat(times, "B T -> (B H ) T", H=self.num_heads)
        times = times.unsqueeze(-1).expand(-1, -1, self.num_heads)
        times = rearrange(times, "B T H   -> (B H T )")
        return times

    @override
    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput
    ) -> tuple[Tensor, Tensor]:
        q_ = q.content * self.inv_sqrt_head_dim
        k_ = k.content * self.inv_sqrt_head_dim
        B = q_.shape[0]
        # get index and send to cuda
        assert q.time is not None, "TimeRoFormer require temporal info"
        assert k.time is not None
        weight_index_q = self.get_time(times=q.time)
        weight_index_k = self.get_time(times=k.time)
        assert q_.shape[0] * q_.shape[1] == weight_index_q.shape[0], (
            f"q {q_.shape} time {weight_index_q.shape}"
        )
        assert q_.shape[-1] == self.dk // self.num_heads, f"dk {self.dk},q {q_.shape}"
        q_ = self.rope(x=rearrange(q_, "B T C -> (B T ) C "), delta=weight_index_q)
        k_ = self.rope(x=rearrange(k_, "B T C -> (B T )  C "), delta=weight_index_k)

        return rearrange(q_, "(B T ) C -> B T C", B=B), rearrange(
            k_, "(B T) C -> B T C ", B=B
        )
