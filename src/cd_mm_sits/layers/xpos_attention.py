import math
from typing import Literal, override

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput, LinearAttention
from cd_mm_sits.layers.xpos import TimeXPOS


class TimeXPOSAttention(LinearAttention):
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
        gamma_init: int = 32,
        gamma_end: int = 512,
        apply_denom: bool = True,
        gammas: list | None = None,
        scale_base: int = 512,
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
        self.head_size = self.dk // self.num_heads
        if gammas is None:
            self.gammas = (
                (
                    1
                    - torch.exp(
                        torch.linspace(
                            math.log(1 / gamma_init), math.log(1 / gamma_end), num_heads
                        )
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
        else:
            self.gammas = gammas
            assert len(self.gammas) == num_heads
        self.xpos = nn.ModuleList(
            [
                TimeXPOS(
                    embed_dim=self.dk // self.num_heads,
                    gamma=gamma,
                    T_base=base,
                    scale_base=scale_base,
                )
                for gamma in self.gammas
            ]
        )

        # TimeDeltaRotaryPE(base=base, d=self.dk // self.num_heads)
        self.apply_denom = apply_denom

    def get_time(self, times: Tensor):
        """
        :returns:
        a weight matrix of shape (B,T,1)
        """
        # times = repeat(times, "B T -> (B H ) T", H=self.num_heads)
        times = times.unsqueeze(-1).expand(-1, -1, self.num_heads)
        times = rearrange(times, "B T H   -> B H T")
        return times

    @override
    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput, xpos: TimeXPOS
    ) -> tuple[Tensor, Tensor]:
        q_ = q.content
        k_ = k.content

        # get index and send to cuda
        assert q.time is not None, "TimeRoFormer require temporal info"
        assert k.time is not None

        q_ = xpos(x=q_, time=q.time)
        k_ = xpos(x=k_, time=q.time, downscale=True)

        return q_, k_

    def forward(
        self,
        query: FlexibleAttnInput,
        key: FlexibleAttnInput,
        value: FlexibleAttnInput,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        average_attn_weights: bool = False,
        need_weights: bool = False,
        eps: float = 1e-10,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if is_causal:
            assert attn_mask is not None
        ### Step 1: input proj with W_Q,W_K,W_V
        q, k, v, head_dim, tgt_len, bsz, src_len = self.input_projection(
            query=query.content,
            key=key.content,
            value=value.content,
            merge_qk_batch_head=False,
        )  # B,H,T,C

        ### Step 2: feature maps on Q and K
        q = self.feature_map(q)
        k = self.feature_map(k)
        ### Step 3: [Optional] reweighting (update Q and K)
        # assert query.mod is not None
        # replace with 0 the masked query idx. (avoid error in emebdding)
        weight_index_q = self.get_time(times=query.time)  # B,H,T
        weight_index_k = self.get_time(times=key.time)
        Y = [
            self.qk_reweighting(
                q=FlexibleAttnInput(
                    q[:, h, ...], time=weight_index_q[:, h, ...], mod=query.mod
                ),
                k=FlexibleAttnInput(
                    k[:, h, ...], time=weight_index_k[:, h, ...], mod=key.mod
                ),
                xpos=self.xpos[h],
            )
            for h in range(self.num_heads)
        ]
        Q, K = zip(*Y, strict=False)
        q_h = torch.stack(Q, dim=1)  # B*H,T,C
        k_h = torch.stack(K, dim=1)  # B*H,T,C
        weights = torch.bmm(
            rearrange(q_h, "B H T C -> (B H ) T C "),
            rearrange(k_h, "B H T C -> (B H ) C T "),
        )

        # TODO check that
        # assert torch.all(~torch.isneginf(weights))  # weights can't be negative
        # weights[weights < 0] = 0  # if distance (in days)

        weights = torch.nan_to_num(
            weights
        )  # if elemnts have important temporal distance it could lead to nan values
        if key_padding_mask is not None:
            key_padding_mask = repeat(
                key_padding_mask, "B S -> (B H) S", H=self.num_heads
            )
            weights = weights.masked_fill(
                key_padding_mask[:, None, :], 0
            )  # attn_mask[:,None,1] is of shape B,1,S
        if attn_mask is not None:
            weights = weights.masked_fill(attn_mask.bool(), 0)
        # (N * h, L, S) -> (N * h, L, S)
        if self.apply_denom:
            denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
            # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
            attn_weights = weights / denom
        else:
            attn_weights = weights
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # print(attn_output.shape)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = rearrange(
            attn_output, "(B h) N C -> B N (C h)", h=self.num_heads, C=head_dim
        )
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        assert torch.all(~torch.isneginf(attn_output))
        if need_weights:
            attn_weights = rearrange(
                attn_weights, "(B H ) L S -> B H L S", H=self.num_heads, B=bsz
            )
            return attn_output, attn_weights.detach()

        return attn_output, None

    def right_attn(
        self,
        query: FlexibleAttnInput,
        key: FlexibleAttnInput,
        value: FlexibleAttnInput,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        average_attn_weights: bool = False,
        need_weights: bool = False,
        eps: float = 1e-10,
        is_causal: bool = False,
    ) -> tuple[Tensor, None]:
        raise NotImplementedError
