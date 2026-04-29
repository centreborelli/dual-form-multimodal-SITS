from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from cd_mm_sits.layers.activation import ModifiedELU
from cd_mm_sits.model.dataclass import State
from cd_mm_sits.ops.attention import (
    parallel_linear_attention_causal,
    parallel_linear_attention_non_causal,
    reccurent_linear_attention,
)


@dataclass
class FlexibleAttnInput:
    content: Tensor
    time: Tensor | None = None
    mod: Tensor | None = None


class LinearAttention(nn.Module):
    """Generic class for kernel-based linear attention
    Its classical form allows for linear attention
    as defined by Katharopoulos 2020"""

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
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.dk = dk if dk is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, self.dk, bias=False)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, self.dk, bias=False)
        # outprojection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # dropout rate
        # causal
        self.causal = causal
        self.state = State()
        assert self.embed_dim % self.num_heads == 0, (
            "embed_dim must be divisible by num_heads"
        )
        assert self.dk % self.num_heads == 0, "dk must be divisible by num_heads"
        self.reweighting = False

    def feature_map(self, x: Tensor):
        """Project the tensor representing either key or query
        in the feature map"""
        return self.act_fun(x)

    def input_projection(
        self,
        query: Tensor,
        key: Tensor | None,
        value: Tensor | None,
        merge_qk_batch_head: bool = True,
    ):
        """Method for inputs query key and value projection

        :param query: B,L,C
        :param key: B,S,C
        :param value: B,S,C
        :param merge_qk_batch_head: bool if set to True output a 3D tensor
        where head dimension has been merged to batch for query and key.
        Else output a 4D tensor of shape B,H,T,C for query and key
        """
        # test for the correctness of the program
        if key is None:
            key = query
        if value is None:
            value = query

        num_heads = self.num_heads
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        qk_head_dim = self.dk // num_heads
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # multihead reshape
        # (N * h, L, d)
        if merge_qk_batch_head:
            q = rearrange(q, "B N (C H) -> (B H ) N C", C=qk_head_dim, H=num_heads)
            k = rearrange(k, "B N (C H) -> (B H ) N C", C=qk_head_dim, H=num_heads)
        else:
            q = rearrange(q, "B N (C H) -> B H  N C", C=qk_head_dim, H=num_heads)
            k = rearrange(k, "B N (C H) -> B H N C", C=qk_head_dim, H=num_heads)
            # (N * h, S, d)
            # v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = rearrange(v, "B N (C H) -> (B H ) N C", C=head_dim, H=num_heads)
        return q, k, v, head_dim, tgt_len, bsz, src_len

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
        """Attention as (QK)V"""
        if is_causal:
            assert attn_mask is not None
        ### Step 1: input proj with W_Q,W_K,W_V
        q, k, v, head_dim, tgt_len, bsz, src_len = self.input_projection(
            query=query.content, key=key.content, value=value.content
        )

        ### Step 2: feature maps on Q and K
        q = self.feature_map(q)
        k = self.feature_map(k)
        ### Step 3: [Optional] reweighting (update Q and K)
        if self.reweighting:
            # assert query.mod is not None
            # replace with 0 the masked query idx. (avoid error in emebdding)

            q, k = self.qk_reweighting(
                FlexibleAttnInput(q, time=query.time, mod=query.mod),
                FlexibleAttnInput(k, time=key.time, mod=key.mod),
            )

        weights = torch.bmm(q, k.transpose(1, 2))

        # TODO check that
        assert torch.all(~torch.isneginf(weights))  # weights can't be negative
        # weights[weights < 0] = 0  # if distance (in days)

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
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # print(attn_output.shape)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = rearrange(
            attn_output, "(B h) N C -> B N (C h)", h=self.num_heads, C=head_dim
        )
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = rearrange(
                attn_weights, "(B H ) L S -> B H L S", H=self.num_heads, B=bsz
            )
            return attn_output, attn_weights.detach()

        return attn_output, None

    def get_act_fun(self, act_fun) -> nn.Module:
        if act_fun == "relu":
            return nn.ReLU()
        elif act_fun == "elu":
            return ModifiedELU()
        else:
            raise NotImplementedError

    def qk_reweighting(
        self, q: FlexibleAttnInput, k: FlexibleAttnInput
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def qk_reccurent_reweighting(
        self,
        x_index: int,
        m: int,
        seq_len: int,
        q: FlexibleAttnInput,
        k: FlexibleAttnInput,
    ):
        raise NotImplementedError

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
        """Attention as Q(KV)"""
        ### Step 1: input proj with W_Q,W_K,W_V
        q, k, v, head_dim, tgt_len, bsz, src_len = self.input_projection(
            query=query.content, key=key.content, value=value.content
        )
        ### Step 2: feature maps on Q and K
        q = self.feature_map(q)
        k = self.feature_map(k)
        ### Step 3: [Optional] reweighting (update Q and K)
        if self.reweighting:
            q, k = self.qk_reweighting(
                FlexibleAttnInput(q, time=query.time, mod=query.mod),
                FlexibleAttnInput(k, time=key.time, mod=key.mod),
            )

        ### Step 4: Prepare mask
        if key_padding_mask is not None:
            key_padding_mask = repeat(
                key_padding_mask, "B S -> (B H) S", H=self.num_heads
            )
            v = v.masked_fill(key_padding_mask[..., None], 0)
            k = k.masked_fill(key_padding_mask[..., None], 0)

        ### Step 5 : Q(KV)
        if attn_mask is not None:  # so it fits my temporal encoder implemementation
            attn_output = parallel_linear_attention_causal(
                q_=q,
                k_=k,
                v=v,
                eps=eps,
                bs=bsz,
            )
        else:
            attn_output = parallel_linear_attention_non_causal(
                q_=q.contiguous(), k_=k.contiguous(), v=v.contiguous(), eps=eps, bs=bsz
            )

        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output, None

    def one_step_forward(
        self,
        x: FlexibleAttnInput,
        state: State,
        m: int,
        x_index: Tensor,
        key_padding_mask: Tensor | None = None,
        eps=1e-6,
        seq_len: int = 32,
    ) -> tuple[Tensor, State]:
        """Iterative method for efficient prediction
        x: (B,1,d) the input which corresponds to the "new token"
        state: buffer which contains pas information
        m: The constant which accounts for the expected context
        taken into account
        x_index: (B,1) the positions of the elements in x
        key_padding_mask: (B 1)
        eps: 1e-6"""

        q, k, v, head_dim, tgt_len, bsz, src_len = self.input_projection(
            query=x.content, key=x.content, value=x.content, merge_qk_batch_head=False
        )
        ### Step 2: feature maps on Q and K

        q = self.feature_map(q)
        k = self.feature_map(k)

        if self.reweighting:
            q, k = self.qk_reccurent_reweighting(
                x_index.long(),
                m,
                seq_len=seq_len,
                q=FlexibleAttnInput(q, x.time, x.mod),
                k=FlexibleAttnInput(k, x.time, x.mod),
            )
        else:
            q = rearrange(q, "B H T C -> (B H ) T C ")
            k = rearrange(k, "B H T C -> (B H ) T C ")
        if key_padding_mask is not None:
            key_padding_mask = repeat(
                key_padding_mask, "B S -> (B H) S", H=self.num_heads
            )  # S equals 1 here
            v = v.masked_fill(key_padding_mask[..., None], 0)
            k = k.masked_fill(key_padding_mask[..., None], 0)

        output, state = reccurent_linear_attention(
            k_n=k, vn=v, q_n=q, state=state, bs=bsz, eps=eps
        )
        if self.has_outproj:
            output = self.out_proj(output)
        return output, state
