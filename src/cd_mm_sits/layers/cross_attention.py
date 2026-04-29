"""
Cross-Attention mechanism employed
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

my_logger = logging.getLogger(__name__)


@dataclass
class ConfigLQMHA:
    n_head: int
    d_k: int
    d_in: int
    n_q: int
    d_v: int


class LearnedQMultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    No W_Q and W_V matrix to project queries and values.
     Value features are split along the heads
    """

    def __init__(self, config: ConfigLQMHA):
        super().__init__()
        self.n_head = config.n_head
        self.d_k = config.d_k
        self.d_in = config.d_in
        self.n_q = config.n_q
        self.d_v = config.d_v
        assert self.d_v // self.n_head, (
            f"d_in {config.d_v} must be divible by n_heads {config.n_head}"
        )
        self.Q = nn.Parameter(
            torch.zeros((config.n_head, config.n_q, config.d_k))
        ).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (config.d_k)))
        self.fc1_k = nn.Linear(config.d_in, config.n_head * config.d_k, bias=False)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (config.d_k)))
        self.fc1_v = nn.Linear(config.d_in, config.d_v, bias=False)

    def forward(
        self, X: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """

        Args:
            X (): sequence which is going to be projected (B,T,C)
            C must be divisible by n_head
            pad_mask (): (B,T) True means the value should take part in attention
        Returns:
        the projected sequence a Tensor of size (B,nq,C)
        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = X.size()
        q = repeat(
            self.Q, "nh nq dk -> nh repeat nq dk", repeat=sz_b
        )  # torch.stack([self.Q for _ in range(sz_b)], dim=1)
        my_logger.debug(f"query{q.shape}")
        q = rearrange(q, "head b nq c -> b head nq c")
        k = self.fc1_k(X).view(sz_b, seq_len, n_head, d_k)
        my_logger.debug(f"key {k.shape}")
        k = rearrange(k, "b t head c -> b head t c")
        my_logger.debug(f"key {k.shape}")
        v = self.fc1_v(X)
        my_logger.debug(f"key {v.shape}")
        v = rearrange(v, "b t (head c) -> b head t c", head=self.n_head)
        if pad_mask is not None:
            my_logger.debug(f"Pad mask shape {pad_mask.shape}")
            pad_mask = repeat(pad_mask, "B T -> B T nh nq", nh=self.n_head, nq=self.n_q)
            pad_mask = rearrange(pad_mask, "b t head nq ->b head nq t")
            my_logger.debug(f"Pad mask shape {pad_mask.shape}")
        # X = torch.stack(X.split(X.shape[-1] // n_head, dim=-1))
        # X = rearrange(X, "head b t c -> b head t c")
        my_logger.debug(f"value {v.shape}")
        my_logger.debug(f"query{q.shape}")
        # q=q.to(v)
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=pad_mask
        )  # B,h,nq,d_in
        my_logger.debug(f"output {output.shape}")
        return rearrange(output, "b h nq c -> b nq (h c)")
