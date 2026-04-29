import torch
from einops import rearrange
from torch import Tensor

from cd_mm_sits.model.dataclass import State


def parallel_linear_attention_non_causal(
    q_: Tensor, k_: Tensor, v: Tensor, bs: int, eps: float = 1e-6
) -> Tensor:
    # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
    kv_ = torch.einsum("nld,nlm->ndm", k_, v)  # corresponds to k_^TV
    # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
    z_ = 1 / torch.clamp_min(
        torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
    )  # the denominator
    # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
    attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
    # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
    attn_output = rearrange(attn_output, "(N h) L d -> N L (d h)", N=bs)
    return attn_output


def reccurent_linear_attention(
    k_n: Tensor, vn: Tensor, q_n: Tensor, state: State, bs: int, eps: float = 1e-6
) -> tuple[Tensor, State]:
    # Compute kv_n using matrix multiplication

    kv_n = torch.bmm(k_n.transpose(1, 2), vn)

    # Update state
    if state.kv_t is not None:
        kv_n += state.kv_t
    if state.k_cum_t is not None:
        k_cum_n = k_n + state.k_cum_t
    else:
        k_cum_n = k_n

    state.kv_t = kv_n
    state.k_cum_t = k_cum_n

    # Compute qkv and denominator
    qkv = torch.bmm(q_n, kv_n)
    denom = torch.bmm(q_n, k_cum_n.transpose(1, 2)).clamp_min_(eps)
    qkv = qkv.div_(denom)

    # Rearrange the output tensor
    attn_output = rearrange(qkv, "(N h) L d -> N L (d h)", N=bs)
    # Compute attention output
    # attn_output = qkv / denom

    return attn_output, state


def normalize_output(q, k, o):
    k_cum = k.cumsum(-2)  # torch.cumsum(k, dim=1)
    z = (q * k_cum).sum(-1, keepdim=True).clamp_min(1e-6)
    return o / z


def parallel_linear_attention_causal(
    q_: Tensor, k_: Tensor, v: Tensor, bs: int, eps: float = 1e-6
) -> Tensor:
    # Compute kv_ and perform cumulative sum
    kv_ = torch.matmul(k_.unsqueeze(-1), v.unsqueeze(-2))
    kv_cum = torch.cumsum(kv_, dim=1)
    # Compute qkv using matrix multiplication
    qkv = torch.matmul(q_.unsqueeze(-2), kv_cum).squeeze_(-2)
    # Compute attention output
    attn_output = normalize_output(q_, k_, qkv)
    # Rearrange the output tensor
    bsh, l, d = attn_output.shape
    h = bsh // bs
    # attn_output = rearrange(attn_output, "(N h) L d -> N L (d h)", N=bs)
    h = bsh // bs
    attn_output = attn_output.reshape(bs, h, l, d)
    attn_output = attn_output.permute(0, 2, 3, 1)
    attn_output = attn_output.reshape(bs, l, -1)
    return attn_output


def parallel_linear_attention_causal_slow(
    q_: Tensor, k_: Tensor, v: Tensor, bs: int, eps: float = 1e-6
) -> Tensor:
    """Clearer implementation but einsum slow down training"""
    kv_ = torch.einsum("nld,nlm->nldm", k_, v)  #
    # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
    kv_cum = torch.cumsum(kv_, dim=1)
    # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)

    qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
    # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
    k_cum = torch.cumsum(k_, dim=1)
    # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
    denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
    # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
    attn_output = qkv / denom.unsqueeze(-1)
    # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
    attn_output = rearrange(attn_output, "(N h) L d -> N L (d h)", N=bs)
    return attn_output
