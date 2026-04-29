import pytest
import torch
from einops import rearrange, repeat

from cd_mm_sits.model.dataclass import State
from cd_mm_sits.ops.attention import (
    parallel_linear_attention_causal,
    parallel_linear_attention_non_causal,
    reccurent_linear_attention,
)


@pytest.mark.critical
def test_parallel_linear_attention_non_causal():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    b, h, t, c = 2, 4, 11, 3
    q_ = torch.randn(b * h, t, 2 * c)
    k_ = torch.randn(b * h, t, 2 * c)
    v = torch.randn(b * h, t, c)
    output = parallel_linear_attention_non_causal(q_=q_, k_=k_, v=v, bs=b)
    print(f"output {output.shape}")
    A = torch.bmm(q_, torch.transpose(k_, 1, 2))
    O_ = torch.bmm(A, v)

    z_ = 1 / torch.clamp_min(
        torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), 1e-6
    )  # the denominator
    print(f"z {z_.shape}")
    output2 = torch.einsum("nl,nld->nld", z_, O_)
    output2 = rearrange(output2, "(B H) n d -> B n (d H)", B=b)
    print(output[0, 0, ...])
    print(output2[0, 0, ...])
    assert output.shape == output2.shape, f"{output.shape}and {output2.shape}"
    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_understand_einsum():
    b, h, t, c = 2, 4, 11, 3
    q_ = torch.randn(b * h, t, 2 * c)
    k_ = torch.randn(b * h, t, 2 * c)
    v = torch.randn(b * h, t, c)
    # out_einsum = torch.einsum("nld,nsd->nls", q_, k_)
    kv_ = torch.bmm(torch.transpose(k_, 1, 2), v)  # corresponds to k_^TV
    num_einsum = torch.bmm(q_, kv_)
    attn_bmm = torch.bmm(q_, torch.transpose(k_, 1, 2))
    num_bmm = torch.bmm(attn_bmm, v)
    # dis = torch.nn.PairwiseDistance()
    assert torch.allclose(num_bmm, num_einsum, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_parallel_linear_attention_causal():
    b, h, t, c = 2, 4, 11, 3
    q_ = torch.randn(b * h, t, 2 * c)
    k_ = torch.randn(b * h, t, 2 * c)
    v = torch.randn(b * h, t, c)
    output = parallel_linear_attention_causal(q_=q_, k_=k_, v=v, bs=b)
    A = torch.bmm(q_, torch.transpose(k_, 1, 2))
    attn_mask = torch.triu(torch.full((t, t), 1), diagonal=1).bool()
    attn_mask = repeat(attn_mask, "s l -> b s l ", b=b * h)
    A = torch.masked_fill(A, attn_mask, 0)
    O_ = torch.bmm(A, v)
    k_cum = torch.cumsum(k_, dim=1)
    denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), 1e-6)
    output2 = O_ / denom.unsqueeze(-1)
    output2 = rearrange(output2, "(B H) n d -> B n (d H)", B=b)
    # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
    dis = torch.nn.PairwiseDistance()
    print(torch.mean(dis(output, output2)))
    assert output2.shape == output.shape
    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_reccurent_linear_attention():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    b, h, t, c = 2, 4, 11, 3
    q_ = torch.randn(b * h, t, 2 * c)
    k_ = torch.randn(b * h, t, 2 * c)
    v = torch.randn(b * h, t, c)
    state = State()
    print(state)
    seq_output = []
    for time in range(t):
        k_n = k_[:, [time], ...]
        q_n = q_[:, [time], ...]
        v_n = v[:, [time], ...]
        print(q_n.shape)
        y_n, state = reccurent_linear_attention(
            k_n=k_n, vn=v_n, q_n=q_n, state=state, bs=2
        )
        seq_output += [y_n]
    rec_output = torch.cat(seq_output, dim=1)
    print(rec_output.shape)
    assert rec_output.shape == (b, t, h * c)
    output = parallel_linear_attention_causal(q_=q_, k_=k_, v=v, bs=b)
    dis = torch.nn.PairwiseDistance()
    print(torch.mean(dis(output, rec_output)))
    print(output[0, 0, ...], rec_output[0, 0, ...])
    assert torch.allclose(output, rec_output, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_replace_einops():
    import torch
    from einops import rearrange

    attn_output = torch.randn(8, 10, 64)  # shape: (bs * h, L, d)
    bs = 2
    h = attn_output.shape[0] // bs  # 8 // 2 = 4

    # Original einops result
    rear = rearrange(attn_output, "(bs h) l d -> bs l (d h)", bs=bs)

    # Manual replacement
    # Step 1: (bs * h, L, d) → (bs, h, L, d)
    attn_output = attn_output.view(bs, h, attn_output.shape[1], attn_output.shape[2])

    # Step 2: (bs, h, L, d) → (bs, L, d, h)
    attn_output = attn_output.permute(0, 2, 3, 1)

    # Step 3: (bs, L, d, h) → (bs, L, d * h)
    attn_output = attn_output.reshape(bs, attn_output.shape[1], -1)

    print("Expected shape:", rear.shape)
    print("Actual shape:  ", attn_output.shape)
    print("Shapes match:  ", rear.shape == attn_output.shape)
    print("Allclose:      ", torch.allclose(rear, attn_output, atol=1e-5, rtol=1e-4))

    assert torch.allclose(rear, attn_output, atol=1e-5, rtol=1e-4)
