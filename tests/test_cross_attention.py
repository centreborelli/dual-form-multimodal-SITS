import torch

from cd_mm_sits.layers.cross_attention import ConfigLQMHA, LearnedQMultiHeadAttention


def test_forward_lq_attn():
    h, dk, d_in, nq = 2, 8, 32, 4
    B, T = 4, 13

    d_v = 16
    config_lqmha = ConfigLQMHA(n_head=h, d_k=dk, d_in=d_in, n_q=nq, d_v=d_v)

    lq_mha = LearnedQMultiHeadAttention(config_lqmha)
    X = torch.randn(B, T, d_in)
    pad_mask = torch.ones(B, T).bool()
    out = lq_mha(X, pad_mask)
    assert out.shape == (B, nq, d_v)
    loss = out.sum()
    loss.backward()

    # Check each parameter individually
    for name, param in lq_mha.named_parameters():
        if param.grad is None:
            print(f"No gradient for: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Is leaf: {param.is_leaf}")
        else:
            print(f"Has gradient: {name}")
