import pytest
import torch

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput
from cd_mm_sits.layers.retention import MultiScaleRetention, SimpleRetention


@pytest.mark.critical
@pytest.mark.parametrize(
    "pos",
    [("xpos"), ("timexpos")],
)
def test_retention_forward(pos):
    d_model, _, hdim = 64, 4, 32
    B, T = 2, 11
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = FlexibleAttnInput(
        torch.randn(B, T, d_model), time=torch.arange(T)[None, :].expand(B, -1)
    )
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    attn_mask = attn_mask.bool()
    retention = SimpleRetention(embed_dim=d_model, head_size=hdim, xpos=pos, gamma=0.4)
    out, weights = retention.forward(
        query=input_x,
        key=input_x,
        value=input_x,
        need_weights=True,
        attn_mask=attn_mask,
    )
    assert weights[0, 0, 1] == 0, f"weights {weights}"
    assert out.shape == (B, T, hdim)


@pytest.mark.critical
@pytest.mark.parametrize(
    "pos,dk",
    [("xpos", None), ("timexpos", 32)],
)
def test_multiscale_retention(pos, dk):
    d_model, num_heads, _ = 64, 4, 64
    B, T = 2, 11
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = FlexibleAttnInput(
        torch.randn(B, T, d_model), time=torch.arange(T)[None, :].expand(B, -1)
    )

    retention = MultiScaleRetention(
        embed_dim=d_model, num_heads=num_heads, xpos=pos, dk=dk
    )
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    attn_mask = attn_mask.bool()
    out, weights = retention.forward(
        query=input_x,
        key=input_x,
        value=input_x,
        need_weights=True,
        attn_mask=attn_mask,
    )
    assert weights.shape == (B, num_heads, T, T)
    assert weights[0, 0, 0, 1] == 0, f"weights {weights}"
    assert out.shape == (B, T, d_model)
