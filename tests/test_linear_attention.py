import pytest
import torch

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput, LinearAttention


@pytest.mark.critical
def test_forward_linear_attention():
    d_model, num_heads, kdim = 64, 4, 64
    attn = LinearAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model
    )
    attn.eval()
    T = 11
    input_x = torch.randn(2, T, d_model)
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    attn_mask = attn_mask.type_as(input_x).bool().to(device=input_x.device)
    output, attn_w = attn.forward(
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        attn_mask=attn_mask,
        is_causal=True,
        need_weights=True,
    )
    output2, _ = attn.right_attn(
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        attn_mask=attn_mask,
        is_causal=True,
        need_weights=False,
    )

    assert output.shape == (2, 11, d_model)
    assert output2.shape == (2, 11, d_model)

    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_forward_linear_attention_dk():
    d_model, num_heads, kdim = 64, 4, 64
    attn = LinearAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model, dk=8
    )
    attn.eval()
    T = 11
    input_x = torch.randn(2, T, d_model)
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    attn_mask = attn_mask.type_as(input_x).bool().to(device=input_x.device)
    output, attn_w = attn.forward(
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        attn_mask=attn_mask,
        is_causal=True,
        need_weights=True,
    )
    output2, _ = attn.right_attn(
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        attn_mask=attn_mask,
        is_causal=True,
        need_weights=False,
    )

    assert output.shape == (2, 11, d_model)
    assert output2.shape == (2, 11, d_model)

    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4)
