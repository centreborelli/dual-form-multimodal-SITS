import pytest
import torch
from einops import repeat

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput
from cd_mm_sits.layers.xpos_attention import TimeXPOSAttention


@pytest.mark.critical
@pytest.mark.parametrize(
    "padding,denom",
    [(True, True), (False, False), (False, True)],
)
def test_timexposattention_forward_padding(padding, denom):
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    times = repeat(torch.arange(T), "T-> B T", B=B)
    attn = TimeXPOSAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        kdim=kdim,
        vdim=d_model,
        act_fun="elu",
        apply_denom=denom,
    )
    attn.eval()
    if padding:
        src_key_padding_mask = torch.zeros(B, T)
        src_key_padding_mask[:, -1] = 1
        src_key_padding_mask = src_key_padding_mask.bool()
    else:
        src_key_padding_mask = None
    input_x = FlexibleAttnInput(torch.randn(B, T, d_model), times)
    output, attn_w = attn.forward(
        query=input_x,
        key=input_x,
        value=input_x,
        key_padding_mask=src_key_padding_mask,
        need_weights=True,
        eps=1e-10,
    )
    assert output.shape == (B, T, d_model)
    assert torch.all(~torch.isneginf(output))
