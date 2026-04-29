import pytest
import torch
from einops import repeat

from cd_mm_sits.layers.linear_attention import FlexibleAttnInput
from cd_mm_sits.layers.roformer_attention import (
    RoFormerAttention,
    TimeRoFormerAttention,
)


@pytest.mark.critical
@pytest.mark.parametrize(
    "padding,attn_name",
    [(True, "time"), (False, "notime")],
)
def test_timeroformerattention_forward_padding(padding, attn_name):
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    times = repeat(torch.arange(T), "T-> B T", B=B)
    if attn_name == "time":
        attn = TimeRoFormerAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=kdim,
            vdim=d_model,
            act_fun="elu",
        )
    else:
        attn = RoFormerAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=kdim,
            vdim=d_model,
            act_fun="elu",
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

    output2, _ = attn.right_attn(
        query=input_x,
        key=input_x,
        value=input_x,
        key_padding_mask=src_key_padding_mask,
        need_weights=True,
        eps=1e-10,
    )
    assert output.shape == (B, T, d_model)
    # print(attn_w[0, ...])
    if attn_w is not None and padding:
        assert attn_w[0, 0, 0, -1] == 0
    dis = torch.nn.MSELoss()
    diff = dis(output, output2)
    print(input_x)
    assert diff < 1e-4, f"dis {diff}"
    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4), (
        f"dis {dis(output, output2)}"
    )
