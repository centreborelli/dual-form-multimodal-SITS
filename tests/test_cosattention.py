import pytest
import torch
from einops import repeat

from cd_mm_sits.layers.cosattention import (
    CosformerAttention,
    TimeCosFormerAttention,
)
from cd_mm_sits.layers.linear_attention import FlexibleAttnInput
from cd_mm_sits.model.dataclass import State

torch.set_float32_matmul_precision("high")


@pytest.mark.critical
def test_cos_attention_forward():
    d_model, num_heads, kdim = 64, 4, 64
    attn = CosformerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model
    )
    input_x = FlexibleAttnInput(torch.randn(2, 11, d_model), None, None)
    output, attn_w = attn.forward(
        input_x, input_x, input_x, attn_mask=None, need_weights=True
    )
    assert output.shape == (2, 11, d_model)


@pytest.mark.critical
def test_cos_attention_forward_dk():
    d_model, num_heads, kdim = 64, 4, 64
    attn = CosformerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model, dk=8
    )
    input_x = FlexibleAttnInput(torch.randn(2, 11, d_model), None, None)
    output, attn_w = attn.forward(
        input_x, input_x, input_x, attn_mask=None, need_weights=True
    )
    assert output.shape == (2, 11, d_model)


@pytest.mark.critical
def test_cos_attention_forward_padding():
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    attn = CosformerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model
    )
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = FlexibleAttnInput(torch.randn(B, T, d_model))
    output, attn_w = attn.forward(
        input_x,
        input_x,
        input_x,
        key_padding_mask=src_key_padding_mask.bool(),
        need_weights=True,
    )
    assert output.shape == (B, T, d_model)
    # print(attn_w[0, ...])
    if attn_w is not None:
        print(attn_w.shape)
        assert attn_w[0, 0, 0, -1] == 0


@pytest.mark.critical
def test_cos_attention_padding_correct():
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    attn = CosformerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model
    )
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = FlexibleAttnInput(torch.randn(B, T, d_model))
    output, attn_w = attn.forward(
        input_x,
        input_x,
        input_x,
        key_padding_mask=src_key_padding_mask.bool(),
        need_weights=True,
    )
    output2, attn_w = attn.right_attn(
        input_x,
        input_x,
        input_x,
        key_padding_mask=src_key_padding_mask.bool(),
        need_weights=True,
    )
    assert output.shape == (B, T, d_model)
    assert torch.allclose(output, output2, atol=1e-5, rtol=1e-4)


@pytest.mark.critical
def test_timecosattention_forward_padding():
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    times = repeat(torch.arange(T), "T-> B T", B=B)
    attn = TimeCosFormerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model
    )
    attn.eval()
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = FlexibleAttnInput(torch.randn(B, T, d_model), times)
    output, attn_w = attn.forward(
        query=input_x,
        key=input_x,
        value=input_x,
        key_padding_mask=src_key_padding_mask.bool(),
        need_weights=True,
    )
    output2, _ = attn.right_attn(
        query=input_x,
        key=input_x,
        value=input_x,
        key_padding_mask=src_key_padding_mask.bool(),
        need_weights=True,
    )
    assert output.shape == (B, T, d_model)
    # print(attn_w[0, ...])
    if attn_w is not None:
        assert attn_w[0, 0, 0, -1] == 0
    dis = torch.nn.MSELoss()
    assert torch.allclose(
        output, output2, atol=1e-5, rtol=1e-4
    ), f"dis {dis(output, output2)}"


@pytest.mark.critical
def test_recurrent_cosattention():
    d_model, num_heads, kdim = 64, 4, 64
    B, T = 2, 11
    attn = CosformerAttention(
        embed_dim=d_model, num_heads=num_heads, kdim=kdim, vdim=d_model, dk=8
    )
    attn.eval()
    src_key_padding_mask = torch.zeros(B, T)
    src_key_padding_mask[:, -1] = 1
    input_x = torch.randn(B, T, d_model)
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    output_para, _ = attn.forward(
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        FlexibleAttnInput(input_x),
        attn_mask=attn_mask,
        key_padding_mask=None,
        need_weights=True,
    )
    state = State()
    l_out = []
    for t in range(T):
        temp_b = FlexibleAttnInput(input_x[:, [t], ...])
        indx = torch.Tensor([t] * B)[:, None] + 1
        out, state = attn.one_step_forward(temp_b, state, x_index=indx, m=T)
        l_out += [out]

    rec_out = torch.cat(l_out, dim=1)

    assert torch.allclose(output_para, rec_out, atol=1e-5, rtol=1e-4)
