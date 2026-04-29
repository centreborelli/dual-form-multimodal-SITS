import pytest
import torch
import torch.nn as nn

from cd_mm_sits.layers.linear_attention import LinearAttention
from cd_mm_sits.layers.transformer_layers import (
    FFLayer,
    VanillaTransformerLayer,
)
from cd_mm_sits.model.decoder import ConvUpSample
from cd_mm_sits.model.mm_sste import MMSSTE
from cd_mm_sits.model.sse import LowResUnet, Unet
from cd_mm_sits.model.template_dual_encoders import DualEncoder
from cd_mm_sits.model.temporal_positional_encoder import PositionalEncoder


@pytest.mark.critical
def test_forward_mm_sste():
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64

    unet1 = Unet(
        inplanes=c_s1,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    unet2 = Unet(
        inplanes=c_s2,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=2, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = nn.Linear(d_model, 2)
    batch_1 = torch.nested.nested_tensor(
        [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
        layout=torch.jagged,
    )
    batch_2 = torch.nested.nested_tensor(
        [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
        layout=torch.jagged,
    )
    doy_1 = torch.nested.nested_tensor(
        [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
    )
    doy_2 = torch.nested.nested_tensor(
        [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
    )

    # ll_times=torch.nested.nested_tensor([torch.arange(2*t1),torch.arange(2*t1)])
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    out_forward = sste(
        sits_s1=batch_1,
        sits_s2=batch_2,
        time_s1=doy_1,
        time_s2=doy_2,
        need_weights=True,
        s1_orbites=doy_1,
    )
    output, idx, we = out_forward.output, out_forward.sorted_idx, out_forward.weights
    T = max(t11 + t21, t12 + t22)
    assert idx.shape[-1] == T
    assert output.shape == (2, T, h, w, 2)
    assert len(we) == 2
    assert we[0].shape == (2 * h * w, 4, T, T)


@pytest.mark.critical
def test_forward_mm_sste_lowresunet():
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64

    unet1 = LowResUnet(
        inplanes=c_s1,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    unet2 = LowResUnet(
        inplanes=c_s2,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=2, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = ConvUpSample(d_model, 2)

    batch_1 = torch.nested.nested_tensor(
        [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
        layout=torch.jagged,
    )
    batch_2 = torch.nested.nested_tensor(
        [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
        layout=torch.jagged,
    )
    doy_1 = torch.nested.nested_tensor(
        [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
    )
    doy_2 = torch.nested.nested_tensor(
        [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
    )

    # ll_times=torch.nested.nested_tensor([torch.arange(2*t1),torch.arange(2*t1)])
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    out_forward = sste(
        sits_s1=batch_1,
        sits_s2=batch_2,
        time_s1=doy_1,
        time_s2=doy_2,
        need_weights=True,
        s1_orbites=doy_1,
    )
    output, idx, we = out_forward.output, out_forward.sorted_idx, out_forward.weights
    T = max(t11 + t21, t12 + t22)
    assert idx.shape[-1] == T
    assert output.shape == (2, T, h, w, 2)
    assert len(we) == 2
    assert we[0].shape == (2 * h // 2 * w // 2, 4, T, T)


def is_sorted_ascending(tensor):
    return torch.all(tensor[:-1] <= tensor[1:])


@pytest.mark.critical
def test_mm_token_embedding():
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64

    unet1 = Unet(
        inplanes=c_s1,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    unet2 = Unet(
        inplanes=c_s2,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = nn.Linear(d_model, 2)
    batch_1 = torch.randn(2, t11, d_model, h, w)
    batch_2 = torch.randn(2, t12, d_model, h, w)
    doy_1 = torch.randn(2, t11)
    doy_2 = torch.randn(2, t12)

    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    output = sste.mm_token_embedding(
        x_s1=batch_1,
        x_s2=batch_2,
        lengths_s1=[t11, t12],
        lengths_s2=[t21, t22],
        time_s1=doy_1,
        time_s2=doy_2,
        s1_orbites=torch.ones_like(doy_1),
    )

    assert is_sorted_ascending(
        output.sorted_times[0, :]
    ), f"sorted times {output.sorted_times[0, :]}"


@pytest.mark.critical
def test_forward_mm_sste_right_prod():
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64

    unet1 = Unet(
        inplanes=c_s1,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    unet2 = Unet(
        inplanes=c_s2,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=2, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = nn.Linear(d_model, 2)
    batch_1 = torch.nested.nested_tensor(
        [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
        layout=torch.jagged,
    )
    batch_2 = torch.nested.nested_tensor(
        [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
        layout=torch.jagged,
    )
    doy_1 = torch.nested.nested_tensor(
        [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
    )
    doy_2 = torch.nested.nested_tensor(
        [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
    )

    # ll_times=torch.nested.nested_tensor([torch.arange(2*t1),torch.arange(2*t1)])
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    out_forward = sste(
        sits_s1=batch_1,
        sits_s2=batch_2,
        time_s1=doy_1,
        time_s2=doy_2,
        need_weights=True,
        s1_orbites=doy_1,
        right_product=True,
    )
    output, idx, we = out_forward.output, out_forward.sorted_idx, out_forward.weights
    T = max(t11 + t21, t12 + t22)
    assert idx.shape[-1] == T
    assert output.shape == (2, T, h, w, 2)
    assert len(we) == 2
    assert we[0].shape == (2 * h * w, 4, T, T)


@pytest.mark.critical
def test_forward_mm_sste_linformer():
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    is_nested = True
    unet1 = Unet(
        inplanes=c_s1,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    unet2 = Unet(
        inplanes=c_s2,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = LinearAttention(embed_dim=d_model, num_heads=4)
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=2, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = nn.Linear(d_model, 2)
    if is_nested:
        batch_1 = torch.nested.nested_tensor(
            [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
            layout=torch.jagged,
        )
        batch_2 = torch.nested.nested_tensor(
            [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
            layout=torch.jagged,
        )
        doy_1 = torch.nested.nested_tensor(
            [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
        )
        doy_2 = torch.nested.nested_tensor(
            [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
        )
        T = max(t11 + t21, t12 + t22)
    else:
        T = t11 + t12
        batch_1 = torch.randn(2, t11, c_s1, h, w)
        batch_2 = torch.randn(2, t12, c_s2, h, w)
        doy_1 = torch.ones_like(torch.randn(2, t11))
        doy_2 = torch.randn(2, t12)

    # ll_times=torch.nested.nested_tensor([torch.arange(2*t1),torch.arange(2*t1)])
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    out_forward = sste(
        sits_s1=batch_1,
        sits_s2=batch_2,
        time_s1=doy_1,
        time_s2=doy_2,
        need_weights=True,
        s1_orbites=doy_1,
        right_product=False,
    )
    output, idx, we = out_forward.output, out_forward.sorted_idx, out_forward.weights

    assert idx.shape[-1] == T
    assert output.shape == (2, T, h, w, 2)
    assert len(we) == 2
    assert we[0].shape == (2 * h * w, 4, T, T)
    assert output.requires_grad
    # Trigger gradient backprop
    output.sum().backward()
    # Check each parameter individually
    for name, param in sste.named_parameters():
        if param.grad is None:
            print(f"No gradient for: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Is leaf: {param.is_leaf}")
        else:
            print(f"Has gradient: {name}")
    # Test for gradient correclty propagated
    for params in sste.parameters():
        assert params.grad is not None
