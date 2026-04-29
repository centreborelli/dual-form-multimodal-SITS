import pytest
import torch
import torch.nn as nn

from cd_mm_sits.layers.cosattention import CosformerAttention
from cd_mm_sits.layers.transformer_layers import (
    FFLayer,
    KernelAttentionLayer,
    VanillaTransformerLayer,
)
from cd_mm_sits.model.basic_baselines import OnlyUnet
from cd_mm_sits.model.decoder import ConvUpSample, SimplePixelShuffleDec
from cd_mm_sits.model.sse import LowResUnet, Unet
from cd_mm_sits.model.sste import SSTE
from cd_mm_sits.model.template_dual_encoders import DualEncoder
from cd_mm_sits.model.temporal_positional_encoder import PositionalEncoder


@pytest.mark.parametrize(
    "convert_to_dyntanh",
    [(False), (True)],
)
@pytest.mark.critical
def test_forward_sste(convert_to_dyntanh):
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64

    unet = Unet(
        inplanes=c,
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
    batch = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
        layout=torch.jagged,
    )
    doy = torch.nested.nested_tensor(
        [torch.arange(t1), torch.arange(t2)],
        layout=torch.jagged,
    )
    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        convert_to_dyntanh=convert_to_dyntanh,
    )
    output, weights, padded_mask = sste(batch, doy, need_weights=True)
    assert output.shape == (2, max(t1, t2), h, w, 2)
    assert padded_mask.shape == (2 * h * w, max(t1, t2))


@pytest.mark.critical
def test_forward_sste_lowres_unet():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64

    unet = LowResUnet(
        inplanes=c,
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
    last_layer = ConvUpSample(d_model, 2)
    batch = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
        layout=torch.jagged,
    )
    doy = torch.nested.nested_tensor(
        [torch.arange(t1), torch.arange(t2)],
        layout=torch.jagged,
    )
    sste = SSTE(sse=unet, temporal_encoder=transformer, tpe=pe, last_layer=last_layer)
    output, *_ = sste(batch, doy, need_weights=True)
    assert output.shape == (2, max(t1, t2), h, w, 2)


@pytest.mark.critical
def test_forward_lowres_sste_nonested():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64
    unet = LowResUnet(
        inplanes=c,
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
    last_layer = ConvUpSample(d_model, 2)
    batch = torch.randn(2, max(t1, t2), c, h, w)
    doy = torch.ones(2, max(t1, t2))
    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    output, *_ = sste(batch, doy, need_weights=True)

    assert output.shape == (2, max(t1, t2), h, w, 2)


def test_forward_onlyunet_nonested():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64
    unet = Unet(
        inplanes=c,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    last_layer = nn.Linear(d_model, 2)
    batch = torch.randn(2, max(t1, t2), c, h, w)
    doy = torch.ones(2, max(t1, t2))
    sste = OnlyUnet(sse=unet, last_layer=last_layer)
    output, weights, padded_mask = sste(batch, doy, need_weights=True)

    assert output.shape == (2, max(t1, t2), h, w, 2)
    assert padded_mask.shape == (2 * h * w, max(t1, t2))


@pytest.mark.critical
def test_forward_lowres_sste_grad_propagation():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64
    unet = LowResUnet(
        inplanes=c,
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
    last_layer = ConvUpSample(d_model, 2)
    batch = torch.randn(2, max(t1, t2), c, h, w)
    doy = torch.ones(2, max(t1, t2))
    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    output, *_ = sste(batch, doy, need_weights=True)

    assert output.shape == (2, max(t1, t2), h, w, 2)

    # Trigger gradient backprop
    output.sum().backward()

    # Test for gradient correclty propagated
    for params in sste.parameters():
        assert params.grad is not None


@pytest.mark.critical
def test_forward_sste_nonested():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64

    unet = Unet(
        inplanes=c,
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
    batch = torch.randn(2, max(t1, t2), c, h, w)
    doy = torch.ones(2, max(t1, t2))
    sste = SSTE(sse=unet, temporal_encoder=transformer, tpe=pe, last_layer=last_layer)
    output, *_ = sste(batch, doy, need_weights=True)
    assert output.shape == (2, max(t1, t2), h, w, 2)


@pytest.mark.critical
def test_one_step_forward_sste():
    t1, _ = 7, 13
    c, h, w = 10, 64, 64
    d_model = 64

    unet = Unet(
        inplanes=c,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
    )
    ffl_layer = FFLayer(
        d_model=d_model,
        dim_feedforward=128,
    )
    mha = CosformerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    num_layers = 2
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = KernelAttentionLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=2, layer=layer)
    pe = PositionalEncoder(d=d_model)
    last_layer = nn.Linear(d_model, 2)

    batch = torch.randn(1, t1, c, h, w)
    doy = torch.arange(t1)[None, :]
    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    sste.eval()
    sste.temporal_encoder.eval()
    states = [None] * num_layers
    outputs = []
    for img_index in range(batch.shape[1]):
        output, states = sste.one_step_forward(
            batch[:, [img_index], ...],
            states=states,
            x_index=torch.Tensor([img_index])[:, None] + 1,
            time=doy[:, [img_index]],
            m=t1,
            key_padding_mask=None,
        )
        outputs += [output]
    outputs = torch.cat(outputs, dim=1)
    output1, *_ = sste(
        batch,
        torch.nested.nested_tensor([torch.arange(t1)], layout=torch.jagged),
        need_weights=True,
    )

    assert torch.allclose(output1, outputs, atol=1e-5, rtol=1e-4)
