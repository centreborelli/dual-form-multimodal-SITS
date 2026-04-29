import pytest
import torch
import torch.nn as nn

from cd_mm_sits.layers.cosattention import CosformerAttention, TimeCosFormerAttention
from cd_mm_sits.layers.dynamic_tanh import convert_ln_to_dyt
from cd_mm_sits.layers.retention import MultiScaleRetention
from cd_mm_sits.layers.transformer_layers import (
    FFLayer,
    KernelAttentionLayer,
    TimeFormerLayer,
    VanillaTransformerLayer,
)
from cd_mm_sits.layers.xpos_attention import TimeXPOSAttention
from cd_mm_sits.model.template_dual_encoders import DualEncoder


@pytest.mark.critical
def test_build_transformer():
    """Test to check whether or not we are able to instantiate a transformer"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    out, weights = transformer(batch=torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 64)
    assert weights[0] is None
    out, weights = transformer(batch=torch.randn(2, 10, 64), need_weights=True)
    assert out.shape == (2, 10, 64)
    assert weights[0].shape == (2, 4, 10, 10)


@pytest.mark.critical
def test_build_cosformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = CosformerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    out, weights = transformer(batch=torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 64)
    assert weights[0] is None
    out, weights = transformer(
        batch=torch.randn(2, 10, 64), need_weights=True, is_causal=True
    )
    if weights[0] is not None:
        assert weights[0][0, 0, 8, 9] == 0
        assert weights[0][0, 0, 9, 9] != 0
        assert weights[0].shape == (2, 4, 10, 10)
    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_build_causal_cosformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = CosformerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)

    out, weights = transformer(
        batch=torch.randn(2, 10, 64), need_weights=True, is_causal=True
    )

    assert out.shape == (2, 10, 64)
    # assert weights[0][0, 0, 8, 9] == 0
    # assert weights[0][0, 0, 9, 9] != 0
    # assert weights[0].shape == (2, 4, 10, 10)


@pytest.mark.critical
def test_build_causal_cosformer_padding():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = CosformerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    pad_mask = torch.zeros(2, 10)
    pad_mask[:, 9] = 1
    pad_mask[1, 8] = 1
    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        key_padding_mask=pad_mask.bool(),
    )

    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_build_causal_timecosformer_left():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = TimeCosFormerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = TimeFormerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)

    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        times=torch.ones(2, 10),
        right_product=True,
    )
    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_build_causal_timecosformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = TimeCosFormerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = TimeFormerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)

    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        times=torch.ones(2, 10),
    )

    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_build_causal_xposformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = TimeXPOSAttention(embed_dim=d_model, num_heads=2, act_fun="elu")
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = TimeFormerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)

    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        times=torch.arange(10)[None, ...].expand(2, -1),
    )

    assert out.shape == (2, 10, 64)
    assert torch.all(~torch.isneginf(out))


@pytest.mark.critical
def test_convert_causal_timecosformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = TimeCosFormerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = TimeFormerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    transformer = convert_ln_to_dyt(transformer)
    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        times=torch.ones(2, 10),
    )

    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_retention_in_transformer():
    """Test to check whether or not we are able to instantiate a transformer
    with cosformer attention"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = MultiScaleRetention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    transformer = convert_ln_to_dyt(transformer)
    out, weights = transformer(
        batch=torch.randn(2, 10, 64),
        need_weights=True,
        is_causal=True,
        times=torch.ones(2, 10),
    )

    assert out.shape == (2, 10, 64)


@pytest.mark.critical
def test_build_recurring_dual_encoders():
    """Test the implementation of the reccurent form of the CosFormer"""
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = CosformerAttention(
        embed_dim=d_model,
        num_heads=4,
    )
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = KernelAttentionLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer = DualEncoder(num_layers=1, layer=layer)
    transformer.eval()
    pad_mask = torch.zeros(2, 10)
    pad_mask[:, 9] = 1
    pad_mask[1, 8] = 1
    states = [None]
    batch = torch.randn(2, 10, 64)
    outputs = []
    for img_index in range(batch.shape[1]):
        # print(batch[:, [img_index], ...].shape)

        out, states = transformer.one_step_forward(
            x=batch[:, [img_index], ...],
            m=10,
            x_index=torch.Tensor([img_index] * 2)[:, None] + 1,
            states=states,
        )
        outputs += [out]
    outputs = torch.cat(outputs, dim=1)
    output1, weights = transformer(
        batch=batch,
        need_weights=True,
        is_causal=True,
    )

    assert outputs.shape == (2, 10, 64)
    print(output1.shape, outputs.shape)
    assert torch.allclose(output1, outputs, atol=1e-5, rtol=1e-4)
