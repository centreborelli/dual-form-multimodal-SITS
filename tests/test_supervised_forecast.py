from pathlib import Path

import hydra
import pytest
import torch
import torch.nn as nn
import torchmetrics
from einops import rearrange
from hydra import compose, initialize
from hydra.utils import instantiate

from cd_mm_sits.data.batch_class import MMForcBatch, SITSBatch
from cd_mm_sits.layers.transformer_layers import FFLayer, VanillaTransformerLayer
from cd_mm_sits.lightning_module.hydra_dataclass import CAWConfig, OptimizerAdamConfig
from cd_mm_sits.lightning_module.losses import FocalLoss
from cd_mm_sits.lightning_module.supervised_forecast import (
    MMSupervisedForecast,
    MonoMForecast,
)
from cd_mm_sits.lightning_module.template_module import TrainConfig
from cd_mm_sits.model.decoder import ConvUpSample, PixelShuffleDec
from cd_mm_sits.model.forecaster import (
    AngleEncoder,
    MMShallowForecast,
    RopeMMShallowForecaster,
    WeatherLQEncoder,
)
from cd_mm_sits.model.mm_sste import MMSSTE
from cd_mm_sits.model.sse import LowResUnet, Unet
from cd_mm_sits.model.sste import SSTE
from cd_mm_sits.model.template_dual_encoders import DualEncoder
from cd_mm_sits.model.temporal_positional_encoder import PositionalEncoder
from cd_mm_sits.model.utils import cat_nested_along_time


def create_rdn_notnested_batch_mm(
    t11: int = 8,
    t12: int = 20,
    t21: int = 20,
    t22: int = 10,
    c_s1: int = 2,
    c_s2: int = 10,
    h: int = 64,
    w: int = 64,
):
    T1 = max(t11, t12)
    T2 = max(t21, t22)
    batch_1 = torch.randn(2, T1, c_s1, h, w)
    doy_1 = torch.zeros(2, T1)
    batch_2 = torch.randn(2, T2, c_s2, h, w)
    doy_2 = torch.randn(2, T2)
    mask1 = torch.zeros(2, T1, h, w)
    mask2 = torch.zeros(2, T2, h, w)
    b_s1 = SITSBatch(
        content=batch_1, time=doy_1, mod="s1", orbites=doy_1, validity_mask=mask1
    )
    b_s2 = SITSBatch(content=batch_2, time=doy_2, mod="s2", validity_mask=mask2)
    return MMForcBatch(s1=b_s1, s2=b_s2)


def create_rdn_nested_batch_forecast(
    t11: int = 8,
    t12: int = 20,
    t21: int = 20,
    t22: int = 10,
    c_s1: int = 2,
    c_s2: int = 10,
    h: int = 64,
    w: int = 64,
    t_w: int = 10,
    c_w: int = 8,
    d_angles: int = 16,
):
    s_1 = SITSBatch(
        content=torch.nested.nested_tensor(
            [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
            layout=torch.jagged,
        ),
        time=torch.nested.nested_tensor(
            [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
        ),
        orbites=torch.nested.nested_tensor(
            [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
        ),
        mod="s1",
        validity_mask=torch.nested.nested_tensor(
            [torch.ones(t11, h, w), torch.randn(t12, h, w)],
            layout=torch.jagged,
        ),
        weather_var=torch.nested.nested_tensor(
            [torch.randn(t11, t_w, c_w), torch.randn(t12, t_w, c_w)],
            layout=torch.jagged,
        ),
        angles=torch.nested.nested_tensor(
            [torch.randn(t11, d_angles), torch.randn(t12, d_angles)],
            layout=torch.jagged,
        ),
    )
    s_2 = SITSBatch(
        content=torch.nested.nested_tensor(
            [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
            layout=torch.jagged,
        ),
        time=torch.nested.nested_tensor(
            [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
        ),
        mod="s2",
        validity_mask=torch.nested.nested_tensor(
            [torch.ones(t21, h, w), torch.ones(t22, h, w)],
            layout=torch.jagged,
        ),
        weather_var=torch.nested.nested_tensor(
            [torch.randn(t21, t_w, c_w), torch.randn(t22, t_w, c_w)],
            layout=torch.jagged,
        ),
    )

    return MMForcBatch(s1=s_1, s2=s_2)


@pytest.mark.critical
def test_supervised_forecast_forward():
    t11, t12 = 20, 23
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    OUT_C = 2
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
    last_layer = nn.Linear(d_model, OUT_C)
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    sh_forecast = MMShallowForecast(OUT_C, 32, 8, 10, 2, n_mod_feat=16, n_layers=1)
    pred_after_n = 5
    pl_module = MMSupervisedForecast(
        train_config=train_config,
        model=sste,
        shallow_forecaster=sh_forecast,
        pred_after_n=pred_after_n,
    )
    batch = create_rdn_notnested_batch_mm(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    print(batch.s1.validity_mask.shape)
    print(batch.s2.validity_mask.shape)
    out = pl_module(batch)
    final_b = batch.s1.content.shape[0] * (
        batch.s1.content.shape[1] + batch.s2.content.shape[1] - pred_after_n - 1
    )

    assert out.last_repr.shape[0] == final_b

    assert out.pred_s1.shape == (out.pred_s1.shape[0], c_s1, h, w)
    assert out.pred_s2.shape == (out.pred_s2.shape[0], c_s2, h, w)
    loss = pl_module.training_step(batch, 0)
    assert loss.requires_grad


@pytest.mark.parametrize("use_delta_times", [(True), (False)])
@pytest.mark.critical
def test_supervised_forecast_forward_nested(use_delta_times):
    t11, t12 = 20, 23
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    OUT_C = 2
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
    last_layer = nn.Linear(d_model, OUT_C)
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    sh_forecast = MMShallowForecast(OUT_C, 32, 8, 10, 2, n_mod_feat=16, n_layers=2)
    pred_after_n = 5
    pl_module = MMSupervisedForecast(
        train_config=train_config,
        model=sste,
        shallow_forecaster=sh_forecast,
        pred_after_n=pred_after_n,
        use_delta_times=use_delta_times,
    )
    batch = create_rdn_nested_batch_forecast(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    out = pl_module(batch)

    assert out.last_repr.shape[0] == 48
    assert out.pred_s1.shape == (out.pred_s1.shape[0], c_s1, h, w)
    assert out.pred_s2.shape == (out.pred_s2.shape[0], c_s2, h, w)
    loss = pl_module.training_step(batch, 0)
    assert loss.requires_grad


@pytest.mark.critical
def test_supervised_forecast_forward_nested_ropeforecast_lowres():
    t11, t12 = 20, 23
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    OUT_C = 2
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
    last_layer = ConvUpSample(d_model, OUT_C)
    sste = MMSSTE(
        sse_s1=unet1,
        sse_s2=unet2,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    sh_forecast = RopeMMShallowForecaster(
        OUT_C, 32, 8, 10, 2, n_mod_feat=16, base=1000, n_layers=3
    )
    pred_after_n = 5
    pl_module = MMSupervisedForecast(
        train_config=train_config,
        model=sste,
        shallow_forecaster=sh_forecast,
        pred_after_n=pred_after_n,
    )
    batch = create_rdn_nested_batch_forecast(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    out = pl_module(batch)

    assert out.last_repr.shape[0] == 48
    assert out.pred_s1.shape == (out.pred_s1.shape[0], c_s1, h, w)
    assert out.pred_s2.shape == (out.pred_s2.shape[0], c_s2, h, w)
    loss = pl_module.training_step(batch, 0)
    loss.backward()
    assert loss.requires_grad


def test_init_from_config_mm_forecast():
    assert Path("./config/mm_supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/",
        job_name="test_forecast_supervised",
        version_base="1.2",
    ):
        config = compose(config_name="mm_forecast.yaml")
        print(config)
        model = instantiate(config.model)
        if config.metrics is not None:
            metrics = (
                torchmetrics.MetricCollection(
                    {
                        name: hydra.utils.instantiate(metric_cfg)
                        for name, metric_cfg in config.metrics.items()
                    }
                ),
            )
        else:
            metrics = None
        train_config = instantiate(config.train.train_config, metrics=metrics)
        pl_module = instantiate(
            config.lightning_module,
            model=model,
            train_config=train_config,
        )
        assert isinstance(pl_module, MMSupervisedForecast)
        assert isinstance(pl_module.shallow_forecaster, RopeMMShallowForecaster)


def test_revert_from_padding():
    _, t1, t2, c = 2, 5, 11, 3
    s1_original = torch.nested.nested_tensor(
        [
            torch.arange(t1).unsqueeze_(-1).expand(-1, c),
            torch.arange(t1 + 1).unsqueeze_(-1).expand(-1, c),
        ],
        layout=torch.jagged,
    )
    s1_mod = torch.nested.nested_tensor(
        [torch.ones(t1, c), torch.ones(t1 + 1, c)], layout=torch.jagged
    )
    s2_mod = torch.nested.nested_tensor(
        [torch.ones(t2, c) * 2, torch.ones(t2 + 2, c) * 2], layout=torch.jagged
    )
    s2_original = torch.nested.nested_tensor(
        [
            torch.arange(t2).unsqueeze_(-1).expand(-1, c) * 2,
            torch.arange(t2 + 2).unsqueeze_(-1).expand(-1, c),
        ],
        layout=torch.jagged,
    )
    print(s1_original.shape, s2_original.shape)
    s1s2_original = cat_nested_along_time(s1_original, s2_original)
    mod = cat_nested_along_time(s1_mod, s2_mod)

    padded_s1s2 = torch.nested.to_padded_tensor(s1s2_original, padding=0.0)

    padded_mod = torch.nested.to_padded_tensor(mod, padding=-1)
    padded_s1s2 = rearrange(padded_s1s2, "B T C -> (B T ) C")
    padded_mod = rearrange(padded_mod, "B T C -> (B T) C ")

    idx_s1 = torch.nonzero(padded_mod[:, 0] == 1, as_tuple=True)[0]
    idx_s2 = torch.nonzero(padded_mod[:, 0] == 2, as_tuple=True)[0]
    pred_s1 = padded_s1s2[idx_s1, ...]
    pred_s2 = padded_s1s2[idx_s2, ...]

    print(padded_mod)
    print(s1_original.values().shape)
    print(pred_s1.shape)
    assert torch.allclose(pred_s1, s1_original.values())
    assert torch.allclose(pred_s2, s2_original.values())


@pytest.mark.parametrize(
    "decoder",
    [("conv"), ("shuffle")],
)
def test_monom_supervised_forecast(decoder):
    t11, t12 = 20, 23
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    OUT_C = 2
    d_angle = 16
    unet = LowResUnet(
        inplanes=c_s1,
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

    if decoder == "conv":
        last_layer = ConvUpSample(d_model, OUT_C)
    else:
        last_layer = PixelShuffleDec(d_model, OUT_C)

    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    sh_forecast = RopeMMShallowForecaster(
        OUT_C,
        32,
        8,
        10,
        2,
        n_mod_feat=16,
        base=1000,
        n_layers=1,
        angle_encoder=AngleEncoder(d_angle, 8),
    )

    pred_after_n = 5
    pl_module = MonoMForecast(
        train_config=train_config,
        model=sste,
        shallow_forecaster=sh_forecast,
        pred_after_n=pred_after_n,
        mod="s1",
    )

    batch = create_rdn_nested_batch_forecast(
        t11=t11,
        t12=t12,
        t21=t21,
        t22=t22,
        c_s1=c_s1,
        c_s2=c_s2,
        h=h,
        w=w,
        d_angles=d_angle,
    )
    last_repr, out, weights, idx = pl_module(batch.s1)

    assert last_repr.shape[0] == 2 * (-pred_after_n - 1 + max(t11, t12))
    assert out.shape == (out.shape[0], c_s1, h, w)
    loss = pl_module.training_step(batch.s1, 0)
    loss.backward()
    assert loss.requires_grad


@pytest.mark.parametrize(
    "decoder",
    [("conv"), ("shuffle")],
)
def test_monom_supervised_forecast_weather(decoder):
    t11, t12 = 20, 23
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    d_model = 64
    OUT_C = 2

    unet = LowResUnet(
        inplanes=c_s1,
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
    if decoder == "conv":
        last_layer = ConvUpSample(d_model, OUT_C)
    else:
        last_layer = PixelShuffleDec(d_model, OUT_C)

    sste = SSTE(
        sse=unet,
        temporal_encoder=transformer,
        tpe=pe,
        last_layer=last_layer,
        is_causal=True,
    )
    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    d_in_w, d_w, t_w = 3, 16, 5
    weather_encoder = WeatherLQEncoder(
        n_head=2, d_k=16, d_in=d_in_w, n_q=2, d_v=d_w, max_len=t_w
    )
    sh_forecast = RopeMMShallowForecaster(
        OUT_C,
        32,
        8,
        10,
        2,
        n_mod_feat=16,
        base=1000,
        n_layers=1,
        weather_encoder=weather_encoder,
    )

    pred_after_n = 5
    pl_module = MonoMForecast(
        train_config=train_config,
        model=sste,
        shallow_forecaster=sh_forecast,
        pred_after_n=pred_after_n,
        mod="s1",
    )
    batch = create_rdn_nested_batch_forecast(
        t11=t11,
        t12=t12,
        t21=t21,
        t22=t22,
        c_s1=c_s1,
        c_s2=c_s2,
        h=h,
        w=w,
        t_w=t_w,
        c_w=d_in_w,
    )
    last_repr, out, weights, idx = pl_module(batch.s1)

    assert last_repr.shape[0] == 2 * (-pred_after_n - 1 + max(t11, t12))
    assert out.shape == (out.shape[0], c_s1, h, w)
    loss = pl_module.training_step(batch.s1, 0)
    loss.backward()
    assert loss.requires_grad
