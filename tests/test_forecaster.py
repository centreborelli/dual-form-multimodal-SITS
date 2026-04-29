import pytest
import torch

from cd_mm_sits.model.forecaster import (
    AngleEncoder,
    InputForecForward,
    MLPWeatherEncoder,
    MMShallowForecast,
    RopeMMShallowForecaster,
    WeatherLQEncoder,
)


@pytest.mark.critical
def test_forward_shallow_forecast():
    inplanes, planes, d_pe, band_s2, band_s1 = 32, 16, 8, 10, 2
    t_pred = 4
    b, _, h, w = 2, 1, 64, 64
    model = MMShallowForecast(
        inplanes, planes, d_pe, band_s2, band_s1, n_mod_feat=16, n_layers=1
    )
    time = torch.randn(b * t_pred)
    lat_repr = torch.randn(b * t_pred, h, w, inplanes)
    input_s1 = InputForecForward(
        input_time=None, lat_repr=lat_repr, target_time=time, mod="s1"
    )
    input_s2 = InputForecForward(
        input_time=None, lat_repr=lat_repr, target_time=time, mod="s2"
    )
    out_s1 = model(input_s1)
    assert out_s1.shape == (b * t_pred, band_s1, h, w)
    out_s2 = model(input_s2)
    assert out_s2.shape == (b * t_pred, band_s2, h, w)


@pytest.mark.critical
@pytest.mark.parametrize("n_layers,use_norm", [(1, True), (4, False)])
def test_forward_rope_forecast(n_layers, use_norm):
    inplanes, planes, d_pe, band_s2, band_s1 = 32, 16, 8, 10, 2
    t_pred = 4
    b, _, h, w = 2, 1, 64, 64
    model = RopeMMShallowForecaster(
        inplanes,
        planes,
        d_pe,
        band_s2,
        band_s1,
        n_mod_feat=16,
        base=1000,
        n_layers=n_layers,
        use_norm=use_norm,
    )
    time = torch.randn(b * t_pred)
    lat_repr = torch.randn(b * t_pred, h, w, inplanes)
    input_s1 = InputForecForward(
        input_time=None, lat_repr=lat_repr, target_time=time, mod="s1"
    )
    input_s2 = InputForecForward(
        input_time=None, lat_repr=lat_repr, target_time=time, mod="s2"
    )
    out_s1 = model(input_s1)
    assert out_s1.shape == (b * t_pred, band_s1, h, w)

    out_s2 = model(input_s2)
    assert out_s2.shape == (b * t_pred, band_s2, h, w)


@pytest.mark.critical
@pytest.mark.parametrize(
    "n_layers,use_norm,w_encoder",
    [(1, True, "lq"), (4, False, "lq"), (4, False, "mlp")],
)
def test_forward_rope_forecast_weather(n_layers, use_norm, w_encoder):
    torch.autograd.set_detect_anomaly(True)
    inplanes, planes, d_pe, band_s2, band_s1 = 32, 16, 8, 10, 2
    d_in_w, d_w, t_w = 3, 16, 5
    d_angles = 4
    if w_encoder == "lq":
        weather_encoder = WeatherLQEncoder(
            n_head=2, d_k=16, d_in=d_in_w, n_q=2, d_v=d_w, max_len=t_w
        )
    elif w_encoder == "mlp":
        weather_encoder = MLPWeatherEncoder(d_in=d_in_w, d_out=2 * d_w, max_len=t_w)
    else:
        raise NotImplementedError
    t_pred = 4
    b, _, h, w = 2, 1, 64, 64
    model = RopeMMShallowForecaster(
        inplanes,
        planes,
        d_pe,
        band_s2,
        band_s1,
        n_mod_feat=16,
        base=1000,
        n_layers=n_layers,
        use_norm=use_norm,
        weather_encoder=weather_encoder,
        angle_encoder=AngleEncoder(d_angles, 8),
    )
    time = torch.randn(b * t_pred)
    lat_repr = torch.randn(b * t_pred, h, w, inplanes)
    aux = torch.randn(b * t_pred, t_w, d_in_w)
    angles_s1 = torch.randn(b * t_pred, d_angles)
    input_s1 = InputForecForward(
        input_time=None,
        lat_repr=lat_repr,
        target_time=time,
        mod="s1",
        weather=aux,
        angles=angles_s1,
    )
    input_s2 = InputForecForward(
        input_time=None, lat_repr=lat_repr, target_time=time, mod="s2", weather=aux
    )
    out_s1 = model(input_s1)
    assert out_s1.shape == (b * t_pred, band_s1, h, w)

    out_s2 = model(input_s2)
    assert out_s2.shape == (b * t_pred, band_s2, h, w)
    hp_dic = model._load_hp()
    print(hp_dic)

    loss = out_s1.sum() + out_s2.sum()
    loss.backward()
    shared_params = ["rope", "weather_encoder", "mm_embedding", "mlp"]
    modal_params = ["out_s1", "out_s2"]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(shared in name for shared in shared_params):
                assert param.grad is not None, (
                    f"Shared param {name} should have gradients"
                )
            elif any(modal in name for modal in modal_params):
                # Modal params might not have gradients - that's OK
                pass

    if isinstance(weather_encoder, WeatherLQEncoder):
        assert (
            hp_dic["forecaster/weather_encoder/n_q"]
            == model.weather_encoder.config_lq_mha.n_q
        ), hp_dic


def test_angle_encoder_forward():
    B, T, C, d = 2, 20, 10, 64
    angle_encoder = AngleEncoder(n_angles=C, d_=d)
    out = angle_encoder(torch.randn(B, T, C))
    assert out.shape == (B, T, d)


# @pytest.mark.parametrize("n_layers", [(1), (4)])
# def test_forward_twope_forecast(n_layers):
#     inplanes, planes, d_pe, band_s2, band_s1 = 32, 16, 8, 10, 2
#     t_pred = 4
#     b, _, h, w = 2, 1, 64, 64
#     model = MMShallow2PeForecast(
#         inplanes,
#         planes,
#         d_pe,
#         band_s2,
#         band_s1,
#         n_mod_feat=16,
#         n_layers=n_layers,
#     )
#     time = torch.randn(b * t_pred)
#     lat_repr = torch.randn(b * t_pred, h, w, inplanes)
#     input_s1 = InputForecForward(
#         input_time=time, lat_repr=lat_repr, target_time=time, mod="s1"
#     )
#     input_s2 = InputForecForward(
#         input_time=time, lat_repr=lat_repr, target_time=time, mod="s2"
#     )
#     out_s1 = model(input_s1)
#     assert out_s1.shape == (b * t_pred, band_s1, h, w)


#     out_s2 = model(input_s2)
#     assert out_s2.shape == (b * t_pred, band_s2, h, w)
