import pytest
import torch
import torch.nn as nn

from cd_mm_sits.model.basic_baselines import OnlyUnet
from cd_mm_sits.model.sse import FCNResNet50, Unet


@pytest.mark.critical
def test_forward_ony_unet():
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
    batch = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)], layout=torch.jagged
    )
    doy = torch.nested.nested_tensor(
        [torch.arange(t1), torch.arange(t2)], layout=torch.jagged
    )
    only_unet = OnlyUnet(sse=unet, last_layer=last_layer)
    output, weights, padd = only_unet(batch, doy, need_weights=True)
    assert output.requires_grad
    output.sum().backward()

    # Test for gradient correclty propagated
    for params in only_unet.parameters():
        assert params.grad is not None

    assert output.shape == (2, max(t1, t2), h, w, 2)


@pytest.mark.critical
def test_forward_only_fcn():
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    d_model = 64

    unet = FCNResNet50(inplanes=c, planes=64)
    last_layer = nn.Linear(d_model, 2)
    batch = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
    )
    doy = torch.nested.nested_tensor([torch.arange(t1), torch.arange(t2)])
    only_unet = OnlyUnet(sse=unet, last_layer=last_layer)
    output, weights, padd = only_unet(batch, doy, need_weights=True)
    assert output.shape == (2, max(t1, t2), h, w, 2)
