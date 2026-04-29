import pytest
import torch

from cd_mm_sits.model.sse import LowResUnet, SMPEncoder, Unet


@pytest.mark.parametrize(
    "norm,k_dec,s_dec,p_dec,skip_conv",
    [("batch", 4, 2, 1, False), ("group", 2, 2, 0, True), ("none", 2, 2, 0, False)],
)
def test_unet_nested_batch(norm, k_dec, s_dec, p_dec, skip_conv):
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    unet = Unet(
        inplanes=c,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        encoder_norm=norm,
        decoding_norm=norm,
        skip_conv_norm=norm,
        str_conv_k_dec=k_dec,
        str_conv_s_dec=s_dec,
        str_conv_p_dec=p_dec,
        return_maps=False,
        gated_skip_conv=skip_conv,
    )
    nt = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
    )
    lengths = [sits.shape[0] for sits in nt]
    input_batch = torch.cat(nt.unbind())  # (B*T, C H W)
    output = unet(input_batch)
    assert output.shape == (t1 + t2, 64, h, w)
    nested_output = torch.nested.nested_tensor(
        list(torch.split(output, lengths, dim=0))
    )
    assert nested_output[0].shape == (t1, 64, h, w)
    padded_out_tensor = torch.nested.to_padded_tensor(nested_output, padding=0.0)
    nested_masks = torch.nested.nested_tensor([torch.zeros(t1), torch.zeros(t2)])
    mask_padded = torch.nested.to_padded_tensor(nested_masks, padding=1).bool()
    assert padded_out_tensor.shape[1] == max(t1, t2)
    assert mask_padded.shape[1] == max(t1, t2)
    assert mask_padded.shape == (2, max(t1, t2))
    assert torch.sum(mask_padded) == max(t2, t1) - min(t1, t2)


@pytest.mark.parametrize(
    "depth,encoder_name", [(4, "efficientnet-b0"), (2, "mobilenet_v2")]
)
def test_smp_sse(depth, encoder_name):
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    model = SMPEncoder(
        inplanes=c, planes=64, encoder_depth=depth, encoder_name=encoder_name
    )
    nt = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
    )
    input_batch = torch.cat(nt.unbind())  # (B*T, C H W)
    output = model(input_batch)
    assert output.shape == (t1 + t2, 64, h, w)


@pytest.mark.parametrize(
    "norm,k_dec,s_dec,p_dec",
    [("batch", 4, 2, 1), ("group", 2, 2, 0), ("none", 2, 2, 0)],
)
def test_lowres_unet(norm, k_dec, s_dec, p_dec):
    t1, t2 = 10, 13
    c, h, w = 10, 64, 64
    unet = LowResUnet(
        inplanes=c,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        return_maps=False,
        encoder_norm=norm,
        decoding_norm=norm,
        skip_conv_norm=norm,
        str_conv_k_dec=k_dec,
        str_conv_s_dec=s_dec,
        str_conv_p_dec=p_dec,
    )
    nt = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)]
    )
    input_batch = torch.cat(nt.unbind())  # (B*T, C H W)
    output = unet(input_batch)
    # Trigger gradient backprop
    output.sum().backward()

    # Test for gradient correclty propagated
    for params in unet.parameters():
        assert params.grad is not None

    assert output.shape == (t1 + t2, 64, h // 2, w // 2)
