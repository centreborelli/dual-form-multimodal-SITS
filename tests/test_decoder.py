import torch

from cd_mm_sits.model.decoder import PixelShuffleDec


def test_pixel_shuffle_dec():
    C, D = 32, 16
    B, T, H, W = 2, 11, 64, 64
    dec = PixelShuffleDec(C, D, 2)
    out = dec(torch.randn(B, T, H, W, C))
    assert out.shape == (B, T, H * 2, W * 2, D)
