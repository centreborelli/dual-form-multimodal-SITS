import pytest
import torch
from einops import repeat

from cd_mm_sits.model.utils import cat_nested_along_time, sort_mm_sits


@pytest.mark.critical
def test_cat_nested_along_time():
    t1, t2, c, h, w = 10, 11, 3, 64, 64
    batch_1 = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)], layout=torch.jagged
    )
    batch_2 = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)], layout=torch.jagged
    )
    out = cat_nested_along_time(batch_1, batch_2)
    assert out.shape[0] == 2
    assert out[0, :, 0, 0, 0].shape == torch.Size([2 * t1])


def is_sorted_fast(t):
    return torch.all(t[:-1] <= t[1:])


@pytest.mark.critical
def test_sort_mm_sits():
    B, T, C = 16, 20, 32
    base = torch.arange(T)
    twod_in = repeat(base, "T -> B T ", B=B)
    threed_in = repeat(base, "T -> B T C ", B=B, C=C)
    out = sort_mm_sits(
        padded_sits=threed_in,
        key_padding_mask=twod_in,
        padded_times=twod_in,
        padded_tpe=threed_in,
        mod=twod_in,
    )
    assert out.mm_sits.shape == out.sorted_tpe.shape
    assert is_sorted_fast(out.mm_sits[0, :, 0])
    assert is_sorted_fast(out.sorted_tpe[0, :, 0])

    assert is_sorted_fast(out.sorted_key_padding_mask[0, :])
    assert is_sorted_fast(out.sorted_times[0, :])


@pytest.mark.critical
def test_compare_bmm_witheinsum():
    k_ = torch.randn(8, 11, 6)
    v = torch.randn(8, 11, 3)
    kv_1 = torch.einsum("nld,nlm->nldm", k_, v)  #
    kv_2 = torch.matmul(k_.unsqueeze(-1), v.unsqueeze(-2))
    assert torch.allclose(kv_1, kv_2, atol=1e-5, rtol=1e-4)
