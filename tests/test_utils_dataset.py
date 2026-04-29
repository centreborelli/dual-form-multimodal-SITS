import pytest
import torch

from cd_mm_sits.data.dataset.utils import (
    perform_center_crop,
    perform_random_crop,
    temporal_crop,
)


@pytest.mark.critical
def test_perform_random_crop():
    T, C, H, W = 11, 3, 64, 64
    min_h, min_w = 32, 32
    sits = torch.randn(T, C, H, W)
    label = torch.randn(T, H, W)
    crop_sits, crop_label = perform_random_crop(sits, label, output_size=(min_h, min_w))
    assert crop_sits.shape == (T, C, min_h, min_w)
    assert crop_label.shape == (T, min_h, min_w)


@pytest.mark.critical
def test_perform_center_crop():
    T, C, H, W = 11, 3, 64, 64
    min_h, min_w = 32, 32
    sits = torch.randn(T, C, H, W)
    label = torch.randn(T, H, W)
    crop_sits, crop_label = perform_center_crop(sits, label, output_size=(min_h, min_w))
    assert crop_sits.shape == (T, C, min_h, min_w)
    assert crop_label.shape == (T, min_h, min_w)


@pytest.mark.critical
def test_temporal_crop():
    T, C, H, W = 11, 3, 64, 64
    max_len = 6

    sits = torch.randn(T, C, H, W)
    label = torch.randn(T, H, W)
    time = torch.randn(T)
    output = temporal_crop(sits, time, label, max_len=max_len, seed=1)
    x, time, label = output.x, output.time, output.label
    assert x.shape[0] == max_len
    assert label.shape[0] == max_len
    assert time.shape[0] == max_len
