from functools import lru_cache
from typing import override

import einops
import torch
import torch.nn as nn

# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


def fixed_pos_embedding(x: torch.Tensor, T_base):
    seq_len, dim = x.shape  # T,C
    inv_freq = 1.0 / (T_base ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd gdimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    """Implementation
    from https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, head_dim, scale_base=512, T_base: float = 1e4):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
        self.T_base = T_base

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos

        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale, self.T_base)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


@staticmethod
@lru_cache(maxsize=128)
def _get_xi_xpos(
    scale: torch.Tensor,
    max_len: int,
    scale_base: int,
    device: torch.device,
    dtype: torch.dtype,
):
    return (scale ** torch.arange(0, max_len, 1).to(scale).div(scale_base)[:, None])[
        None, ...
    ]  # 1,T,D


def _get_xi_xpostime(
    scale: torch.Tensor,
    time: torch.Tensor,
    scale_base: int,
):
    return scale[None, ...] ** time.div(scale_base)[..., None]  # B,T,D


@staticmethod
@lru_cache(maxsize=128)
def _get_theta_xpos(d: int, base: int, device: torch.device, dtype: torch.dtype):
    return 1 / (base ** (torch.arange(0, d, 2, device=device, dtype=dtype) / d))


def fixed_pos_embedding_3d(theta: torch.Tensor, x_shape) -> torch.Tensor:
    """
    theta a tensor of size D/2
    x tuple containing (B,T)
    Returns theta_i T"""

    index = (
        torch.arange(0, x_shape[1], 1, device=theta.device)
        .unsqueeze(0)
        .expand(x_shape[0], -1)
    )
    return einops.einsum(index, theta, "b t, d -> b t d")


def fixed_time_pos_embedding_3d(
    theta: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """
    theta a tensor of size D/2
    x tuple containing (B,T)
    Returns theta B, T, D/2"""
    return einops.einsum(time, theta, "b t, d -> b t d")


class MyXpos(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        gamma: float = 0.4,
        T_base: float = 1e4,
        scale_base: int = 512,
    ) -> None:
        super().__init__()
        self.scale_base = scale_base  # unsure it is useful
        self.gamma = gamma
        self.T_base = T_base
        head_dim = embed_dim
        self.head_dim = head_dim
        self.register_buffer(
            "scale",
            (torch.arange(0, head_dim, 2) + self.gamma * head_dim)
            / ((1 + self.gamma) * head_dim),
        )

    def _load_hp(self) -> dict:
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "base": self.T_base,
            suffix + "d": self.head_dim,
            suffix + "gamma": self.gamma,
        }

    def forward(self, x: torch.Tensor, downscale=False):
        length = x.shape[1]
        decay = _get_xi_xpos(
            self.scale,
            max_len=length,
            scale_base=self.scale_base,
            dtype=x.dtype,
            device=x.device,
        )  # B,T,D/2
        x_sin = torch.empty(x.shape, device=x.device)
        if downscale:
            decay = 1 / decay
        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D
        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta_xpos(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2
        theta_t = fixed_pos_embedding_3d(theta=theta, x_shape=x.shape)  # B,T,D/2
        decay_full = torch.empty(1, x.shape[1], x.shape[2], device=x.device)  # 1,T,D
        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t
        decay_full[..., 0::2] = decay
        decay_full[..., 1::2] = decay

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D
        return torch.mul(rope, decay_full)


class TimeXPOS(MyXpos):
    """A modified version of XPOS that changes according to acquisition time"""

    @override
    def forward(self, x, time, downscale=False):
        decay = _get_xi_xpostime(
            self.scale, time=time, scale_base=self.scale_base
        )  # 1,T,D/2
        x_sin = torch.empty(x.shape, device=x.device)
        if downscale:
            decay = 1 / decay
            # depends on decay maps ect ..
        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D

        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta_xpos(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2

        theta_t = fixed_time_pos_embedding_3d(theta=theta, time=time)

        decay_full = torch.empty(
            x.shape[0], x.shape[1], x.shape[2], device=x.device
        )  # 1,T,D

        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t
        decay_full[..., 0::2] = decay
        decay_full[..., 1::2] = decay

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D

        return torch.mul(rope, decay_full)


class RetNetTimeXpos(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        h_index,
        T_base: float = 1e4,
        scale_base: int = 512,
    ) -> None:
        super().__init__()
        self.scale_base = scale_base  # unsure it is useful

        self.T_base = T_base
        head_dim = embed_dim
        self.head_dim = head_dim
        self.register_buffer("scale", torch.Tensor([1 - 2 ** (-5 - h_index)]))

    def _load_hp(self) -> dict:
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "base": self.T_base,
            suffix + "d": self.head_dim,
            suffix + "scale": self.scale,
        }

    def forward(self, x, time, downscale=False):
        decay = _get_xi_xpostime(
            self.scale, time=time, scale_base=self.scale_base
        )  # 1,T,D
        decay.expand(-1, -1, self.head_dim)
        x_sin = torch.empty(x.shape, device=x.device)
        if downscale:
            decay = 1 / decay
        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D

        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta_xpos(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2

        theta_t = fixed_time_pos_embedding_3d(theta=theta, time=time)

        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D

        return torch.mul(rope, decay)


class RetNetXpos(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        h_index,
        T_base: float = 1e4,
        scale_base: int = 512,
    ) -> None:
        super().__init__()
        self.scale_base = scale_base  # unsure it is useful

        self.T_base = T_base
        head_dim = embed_dim
        self.head_dim = head_dim
        self.register_buffer("scale", torch.Tensor([1 - 2 ** (-5 - h_index)]))

    def _load_hp(self) -> dict:
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "base": self.T_base,
            suffix + "d": self.head_dim,
            suffix + "scale": self.scale,
        }

    def forward(self, x, downscale=False):
        length = x.shape[1]
        decay = _get_xi_xpos(
            self.scale,
            scale_base=self.scale_base,
            max_len=length,
            dtype=x.dtype,
            device=x.device,
        )  # 1,T,D
        decay.expand(-1, -1, self.head_dim)
        x_sin = torch.empty(x.shape, device=x.device)
        if downscale:
            decay = 1 / decay
        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D

        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta_xpos(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2
        theta_t = fixed_pos_embedding_3d(theta=theta, x_shape=x.shape)  # B,T,D/2

        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D

        return torch.mul(rope, decay)
