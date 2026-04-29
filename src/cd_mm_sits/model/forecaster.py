from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from cd_mm_sits.layers.cross_attention import ConfigLQMHA, LearnedQMultiHeadAttention
from cd_mm_sits.model.decoder import conv3x3
from cd_mm_sits.model.temporal_positional_encoder import (
    PositionalEncoder,
    TimeDeltaRotaryPE,
)


@dataclass
class InputForecForward:
    lat_repr: torch.Tensor
    target_time: torch.Tensor
    mod: Literal["s2", "s1"]
    input_time: torch.Tensor | None
    weather: torch.Tensor | None = None
    angles: torch.Tensor | None = None


class WeatherLQEncoder(nn.Module):
    """Using a mechanism inspired by learnable
    queries given a sequence of weather var, it outputs
    a fixed size weather token"""

    def __init__(
        self,
        n_head: int,
        d_k: int,
        d_in: int,
        n_q: int,
        d_v: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.config_lq_mha = ConfigLQMHA(
            n_head=n_head, d_k=d_k, d_in=d_in, n_q=n_q, d_v=d_v
        )
        self.lq_mha = LearnedQMultiHeadAttention(self.config_lq_mha)
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, d_in)
        self.d_w = n_q * d_v

    def _load_hp(self) -> dict:
        suffix = "weather_encoder/"
        return {
            suffix + "n_q": self.config_lq_mha.n_q,
            suffix + "n_head": self.config_lq_mha.n_head,
            suffix + "d_v": self.config_lq_mha.d_v,
            suffix + "d_k": self.config_lq_mha.d_k,
            suffix + "max_len": self.max_len,
        }

    def forward(self, X):
        """X: a tensor (B*T,Tbef,C) composed for each timestamp T: of the C var value for the previous Tbef timesteps"""
        BT, Tbef, C = X.shape
        tpe = self.pe(torch.arange(Tbef).to(device=X.device))  # Tbef, pe_dim
        tpe = tpe.unsqueeze(0).expand(BT, -1, -1)
        X = X + tpe
        X = self.lq_mha(X, None)
        return rearrange(X, "BT nq C -> BT (nq C)")


class MLPWeatherEncoder(nn.Module):
    """A simple MLP to compute the weather feature"""

    def __init__(self, d_in: int, d_out: int, max_len: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_w = d_out
        self.max_len = max_len
        self.linear = nn.Linear(d_in * max_len, d_out)

    def _load_hp(self) -> dict:
        suffix = "weather_encoder/"
        return {
            suffix + "d_out_w": self.d_w,
            suffix + "max_len": self.max_len,
        }

    def forward(self, X):
        """X: a tensor (B*T,Tbef,C) composed for each timestamp T: of the C var value for the previous Tbef timesteps"""
        assert X.shape[1] == self.max_len, (
            f"Incorrect MLPWeatherEncoder configuration past_len should be {X.shape[1]}"
        )
        X = rearrange(X, "BT Tbef C -> BT (Tbef C)")
        print(X.shape)
        return self.linear(X)


class AngleEncoder(nn.Module):
    """Encoder for S1 angles"""

    def __init__(self, n_angles: int, d_: int) -> None:
        super().__init__()
        self.d_ = d_
        self.layer = nn.Linear(2 * n_angles, d_)

    def forward(self, angles: torch.Tensor):
        """angles: (B,T,C), angles should be in radian !"""

        return self.layer(torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1))


class ForeCMLP(nn.Module):
    def __init__(self, inplanes: int, planes: int, n_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers += [nn.Linear(inplanes, planes)]
            else:
                layers += [nn.Linear(planes, planes)]
            layers += [
                nn.ReLU(),
                nn.Linear(planes, planes),
                nn.ReLU(),
                nn.LayerNorm(planes),
            ]
        if n_layers > 1:
            layers += []
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return rearrange(self.layers(x), "BT H W C -> BT C H W")


class TemplateForecaster(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        d_pe: int,
        bands_s2: int,
        bands_s1: int,
        n_layers: int,
        n_mod_feat: int = 16,
        weather_encoder: None | WeatherLQEncoder | MLPWeatherEncoder = None,
        use_norm: bool = True,
        angle_encoder: AngleEncoder | None = None,
    ) -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoder(d=d_pe)
        self.d_pe = d_pe
        self.n_mod_feat = n_mod_feat
        self.n_layers = n_layers
        if weather_encoder is not None:
            assert isinstance(weather_encoder, (WeatherLQEncoder, MLPWeatherEncoder))
            d_w_var = weather_encoder.d_w
        else:
            d_w_var = 0
        if angle_encoder is not None:
            d_angle = angle_encoder.d_
        else:
            d_angle = 0
        self.d_angle = d_angle
        if use_norm:
            self.mlp = ForeCMLP(
                inplanes + d_pe + n_mod_feat + d_w_var + d_angle,
                planes,
                n_layers=n_layers,
            )
        else:
            self.mlp = ForeCMLPNoNorm(
                inplanes + d_pe + n_mod_feat + d_w_var + d_angle,
                planes,
                n_layers=n_layers,
            )
        self.out_s2 = conv3x3(planes, bands_s2)
        self.out_s1 = conv3x3(planes, bands_s1)
        self.mod_embedding = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=n_mod_feat
        )
        self.weather_encoder = weather_encoder
        self.angle_encoder = angle_encoder
        # a bit convoluted, but necessary to have the tensor on the right device
        self.d_mod_index = {"s1": 0, "s2": 1}
        d_mods = torch.tensor([0, 1], dtype=torch.int64)
        # for some reasons, `self.d_mods = d_mods` is not registering the buffer correctly
        self.register_buffer("d_mods", d_mods)
        if self.angle_encoder is not None:
            self.angle_empty_feat = nn.Parameter(torch.zeros(d_angle)).requires_grad_(
                True
            )
            nn.init.normal_(
                self.angle_empty_feat,
                mean=0,
                std=np.sqrt(2.0 / (self.angle_encoder.d_)),
            )
        # useful as a placeholder for S2 data in the forecaster (not using angle yet)

    def _load_hp(self) -> dict:
        suffix = "forecaster/"
        if self.weather_encoder is not None:
            w_enc_d = {
                f"{suffix}{key}": value
                for key, value in self.weather_encoder._load_hp().items()
            }
        else:
            w_enc_d = {}
        w_enc_d.update(
            {
                f"{suffix}{key}": value
                for key, value in self.positional_encoder._load_hp().items()
            }
        )
        w_enc_d.update(
            {
                suffix + "n_mod_feat": self.n_mod_feat,
                suffix + "weather_encoder": type(self.weather_encoder).__name__,
                suffix + "angle_encoder": type(self.angle_encoder).__name__,
                suffix + "n_layers": self.n_layers,
                suffix + "d_angle": self.d_angle,
            }
        )
        return w_enc_d

    def mm_embedding(self, mod) -> torch.Tensor:
        m = self.d_mods[self.d_mod_index[mod]]
        return self.mod_embedding(m)

    def _encode_weather(
        self, aux_var: torch.Tensor | None, h, w
    ) -> torch.Tensor | None:
        if self.weather_encoder is not None:
            assert aux_var is not None
            aux_var = self.weather_encoder(aux_var)
            return aux_var.unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1)
        else:
            return None

    def _encode_angle(
        self, angle_tensor: torch.Tensor | None, h: int, w: int, bt: int
    ) -> torch.Tensor | None:
        if (self.angle_encoder is not None) and angle_tensor is not None:
            angle_var = self.angle_encoder(angle_tensor).unsqueeze(-1).unsqueeze(-1)

            return rearrange(angle_var.expand(-1, -1, h, w), "bt c h w -> bt h w c")
        elif self.angle_encoder is not None:
            return repeat(self.angle_empty_feat, "d -> bt h w d", h=h, w=w, bt=bt)
        else:
            return None


class ForeCMLPNoNorm(nn.Module):
    def __init__(self, inplanes: int, planes: int, n_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers += [nn.Linear(inplanes, planes)]
            else:
                layers += [nn.Linear(planes, planes)]
            layers += [
                nn.ReLU(),
                nn.Linear(planes, planes),
                nn.ReLU(),
            ]
        if n_layers > 1:
            layers += []
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return rearrange(self.layers(x), "BT H W C -> BT C H W")


class MMShallowForecast(TemplateForecaster):
    """A MLP which performs a Forecast from a latent representation"""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        d_pe: int,
        bands_s2: int,
        bands_s1: int,
        n_layers: int,
        n_mod_feat: int = 16,
        weather_encoder: None | WeatherLQEncoder = None,
        use_norm: bool = True,
        angle_encoder: AngleEncoder | None = None,
    ) -> None:
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            d_pe=d_pe,
            bands_s2=bands_s2,
            bands_s1=bands_s1,
            n_layers=n_layers,
            n_mod_feat=n_mod_feat,
            weather_encoder=weather_encoder,
            use_norm=use_norm,
            angle_encoder=angle_encoder,
        )
        self.positional_encoder = PositionalEncoder(d=self.d_pe)

    def forward(self, inputs: InputForecForward) -> torch.Tensor:
        """
        It performs a forward of a SITS per modality !
        (but the lat_repr can come from other modality)
        time: Tensor B,T, time of the target
        mod: str, modality of the target
        lat_repr: Tensor BT,C,H,W
        """
        BT, H, W, C = inputs.lat_repr.shape
        tpe = self.positional_encoder.forward_1d(inputs.target_time)  # B*T,d
        lat_repr = inputs.lat_repr
        tpe = (
            tpe.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, lat_repr.shape[-3], lat_repr.shape[-2])
        )  # B,T,d,H,W
        tpe = rearrange(tpe, "BT d H W -> BT H W d")
        aux_var = self._encode_weather(inputs.weather, h=H, w=W)
        angle_var = self._encode_angle(inputs.angles, h=H, w=W, bt=BT)

        mod_emb = (
            self.mm_embedding(inputs.mod).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # 1,1,1,1,d
        mod_emb = mod_emb.expand(tpe.shape[0], tpe.shape[1], tpe.shape[2], -1)
        list_x = [lat_repr, tpe, mod_emb]
        if aux_var is not None:
            list_x += [aux_var]
        if angle_var is not None:
            list_x += [angle_var.to(device=lat_repr.device)]
        x = torch.cat(list_x, dim=-1)  # BT,H,W,C+d_mod
        x = self.mlp(x)
        if inputs.mod == "s2":
            return self.out_s2(x)

        elif inputs.mod == "s1":
            return self.out_s1(x)
        else:
            raise NotImplementedError


class RopeMMShallowForecaster(TemplateForecaster):
    """MM Forecast which encodes temporal information using Rotary PE"""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        d_pe: int,
        bands_s2: int,
        bands_s1: int,
        n_layers: int,
        n_mod_feat: int = 16,
        base: int = 1000,
        use_norm: bool = True,
        weather_encoder: None | WeatherLQEncoder = None,
        angle_encoder: AngleEncoder | None = None,
    ) -> None:
        d_pe = 0  # so lin1 is correctly initialized
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            d_pe=d_pe,
            bands_s2=bands_s2,
            bands_s1=bands_s1,
            n_mod_feat=n_mod_feat,
            n_layers=n_layers,
            weather_encoder=weather_encoder,
            use_norm=use_norm,
            angle_encoder=angle_encoder,
        )
        self.rope = TimeDeltaRotaryPE(base=base, d=inplanes)

    def forward(self, inputs: InputForecForward):
        lat_repr = inputs.lat_repr
        BT, H, W, C = lat_repr.shape

        delta_time = inputs.target_time

        delta_time = (
            delta_time.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, lat_repr.shape[-3], lat_repr.shape[-2])
        )  # B,T,d,H,W
        aux_var = self._encode_weather(inputs.weather, h=H, w=W)

        angle_var = self._encode_angle(inputs.angles, h=H, w=W, bt=BT)

        delta_time = rearrange(delta_time, "BT H W -> (BT H W)")
        lat_repr = rearrange(lat_repr, "BT H W C -> (BT H W ) C ")
        rotated_x = self.rope(x=lat_repr, delta=delta_time)
        rotated_x = rearrange(rotated_x, "(BT H W) C-> BT H W C", BT=BT, H=H, W=W)

        mod_emb = (
            self.mm_embedding(inputs.mod).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # 1,1,1,1,d
        mod_emb = mod_emb.expand(BT, H, W, -1)
        list_x = [rotated_x, mod_emb]

        if aux_var is not None:
            list_x += [aux_var]
        if angle_var is not None:
            list_x += [angle_var.to(device=lat_repr.device)]
        x = torch.cat(list_x, dim=-1)  # BT,H,W,C+d_mod
        x = self.mlp(x)

        if inputs.mod == "s2":
            return self.out_s2(x)

        elif inputs.mod == "s1":
            return self.out_s1(x)
        else:
            raise NotImplementedError


class MMShallow2PeForecast(nn.Module):
    """TO FIX !!! (DOES NOT WORK)
    A MLP which performs a Forecast from a latent representation"""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        d_pe: int,
        bands_s2: int,
        bands_s1: int,
        n_layers: int,
        n_mod_feat: int = 16,
        weather_encoder: None | WeatherLQEncoder = None,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoder(d=d_pe)
        if weather_encoder is not None:
            d_w_var = weather_encoder.lq_mha.d_v * weather_encoder.lq_mha.n_q

        else:
            d_w_var = 0
        if use_norm:
            self.mlp = ForeCMLP(
                inplanes + d_pe + n_mod_feat + d_w_var, planes, n_layers=n_layers
            )
        else:
            self.mlp = ForeCMLPNoNorm(
                inplanes + d_pe + n_mod_feat + d_w_var, planes, n_layers=n_layers
            )
        self.out_s2 = conv3x3(planes, bands_s2)
        self.out_s1 = conv3x3(planes, bands_s1)
        self.mod_embedding = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=n_mod_feat
        )
        # a bit convoluted, but necessary to have the tensor on the right device
        self.d_mod_index = {"s1": 0, "s2": 1}
        d_mods = torch.tensor([0, 1], dtype=torch.int64)
        self.weather_encoder = weather_encoder
        # for some reasons, `self.d_mods = d_mods` is not registering the buffer correctly
        self.register_buffer("d_mods", d_mods)

    def forward(self, inputs: InputForecForward) -> torch.Tensor:
        """
        It performs a forward of a SITS per modality !
        (but the lat_repr can come from other modality)
        time: Tensor B,T, time of the target
        mod: str, modality of the target
        lat_repr: Tensor BT,C,H,W
        """
        h, w = inputs.lat_repr.shape[-3], inputs.lat_repr.shape[-2]
        tpe_input = self._compute_tpe(time=inputs.input_time, h=h, w=w)
        tpe_target = self._compute_tpe(time=inputs.target_time, h=h, w=w)
        aux_var = self._encode_weather(inputs.weather, h=h, w=w)
        mod_emb = (
            self.mm_embedding(inputs.mod).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # 1,1,1,1,d

        mod_emb = mod_emb.expand(
            tpe_input.shape[0], tpe_input.shape[1], tpe_input.shape[2], -1
        )
        lat_repr = inputs.lat_repr
        # print(lat_repr.shape, tpe_input.shape, tpe_target.shape, mod_emb.shape)
        x = torch.cat(
            [lat_repr, tpe_input, tpe_target, mod_emb], dim=-1
        )  # B,T,d_pe+C+d_mod
        if aux_var is not None:
            x = torch.cat([x, aux_var], dim=-1)
        x = self.mlp(x)
        if inputs.mod == "s2":
            return self.out_s2(x)

        elif inputs.mod == "s1":
            return self.out_s1(x)
        else:
            raise NotImplementedError

    def _compute_tpe(self, time: torch.Tensor, h, w):
        tpe = self.positional_encoder.forward_1d(time)  # B*T,d
        tpe = tpe.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, h, w)  # B,T,d,H,W
        return rearrange(tpe, "BT d H W -> BT H W d")

    def mm_embedding(self, mod) -> torch.Tensor:
        m = self.d_mods[self.d_mod_index[mod]]
        return self.mod_embedding(m)

    def _encode_weather(self, aux_var: torch.Tensor | None, h, w):
        if self.weather_encoder is not None:
            assert aux_var is not None
            aux_var = self.weather_encoder(aux_var)
            return aux_var.unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1)
        else:
            return None
