"""
File which contains relevant spectral spatial encoder.
These architectures process single images and do not
take into account any temporal dimension.
"""

from dataclasses import dataclass
from typing import Literal, no_type_check

import segmentation_models_pytorch as smp
import torch
from torch import Tensor
from torch import nn as nn
from torchvision.models.segmentation.fcn import (
    FCNHead,
    fcn_resnet50,
)


@dataclass
class UnetConfig:
    inplanes: int
    planes: int
    encoder_widths: list
    decoder_widths: list
    encoder_norm: Literal["group", "batch"] = "group"
    padding_mode: str = "reflect"
    decoding_norm: Literal["group", "batch"] = "group"
    return_maps: bool = False
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1
    skip_conv_norm: Literal["group", "batch"] = "group"
    momentum: float = 0.1


class Unet(nn.Module):
    """
    Inspired by
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        encoder_widths: list,
        decoder_widths: list,
        encoder_norm: Literal["group", "batch", "none"] = "group",
        padding_mode: str = "reflect",
        decoding_norm: Literal["group", "batch", "none"] = "group",
        return_maps: bool = False,
        str_conv_k_enc: int = 4,
        str_conv_s_enc: int = 2,
        str_conv_p_enc: int = 1,
        str_conv_k_dec: int = 4,
        str_conv_s_dec: int = 2,
        str_conv_p_dec: int = 1,
        skip_conv_norm: Literal["group", "batch", "none"] = "group",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
        gated_skip_conv: bool = False,
    ):
        """
        Parameters
        ----------
        inplanes : input number of channels
        planes :
        encoder_widths : list of the number of channels in the Unet encoder
        decoder_widths : list of the number of channels in the Unet decoder
        encoder_norm : normalisation type description
        padding_mode : padding type as defined in Conv2d
        decoding_norm : normalisation in the decoder
        return_maps : wether all intermetdiate eafture maps are returned
        str_conv_k : kernel size
        str_conv_s : stride size
        str_conv_p : padding size
        skip_conv_norm : normalisation in the skip connections"""
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation
        self.planes = planes
        self.gated_skip_conv = gated_skip_conv
        self.str_conv_k_enc = str_conv_k_enc
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.d_model = planes
        self.encoder_norm = encoder_norm
        self.decoder_norm = decoding_norm
        self.decoder_widths = decoder_widths
        self.in_conv = ConvBlock(
            nkernels=[inplanes] + [encoder_widths[0], encoder_widths[0]],
            norm=encoder_norm,
            padding_mode=padding_mode,
            momentum=momentum,
            activation=activation,
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], planes],
            norm=decoding_norm,
            padding_mode=padding_mode,
            momentum=momentum,
            activation=activation,
        )
        self.n_stages = len(encoder_widths)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k_enc,
                s=str_conv_s_enc,
                p=str_conv_p_enc,
                norm=encoder_norm,
                padding_mode=padding_mode,
                momentum=momentum,
                activation=activation,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k_dec,
                    s=str_conv_s_dec,
                    p=str_conv_p_dec,
                    norm=decoding_norm,
                    padding_mode=padding_mode,
                    skip_conv_norm=skip_conv_norm,
                    momentum=momentum,
                    activation=activation,
                    gated_skip_conv=gated_skip_conv,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )

    def _load_hp(self) -> dict:
        suffix = "low_res_unet/"
        return {
            suffix + "planes": self.planes,
            suffix + "activation": type(self.activation).__name__,
            suffix + "gated_skip_conv": self.gated_skip_conv,
            suffix + "encoder_norm": self.encoder_norm,
            suffix + "decoder_norm": self.decoder_norm,
            suffix + "str_conv_k_enc": self.str_conv_k_enc,
        }

    @no_type_check
    def forward(self, input: Tensor):
        """

        Parameters
        ----------
        input : Tensor (B,C,H,W)
        Returns
        -------
        either a Tensor of size (B,C,H,W)  or if returns_map is s
        et to True a Tensor of size
        B,C,H,W and well as a list of all intermediate feature maps
        """
        dtype = input.dtype
        out = self.in_conv(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
            feature_maps.append(out)
            # print(out.shape)
        if self.return_maps:
            maps = [out]
        # print([out.shape for out in feature_maps])
        for i in range(self.n_stages - 1):
            skip = feature_maps[-(i + 2)]
            #  print(skip.shape, out.shape)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)
            #            out = rearrange(out, "b c h w -> b h w c")
        out = self.out_conv(out)
        if self.return_maps:
            return out.to(dtype), maps.to(dtype)

        return out.to(dtype)


class UpConvBlock(nn.Module):
    @no_type_check
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        final_out=None,
        norm: Literal["group", "batch", "none"] = "batch",
        d_skip=None,
        padding_mode="reflect",
        skip_conv_norm: Literal["group", "batch", "none"] = "batch",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
        gated_skip_conv: bool = False,
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        if activation is None:
            activation = nn.ReLU()
        if skip_conv_norm == "batch":
            skip_norm_begin = nn.BatchNorm2d(d, momentum=momentum)
            skip_norm_end = nn.BatchNorm2d(d_out, momentum=momentum)
        elif skip_conv_norm == "group":
            skip_norm_begin = nn.GroupNorm(num_groups=4, num_channels=d)
            skip_norm_end = nn.GroupNorm(num_groups=4, num_channels=d_out)
        else:
            skip_norm_begin = nn.Sequential()
            skip_norm_end = nn.Sequential()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            skip_norm_begin,
            activation,
        )
        if gated_skip_conv:
            self.gate = nn.Identity()  # OBSOLETE
        else:
            self.gate = nn.Identity()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d_in, d_out, kernel_size=3, padding=1, padding_mode=padding_mode),
            skip_norm_end,
            activation,
        )
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=d_in,
        #         out_channels=d_out,
        #         kernel_size=k,
        #         stride=s,
        #         padding=p,
        #     ),
        #     skip_norm_end,
        #     nn.ReLU(),
        # )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out],
            norm=norm,
            padding_mode=padding_mode,
            momentum=momentum,
        )
        if final_out is None:
            final_out = d_out

        self.conv2 = ConvLayer(
            nkernels=[d_out, final_out],
            norm=norm,
            padding_mode=padding_mode,
            momentum=momentum,
        )

    @no_type_check
    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat(
            [out, self.gate.to(device=input.device)(self.skip_conv(skip))], dim=1
        )
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        norm="batch",
        padding_mode="reflect",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
            activation=activation,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
            activation=activation,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class ConvLayer(nn.Module):
    @no_type_check
    def __init__(
        self,
        nkernels,
        norm: Literal["group", "batch", "none", "instance"] = "batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        layers = []
        if norm == "batch":
            nl = lambda num_feats: nn.BatchNorm2d(num_feats, momentum=momentum)  # noqa: E731
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":

            def group_norm(num_feats):
                return nn.GroupNorm(
                    num_channels=num_feats,
                    num_groups=n_groups,
                )

            nl = group_norm
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if i < len(nkernels) - 1:
                layers.append(activation)
        self.conv = nn.Sequential(*layers)

    @no_type_check
    def forward(self, input):
        return self.conv(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        nkernels,
        norm: Literal["group", "batch", "none"] = "batch",
        last_relu=True,
        padding_mode="reflect",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
            momentum=momentum,
            activation=activation,
        )

    def forward(self, input):
        return self.conv(input)


class FCNResNet50(nn.Module):
    def __init__(self, inplanes: int, planes: int, pretrained: bool = False):
        super().__init__()
        self.d_model = planes
        model = fcn_resnet50(pretrained=pretrained)
        model.classifier = FCNHead(2048, planes)
        model.backbone.conv1 = torch.nn.Conv2d(
            inplanes, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = model

    def forward(self, input: Tensor):
        """

        Parameters
        ----------
        input : Tensor (B,C,H,W)
        Returns
        -------
        a Tensor of size (B,C,H,W)
        """
        return self.backbone(input)["out"]


class SMPEncoder(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        encoder_name: str = "efficientnet-b0",
        encoder_depth: int = 4,
        encoder_weights: str | None = None,
    ):
        super().__init__()
        self.d_model = planes
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=inplanes,
            classes=planes,
            encoder_depth=encoder_depth,
            decoder_channels=tuple(256 // (2**i) for i in range(encoder_depth)),
        )

    def forward(self, input: Tensor):
        return self.backbone(input)


class SMPViTEncoder(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        encoder_name: str = "efficientnet-b0",
        encoder_depth: int = 4,
        encoder_weights: str | None = None,
    ):
        super().__init__()
        self.d_model = planes
        self.backbone = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=inplanes,
            classes=planes,
            encoder_depth=encoder_depth,
            decoder_channels=tuple(256 // (2**i) for i in range(encoder_depth)),
        )

    def forward(self, input: Tensor):
        return self.backbone(input)


class LowResUnet(nn.Module):
    """
    This Unet does not output a feature map at the same resolution
    as the input but H/2,W/2. It does not perform the last upsampling step

    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        encoder_widths: list,
        decoder_widths: list,
        encoder_norm: Literal["group", "batch", "none"] = "group",
        padding_mode: str = "reflect",
        decoding_norm: Literal["group", "batch", "none"] = "group",
        return_maps: bool = False,
        str_conv_k_enc: int = 4,
        str_conv_s_enc: int = 2,
        str_conv_p_enc: int = 1,
        str_conv_k_dec: int = 4,
        str_conv_s_dec: int = 2,
        str_conv_p_dec: int = 1,
        skip_conv_norm: Literal["group", "batch", "none"] = "group",
        momentum: float = 0.1,
        activation: nn.Module | None = None,
        gated_skip_conv: bool = False,
    ):
        """
        Parameters
        ----------
        inplanes : input number of channels
        planes :
        encoder_widths : list of the number of channels in the Unet encoder
        decoder_widths : list of the number of channels in the Unet decoder
        encoder_norm : normalisation type description
        padding_mode : padding type as defined in Conv2d
        decoding_norm : normalisation in the decoder
        return_maps : wether all intermetdiate eafture maps are returned
        str_conv_k : kernel size
        str_conv_s : stride size
        str_conv_p : padding size
        skip_conv_norm : normalisation in the skip connections"""
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.gated_skip_conv = gated_skip_conv
        self.activation = activation
        self.planes = planes
        self.str_conv_k_enc = str_conv_k_enc
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.d_model = planes
        self.decoder_widths = decoder_widths
        self.encoder_norm = encoder_norm
        self.decoder_norm = decoding_norm
        self.str_conv_k_enc = str_conv_k_enc
        self.in_conv = ConvBlock(
            nkernels=[inplanes] + [encoder_widths[0], encoder_widths[0]],
            norm=encoder_norm,
            padding_mode=padding_mode,
            momentum=momentum,
            activation=activation,
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], planes],
            norm=decoding_norm,
            padding_mode=padding_mode,
            momentum=momentum,
            activation=activation,
        )
        self.n_stages = len(encoder_widths)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k_enc,
                s=str_conv_s_enc,
                p=str_conv_p_enc,
                norm=encoder_norm,
                padding_mode=padding_mode,
                momentum=momentum,
                activation=activation,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k_dec,
                    s=str_conv_s_dec,
                    p=str_conv_p_dec,
                    norm=decoding_norm,
                    padding_mode=padding_mode,
                    skip_conv_norm=skip_conv_norm,
                    momentum=momentum,
                    activation=activation,
                    gated_skip_conv=gated_skip_conv,
                )
                for i in range(self.n_stages - 1, 1, -1)
            ]
        )

    def _load_hp(self) -> dict:
        suffix = "low_res_unet/"
        return {
            suffix + "planes": self.planes,
            suffix + "activation": type(self.activation).__name__,
            suffix + "gated_skip_conv": self.gated_skip_conv,
            suffix + "encoder_norm": self.encoder_norm,
            suffix + "decoder_norm": self.decoder_norm,
            suffix + "str_conv_k_enc": self.str_conv_k_enc,
        }

    @no_type_check
    def forward(self, input: Tensor):
        """

        Parameters
        ----------
        input : Tensor (B,C,H,W)
        Returns
        -------
        either a Tensor of size (B,C,H,W)  or if returns_map is s
        et to True a Tensor of size
        B,C,H,W and well as a list of all intermediate feature maps
        """
        dtype = input.dtype
        out = self.in_conv(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
            feature_maps.append(out)
            # print(out.shape)
        if self.return_maps:
            maps = [out]
        # print([out.shape for out in feature_maps])
        for i in range(self.n_stages - 2):
            skip = feature_maps[-(i + 2)]
            #  print(skip.shape, out.shape)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)
            #            out = rearrange(out, "b c h w -> b h w c")
        out = self.out_conv(out)
        if self.return_maps:
            return out.to(dtype), maps.to(dtype)

        return out.to(dtype)
