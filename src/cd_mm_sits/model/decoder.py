import torch
import torch.nn as nn
from einops import rearrange


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
        padding_mode="reflect",
    )


class ConvBasicDecoder(nn.Module):
    def __init__(self, inplanes: int, planes: int, dropout: float = 0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.conv2 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """x: Tensor (B T H W C)"""
        nested_ = x.is_nested
        b, *_ = x.shape
        if nested_:
            offsets = x.offsets()
            x = rearrange(x.values(), "BT H W C -> BT C H W")
            x = self.conv1(x)
            x = self.dropout(self.relu(x))
            x = self.dropout(self.conv2(x))
            x = torch.nested.nested_tensor_from_jagged(values=x, offsets=offsets)
            x = rearrange(x, "B T C H W -> B T H W C", B=b)
            return x

        else:
            x = rearrange(x, "B T H W C -> (B T) C H W")
            x = self.conv1(x)
            x = self.dropout(self.relu(x))
            x = self.dropout(self.conv2(x))
            return rearrange(x, "(B T) C H W -> B T H W C", B=b)


class ConvUpSample(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        dropout: float = 0,
        k: int = 4,
        s: int = 2,
        p: int = 1,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(inplanes)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=inplanes,
                kernel_size=k,
                stride=s,
                padding=p,
            ),
            self.norm,
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        nested_ = x.is_nested
        b, *_ = x.shape
        if nested_:  # obsolete
            offsets = x.offsets()
            x = rearrange(x.values(), "BT H W C -> BT C H W")
            x = self.up(x)
            x = self.conv1(x)

            x = torch.nested.nested_tensor_from_jagged(values=x, offsets=offsets)

            x = rearrange(x, "B T C H W -> B T H W C", B=b)
            return x
        else:
            x = rearrange(x, "B T  H W C  -> (B T) C H W")
            x = self.up(x)
            x = self.conv1(x)
            return rearrange(x, "(B T) C H W -> B T H W C", B=b)


class PixelShuffleDec(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        upscale_factor: int = 2,
        dropout: float = 0,
    ):
        super().__init__()
        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv1 = conv3x3(inplanes // 4, planes)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(inplanes)

    def forward(self, x: torch.Tensor):
        nested_ = x.is_nested
        b, *_ = x.shape
        if nested_:  # obsolete
            offsets = x.offsets()
            x = rearrange(x.values(), "BT H W C -> BT C H W")
            x = self.up(x)
            x = self.conv1(x)

            x = torch.nested.nested_tensor_from_jagged(values=x, offsets=offsets)

            x = rearrange(x, "B T C H W -> B T H W C", B=b)
            return x
        else:
            x = rearrange(x, "B T  H W C  -> (B T) C H W")
            x = self.up(x)
            x = self.conv1(x)
            return rearrange(x, "(B T) C H W -> B T H W C", B=b)


class SimplePixelShuffleDec(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        upscale_factor: int = 2,
        dropout: float = 0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(inplanes, inplanes * (upscale_factor**2))
        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.relu = nn.GELU()
        self.out = nn.Linear(inplanes, planes)

    def forward(self, x: torch.Tensor):
        nested_ = x.is_nested
        b, *_ = x.shape
        if nested_:  # obsolete
            x = self.in_layer(x)
            offsets = x.offsets()
            x = rearrange(x.values(), "BT H W C -> BT C H W")
            x = self.up(x)
            x = torch.nested.nested_tensor_from_jagged(values=x, offsets=offsets)
            x = rearrange(x, "B T C H W -> B T H W C", B=b)
            return self.out(x)
        else:
            x = self.in_layer(x)
            x = rearrange(x, "B T  H W C  -> (B T) C H W")
            x = self.up(x)
            return self.out(rearrange(x, "(B T) C H W -> B T H W C", B=b))


class BilinearUpsample(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        upscale_factor: int = 2,
        dropout: float = 0,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=upscale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(
                inplanes, planes, kernel_size=3, padding=1, padding_mode=padding_mode
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        nested_ = x.is_nested
        b, *_ = x.shape
        if nested_:  # obsolete
            offsets = x.offsets()
            x = rearrange(x.values(), "BT H W C -> BT C H W")
            x = self.up(x)
            x = torch.nested.nested_tensor_from_jagged(values=x, offsets=offsets)

            x = rearrange(x, "B T C H W -> B T H W C", B=b)
            return x
        else:
            x = rearrange(x, "B T  H W C  -> (B T) C H W")
            x = self.up(x)
            return rearrange(x, "(B T) C H W -> B T H W C", B=b)
