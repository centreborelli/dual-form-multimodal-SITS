from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor


def nested_safe_to(tensor, device):
    if tensor.is_nested:
        return torch.nested.nested_tensor(tensor.unbind(), device=device)
    else:
        return tensor.to(device)


@dataclass
class SITSBatch:
    """class for a mono-modal SITS"""

    content: Tensor
    time: Tensor
    # TODO: remove mod and move orbites to a S1SITS class
    mod: Literal["s1", "s2"]
    orbites: Tensor | None = None
    validity_mask: Tensor | None = None
    weather_var: Tensor | None = None
    angles: Tensor | None = None  # should be in radian

    def pin_memory(self):
        self.content = self.content.pin_memory()
        self.time = self.time.pin_memory()
        if self.orbites is not None:
            self.orbites = self.orbites.pin_memory()
        if self.validity_mask is not None:
            self.validity_mask = self.validity_mask.pin_memory()
        if self.weather_var is not None:
            self.weather_var = self.weather_var.pin_memory()
        if self.angles is not None:
            self.angles = self.angles.pin_memory()
        return self

    def to_device(self, device):
        self.content = self.content.to(device)
        self.time = self.time.to(device)
        if self.orbites is not None:
            self.orbites = self.orbites.to(device)
        if self.validity_mask is not None:
            self.validity_mask = self.validity_mask.to(device)
        if self.weather_var is not None:
            self.weather_var = self.weather_var.to(device)
        if self.angles is not None:
            self.angles = self.angles.to(device)
        return self


@dataclass
class CDSupBatch:
    """
    Class for supervised change detection. Labels are provided for each image
    """

    sits: Tensor  # B,T,C,H,W
    time: Tensor  # B,T
    label: Tensor  # B,T,H,W
    extracted_ids: list[str | None] | None = None
    orbites: Tensor | None = None
    mask_site: Tensor | None = None

    def __post_init__(self):
        # Ensure all fields are Tensors
        for name, tensor in [
            ("sits", self.sits),
            ("label", self.label),
            ("time", self.time),
        ]:
            assert isinstance(tensor[0], Tensor), f"{name} must be a torch.Tensor"

    def pin_memory(self):
        """
        To optimize data transfer
        """
        self.sits = self.sits.pin_memory()
        self.time = self.time.pin_memory()
        self.label = self.label.pin_memory()
        if self.orbites is not None:
            self.orbites = self.orbites.pin_memory()
        if self.mask_site is not None:
            self.mask_site = self.mask_site.pin_memory()
        return self

    def to_device(self, device):
        """Transfer batch on a specific device. Nested Tensor suffer
        from som e bugs with the .to() method.
        Once this bug is fixed should be simplified
        """

        self.sits = self.sits.to(device=device)
        self.label = self.label.to(device=device)
        self.time = self.time.to(device)
        if self.orbites is not None:
            self.orbites = self.orbites.to(device)
        if self.mask_site is not None:
            self.mask_site = self.mask_site.to(device)
        return self


@dataclass
class MMSpSupBatch:
    s1: CDSupBatch
    s2: CDSupBatch
    mask_site: Tensor | None  # 0 means the pixel should be ignored

    def pin_memory(self):
        self.s1 = self.s1.pin_memory()
        self.s2 = self.s2.pin_memory()
        if self.mask_site is not None:
            self.mask_site = self.mask_site.pin_memory()
        return self

    def to_device(self, device):
        self.s2 = self.s2.to_device(device=device)
        self.s1 = self.s1.to_device(device=device)
        if self.mask_site is not None:
            self.mask_site = self.mask_site.to(device)
        return self


@dataclass
class MMForcBatch:
    """Class of the batch used to perform supervised forecast"""

    s1: SITSBatch  # the input s1
    s2: SITSBatch  # the input s2

    def pin_memory(self):
        self.s1 = self.s1.pin_memory()
        self.s2 = self.s2.pin_memory()
        return self

    def to_device(self, device):
        self.s2 = self.s2.to_device(device=device)
        self.s1 = self.s1.to_device(device=device)
        return self

    def extract_s1_idx(self):
        if self.s1.content.is_nested:
            offsets = self.s1.time.offsets()
            global_idx = torch.arange(
                self.s1.time.values().shape[0], device=offsets.device
            )
            return torch.nested.nested_tensor_from_jagged(
                values=global_idx, offsets=offsets
            )
        else:
            B = self.s1.content.shape[0]
            T = self.s1.content.shape[1]
            return rearrange(torch.arange(B * T), "(B T ) -> B T ", B=B, T=T)

    def extract_s2_idx(self):
        if self.s2.content.is_nested:
            offsets = self.s2.time.offsets()
            global_idx = torch.arange(
                self.s2.time.values().shape[0], device=offsets.device
            )

            return torch.nested.nested_tensor_from_jagged(
                values=global_idx, offsets=offsets
            )

        else:
            B = self.s2.content.shape[0]
            T = self.s2.content.shape[1]
            return rearrange(torch.arange(B * T), "(B T ) -> B T ", B=B, T=T)


@dataclass
class DeforBatch:
    """
    Class for supervised change detection. Labels are provided for each image
    """

    sits: Tensor  # B,T,C,H,W
    time: Tensor  # B,T
    label: Tensor  # B,H,W
    idx_bef: Tensor  # B
    idx_after: Tensor  # B
    batch_idx: Tensor  # B
    extracted_ids: list[str | None] | None = None
    lengths: None | list = None

    def __post_init__(self):
        # Ensure all fields are Tensors
        for name, tensor in [
            ("sits", self.sits),
            ("label", self.label),
            ("time", self.time),
        ]:
            assert isinstance(tensor[0], Tensor), f"{name} must be a torch.Tensor"

    def pin_memory(self):
        """
        To optimize data transfer
        """
        self.sits = self.sits.pin_memory()
        self.time = self.time.pin_memory()
        self.label = self.label.pin_memory()
        self.idx_bef = self.idx_bef.pin_memory()
        self.idx_after = self.idx_after.pin_memory()
        return self

    def to_device(self, device):
        """Transfer batch on a specific device. Nested Tensor suffer
        from som e bugs with the .to() method.
        Once this bug is fixed should be simplified
        """

        self.sits = self.sits.to(device=device)
        self.label = self.label.to(device)
        self.time = self.time.to(device)
        self.idx_bef = self.idx_bef.to(device)
        self.idx_after = self.idx_after.to(device)
        return self

    def get_lengths(self):
        if self.lengths is None:
            self.lengths = [sits.shape[0] for sits in self.sits]
        return self.lengths
