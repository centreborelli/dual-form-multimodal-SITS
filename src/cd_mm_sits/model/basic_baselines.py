import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from cd_mm_sits.model.sse import FCNResNet50, Unet


class OnlyUnet(nn.Module):
    def __init__(
        self,
        sse: Unet | FCNResNet50,
        last_layer: nn.Module,
    ):
        super().__init__()
        self.sse = sse
        self.last_layer = last_layer
        self.save_hp = {
            "sse": type(sse).__name__,
            "last_layer": type(last_layer).__name__,
        }

    def forward(
        self,
        batch: Tensor,
        time: Tensor,
        need_weights: bool = False,
        right_product: bool = False,
        return_nested: bool = False,
    ) -> tuple[Tensor, None, Tensor]:
        lengths = [sits.shape[0] for sits in batch]
        x = torch.cat(batch.unbind())
        print(x.shape)
        x = self.sse(x)
        bt, c, h, w = x.shape
        print(x.shape)

        # Use pad_sequence instead of nested tensor operations
        split_tensors = torch.split(x, lengths, dim=0)
        padded_x = pad_sequence(split_tensors, batch_first=True, padding_value=0.0)

        # Create padding mask vectorized (no nested tensors)
        batch_size = len(lengths)
        max_len = max(lengths)
        lengths_tensor = torch.tensor(lengths, device=x.device)
        range_tensor = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len
        )
        key_padding_mask = range_tensor >= lengths_tensor.unsqueeze(1)

        key_padding_mask = repeat(key_padding_mask, "B T -> (B H W) T", H=h, W=w)
        padded_x = rearrange(padded_x, "B T C H W -> B T H W C")

        # step 3: the temporal transformer

        # step 4: last decoding layer
        return self.last_layer(padded_x), None, key_padding_mask
