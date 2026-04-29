"""
File for spectro-spatio-temporal encoder architecture
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from cd_mm_sits.layers.dynamic_tanh import convert_ln_to_dyt
from cd_mm_sits.layers.linear_attention import LinearAttention
from cd_mm_sits.model.dataclass import State
from cd_mm_sits.model.sse import LowResUnet, Unet
from cd_mm_sits.model.template_dual_encoders import DualEncoder
from cd_mm_sits.model.temporal_positional_encoder import PositionalEncoder
from cd_mm_sits.model.utils import _upsample_2D_tensor


@dataclass
class ConfigSSTE:
    sse: Unet
    temporal_encoder: DualEncoder
    tpe: PositionalEncoder
    last_layer: nn.Module


class SSTE(nn.Module):
    def __init__(
        self,
        sse: Unet | LowResUnet,
        temporal_encoder: DualEncoder,
        tpe: PositionalEncoder,
        last_layer: nn.Module,
        is_causal: bool = False,
        convert_to_dyntanh: bool = False,
    ):
        super().__init__()
        self.sse = sse
        self.tpe = tpe
        if convert_to_dyntanh:
            self.temporal_encoder = convert_ln_to_dyt(temporal_encoder)
        else:
            self.temporal_encoder = temporal_encoder
        self.last_layer = last_layer
        self.is_causal = is_causal
        self.save_hp = {
            "sse": type(sse).__name__,
            "temporal_encoder": type(temporal_encoder).__name__,
            "num_layers": self.temporal_encoder.num_layers,
            "is_causal": is_causal,
            "last_layer": type(last_layer).__name__,
            "tpe": type(tpe).__name__,
            "attention": type(temporal_encoder.layers[0].attn_block).__name__,
            "d_model": self.sse.d_model,
        }
        if isinstance(self.sse, (LowResUnet, Unet)):
            self.save_hp.update(self.sse._load_hp())
        if isinstance(temporal_encoder.layers[0].attn_block, LinearAttention):
            self.save_hp.update(
                {
                    "attention_act_fn": type(
                        temporal_encoder.layers[0].attn_block.act_fun
                    ).__name__
                }
            )

    def forward(
        self,
        batch: Tensor,
        time: Tensor,
        need_weights: bool = False,
        right_product: bool = False,
        return_nested: bool = False,
    ) -> tuple[Tensor, list[Tensor | None], Tensor]:
        """Forward function for a spectral spatial temporal encoder
        :param batch: a nested tensor B,T,C,H,W, T can have various dimension
        for each sample of the batch
        :param time: contains temporal information which are encoded
        by the temporal positional encoder (i.e DOY)
        :param key_padding_mask:  mask required if sequence
        has been padded. mask of shape B,T. True means that the
        key should be ignore
        :param need_weights: if True, a list of the attention score across
        each head is returned.
        :param is_causal: means is it auto-regressive. If yes create the
        auto-regressive mask. This is different from the is_causal
        in pytorch transformer
        :returns:
        """

        # step 1: each image is processed by a a Unet, use batch of nested tensor ?
        # b, t, c, h, w = batch.shape
        high_res_h, high_res_w = batch.shape[-2], batch.shape[-1]
        lengths = [sits.shape[0] for sits in batch]
        if batch.is_nested:
            x = torch.cat(batch.unbind())
        else:
            x = rearrange(batch, "b t c h w -> (b t) c h w")
        x = self.sse(x)
        *_, h, w = x.shape
        if time.is_nested:
            # :coche_blanche: FIX: Use pad_sequence instead of nested tensor operations
            split_tensors = torch.split(x, lengths, dim=0)
            padded_x = pad_sequence(split_tensors, batch_first=True, padding_value=0.0)
            # :coche_blanche: FIX: Create padding mask vectorized (preserves gradients)
            batch_size = len(lengths)
            max_len = max(lengths)
            lengths_tensor = torch.tensor(lengths, device=x.device)
            range_tensor = torch.arange(max_len, device=x.device).expand(
                batch_size, max_len
            )
            key_padding_mask = range_tensor >= lengths_tensor.unsqueeze(1)
            # :coche_blanche: FIX: Handle time tensor with pad_sequence
            time_sequences = time.unbind()
            padded_time = pad_sequence(
                time_sequences, batch_first=True, padding_value=0
            )
        else:
            padded_x = rearrange(x, "(B T) C H W -> B T C H W", B=batch.shape[0])
            key_padding_mask = torch.zeros(
                (batch.shape[0], batch.shape[1]), device=x.device
            ).bool()
            padded_time = time
        *_, h, w = padded_x.shape
        padded_x = rearrange(padded_x, "B T C H W -> (B H W) T C")
        key_padding_mask = repeat(key_padding_mask, "B T -> (B H W) T", H=h, W=w)
        tpe_batch = self.tpe(padded_time.type_as(x))  # B,T
        tpe_batch = repeat(tpe_batch, "B T C -> (B H W) T C", H=h, W=w)
        padded_x = padded_x + tpe_batch  # we decide here to concatenate
        # maybe integrate more flexibility
        padded_x = padded_x.to(x)  # :coche_blanche: FIX: Proper .to() call
        # step 3: the temporal transformer
        padded_x, weights = self.temporal_encoder(
            batch=padded_x,
            times=repeat(padded_time, "B T -> (B H W) T", H=h, W=w),
            key_padding_mask=key_padding_mask,
            is_causal=self.is_causal,
            need_weights=need_weights,
            right_product=right_product,
        )
        padded_x = rearrange(padded_x, "(B H W) T C -> B T H W C", H=h, W=w)
        if high_res_w != w:
            key_padding_mask = _upsample_2D_tensor(
                key_padding_mask, h=h, w=w, high_res_h=high_res_h, high_res_w=high_res_w
            )
        # :coche_blanche: FIX: Always use regular tensors for gradient preservation
        out = self.last_layer(padded_x)

        if return_nested and time.is_nested:
            # Split back to sequences for nested output (outside gradient graph)
            outputs = []
            for i, length in enumerate(lengths):
                outputs.append(out[i, :length])
                out = torch.nested.nested_tensor(outputs)  # No layout=torch.jagged
        return out, weights, key_padding_mask

    def one_step_forward(
        self,
        batch: Tensor,
        x_index: Tensor,
        time: Tensor,
        m: int,
        states: list[State] | list[None],
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[State]]:
        """Iterative way of processing elements
        :param batch: B,1,C
        :param x_index: B,1
        :param time: B,1
        :param key_padding_mask: B,1
        :returns:

        """
        b, t, c, h, w = batch.shape
        x = self.sse(rearrange(batch, "B t c h w-> (B t) c h w"))
        if key_padding_mask is not None:
            key_padding_mask = repeat(key_padding_mask, "B T -> (B H W ) T", H=h, W=w)
        tpe_batch = self.tpe(time.type_as(x))  # B,T
        tpe_batch = repeat(tpe_batch, "B T C-> (B H W) T C", H=h, W=w)  # T=1
        x = rearrange(x, "(B T) C H W -> (B H W) T C", H=h, W=w, B=b, T=t)
        padded_x = x + tpe_batch  # we decide here to concatenate
        # maybe integrate more flexibility
        padded_x = padded_x.to(x)

        # step 3: the temporal transformer
        padded_x, states = self.temporal_encoder.one_step_forward(
            x=padded_x,
            x_index=repeat(x_index, "B T -> (B H W) T", H=h, W=w),
            m=m,
            states=states,
            key_padding_mask=key_padding_mask,
        )

        padded_x = rearrange(padded_x, "(B H W) T C -> B T H W C", H=h, W=w)

        # step 4: last decoding layer
        return self.last_layer(padded_x), states
