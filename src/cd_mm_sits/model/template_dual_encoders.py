"""File which describe the backbone of all models which allows to perform
high parallelization as well as efficient inference"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from cd_mm_sits.layers.transformer_layers import (
    EncoderLayer,
    KernelAttentionLayer,
    TimeFormerLayer,
)
from cd_mm_sits.model.dataclass import State


@dataclass(kw_only=True)
class ConfigDualEncoder:
    """Config class for dual encoder generic class"""

    num_layers: int
    layer: EncoderLayer


class DualEncoder(nn.Module):
    """Original class for dual transformer, which allows
    a parallel forward as well as an iterative one

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self, num_layers: int, layer: EncoderLayer):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [copy.deepcopy(layer) for lay in range(num_layers)]
        )
        if isinstance(layer, TimeFormerLayer):
            self.layer_name = "timecos"
        else:
            self.layer_name = "else"  # no need for time
            # or mod information in forward

    def forward(
        self,
        batch: Tensor,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
        times: Tensor | None = None,
        mod: Tensor | None = None,
        right_product: bool = False,
    ) -> tuple[Tensor, list[Tensor | None]]:
        """By default

        Parameters
        ----------
        :param batch: a tensor of size B,T,C
        :param key_padding_mask: mask required if sequence
        has been padded. mask of shape B,T. True means that the
        key should be ignore
        :param is_causal: This is not equivalent to the is_causal
        parameter in pytorch. Here, it means I will create the attn_mask.

        :param need_weights: if True, a list of the attention
        score across each head is returned.
        :param times : mandatory if layer is a TimeFormerLayer
        :returns:
        - a sequence of dimension B,T,C
        - if need_weights is set to True output a sequence of tensor which
        corresponds for each layer to the attention matrix.
        Is set to False, a list of None
        :param right_product, when possible (for linear attentiononly )
        returns the attention (QK)V and not Q(KV)
        """

        l_weights = []
        T = batch.shape[1]
        # TODO not sure it works if need_weights=False
        if is_causal:
            attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
            attn_mask = attn_mask.type_as(batch).bool().to(device=batch.device)
        else:
            attn_mask = None

        for lay in self.layers:
            if self.layer_name == "mmtimecos":
                assert times is not None
                assert mod is not None
                batch, weights = lay(
                    batch=batch,
                    times=times,
                    mod=mod,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    right_product=right_product,
                    is_causal=is_causal,
                )
            elif self.layer_name == "timecos":
                assert times is not None
                batch, weights = lay(
                    batch=batch,
                    times=times,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    right_product=right_product,
                    is_causal=is_causal,
                )

            else:
                batch, weights = lay(
                    batch=batch,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    right_product=right_product,
                    is_causal=is_causal,
                )
            l_weights += [weights]
        return batch, l_weights

    def one_step_forward(
        self,
        x: Tensor,
        m: int,
        x_index: Tensor,
        states: list[State] | list[None],
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[State]]:
        """Efficient prediction, one token can be processed a time

        Parameters
        ----------
        :param x: (B,1,C)
        :param states: list which for each layer contains past information
        :param x_index: (B,1) the index of the token
        :param key_padding_mask: (B,1) a boolean tensor

        Raises
        ------
        NotImplementedError
            _description_
        """

        if not isinstance(self.layers[0], KernelAttentionLayer):
            raise NotImplementedError
        else:
            assert len(states) == len(
                self.layers
            ), f"states {len(states)} layers {len(self.layers)}"
            for idx_lay, lay in enumerate(self.layers):
                x, state = lay.one_step_forward(
                    x=x,
                    x_index=x_index,
                    m=m,
                    state=states[idx_lay],
                    key_padding_mask=key_padding_mask,
                )
                states[idx_lay] = state
            return x, states
