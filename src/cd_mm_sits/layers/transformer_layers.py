"""File which presents layers used in various models"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cd_mm_sits.layers.cosattention import (
    CosformerAttention,
    TimeCosFormerAttention,
)
from cd_mm_sits.layers.linear_attention import FlexibleAttnInput, LinearAttention
from cd_mm_sits.layers.retention import MultiScaleRetention
from cd_mm_sits.layers.roformer_attention import (
    RoFormerAttention,
    TimeRoFormerAttention,
)
from cd_mm_sits.layers.xpos_attention import TimeXPOSAttention
from cd_mm_sits.model.dataclass import State


class FFLayer(nn.Module):
    """Feed Forward class as defined in a Transformer
    inspired by https://github.com/pytorch/
    pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/
    torch/nn/modules/transformer.py#L940
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        activation: nn.Module | None = None,
    ):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias)
        if activation is None:
            activation = torch.nn.ReLU()
        self.activation = activation
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """The feed forward function as implemented in pytorch transformer

        :param x: B,T,C
        :returns: Tensor B,T,C

        """

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


@dataclass(kw_only=True)
class ConfigFFLayer:
    """Config for feed-forward layer as defined in the Vaswani Transformer"""

    d_model: int
    dim_feedforward: int
    _target_: Any = FFLayer
    dropout: float = 0.1
    bias: bool = True
    layer_norm_eps: float = 1e-5
    norm_first: bool = False
    activation: torch.nn.ReLU | torch.nn.GELU = torch.nn.ReLU()


class EncoderLayer(nn.Module):
    """
    Attempt for a generic class for a layer in the considered transformer inspired model
    Such a layer is composed of two blocks
    - an attention or pseudo block
    - a feed-forward block
    Parameters
    ----------
    nn : _type_
        _description_dim_feedforward,
    """

    def __init__(
        self,
        attn_block: (
            nn.MultiheadAttention
            | CosformerAttention
            | RoFormerAttention
            | TimeCosFormerAttention
            | MultiScaleRetention
        ),
        feed_forward: FFLayer,
        attn_dropout: float = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_block: (
            nn.MultiheadAttention
            | CosformerAttention
            | TimeCosFormerAttention
            | MultiScaleRetention
            | RoFormerAttention
        ) = attn_block
        self.feed_forward: FFLayer = feed_forward
        self.dropout = nn.Dropout(attn_dropout)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Subclasses must implement this method."""
        raise NotImplementedError

    @abstractmethod
    def one_step_forward(
        self,
        x: Tensor,
        m: int,
        x_index: Tensor,
        state: State | None = None,
    ):
        raise NotImplementedError


@dataclass(kw_only=True)
class ConfigEncoderLayer:
    """Config class for hydra instantiation"""

    attn_block: nn.MultiheadAttention
    feed_forward: FFLayer
    _target_: Any = EncoderLayer
    attn_dropout: float = 0


class TemplateFormerLayer(EncoderLayer):
    def __init__(
        self,
        norm1: nn.LayerNorm,
        norm2: nn.LayerNorm,
        norm_first: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm_first = norm_first
        self.norm1 = norm1
        self.norm2 = norm2


class VanillaTransformerLayer(TemplateFormerLayer):
    def forward(
        self,
        batch: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        need_weights: bool = False,
        right_product: bool = False,
    ):
        x = batch
        if self.norm_first:
            x = self.norm1(x)
            out_attn, weights = self._sa_block(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                right_product=right_product,
            )

            x += out_attn
            x = x + self.feed_forward(self.norm2(x))
        else:
            out_attn, weights = self._sa_block(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                right_product=right_product,
            )
            x = self.norm1(x + out_attn)
            x = self.norm2(x + self.feed_forward(x))
        if weights is not None:
            weights = weights.detach()

        return x, weights

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        need_weights: bool = False,
        right_product: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Self-attention block, it means the query, key and value are
        each projection of the same input sequence x
        :param x: a tensor of size B,T,C
        :param attn_mask: a tensor of size BxH,T,T
        :param key_padding_mask: mask required if sequence
        has been padded. mask of shape B,T
        :param is_causal: I have not understood this param yet
        :param need_weights: if True, a list of the attention score
        :param time: mandatory only if attn_block is type TimeCosFormerAttention
        across each head is displayed.
        """
        if isinstance(self.attn_block, nn.MultiheadAttention):
            x, attn_weights = self.attn_block(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
                average_attn_weights=False,
            )
        elif isinstance(
            self.attn_block,
            (
                LinearAttention,
                MultiScaleRetention,
                CosformerAttention,
                RoFormerAttention,
            ),
        ) and (not right_product):
            x, attn_weights = self.attn_block(
                query=FlexibleAttnInput(x),
                key=FlexibleAttnInput(
                    x,
                ),
                value=FlexibleAttnInput(x),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
                average_attn_weights=False,
            )
        else:
            x, attn_weights = self.attn_block.right_attn(
                query=FlexibleAttnInput(x),
                key=FlexibleAttnInput(
                    x,
                ),
                value=FlexibleAttnInput(x),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
                average_attn_weights=False,
            )

        return self.dropout(x), attn_weights


class KernelAttentionLayer(VanillaTransformerLayer):
    def __post_init__(self):
        assert isinstance(self.attn_block, CosformerAttention)

    def _rec_sa_block(
        self,
        x: Tensor,
        m: int,
        x_index: Tensor,
        state: State,
        key_padding_mask: Tensor | None,
    ) -> tuple[Tensor, State]:
        """
        Self-attention block, it means the query, key and value are
        each projection of the same input sequence x
        :param x: a tensor of size B,1,C
        :param m: the constant used to compute attention reweighting
        :param x_index: the index of the element x within the seq
        :param state: contains the buffer information
        :param key_padding_mask: mask required if sequence
        has been padded. mask of shape B,T
        across each head is displayed.
        """

        x, state = self.attn_block.one_step_forward(
            FlexibleAttnInput(x),
            m=m,
            state=state,
            key_padding_mask=key_padding_mask,
            x_index=x_index,
        )
        return x, state

    def one_step_forward(
        self,
        x: Tensor,
        m: int,
        x_index: Tensor,
        state: State | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, State]:
        """The function which allow to process
        a sequence in a recurrent way
        :param x: (B,1,C) the token
        :param state: contains past information
        :param x_index: (B,1) the index of the token
        :param key_padding_mask: (B,1) a boolean tensor
        :returns:

        """
        if state is None:
            state = State()
        if self.norm_first:
            x = self.norm1(x)
            out_attn, state = self._rec_sa_block(
                x,
                m=m,
                x_index=x_index,
                state=state,
                key_padding_mask=key_padding_mask,
            )
            x += out_attn
            x = x + self.feed_forward(self.norm2(x))
        else:
            out_attn, state = self._rec_sa_block(
                x,
                m=m,
                state=state,
                x_index=x_index,
                key_padding_mask=key_padding_mask,
            )

            x = self.norm1(x + out_attn)
            x = self.norm2(x + self.feed_forward(x))

        return x, state


@dataclass(kw_only=True)
class ConfigVanillaTransformerLayer(ConfigEncoderLayer):
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    _target_: Any = VanillaTransformerLayer
    norm_first: bool = False


class TimeFormerLayer(TemplateFormerLayer):
    def __post_init__(self):
        assert isinstance(self.attn_block, TimeCosFormerAttention)

    def forward(
        self,
        batch: Tensor,
        times: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        need_weights: bool = False,
        right_product: bool = False,
    ):
        x = batch
        if self.norm_first:
            x = self.norm1(x)
            out_attn, weights = self._sa_block(
                x,
                times=times,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                right_product=right_product,
            )

            x += out_attn
            x = x + self.feed_forward(self.norm2(x))
        else:
            out_attn, weights = self._sa_block(
                x,
                times=times,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                right_product=right_product,
            )
            x = self.norm1(x + out_attn)
            x = self.norm2(x + self.feed_forward(x))

        return x, weights

    def _sa_block(
        self,
        x: Tensor,
        times: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        need_weights: bool = False,
        right_product: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        assert isinstance(
            self.attn_block,
            TimeCosFormerAttention
            | TimeRoFormerAttention
            | TimeXPOSAttention
            | MultiScaleRetention,
        )
        if right_product:
            x, attn_weights = self.attn_block.right_attn(
                query=FlexibleAttnInput(x, times),
                key=FlexibleAttnInput(x, times),
                value=FlexibleAttnInput(x, times),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
                average_attn_weights=False,
            )
        else:
            x, attn_weights = self.attn_block.forward(
                query=FlexibleAttnInput(x, times),
                key=FlexibleAttnInput(x, times),
                value=FlexibleAttnInput(x, times),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                is_causal=is_causal,
                average_attn_weights=False,
            )

        return x, attn_weights
