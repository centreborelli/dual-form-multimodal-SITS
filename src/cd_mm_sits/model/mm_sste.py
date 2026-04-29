from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from cd_mm_sits.constant.model import PAD_MOD_IGNORE, S1_IDX, S2_IDX
from cd_mm_sits.layers.linear_attention import LinearAttention
from cd_mm_sits.model.dataclass import State
from cd_mm_sits.model.sse import LowResUnet, Unet
from cd_mm_sits.model.template_dual_encoders import DualEncoder
from cd_mm_sits.model.temporal_positional_encoder import PositionalEncoder
from cd_mm_sits.model.utils import (
    TokenizationOut,
    _upsample_2D_tensor,
    cat_list_along_time,
    cat_nested_along_time,
    sort_mm_sits,
)


@dataclass
class OutForwardMMSSTE:
    output: Tensor
    sorted_idx: Tensor
    sorted_mod: Tensor
    sorted_times: Tensor
    weights: list | None = None


class MMSSTE(nn.Module):
    def __init__(
        self,
        sse_s1: Unet | LowResUnet,
        sse_s2: Unet | LowResUnet,
        temporal_encoder: DualEncoder,
        tpe: PositionalEncoder,
        last_layer: nn.Module,
        n_mod_feat: int = 16,
        is_causal: bool = False,
        d_model: int = 64,
    ):
        super().__init__()
        d_model = sse_s1.d_model
        d_pe = tpe.d
        num_layers = temporal_encoder.num_layers
        self.sse_s1 = sse_s1  # torch.compile(sse_s1)
        self.sse_s2 = sse_s2  # torch.compile(sse_s2)
        self.tpe = tpe  # torch.compile(tpe)
        self.temporal_encoder = temporal_encoder  # torch.compile(temporal_encoder)
        self.last_layer = last_layer
        self.is_causal = is_causal
        self.modality_embedding = torch.nn.Embedding(
            num_embeddings=176, embedding_dim=n_mod_feat
        )  # one for each relative orbite of S1
        # and indices 0 is given to S2
        self.d_model = d_model
        self.n_mod_feat = n_mod_feat
        self.token_layer = torch.nn.Linear(d_model + d_pe + n_mod_feat, self.d_model)
        self.save_hp = {
            "sse": type(sse_s1).__name__,
            "sse_d": d_model,
            "temporal_encoder": type(temporal_encoder).__name__,
            "num_layers": num_layers,
            "is_causal": is_causal,
            "last_layer": type(last_layer).__name__,
            "tpe": type(tpe).__name__,
            "attention": type(temporal_encoder.layers[0].attn_block).__name__,
            "d_model": self.d_model,
            "n_mod_feat": self.n_mod_feat,
        }
        if isinstance(self.sse_s1, (LowResUnet, Unet)):
            self.save_hp.update(self.sse_s1._load_hp())
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
        sits_s1: Tensor,
        sits_s2: Tensor,
        time_s1: Tensor,
        time_s2: Tensor,
        s1_orbites: Tensor,
        need_weights: bool = False,
        right_product: bool = False,
        return_nested: bool = False,
    ) -> OutForwardMMSSTE:
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
        :param right_product, when possible (for linear attentiononly )
        returns the attention (QK)V and not Q(KV)
        :returns:

        """

        # step 1: For each modality, each image is processed by a a Unet
        # extract the length of each sits of each mod

        # reshape data into (b*t, c h w)
        high_res_h, high_res_w = sits_s2.shape[-2], sits_s2.shape[-1]
        if sits_s1.is_nested:
            lengths_s2 = [s2.shape[0] for s2 in sits_s2]
            lengths_s1 = [s1.shape[0] for s1 in sits_s1]
            sits_s2 = torch.cat(sits_s2.unbind())
            sits_s1 = torch.cat(sits_s1.unbind())
            x_s1 = self.sse_s1(sits_s1)
            x_s2 = self.sse_s2(sits_s2)
            h, w = x_s1.shape[-2], x_s1.shape[-1]
            # reshape into b,t,sse.d_model,h,w
            output_s1 = torch.split(x_s1, lengths_s1, dim=0)
            output_s2 = torch.split(x_s2, lengths_s2, dim=0)

            # nested_output_s1 = torch.nested.nested_tensor(
            #     split_s1, layout=torch.jagged, requires_grad=True
            # )  # recombine  he output of the unet using nested tensor
            # nested_output_s2 = torch.nested.nested_tensor(
            #     split_s2, layout=torch.jagged, requires_grad=True
            # )
        else:
            lengths_s2 = [s2.shape[0] for s2 in sits_s2]
            lengths_s1 = [s1.shape[0] for s1 in sits_s1]
            B = sits_s2.shape[0]
            sits_s2 = rearrange(sits_s2, "B T C H W -> (B T) C H W")
            sits_s1 = rearrange(sits_s1, "B T C H W -> (B T) C H W")

            x_s1 = self.sse_s1(sits_s1)
            x_s2 = self.sse_s2(sits_s2)
            h, w = x_s1.shape[-2], x_s1.shape[-1]
            output_s1 = rearrange(x_s1, "(B T) C H W -> B T C H W", B=B)
            output_s2 = rearrange(x_s2, "(B T) C H W -> B T C H W", B=B)

        # step 2: prepare the pixel-level heterogeneous SITS
        # add the token embedding+tpe+padding
        transformer_input = self.mm_token_embedding(
            x_s1=output_s1,
            x_s2=output_s2,
            lengths_s1=lengths_s1,
            lengths_s2=lengths_s2,
            time_s1=time_s1,
            time_s2=time_s2,
            s1_orbites=s1_orbites,
        )

        # step 3: the temporal transformer
        padded_x, weights = self.temporal_encoder(
            batch=transformer_input.mm_sits,
            times=transformer_input.sorted_times,
            mod=transformer_input.sorted_mod,
            key_padding_mask=transformer_input.sorted_key_padding_mask,
            is_causal=self.is_causal,
            need_weights=need_weights,
            right_product=right_product,
        )

        padded_x = rearrange(padded_x, "(B H W) T C -> B T H W C", H=h, W=w)
        # step 4: last decoding layer
        if high_res_w != w:
            sorted_idx = _upsample_2D_tensor(
                sorted_tensor=transformer_input.sorted_idx,
                h=h,
                w=w,
                high_res_h=high_res_h,
                high_res_w=high_res_w,
            )
            sorted_mod = _upsample_2D_tensor(
                sorted_tensor=transformer_input.sorted_mod,
                h=h,
                w=w,
                high_res_h=high_res_h,
                high_res_w=high_res_w,
            )
            sorted_times = _upsample_2D_tensor(
                sorted_tensor=transformer_input.sorted_times,
                h=h,
                w=w,
                high_res_h=high_res_h,
                high_res_w=high_res_w,
            )
        else:
            sorted_idx = transformer_input.sorted_idx
            sorted_mod = transformer_input.sorted_mod
            sorted_times = transformer_input.sorted_times
        if return_nested:
            # Look at SSTE implementation
            raise NotImplementedError
        return OutForwardMMSSTE(
            output=self.last_layer(padded_x),
            sorted_idx=sorted_idx.to(device=padded_x.device),
            weights=weights,
            sorted_mod=sorted_mod.to(device=padded_x.device),
            sorted_times=sorted_times.to(device=padded_x.device),
        )

    def one_step_forward(
        self,
        sits_s1: Tensor,
        sits_s2: Tensor,
        time_s1: Tensor,
        time_s2: Tensor,
        states: list[State] | list[None],
        s1_orbites: Tensor,
        need_weights: bool = False,
    ) -> tuple[Tensor, list[State]]:
        """Iterative way of processing elements
        :param batch: B,1,C
        :param x_index: B,1
        :param time: B,1
        :param key_padding_mask: B,1
        :returns:

        """
        raise NotImplementedError

    @torch._dynamo.allow_in_graph
    def mm_token_embedding(
        self,
        x_s1: Tensor | list,
        x_s2: Tensor | list,
        lengths_s1: list[int],
        lengths_s2: list[int],
        time_s1: Tensor,
        time_s2: Tensor,
        s1_orbites: Tensor,
    ) -> TokenizationOut:
        """Given mm inputs create a sequence of heterogeneous tokens.
        This function merge the different modality,
        integrate the PE for each modality along a modality token integration
        :param x_s1: B,T,C,H,W
        :param x_s2: B,T,C,H,W
        :param all_times: Tensor or shape time_s1 followed by time_s2
        :returns:

        """

        if isinstance(x_s2, (list, tuple)):
            assert isinstance(x_s1, (list, tuple))
            device = x_s1[0].device
            l_tok_s2 = [
                torch.full((seq.size(0),), S2_IDX, dtype=torch.long, device=device)
                for seq in x_s2
            ]
            l_tok_s1 = [seq.to(dtype=torch.long, device=device) for seq in s1_orbites]

            tokens = cat_list_along_time(l_tok_s1, l_tok_s2, padding_value=100)
            s1_mod = torch.nested.nested_tensor(
                [
                    torch.full((seq.size(0),), S1_IDX, dtype=torch.long, device=device)
                    for seq in x_s1
                ],
                layout=torch.jagged,
            )
            s2_mod = torch.nested.nested_tensor(
                l_tok_s2,
                layout=torch.jagged,
            )
            tokens = self.modality_embedding(tokens)

            t, c, h, w = x_s1[0].shape

            padd_mm_sits = cat_list_along_time(x_1=x_s1, x_2=x_s2)
            lengths = np.sum([lengths_s1, lengths_s2], axis=0)
            batch_size = len(lengths)
            max_len = np.max(lengths)
            lengths_tensor = torch.tensor(lengths, device=device)
            range_tensor = torch.arange(max_len, device=device).expand(
                batch_size, max_len
            )
            key_padding_mask = (range_tensor >= lengths_tensor.unsqueeze(1)).bool()

        else:
            assert isinstance(x_s2, Tensor)
            assert isinstance(x_s1, Tensor)
            assert not x_s2.is_nested
            b, t, c, h, w = x_s1.shape
            token_s2 = (
                torch.ones((x_s2.shape[0], x_s2.shape[1]), device=x_s1.device) * S2_IDX
            ).int()
            s2_mod = token_s2
            s1_mod = (
                torch.ones((x_s1.shape[0], x_s1.shape[1]), device=x_s1.device) * S1_IDX
            ).int()

            padd_mm_sits = cat_nested_along_time(x_s1, x_s2)
            key_padding_mask = torch.zeros(
                (padd_mm_sits.shape[0], padd_mm_sits.shape[1]), device=x_s1.device
            ).bool()
            # token vector from embedding
            token_s2 = self.modality_embedding(token_s2)  # B,T,n_mod_feat
            token_s1 = self.modality_embedding(
                s1_orbites.to(dtype=torch.long),
            )  # B,T,n_mod_feat

            tokens = cat_nested_along_time(token_s1, token_s2)  # b,tS1+tS2,n_mod_feat
        mod = cat_nested_along_time(s1_mod, s2_mod)
        if tokens.is_nested:
            tokens = torch.nested.to_padded_tensor(
                tokens, padding=100
            )  # B,T,n_mod_feat
        if mod.is_nested:
            mod = torch.nested.to_padded_tensor(mod, padding=PAD_MOD_IGNORE)

        tokens = tokens.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, -1, h, w)
        # tokens = rearrange(tokens, "b t h w c -> b t c h w")
        # we need to padd to apply argsort
        # Integrate temporal Encoding info

        mod = mod.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, h, w)
        all_times = cat_nested_along_time(time_s1, time_s2)  # b,t1+t2
        if all_times.is_nested:
            all_times = torch.nested.to_padded_tensor(
                all_times, padding=1e4
            )  # not padding time at 0 cause we want to keep padded
        # elements at the end of the ts after sorting
        tpe_time = self.tpe(all_times)
        tpe_time = (
            tpe_time.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, -1, h, w)
        )  # equivalent to repeat(tpe_time, "B T C-> B T C H W", H=h, W=w)

        padded_mm_sits = torch.cat([padd_mm_sits, tokens, tpe_time], dim=2)
        # create key padding mask

        key_padding_mask = (
            key_padding_mask.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, h, w)
        )
        key_padding_mask = rearrange(key_padding_mask, "B T H W -> (B H W ) T")

        padded_mm_sits = rearrange(padded_mm_sits, "B T C H W -> (B H W) T C", H=h, W=w)
        padded_times = all_times.unsqueeze_(-1).unsqueeze_(-1).expand(-1, -1, h, w)
        padded_times = rearrange(padded_times, "B T H W -> (B H W) T ")
        tpe_time = rearrange(tpe_time, "B T C H W -> (B H W) T C")
        mod = rearrange(mod, "B T H W -> (B H W) T")
        # Sort elements based on their acquisition date
        # Mandatory to ensure non-regressive masking

        out = sort_mm_sits(
            padded_sits=padded_mm_sits,
            mod=mod,
            key_padding_mask=key_padding_mask,
            padded_times=padded_times,
            padded_tpe=tpe_time,
        )
        out.mm_sits = self.token_layer(out.mm_sits)

        return out

    def tokenize_one_mod(self, token: Tensor, x_: Tensor):
        """

        :param token: B,T,H,W,n_mod_feat
        :param x_: B,T,C,H,W
        :param time: B,T
        :returns:

        """
        b, t, h, w, c = token.shape

        token = rearrange(token, "B T H W C -> B T C H W ")
        return torch.cat([x_, token], dim=2)
