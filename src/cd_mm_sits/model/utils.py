from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@torch._dynamo.disable
def cat_nested_along_time(x_1: torch.Tensor, x_2: torch.Tensor):
    """cat nested tensor along their ragged dim
    in the case where the ragged dimension is 2nd one.
    This should be replaced in the future by a torch.nested.cat
    function once this function exist
    s1: B,t,*
    s2=B,T,*"""
    bs = x_1.shape[0]
    if x_1.is_nested:
        merged_list = [torch.cat([x_1[b], x_2[b]], dim=0) for b in range(bs)]
        return torch.nested.nested_tensor(merged_list, layout=torch.jagged)
    else:
        return torch.cat([x_1, x_2], dim=1)


def cat_list_along_time(x_1: list, x_2: list, padding_value: float = 0.0) -> Tensor:
    """cat nested tensor along their ragged dim
    in the case where the ragged dimension is 2nd one.
    This should be replaced in the future by a torch.nested.cat
    function once this function exist
    s1: B,t,*
    s2=B,T,*"""
    bs = len(x_1)
    merged_list = [torch.cat([x_1[b], x_2[b]], dim=0) for b in range(bs)]
    return pad_sequence(merged_list, batch_first=True, padding_value=padding_value)


@dataclass
class TokenizationOut:
    mm_sits: Tensor  # (B H W) T C
    sorted_times: Tensor  # (B H W) T
    sorted_tpe: Tensor  # (B H W ) T
    sorted_idx: Tensor  # (B H W), T
    sorted_key_padding_mask: Tensor  # (B H W) T
    sorted_mod: Tensor  # (B H W) T


def sort_mm_sits(
    padded_sits: Tensor,
    key_padding_mask: Tensor,
    padded_times: Tensor,
    padded_tpe: Tensor,
    mod: Tensor,
):
    """Return the MM SITS along with its mask
    and label sorted across padded_time information

    :param padded_sits: B,T,C
    :param key_padding_mask: B,T
    :param padded_times: B,T
    :param padded_tpe: B,T,C
    :param padded_labels: B,T
    :returns:

    """
    idx = torch.argsort(padded_times, dim=-1)  # B,T

    idx_3d = repeat(idx, "B T -> B T C", C=padded_sits.shape[-1])
    sorted_times = torch.gather(padded_times, 1, index=idx)
    sorted_mod = torch.gather(mod, 1, index=idx)
    sorted_mm_sits = torch.gather(padded_sits, 1, index=idx_3d)
    sorted_tpe = torch.gather(
        padded_tpe, 1, index=repeat(idx, "B T -> B T C", C=padded_tpe.shape[-1])
    )
    sorted_key_padding_mask = torch.gather(key_padding_mask, 1, index=idx)
    return TokenizationOut(
        mm_sits=sorted_mm_sits,
        sorted_idx=idx,
        sorted_times=sorted_times,
        sorted_mod=sorted_mod,
        sorted_key_padding_mask=sorted_key_padding_mask,
        sorted_tpe=sorted_tpe,
    )


def cat_and_padd(x_1: Tensor, x_2: Tensor, padding=0):
    # Convert nested to padded tensors
    padded_1 = x_1.to_padded_tensor(padding=padding)
    padded_2 = x_2.to_padded_tensor(padding=padding)

    # Concat along time (dim=1) assuming shape (B, T, C)
    return torch.cat([padded_1, padded_2], dim=1)


def _upsample_2D_tensor(sorted_tensor: Tensor, h, w, high_res_w, high_res_h):
    assert len(sorted_tensor.shape) == 2
    sorted_tensor = rearrange(sorted_tensor, "(B H W) T -> B T H W ", H=h, W=w)
    sorted_tensor = repeat(
        sorted_tensor,
        "B T H W -> B T (H repeat_h) (W repeat_w)",
        repeat_w=high_res_w // w,
        repeat_h=high_res_h // h,
    )
    return rearrange(sorted_tensor, "B T H W -> (B H W) T")


def _upsample_3D_tensor(sorted_tensor: Tensor, h, w, high_res_w, high_res_h):
    assert len(sorted_tensor.shape) == 3
    sorted_tensor = rearrange(sorted_tensor, "(B H W) T C -> B T H W C", H=h, W=w)
    sorted_tensor = repeat(
        sorted_tensor,
        "B T H W C -> B T (H repeat_h) (W repeat_w) C",
        repeat_w=high_res_w // w,
        repeat_h=high_res_h // h,
    )
    return rearrange(sorted_tensor, "B T H W C -> (B H W) T C")
