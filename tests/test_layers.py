import pytest
import torch
import torch.nn as nn

from cd_mm_sits.layers.cosattention import CosformerAttention
from cd_mm_sits.layers.transformer_layers import FFLayer, KernelAttentionLayer
from cd_mm_sits.model.dataclass import State


@pytest.mark.critical
def test_kernel_attention_one_step():
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = CosformerAttention(embed_dim=d_model, num_heads=4, has_outproj=False)
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer = KernelAttentionLayer(
        attn_block=mha,
        feed_forward=ffl_layer,
        norm1=norm1,
        norm2=norm2,
    )
    T = 10
    layer.eval()
    batch = torch.randn(2, T, 64)
    state = State()
    outputs = []
    for img_index in range(T):
        # print(batch[:, [img_index], ...].shape)
        out, state = layer.one_step_forward(
            x=batch[:, [img_index], ...],
            m=10,
            state=state,
            x_index=torch.Tensor([img_index] * 2)[:, None] + 1,
        )
        outputs += [out]
        # print(f"state {state.kv_t[0, 0]}")
    outputs = torch.cat(outputs, dim=1)
    attn_mask = torch.triu(torch.full((T, T), 1), diagonal=1)
    attn_mask = attn_mask.type_as(batch).bool()
    outputs_paralell, _ = layer.forward(
        batch=batch,
        attn_mask=attn_mask,
        is_causal=True,
        key_padding_mask=None,
    )
    print(outputs[0, :, 0])
    print(outputs_paralell[0, :, 0])
    dis = torch.nn.PairwiseDistance()
    print(torch.mean(dis(outputs, outputs_paralell)))

    assert torch.allclose(outputs_paralell, outputs, atol=1e-5, rtol=1e-4)
