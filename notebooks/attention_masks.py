import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from cd_mm_sits.layers.cosattention import CosformerAttention
    from cd_mm_sits.layers.transformer_layers import (
        FFLayer,
        VanillaTransformerLayer,
    )
    from cd_mm_sits.model.template_dual_encoders import DualEncoder

    return (
        CosformerAttention,
        DualEncoder,
        FFLayer,
        VanillaTransformerLayer,
        nn,
        plt,
        torch,
    )


@app.cell
def _(CosformerAttention, DualEncoder, FFLayer, VanillaTransformerLayer, nn):
    d_model = 64
    ffl_layer = FFLayer(d_model=d_model, dim_feedforward=128)
    mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=4, dropout=0, batch_first=True
    )
    cos_attn = CosformerAttention(num_heads=4, embed_dim=d_model, causal=True).to("cpu")
    norm1 = nn.LayerNorm(normalized_shape=d_model)
    norm2 = nn.LayerNorm(normalized_shape=d_model)
    layer_regular = VanillaTransformerLayer(
        attn_block=mha, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
    )
    transformer_regular = DualEncoder(num_layers=1, layer=layer_regular)
    transformer_cos = DualEncoder(
        num_layers=1,
        layer=VanillaTransformerLayer(
            attn_block=cos_attn, feed_forward=ffl_layer, norm1=norm1, norm2=norm2
        ),
    )
    return transformer_cos, transformer_regular


@app.cell
def _(mo):
    mo.md(r"""## Regular attention (no masking)""")
    return


@app.cell
def _(torch, transformer_cos, transformer_regular):
    out_regular, weights_regular = transformer_regular(
        batch=torch.randn(2, 10, 64), need_weights=True
    )
    out_cos, weights_cos = transformer_cos(
        batch=torch.randn(2, 10, 64), need_weights=True
    )
    return (weights_regular,)


@app.cell
def _(plt):
    fig, ax = plt.subplots()
    return (ax,)


@app.cell
def _(ax, weights_regular):
    ax.imshow(weights_regular[0][0, 0, ...].detach().numpy())
    return


@app.cell
def _(mo):
    mo.md(r"""## Padding attention""")
    return


@app.cell
def _(torch):
    def check_attn_matrix_padd(transformer, B=2, T=10, C=64):
        padd_mask = torch.zeros((B, T))
        padd_mask[0, 7:] = 1
        padd_mask[1, 8:] = 1
        padd_mask = padd_mask.bool()
        out, weights = transformer(
            batch=torch.randn(B, T, C), need_weights=True, key_padding_mask=padd_mask
        )
        return out, weights

    return (check_attn_matrix_padd,)


@app.cell
def _(ax, check_attn_matrix_padd):
    ax.imshow(check_attn_matrix_padd()[1][0][0, 0, ...].detach().numpy())
    return


@app.cell
def _(mo):
    mo.md(r"""## Mask the attention score""")
    return


@app.cell
def _(torch, transformer):
    def check_attn_matrix_attn_score(B=2, T=10, C=64, H=4):
        out, weights = transformer(
            batch=torch.randn(B, T, C), need_weights=True, is_causal=True
        )
        return out, weights

    return (check_attn_matrix_attn_score,)


@app.cell
def _(ax, check_attn_matrix_attn_score):
    ax.imshow(check_attn_matrix_attn_score()[1][0][0, 0, ...].detach().numpy())
    return


@app.cell
def _(mo):
    mo.md(r"""## Combine masks""")
    return


@app.cell
def _(torch, transformer):
    def check_attn_matrix_masks(B=2, T=10, C=64, H=4):
        padd_mask = torch.zeros((B, T))
        padd_mask[0, 7:] = 1
        padd_mask[1, 8:] = 1
        padd_mask = padd_mask.bool()
        out, weights = transformer(
            batch=torch.randn(B, T, C),
            need_weights=True,
            key_padding_mask=padd_mask,
            is_causal=True,
        )
        return out, weights

    return (check_attn_matrix_masks,)


@app.cell
def _(ax, check_attn_matrix_masks):
    ax.imshow(check_attn_matrix_masks()[1][0][1, 0, ...].detach().numpy())
    return


if __name__ == "__main__":
    app.run()
