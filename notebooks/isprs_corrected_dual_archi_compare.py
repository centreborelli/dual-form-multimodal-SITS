import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    PATH_RES="/home/iris/Documents/detevent/trainings/isprs_supervised_forecast_v6"
    RES_NAME="test_metrics.pt"
    return (PATH_RES,)


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import matplotlib.pyplot as plt
    return OmegaConf, Path, instantiate, pl, plt, torch


@app.cell
def _(OmegaConf, instantiate, pl):
    def extract_path_config(path):
        path_config=path.parents[3].joinpath(".hydra").joinpath("config.yaml")
        assert path_config.exists()
        return path_config
    def extract_hp(config_path):
        d_params={}
        #print(config_path)
        config=OmegaConf.load(config_path)
        d_params["path"]=config_path
        d_params["d_model"]=config.shared.d_model
        d_params["batch_size"]=config.train.train_config.batch_size
        d_params["lr"]=config.train.train_config.lr
        d_params["model"]=config.model._target_
        d_params["num_heads"]=config.model.temporal_encoder.layer.attn_block.num_heads
        d_params["attention"]=config.model.temporal_encoder.layer.attn_block._target_
        d_params["is_causal"]=config.model.is_causal
        d_params["num_layers"]=config.model.temporal_encoder.num_layers
        try: 
            d_params["w_s2"]=config.lightning_module.w_s2
        except:
            d_params["w_s2"]=float(1)
        try:
            d_params["dk"]=config.model.temporal_encoder.layer.attn_block.dk
        except:
            d_params["dk"]=64
        try:
            d_params["xpos"]=str(config.model.temporal_encoder.layer.attn_block.xpos)
            #print(config.model.temporal_encoder.layer.attn_block.xpos)
            if d_params["xpos"]=="timexpos":
                d_params["attention"]="TimeRetention"
            elif d_params["xpos"]=="xpos":
                d_params["attention"]="Retention"
        except:
            d_params["xpos"]="none"
        d_params["max_len"]=config.datamodule.max_len
        d_params["seed"]=config.seed
        d_params["loss"]=type(instantiate(config.train.train_config.loss)).__name__
        d_params["decoder"]=type(instantiate(config.model.last_layer)).__name__
        if  "MMSSTE" in d_params["model"]:
            d_params["mod"]="s1s2"
        else:
            d_params["mod"]="s2"
        #d_params["training_path"]=config_path
        return pl.DataFrame(d_params)
    return (extract_hp,)


@app.cell
def _():
    return


@app.cell
def _(PATH_RES, Path, extract_hp, pl, torch):
    all_res=[]
    for path_res in Path(PATH_RES).rglob("test_metrics_max_len*.pt"):
        print(path_res)
        test_len=int(path_res.parts[-1].split("_")[-1].split(".")[0][3:])
        #print(test_len)
        if test_len<90:
            config_path=path_res.parents[0].joinpath(".hydra").joinpath("config.yaml")
            assert config_path.exists()
            #print(config_path)
            pl_df=extract_hp(config_path)
            res=torch.load(path_res,map_location=torch.device('cpu'))
            all_res+=[pl.concat([pl_df,pl.DataFrame(res),pl.DataFrame({"test_len":test_len})],how="horizontal")]
    for path_res in Path(PATH_RES).rglob("test_metrics_True_max_len*.pt"):
        print(path_res)
        test_len=int(path_res.parts[-1].split("_")[-1].split(".")[0][3:])
        #print(test_len)
        if test_len<90:
            config_path=path_res.parents[0].joinpath(".hydra").joinpath("config.yaml")
            assert config_path.exists()
            #print(config_path)
            pl_df=extract_hp(config_path)
            res=torch.load(path_res,map_location=torch.device('cpu'))
            all_res+=[pl.concat([pl_df,pl.DataFrame(res),pl.DataFrame({"test_len":test_len})],how="horizontal")]
    return (all_res,)


@app.cell
def _(all_res, pl):
    final_df=pl.concat(all_res)
    return (final_df,)


@app.cell
def _(final_df):
    final_df.get_column("attention").unique()
    return


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df):
    final_df.sort("test_len").group_by("attention","mod","dk","is_causal","test_len","w_s2","num_heads","d_model","decoder","max_len").len().sort("attention")
    return


@app.cell
def _(final_df, pl):
    final_df.with_columns(sum=pl.sum_horizontal("mse_s1","mse_s2")).filter(pl.col("attention")=='cd_mm_sits.layers.cosattention.CosformerAttention').filter(pl.col("num_heads")==4).filter(pl.col("seed")==1).sort("test_len")
    return


@app.cell
def _(final_df, pl):
    final_df.filter(pl.col("num_heads")==4).filter(pl.col("w_s2")==0.1).filter(pl.col("dk")==64).filter(pl.col("mod")=="s1s2").sort("test_len").group_by("attention","test_len","is_causal").len()
    return


@app.cell
def _(final_df):
    final_df.get_column("mod").unique()
    return


@app.cell
def _(final_df):
    final_df.get_column("attention").unique().to_list()

    return


@app.cell
def _():
    return


@app.cell
def _(final_df, pl):
    final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("attention")=="cd_mm_sits.layers.cosattention.CosformerAttention").filter(pl.col("dk")==64).filter(pl.col("num_heads")==4).filter(pl.col("w_s2")==0.1).filter(pl.col("mod")=="s1s2").unique().sort("test_len")
    return


@app.cell
def _(final_df):
    final_df.get_column("attention").unique().to_list()
    return


@app.cell
def _(final_df, pl, plt):
    def create_model_colormap(models):
        """Create colormap grouping base models and their time variants"""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        # Define base model groups and their colors
        base_groups = {
            "multiheadattention": "#FF8C00",  # Dark Orange
            "linear": "#4169E1",              # Royal Blue
            "cosformer": "#2E8B57",          # Sea Green
            "roformer": "#DC143C",           # Crimson
            "retention": "#9932CC",          # Dark Orchid
        }
        color_map = {}
        for model in models:
            model_lower = model.lower()
            # Find which base group this model belongs to
            for base_name, base_color in base_groups.items():
                if base_name in model_lower:
                    if model.lower().startswith("time"):
                        # Create lighter version for time variants
                        base_rgb = mcolors.to_rgb(base_color)  # Fixed function call
                        lighter_rgb = tuple(np.array(base_rgb) * 0.7 + 0.3)
                        color_map[model] = (lighter_rgb, f"Time {base_name.title()}")
                    else:
                        color_map[model] = (base_color, base_name.title())
                    break
        return color_map
    # Usage in your code:
    models = [
        "LinearAttention",
        "MultiheadAttention",
        "CosformerAttention",
        "TimeCosFormerAttention",
        "Retention",
        "TimeRetention",
        "RoFormerAttention",
        "TimeRoFormerAttention"
    ]
    # Create the colormap
    model_colors = create_model_colormap(models)
    # Convert to your d_colors format with markers
    markers = ["o", "s", "^", "d", "v", "p"]
    d_colors = {}
    for i, model in enumerate(models):
        if model in model_colors:
            color = model_colors[model][0]
            d_colors[model] = (color, markers[i % len(markers)])
    fig, ax = plt.subplots(2, 3, figsize=(16, 7))
    plt.subplots_adjust(
        left=0.06, right=0.98,
        top=0.96, bottom=0.18,  # Increased bottom margin for legend
        hspace=0.20, wspace=0.1
    )
    def plot_mse_as_fn_len(stats_subdf, ax, mark, color, metric="sum", alpha=0.1, linestyle='-'):
        if len(stats_subdf) > 0:
            test_lens = stats_subdf.get_column("test_len")
            ax.plot(test_lens, stats_subdf.get_column(f"{metric}_mean"),
                    marker=mark, color=color, linestyle=linestyle)
            ax.fill_between(test_lens,
                           stats_subdf.get_column(f"{metric}_mean") - stats_subdf.get_column(f"{metric}_std"),
                           stats_subdf.get_column(f"{metric}_mean") + stats_subdf.get_column(f"{metric}_std"),
                           alpha=alpha, color=color)
        return ax
    dk = 64
    l_models=[
      "TimeRetention",
      "torch.nn.MultiheadAttention",
      "cd_mm_sits.layers.linear_attention.LinearAttention",
      "Retention",
      "cd_mm_sits.layers.roformer_attention.TimeRoFormerAttention",
      "cd_mm_sits.layers.roformer_attention.RoFormerAttention",
      "cd_mm_sits.layers.cosattention.TimeCosFormerAttention",
      "cd_mm_sits.layers.cosattention.CosformerAttention"
    ]
    for attn in l_models:
        subdf_s2s1 = final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("attention")==attn).filter(pl.col("dk")==dk).filter(pl.col("num_heads")==4).filter(pl.col("w_s2")==0.1).filter(pl.col("mod")=="s1s2").sort("test_len")
        print(len(subdf_s2s1))
        subdf_s2 = final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("attention")==attn).filter(pl.col("dk")==dk).filter(pl.col("num_heads")==4).filter(pl.col("mod")=="s2").sort("test_len")
        stats_subdf_s2s1 = (
            subdf_s2s1.group_by("test_len")
            .agg([
                pl.col("mse_s1").mean().alias("mse_s1_mean"),
                pl.col("mse_s1").std().alias("mse_s1_std"),
                pl.col("mse_s2").mean().alias("mse_s2_mean"),
                pl.col("mse_s2").std().alias("mse_s2_std"),
                pl.col("sum").mean().alias("sum_mean"),
                pl.col("sum").std().alias("sum_std")
            ])
            .sort("test_len")
        )
        stats_subdf_s2 = (
            subdf_s2.group_by("test_len")
            .agg([
                pl.col("mse_s2").mean().alias("mse_s2_mean"),
                pl.col("mse_s2").std().alias("mse_s2_std")
    ])
            .sort("test_len")
        )
        suffix = attn.split(".")[-1]
        color, mark = d_colors[suffix]
        # Determine line style based on whether "Time" is in the attention name
        linestyle = 'dashed' if 'Time' in suffix else '-'
        # Plot with error bars (mean ± std)
        plot_mse_as_fn_len(stats_subdf_s2s1, ax[1,0], mark, color=color, metric="mse_s1", alpha=0.03, linestyle=linestyle)
        plot_mse_as_fn_len(stats_subdf_s2s1, ax[1,1], mark, color=color, metric="mse_s2", alpha=0.03, linestyle=linestyle)
        plot_mse_as_fn_len(stats_subdf_s2s1, ax[1,2], mark, color=color, metric="sum", alpha=0.03, linestyle=linestyle)
        plot_mse_as_fn_len(stats_subdf_s2, ax[0,1], mark, color=color, metric="mse_s2", alpha=0.03, linestyle=linestyle)
    # Set titles
    ax[1,0].set_title(r"MM Forecast $L_{\text{S1}}$", fontweight='bold')
    ax[1,1].set_title(r"MM Forecast $L_{\text{S2}}$", fontweight='bold')
    ax[0,1].set_title(r"S2 Forecast $L_{\text{S2}}$", fontweight='bold')
    ax[1,2].set_title(r"MM Forecast $L_{\text{S1}}+w_{S2} L_{\text{S2}}$", fontweight='bold')
    # Turn off unused subplots
    ax[0,0].set_axis_off()
    ax[0,2].set_axis_off()
    # Remove individual legends from all subplots
    for a in ax.flat:
        if a.get_legend():
            a.get_legend().remove()
    # Create mapping for labels
    d_map_labels = dict(zip([
        "Linear",
        "Multiheadattention",
        "Cosformer",
        "Time Cosformer",
        "Roformer",
        "Time Roformer",
        "Retention",
        "Time Retention"
    ], ["Linear", "Transformer", "Cosformer",
        "Time Cosformer", "LinRoformer",
        "Time LinRoformer",
        "Retention",
        "Time Retention"]))
    # Create common legend at bottom
    labels, handles = [], []
    for att, (color, mark) in d_colors.items():
        model_label = model_colors[att][1] if att in model_colors else att
        linestyle = '--' if 'Time' in model_label else '-'
        print(model_label,linestyle)
        line = plt.Line2D([0], [0], 
                         color=color, 
                         marker=mark, 
                         linestyle=linestyle,
                         markersize=10,           # Larger marker
                         linewidth=2,             # Thicker line
                         markerfacecolor=color)     # Edge thickness
        handles.append(line)
        mapped_label = d_map_labels.get(model_label, model_label)
        labels.append(mapped_label)
    fig.legend(handles, labels,
              bbox_to_anchor=(0.5, 0.02),
              loc='lower center',
              ncol=min(4, len(labels)),
              fontsize=15,
              frameon=False)
    return attn, dk, fig


@app.cell(hide_code=True)
def _(fig):
    fig
    return


@app.cell
def _(fig):
    fig.savefig("all_res_corr.pdf",dpi=300, bbox_inches='tight', pad_inches=0.1)

    return


@app.cell
def _(final_df, pl):
    final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("dk")==16).filter(pl.col("num_heads")==4).sort("test_len")
    return


@app.cell
def _(attn, dk, final_df, pl):
    final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("attention")==attn).filter(pl.col("dk")==dk).filter(pl.col("num_heads")==4).filter(pl.col("w_s2")==0.1).filter(pl.col("mod")=="s1s2").sort("sum")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
