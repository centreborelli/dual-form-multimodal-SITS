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
        d_params["attention"]=config.model.temporal_encoder.layer.attn_block._target_.split(".")[-1]
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
def _(PATH_RES, Path, extract_hp, pl, torch):
    all_res=[]
    for path_res in Path(PATH_RES).rglob("test_metrics_False_max_len*.pt"):
        #print(path_res)
        test_len=int(path_res.parts[-1].split("_")[-1].split(".")[0][3:])
        #print(test_len)
        if test_len<90:
            config_path=path_res.parents[0].joinpath(".hydra").joinpath("config.yaml")
            assert config_path.exists()
            #print(config_path)
            pl_df=extract_hp(config_path)
            res=torch.load(path_res,map_location=torch.device('cpu'))
            print(path_res)
            all_res+=[pl.concat([pl_df,pl.DataFrame(res),pl.DataFrame({"test_len":test_len})],how="horizontal")]

    return (all_res,)


@app.cell
def _(all_res):
    all_res
    return


@app.cell
def _(all_res, pl):
    final_df=pl.concat(all_res)
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df, pl):
    subdf_s2s1=final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("dk")==64).filter(pl.col("num_heads")==4).filter(pl.col("mod")=="s1s2").sort("test_len")

    subdf_s2=final_df.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2")).filter(pl.col("dk")==64).filter(pl.col("num_heads")==4).filter(pl.col("mod")=="s2").sort("test_len")
    return subdf_s2, subdf_s2s1


@app.cell
def _(subdf_s2s1):
    subdf_s2s1["attention"].unique()
    return


@app.cell
def _(subdf_s2s1):
    subdf_s2s1.group_by("test_len","attention","is_causal").len()
    return


@app.cell
def _():
    return


@app.cell
def _(pl, subdf_s2):
    subdf_s2.group_by("test_len","attention").agg([
            pl.col("mse_s2").mean().alias("mse_s2_mean"),
            pl.col("mse_s2").std().alias("mse_s2_std")
        ]).sort("test_len")
    return


@app.cell
def _(pl, subdf_s2):
    subdf_s2.filter(pl.col("test_len")==16).group_by("attention").len()
    return


@app.cell
def _(subdf_s2):
    subdf_s2["attention"].unique().to_list()
    return


@app.cell
def _():
    Label_names=dict(zip([
    "RoFormerAttention"
    "Retention"
    "TimeRoFormerAttention"
    "MultiheadAttention"
    "LinearAttention"
    "TimeCosFormerAttention"
    "MMTimeCosAttention"
    "TimeRetention"
    "CosformerAttention"

    ],[
    "LinRoFormer"
    "Retention"
    "Time LinRoFormer"
    "Transformer"
    "Linear"
    "Time CosFormer"
    "MM-Time CosFormer"
    "Time Retention"
    "Cosformer"
    ]))
    return


@app.cell
def _(pl, plt):
    import seaborn as sns
    import pandas as pd
    def create_model_colormap(models):
        """Create colormap grouping base models and their time variants"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Define base model groups and their colors
        base_groups = {"multiheadattention":"#FF8C00",
                       "linear": "#4169E1",        # Royal Blue 
            "cosformer": "#2E8B57",     # Sea Green 
            "roformer": "#DC143C",      # Crimson
            "retention": "#9932CC",      # Dark Orchid
        }

        color_map = {}

        for model in models:
            model_lower = model.lower()

            # Find which base group this model belongs to
            for base_name, base_color in base_groups.items():
                if base_name in model_lower:
                    print(model_lower)
                    if model.lower().startswith("time"):
                        # Create lighter version for time variants
                        base_rgb = plt.cm.colors.to_rgb(base_color)
                        lighter_rgb = tuple(np.array(base_rgb) * 0.8 + 0.3)
                        color_map[model] = (lighter_rgb, f"Time {base_name.title()}")
                    elif model.lower().startswith("mm"):
                        # Create lighter version for time variants
                        base_rgb = plt.cm.colors.to_rgb(base_color)
                        lighter_rgb = tuple(np.array(base_rgb) * 0.6 + 0.3)
                        color_map[model] = (lighter_rgb, f"Time {base_name.title()}")
                    else:
                        color_map[model] = (base_color, base_name.title())
                    break

        return color_map

    def plot_boxplot(subdf, test_len, d_colors, ax, fig, suffix_mod="s2", mod="mse_s2"):
        df_plot = subdf.filter(pl.col("test_len")==test_len).sort("attention").to_pandas()

        # Get unique attention types from the data
        unique_attentions = df_plot['attention'].unique()
        # Create improved colormap if d_colors doesn't have good grouping
        if len(d_colors) != len(unique_attentions) or not all(att in d_colors for att in unique_attentions):
            improved_colors = create_model_colormap(unique_attentions)
            # Convert to same format as d_colors
            d_colors = {att: (color, label) for att, (color, label) in improved_colors.items()}

        attention_order = list(d_colors.keys())
        df_plot['attention'] = pd.Categorical(df_plot['attention'], categories=attention_order, ordered=True)
        df_plot = df_plot.sort_values('attention')

        # Create colors list in the same order
        colors = [d_colors[att][0] for att in attention_order if att in df_plot['attention'].unique()]
        #assert len(colors) == len(df_plot['attention'].unique()),f"colors {colors} "

        # Create the boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_plot, x='test_len', y=mod, hue='attention',
                    hue_order=attention_order, palette=colors, ax=ax)
        ax.set_xlabel(" ")
        ax.set_ylabel("")

        # Customize the plot
        if mod=="mse_s2":
            metric=r"$L_{S2}$"
        else:
            metric="$L_{S1}+w_{S2}L_{S2}$"
        ax.set_title(f"{metric}, "+r'$n_{dates}$='+f"{test_len}, {suffix_mod}", fontsize=14, fontweight='bold')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # Improve aesthetics
        ax.legend().remove()
        ax.set_xticks([])
        sns.despine()
        #plt.grid(True, alpha=0.3)
        #plt.tight_layout()
        return fig
    return create_model_colormap, plot_boxplot


@app.cell
def _(create_model_colormap, pl, plot_boxplot, plt, subdf_s2, subdf_s2s1):
    plt.rcParams.update({
        'text.usetex': False,  # Disable LaTeX
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'dejavuserif',  # or 'cm' for Computer Modern
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig,ax=plt.subplots(2,2,figsize=(9,7),sharey='row',)
    plt.subplots_adjust(
        left=0.06, right=0.98,
        top=0.96, bottom=0.1,
        hspace=0.1, wspace=0.06
    )

    models = ["LinearAttention","MultiheadAttention",
        "CosformerAttention",
        "TimeCosFormerAttention", 
        "RoFormerAttention",
        "TimeRoFormerAttention",
        "Retention",
        "TimeRetention",
    ]
    subdf_s2s1.with_columns(sum=pl.col("mse_s1") + 0.1 * pl.col("mse_s2"))
    d_colors = create_model_colormap(models)
    plot_boxplot(subdf_s2,8,ax=ax[0,0],d_colors=d_colors,fig=fig,suffix_mod="S2")
    plot_boxplot(subdf_s2,16,ax=ax[0,1],d_colors=d_colors,fig=fig,suffix_mod="S2")
    plot_boxplot(subdf_s2s1,8,ax=ax[1,0],d_colors=d_colors,fig=fig,suffix_mod="S1 + S2",mod="sum")
    plot_boxplot(subdf_s2s1,16,ax=ax[1,1],d_colors=d_colors,fig=fig,suffix_mod="S1 +S2",mod="sum")

    handles = []
    labels = []
    d_map_labels=dict(zip([
      "Linear",
      "Multiheadattention",
      "Cosformer",
      "Time Cosformer",
      "Roformer",
      "Time Roformer",
      "Retention",
      "Time Retention"
    ],["Linear","Transformer","CosFormer",
      "Time CosFormer","LinRoformer",
      "Time LinRoformer",
      "Retention",
      "Time Retention"]))
    for att, (color, label) in d_colors.items():
        handles.append(plt.Rectangle((0,0),1,1, color=color))
        labels.append(d_map_labels[label])
    fig.legend(handles, labels, 
              bbox_to_anchor=(0.5, 0), 
              loc='lower center', 
              ncol=min(4, len(labels)),fontsize=13,frameon=False)

    return fig, labels


@app.cell
def _(labels):
    labels
    return


@app.cell
def _(fig):
    fig.savefig("boxplot_test.pdf")
    return


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(subdf_s2):
    subdf_s2.with_columns("mse_s2").group_by("attention").mean()
    return


@app.cell
def _(subdf_s2s1):
    subdf_s2s1.with_columns("mse_s2").group_by("attention","test_len").mean().sort("sum")
    return


if __name__ == "__main__":
    app.run()
