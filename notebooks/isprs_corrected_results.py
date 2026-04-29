import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    PATH_RES="/home/iris/Documents/detevent/trainings/isprs_mm_sp_benchmark_v8"
    RES_NAME="test_metrics_max_len_s264_s164.pt"
    return PATH_RES, RES_NAME


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    return OmegaConf, Path, instantiate, pl, torch


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
        d_params["d_model"]=config.shared.d_model
        d_params["batch_size"]=config.train.train_config.batch_size
        d_params["lr"]=config.train.train_config.lr
        d_params["model"]=config.model._target_
        try:
            d_params["attention"]=config.model.temporal_encoder.layer.attn_block._target_
            d_params["num_heads"]=config.model.temporal_encoder.layer.attn_block.num_heads
            d_params["name"]=d_params["attention"].split(".")[-1]
            d_params["is_causal"]=config.model.is_causal
            d_params["num_layers"]=config.model.temporal_encoder.num_layers
        except:
            d_params["attention"]=None
            d_params["name"]="None"
        #d_params["model"]=instantiate(config.model)
        try:
            d_params["dk"]=config.model.temporal_encoder.layer.attn_block.dk
        except:
            d_params["dk"]=None
        try:
            d_params["xpos"]=str(config.model.temporal_encoder.layer.attn_block.xpos)
            print(config.model.temporal_encoder.layer.attn_block.xpos)
            if d_params["xpos"]=="timexpos":
                d_params["attention"]="TimeRetention"
                d_params["name"]="TimeRetention"
            elif d_params["xpos"]=="xpos":
                d_params["attention"]="Retention"
                d_params["name"]="Retention"
        except:
            d_params["xpos"]="none"
        d_params["s2_max_len"]=config.datamodule.max_len_s2
        d_params["expe"]=config.datamodule.fold_expe
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
def _(PATH_RES, Path, RES_NAME, extract_hp, pl, torch):
    all_res=[]
    for path_res in Path(PATH_RES).rglob(RES_NAME):
        config_path=path_res.parents[0].joinpath(".hydra").joinpath("config.yaml")
        assert config_path.exists()
        pl_df=extract_hp(config_path)
        pl_df=pl_df.with_columns(
        pl.col("dk").fill_null(0).cast(pl.Int64))
        res=torch.load(path_res,map_location=torch.device('cpu'))
        all_res+=[pl.concat([pl_df,pl.DataFrame(res)],how="horizontal")]

    return (all_res,)


@app.cell
def _(all_res, pl):
    final_df=pl.concat(all_res)
    return (final_df,)


@app.cell
def _(final_df):
    final_df["accuracy","f1_score","miou","expe","attention","model","mod","name","dk"]
    return


@app.cell
def _(final_df):
    final_df["accuracy","f1_score","miou","name","mod","dk","is_causal","num_heads"].group_by("name","mod","dk","is_causal").mean().sort("miou")
    return


@app.cell
def _(final_df):
    final_df["accuracy","f1_score","miou","name","mod","dk","is_causal","num_heads"].group_by("name","mod","dk","is_causal").count()
    return


@app.cell
def _(final_df, pl):
    import matplotlib.pyplot as plt
    stats_df = (
        final_df.select(["accuracy", "f1_score", "miou", "name", "mod", "dk", "is_causal", "num_heads"])
        .group_by(["name", "mod", "dk", "is_causal"])
        .agg([
            pl.col("accuracy").mean().alias("acc_mean"),
            pl.col("accuracy").std().alias("acc_std"),
            pl.col("f1_score").mean().alias("f1_mean"),
            pl.col("f1_score").std().alias("f1_std"),
            pl.col("miou").mean().alias("miou_mean"),
            pl.col("miou").std().alias("miou_std")
        ])
        .sort("miou_mean")
    )

    # Create labels combining group columns
    stats_df = stats_df.with_columns(
        pl.concat_str([
            pl.col("name"),
            pl.col("mod"),
            pl.col("dk").cast(pl.Utf8),
            pl.col("is_causal").cast(pl.Utf8)
        ], separator="_").alias("label")
    )

    # Convert to pandas for easier plotting
    stats_pd = stats_df.to_pandas()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [("accuracy", "acc"), ("f1_score", "f1"), ("miou", "miou")]

    for i, (metric, prefix) in enumerate(metrics):
        means = stats_pd[f"{prefix}_mean"]
        stds = stats_pd[f"{prefix}_std"]

        axes[i].errorbar(range(len(means)), means, yerr=stds, 
                         marker='o', capsize=5, capthick=2)
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        axes[i].set_xticks(range(len(stats_pd)))
        axes[i].set_xticklabels(stats_pd["label"], rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (stats_df,)


@app.cell
def _(stats_df):
    # Simpler alternative using Python f-strings (recommended)

    def format_latex_table_simple(stats_df):
        """Simpler approach using Python formatting with mod as separate column"""

        rows = stats_df.to_dicts()
        stats_df.sort("f1_mean")
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
        latex_table += "Model & Mod & Accuracy & F1 Score & IoU \\\\\n\\hline\n"

        for row in rows:
            print(row)
            if row['name']=="MultiheadAttention":
                name=row['name']
                causal=row['is_causal']
                model = f"{name}, causal={causal}"
            else:
                model = row['name']
            mod = f"{row['mod']}"
            accuracy = f"${row['acc_mean']:.2f} \\pm {row['acc_std']:.2f}$"
            f1_score = f"${row['f1_mean']:.2f} \\pm {row['f1_std']:.2f}$"
            iou = f"${row['miou_mean']:.2f} \\pm {row['miou_std']:.2f}$"

            latex_table += f"{model} & {mod} & {accuracy} & {f1_score} & {iou} \\\\\n"

        latex_table += "\\hline\n\\end{tabular}\n"
        latex_table += "\\caption{Model Performance Comparison}\n"
        latex_table += "\\label{tab:model_results}\n"
        latex_table += "\\end{table}"

        return latex_table

    # Usage
    latex_output = format_latex_table_simple(stats_df)
    print(latex_output)


    return


if __name__ == "__main__":
    app.run()
