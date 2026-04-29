import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    PATH_TO_DATASET = "/home/iris/Documents/datasets/solar_panels_npy"
    return (PATH_TO_DATASET,)


@app.cell
def _():
    import datetime
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from einops import rearrange

    return Path, datetime, np, pd, plt, rearrange


@app.cell
def _(PATH_TO_DATASET, Path):
    with open(Path(PATH_TO_DATASET).joinpath("sites.txt")) as f:
        txt = f.read()
    l_id = txt.split("\n")[:-1]
    len(l_id)
    return (l_id,)


@app.cell
def _(PATH_TO_DATASET, Path):
    l_files = [file for file in Path(PATH_TO_DATASET).rglob(pattern="*l1c*.npy")]
    return (l_files,)


@app.cell
def _(l_files):
    l_files[0]
    return


@app.cell
def _(l_files):
    len(l_files)
    return


@app.cell
def _(l_id, pd):
    df_files = pd.DataFrame(l_id, columns=["image_id"])
    return (df_files,)


@app.cell
def _(df_files):
    extract_id = df_files["image_id"].iloc[2]
    return (extract_id,)


@app.cell
def _(PATH_TO_DATASET, Path, extract_id, np):
    img = np.load(Path(PATH_TO_DATASET).joinpath(f"{extract_id}_l1c.npy"))
    time = np.load(Path(PATH_TO_DATASET).joinpath(f"{extract_id}_timestamp.npy"))
    label = np.load(Path(PATH_TO_DATASET).joinpath(f"{extract_id}_mask.npy"))
    return img, label, time


@app.cell
def _(img):
    img.shape

    return


@app.cell
def _(label):
    print(label.shape)
    return


@app.cell
def _(datetime, time):
    corrected_time = [datetime.datetime.fromtimestamp(t / 1000).date() for t in time]
    return (corrected_time,)


@app.cell
def _(corrected_time):
    print(corrected_time[0])
    return


@app.cell
def _(PATH_TO_DATASET, Path, datetime, np, plt, rearrange):
    def plot_one_sample(extracted_id, max_pix=128):
        img = np.load(Path(PATH_TO_DATASET).joinpath(f"{extracted_id}_l1c.npy"))
        time = np.load(Path(PATH_TO_DATASET).joinpath(f"{extracted_id}_timestamp.npy"))
        corrected_time = [datetime.datetime.fromtimestamp(t / 1000) for t in time]
        label = np.load(Path(PATH_TO_DATASET).joinpath(f"{extracted_id}_mask.npy"))
        fig, ax = plt.subplots(
            2,
            img.shape[0],
            figsize=(3 * img.shape[0], 6),
        )
        max_pix = min(label.shape[1], label.shape[2])
        for t in range(img.shape[0]):
            ax[0, t].imshow(
                rearrange(
                    img[t, :max_pix, :max_pix, [0, 1, 2]] / 5000, "c h w -> h w c"
                )
            )
            ax[1, t].imshow(label[t, :max_pix, :max_pix])
            ax[0, t].set_axis_off()
            ax[0, t].set_title(corrected_time[t])
            ax[1, t].set_title(corrected_time[t])
        plt.tight_layout()
        plt.show()
        fig.savefig(
            f"/home/iris/Documents/detevent/meetings/images/visu_{extracted_id}.pdf"
        )

    return (plot_one_sample,)


@app.cell
def _(df_files, plot_one_sample):
    for i in range(10):
        plot_one_sample(extracted_id=df_files["image_id"].iloc[i])
    return


@app.cell
def _(label, np):
    value, count = np.unique(label, return_counts=True)
    return count, value


@app.cell
def _(count, value):
    dict(zip(value, count, strict=False))
    return


@app.cell
def _(PATH_TO_DATASET, Path, np, pd):
    def compute_stats(l_files):
        df = pd.DataFrame(columns=["No change", "Change"])
        for img_id in l_files:
            label = np.load(Path(PATH_TO_DATASET).joinpath(f"{img_id}_mask.npy"))
            value, count = np.unique(label, return_counts=True)
            dict_count = dict(zip(value, count, strict=False))
            new_row = pd.DataFrame(
                {"No change": dict_count[False], "Change": dict_count[True]}, index=[0]
            )
            df = pd.concat([df, new_row], ignore_index=True)
        return df

    def compute_length(l_files):
        df = pd.DataFrame(columns=["length"])
        for img_id in l_files:
            sits = np.load(Path(PATH_TO_DATASET).joinpath(f"{img_id}_l1c.npy"))
            new_row = pd.DataFrame({"length": sits.shape[0]}, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        return df

    return compute_length, compute_stats


@app.cell
def _(compute_length, l_id):
    df_length = compute_length(l_id)
    return (df_length,)


@app.cell
def _(df_length, plt):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    df_length["length"].hist(bins=20, ax=ax)
    ax.set_title("Distribution of the length of the SITS")

    return


@app.cell
def _(compute_stats, l_id):
    df_counts = compute_stats(l_id)
    return (df_counts,)


@app.cell
def _(df_counts):
    df_counts.sum()
    return


@app.cell
def _(corrected_time):
    type(corrected_time[0])
    return


@app.cell
def _(corrected_time, datetime):
    # Define the base date
    base_date = datetime.datetime(2015, 1, 1)

    # Calculate the difference
    delta = corrected_time[0] - base_date

    # Get difference in days
    days_diff = delta.days
    return (days_diff,)


@app.cell
def _(days_diff):
    type(days_diff)

    return


if __name__ == "__main__":
    app.run()
