import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from torch import Tensor


def plot_one_mod(
    pred: Tensor, target: Tensor, bands: list | int, normalize: bool = False
):
    BT, *_ = pred.shape
    fig, ax = plt.subplots(2, BT, figsize=(3 * BT, 6))

    pred = _reshape_rgb_img(pred[:, bands, ...]).cpu().numpy()
    target = _reshape_rgb_img(target[:, bands, ...]).cpu().numpy()

    if normalize:
        # compute stats only using the target as the pred wont be stable
        # aggregate all bands, otherwise it change the color aspect
        # NOTE: because S2 normalization is per band, the color will change compared to TCI anyway
        # TODO: this aggregates all images of the batch, without consideration of the SITE
        vmin, vmax = np.quantile(target, (0.05, 0.99))
    else:
        vmin = 0.0
        vmax = 1.0

    for bt in range(BT):
        p = pred[bt, ...]
        t = target[bt, ...]

        if normalize:
            p = (p - vmin) / (vmax - vmin)
            p = np.clip(p, 0.0, 1.0) * 255
            p = p.astype(np.uint8)

            t = (t - vmin) / (vmax - vmin)
            t = np.clip(t, 0.0, 1.0) * 255
            t = t.astype(np.uint8)
        else:
            # imshow will complain about clipping values etc
            pass

        ax[0, bt].imshow(p)
        ax[1, bt].imshow(t)
    return fig, ax


def _reshape_rgb_img(img: Tensor):
    if len(img.shape) == 4:
        return rearrange(img, "B C H W -> B H W C")
    if img.shape[0] == 3:
        return rearrange(img, "C H W -> H W C")
    elif len(img.shape) == 2:
        return img
    else:
        return img
