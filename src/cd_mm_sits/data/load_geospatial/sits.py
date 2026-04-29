"""File which contains functions relevant to
open .tif images
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import rioxarray
from rasterio.features import rasterize

from cd_mm_sits.constant.data import S1, S2_10M, S2_20M, S2_MASK
from cd_mm_sits.constant.label import IGNORE_LABEL


@dataclass
class OneAcquisition:
    raster: np.ndarray
    label: np.ndarray
    mask: np.ndarray
    orbite: None | int = None


def open_s2_tif_image(
    path: Path,
    bands_10m: list | None = None,
    bands_20m: list | None = None,
    bands_mask: list | None = None,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    geometry: Any | None = None,
) -> dict:
    """load and resample S2 bands
    :param path:
    :param bands_10m:
    :param bands_20m:
    :param bands_mask:
    :param resampling:
    :returns:
    a dict with as keys the band names and as values numpy array
    """

    if bands_10m is None:
        bands_10m = S2_10M
    if bands_20m is None:
        bands_20m = S2_20M
    if bands_mask is None:
        bands_mask = S2_MASK
    d_bands = {}
    assert len(bands_10m) > 0
    for band in bands_10m:
        path_band = path.joinpath(f"{band}.tif")
        band_img_10m = rioxarray.open_rasterio(path_band)

        d_bands.update({band: band_img_10m.to_numpy()[0, ...]})
    for band in bands_20m:
        path_band = path.joinpath(f"{band}.tif")
        band_img_20m = rioxarray.open_rasterio(path_band)
        band_img_20m = band_img_20m.rio.reproject_match(
            band_img_10m, resampling=resampling
        )

        d_bands.update({band: band_img_20m.to_numpy()[0, ...]})
    for band in bands_mask:
        path_band = path.joinpath(f"{band}.tif")
        band_img_mask = rioxarray.open_rasterio(path_band)
        band_img_mask = band_img_mask.rio.reproject_match(
            band_img_10m, resampling=resampling
        )

        d_bands.update({band: band_img_mask.to_numpy()[0, ...]})
    if geometry is not None:
        geom = geometry.to_crs(band_img_10m.rio.crs)
        geom_raster = rasterize(
            [(geom_val, 1) for geom_val in geom.values],
            out_shape=band_img_10m.shape[-2:],
            out=None,
            transform=band_img_10m.rio.transform(),
            all_touched=True,
            fill=0,
            default_value=1,
            dtype="uint8",
        )
        d_bands.update({"mask_site": geom_raster})
    return d_bands


def open_s1_tif(
    path_s1: Path,
    s1_bands: list | None = None,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    opt: Literal["sp", "gp"] = "sp",
) -> tuple[dict, int]:
    """Load S1"""
    if s1_bands is None:
        s1_bands = S1
    d_bands = {}

    for band in s1_bands:
        path_band = path_s1.joinpath(f"{band}.tif")
        band_img = rioxarray.open_rasterio(path_band)
        d_bands.update({band: band_img.to_numpy()[0, ...]})
    # extract relative orbite file
    if opt == "sp":
        with open(path_s1.joinpath("stac.json")) as f:
            data = json.load(f)
            relative_orbite = int(data["properties"]["sentinel1:relative_orbit_number"])
    elif opt == "gp":
        with open(path_s1.joinpath("metadata.json")) as f:
            data = json.load(f)

            relative_orbite = int(data["sentinel1:relative_orbit_number"])
    else:
        raise NotImplementedError
    return d_bands, relative_orbite


def extract_s2_date(filename: str) -> str:
    """return the acquisition date of .tif image

    :param filename:
    :returns:

    """

    match = re.search(r"_(\d{8})T\d{6}", filename)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj.strftime("%Y-%m-%d")
    raise ValueError(f"Could not extract date from filename: {filename}")


def open_labels(path: Path, opt: Literal["sp", "gp"] = "sp") -> tuple[np.ndarray, str]:
    """Open .tif label along its acquisition date"""
    label_array = rioxarray.open_rasterio(path)
    if opt == "sp":
        date = extract_s2_date(path.parts[-1])
    elif opt == "gp":
        match = re.search(r"T(\d{4}-\d{2}-\d{2})", path.parts[-1])
        assert match is not None
        date = match.group(1)
    else:
        raise NotImplementedError
    return label_array.to_numpy()[0, ...], date


def open_all_labels(
    path_dataset: str,
    sites_id: str,
    opt: Literal["sp", "gp"] = "sp",
) -> dict:
    """
    :param path_dataset:
    :param sites_id:
    :returns:

    """

    list_labels = sorted(
        [path for path in Path(path_dataset).rglob(f"{sites_id}*.tif")]
    )
    # print(list_labels)
    d_labels = {}
    for label_path in list_labels:
        label_array, dates = open_labels(label_path, opt=opt)
        d_labels.update({dates: label_array})
    return d_labels


def find_label(date_s2: str, d_labels: dict, buff_days: int = 0) -> Any:
    """return the label if exist for and S2 date. Allows for label before the
    acquisition (control by buff_days)"""
    date_obj = datetime.strptime(date_s2, "%Y-%m-%d").date()
    possible_date = [
        (date_obj - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, buff_days)
    ]
    for date in possible_date:
        if date in d_labels.keys():
            return d_labels[date]
    return None


def open_s2_sits(
    list_image: list[Path],
    d_labels: dict | None = None,
    selected_bands: list | None = None,
    selected_mask: list | None = None,
    ignore_label=IGNORE_LABEL,
    geometry: Any | None = None,
    buff_days: int = 0,
) -> tuple[dict, np.ndarray | None, list]:
    if selected_bands is None:
        selected_bands = S2_10M + S2_20M
    if selected_mask is None:
        selected_mask = S2_MASK
    d_sits = {}
    site_mask = None
    date_labels = []
    for idx, path in enumerate(list_image):
        if idx == 0:
            d_img = open_s2_tif_image(path=path, geometry=geometry)
            if "mask_site" in d_img.keys():
                site_mask = d_img["mask_site"]  # load as a raster the site mask
            else:
                site_mask = None
        else:
            d_img = open_s2_tif_image(path=path)

        s2_img = np.stack([d_img[band_name] for band_name in selected_bands], axis=0)

        s2_mask = np.stack([d_img[band_name] for band_name in selected_mask], axis=0)
        date_s2 = extract_s2_date(path.parts[-1])

        if d_labels is not None:
            label = find_label(date_s2, d_labels, buff_days)
            if label is None:
                print(f"label not found {date_s2} {s2_img.shape}")
                label = np.ones((s2_img.shape[1], s2_img.shape[2])) * ignore_label
            else:
                print(f"label found {date_s2}")
                date_labels += [date_s2]
        else:
            label = np.ones((s2_img.shape[1], s2_img.shape[2])) * ignore_label

        acquisition = OneAcquisition(raster=s2_img, label=label, mask=s2_mask[0, ...])

        d_sits.update({date_s2: acquisition})
    print(f"my sits s2 {len(d_sits)}")
    return d_sits, site_mask, date_labels


def open_s1_sits(
    list_image: list,
    ignore_label: int = IGNORE_LABEL,
    s1_bands: list | None = None,
    opt: Literal["sp", "gp"] = "sp",
):
    d_sits = {}

    if s1_bands is None:
        s1_bands = S1
    for path in list_image:
        d_img, orbite_number = open_s1_tif(path_s1=path, s1_bands=s1_bands, opt=opt)
        s1_img = np.stack([d_img[band_name] for band_name in s1_bands], axis=0)
        label = np.ones((s1_img.shape[1], s1_img.shape[2])) * ignore_label
        mask = np.ones((s1_img.shape[1], s1_img.shape[2]))
        date_s1 = extract_s2_date(str(path))
        d_sits.update(
            {
                date_s1: OneAcquisition(
                    raster=s1_img, label=label, mask=mask, orbite=orbite_number
                )
            }
        )

    return d_sits
