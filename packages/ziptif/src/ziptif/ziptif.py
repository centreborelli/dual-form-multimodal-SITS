from __future__ import annotations

import abc
import datetime
import json
import struct
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, override

import numpy as np
import polars as pl
import tifffile
import zstd

COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1


class IBuffer(Protocol):
    @abc.abstractmethod
    def read_at(self, offset: int, length: int) -> bytes: ...


@dataclass(frozen=True)
class OffsetedBuffer(IBuffer):
    buffer: bytes
    offset: int

    @override
    def read_at(self, offset: int, length: int) -> bytes:
        offset_in_buffer = offset - self.offset
        return self.buffer[offset_in_buffer : offset_in_buffer + length]


@dataclass(frozen=True)
class MultiOffsetedBuffer(IBuffer):
    buffers: list[OffsetedBuffer]

    @override
    def read_at(self, offset: int, length: int) -> bytes:
        # find the buffer that contains the offset
        for buf in self.buffers:
            if buf.offset <= offset < buf.offset + len(buf.buffer):
                # found the buffer
                offset_in_buffer = offset - buf.offset
                if offset_in_buffer + length <= len(buf.buffer):
                    # the requested length is fully contained in this buffer
                    return buf.buffer[offset_in_buffer : offset_in_buffer + length]
                else:
                    raise ValueError(
                        f"Requested length {length} at offset {offset} spans multiple buffers"
                    )
        raise ValueError(f"Offset {offset} out of range of available buffers")


@dataclass(frozen=True)
class Tifs:
    """
    Represents a set of tiles of the same band, for a given date and site.
    They can be combined to form a full image, but the tiles might not be contiguous.

    Binary format
    -------------
    - dtype (4 bytes, utf-8, padded with \0)
    - tile shape (2x2 bytes)
    - n tiles x, y (2x1 bytes)
    - n channels (1 byte)
    - nodata value (8 bytes, float64)
    - for each tile (in row-major order):
        - offset (8 bytes, uint64)
    - for each tile (in row-major order):
        - length (4 bytes, uint32)
    - for each tile (in row-major order):
        - compression (1 byte, uint8) 0 is no compression, 1 is zstd
    """

    dtype: str
    tile_shape: tuple[int, int]
    n_tile_x: int
    n_tile_y: int
    channels: int
    nodata: float  # might be casted to dtype
    offsets: list[int]
    """
    layout: row-major
    in bytes from the start of the zip file to the start of the tile data
    """
    lengths: list[int]
    compressions: list[int]
    """ 0 is no compression, 1 is zstd """

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.tile_shape[0] * self.n_tile_y,
            self.tile_shape[1] * self.n_tile_x,
            self.channels,
        )

    @staticmethod
    def from_bytes(b: bytes) -> Tifs:
        fmt = (
            "<"
            "4s"  # dtype
            "HH"  # type shapes
            "BB"  # n tiles x, y
            "B"  # n channels
            "d"  # nodata value
        )
        header_size = struct.calcsize(fmt)
        (
            b_dtype,
            tile_h,
            tile_w,
            n_tile_x,
            n_tile_y,
            channels,
            nodata,
        ) = struct.unpack(fmt, b[:header_size])
        dtype = b_dtype.rstrip(b"\0").decode("utf-8")

        n_tiles = n_tile_x * n_tile_y
        assert len(b) == header_size + n_tiles * (8 + 4 + 1), (
            f"expected {header_size + n_tiles * (8 + 4 + 1)} bytes, got {len(b)}"
        )

        offsets = list(
            struct.unpack(f"<{n_tiles}Q", b[header_size : header_size + n_tiles * 8])
        )
        lengths = list(
            struct.unpack(
                f"<{n_tiles}I",
                b[header_size + n_tiles * 8 : header_size + n_tiles * (8 + 4)],
            )
        )
        compressions = list(
            struct.unpack(
                f"<{n_tiles}B",
                b[
                    header_size + n_tiles * (8 + 4) : header_size
                    + n_tiles * (8 + 4 + 1)
                ],
            )
        )

        return Tifs(
            dtype=dtype,
            tile_shape=(tile_h, tile_w),
            n_tile_x=n_tile_x,
            n_tile_y=n_tile_y,
            channels=channels,
            nodata=nodata,
            offsets=offsets,
            lengths=lengths,
            compressions=compressions,
        )

    def to_bytes(self) -> bytes:
        n_tiles = len(self.offsets)
        assert n_tiles == len(self.lengths) == len(self.compressions)
        b_dtype = np.dtype(self.dtype).str.encode("utf-8")
        assert len(b_dtype) <= 4
        b_dtype = b_dtype.ljust(4, b"\0")

        fmt = (
            "<"
            "4s"  # dtype
            "HH"  # type shapes
            "BB"  # n tiles x, y
            "B"  # n channels
            "d"  # nodata value
        )
        b = struct.pack(
            fmt,
            b_dtype,
            self.tile_shape[0],
            self.tile_shape[1],
            self.n_tile_x,
            self.n_tile_y,
            self.channels,
            self.nodata,
        )
        b += struct.pack(f"<{n_tiles}Q", *self.offsets)
        b += struct.pack(f"<{n_tiles}I", *self.lengths)
        b += struct.pack(f"<{n_tiles}B", *self.compressions)
        return b

    def decode_from_buffer(
        self,
        buffer: IBuffer,
        roi: tuple[int, int, int, int] | None,
        assert_single_tile: bool = False,
    ) -> np.ndarray:
        shape = self.shape
        if roi is None:
            roi = (0, 0, *shape[:2])

        y, x, h, w = roi
        tile_x0 = x // self.tile_shape[1]
        tile_y0 = y // self.tile_shape[0]
        tile_x1 = (x + w + self.tile_shape[1] - 1) // self.tile_shape[1]
        tile_y1 = (y + h + self.tile_shape[0] - 1) // self.tile_shape[0]

        # fast path if there is a single tile to decode
        if tile_x1 == tile_x0 + 1 and tile_y1 == tile_y0 + 1:
            tile_index = tile_y0 * self.n_tile_x + tile_x0
            offset = self.offsets[tile_index]
            length = self.lengths[tile_index]

            buf = buffer.read_at(offset, length)
            tile = _decode_tile(
                self.compressions[tile_index],
                buf,
                self.dtype,
                (self.tile_shape[0], self.tile_shape[1], self.channels),
            )

            y0 = tile_y0 * self.tile_shape[0]
            x0 = tile_x0 * self.tile_shape[1]
            ry0 = max(0, y0 - y)
            rx0 = max(0, x0 - x)
            ry1 = min(h, y0 + self.tile_shape[0] - y)
            rx1 = min(w, x0 + self.tile_shape[1] - x)

            ty0 = ry0 + y - y0
            tx0 = rx0 + x - x0
            ty1 = ry1 + y - y0
            tx1 = rx1 + x - x0

            return tile[ty0:ty1, tx0:tx1, ...]

        if assert_single_tile:
            raise ValueError(
                f"ROI spans multiple tiles, but assert_single_tile is True: {roi=} {self=}"
            )

        full_image = np.full(
            (h, w, self.channels),
            fill_value=self.nodata,
            dtype=self.dtype,
        )
        for ty in range(tile_y0, tile_y1):
            for tx in range(tile_x0, tile_x1):
                tile_index = ty * self.n_tile_x + tx
                offset = self.offsets[tile_index]
                length = self.lengths[tile_index]

                buf = buffer.read_at(offset, length)
                tile = _decode_tile(
                    self.compressions[tile_index],
                    buf,
                    self.dtype,
                    (self.tile_shape[0], self.tile_shape[1], self.channels),
                )

                # Compute where to place the tile in the ROI image
                y0 = ty * self.tile_shape[0]
                x0 = tx * self.tile_shape[1]
                y1 = min(y0 + self.tile_shape[0], shape[0])
                x1 = min(x0 + self.tile_shape[1], shape[1])

                ry0 = max(0, y0 - y)
                rx0 = max(0, x0 - x)
                ry1 = min(h, y1 - y)
                rx1 = min(w, x1 - x)

                ty0 = ry0 + y - y0
                tx0 = rx0 + x - x0
                ty1 = ry1 + y - y0
                tx1 = rx1 + x - x0

                full_image[ry0:ry1, rx0:rx1, ...] = tile[ty0:ty1, tx0:tx1, ...]

        return full_image

    def get_read_ranges(self, roi: tuple[int, int, int, int]) -> list[tuple[int, int]]:
        """Get the list of (offset, length) for the tiles that intersect the roi."""
        tile_shape = self.tile_shape
        tiles_per_row = self.n_tile_x

        ry, rx, rh, rw = roi
        tile_x0 = rx // tile_shape[1]
        tile_y0 = ry // tile_shape[0]
        tile_x1 = (rx + rw + tile_shape[1] - 1) // tile_shape[1]
        tile_y1 = (ry + rh + tile_shape[0] - 1) // tile_shape[0]

        ranges: list[tuple[int, int]] = []
        for ty in range(tile_y0, tile_y1):
            for tx in range(tile_x0, tile_x1):
                tile_index = ty * tiles_per_row + tx
                offset = self.offsets[tile_index]
                length = self.lengths[tile_index]
                ranges.append((offset, length))

        return ranges


if __name__ == "__main__":
    tifs = Tifs(
        dtype="<u2",
        tile_shape=(256, 256),
        n_tile_x=4,
        n_tile_y=4,
        channels=1,
        nodata=-9999.0,
        offsets=[i * 1000 for i in range(16)],
        lengths=[1000 for _ in range(16)],
        compressions=[0 for _ in range(16)],
    )
    b = tifs.to_bytes()
    tifs2 = Tifs.from_bytes(b)
    assert tifs == tifs2


@dataclass(frozen=True)
class Site:
    """
    Binary format
    -------------

    - site_id length (1 byte) + site_id (utf-8)
    - zip_id length (1 byte) + zip_id (utf-8)
    - tile_properties length (4 bytes) + tile_properties (json utf-8)
    - n_dates (2 bytes)
    - n_bands (1 byte)
    - for each band:
        - band name length (1 byte) + band name (utf-8)
        - for each date:
            - length of the Tifs structure
            - tifs (see Tifs.to_bytes)
    - acquisitions_metadata length (4 bytes) + acquisitions_metadata (polars arrow binary format)
    """

    site_id: str
    zip_id: str
    tile_properties: dict[str, Any]
    acquisitions: dict[str, list[Tifs]]
    """ dict band -> list[tifs] (on per date)
    The list might contain a single element if there is no notion of time (DEM)."""
    acquisitions_metadata: pl.DataFrame
    """
    For time series data:
        Should contain at a datetime column
        One row per date (= len(acquisitions[...])).
    For static data:
        Can be empty or contain some metadata (eg DEM info).
        Single row dataframe.
    """

    @property
    def n_dates(self):
        return len(self.acquisitions_metadata)

    @staticmethod
    def from_bytes(b: bytes) -> Site:
        offset = 0

        site_id_len = b[offset]
        offset += 1
        site_id = b[offset : offset + site_id_len].decode("utf-8")
        offset += site_id_len

        zip_id_len = b[offset]
        offset += 1
        zip_id = b[offset : offset + zip_id_len].decode("utf-8")
        offset += zip_id_len

        (tile_properties_len,) = struct.unpack("<I", b[offset : offset + 4])
        offset += 4
        tile_properties = json.loads(
            b[offset : offset + tile_properties_len].decode("utf-8")
        )
        offset += tile_properties_len

        (n_dates,) = struct.unpack("<H", b[offset : offset + 2])
        offset += 2

        n_bands = b[offset]
        offset += 1

        acquisitions: dict[str, list[Tifs]] = {}
        for _ in range(n_bands):
            band_name_len = b[offset]
            offset += 1
            band_name = b[offset : offset + band_name_len].decode("utf-8")
            offset += band_name_len

            tifs_list = []
            for _ in range(n_dates):
                (tifs_size,) = struct.unpack("<I", b[offset : offset + 4])
                offset += 4

                tifs_bytes = b[offset : offset + tifs_size]
                tifs = Tifs.from_bytes(tifs_bytes)
                tifs_list.append(tifs)

                offset += tifs_size

            acquisitions[band_name] = tifs_list

        (acquisitions_metadata_len,) = struct.unpack("<I", b[offset : offset + 4])
        offset += 4
        acquisitions_metadata = pl.read_ipc(
            b[offset : offset + acquisitions_metadata_len]
        )
        assert len(acquisitions_metadata) == n_dates, (
            f"expected {n_dates} rows in acquisitions_metadata, got {len(acquisitions_metadata)}"
        )
        offset += acquisitions_metadata_len

        assert offset == len(b), f"expected offset {offset} to be {len(b)}"

        return Site(
            site_id=site_id,
            zip_id=zip_id,
            tile_properties=tile_properties,
            acquisitions=acquisitions,
            acquisitions_metadata=acquisitions_metadata,
        )

    def to_bytes(self) -> bytes:
        bl: list[bytes] = []

        n_dates = len(next(iter(self.acquisitions.values())))
        assert all(len(v) == n_dates for v in self.acquisitions.values()), (
            "all bands must have the same number of dates"
        )
        n_bands = len(self.acquisitions)
        assert n_bands < 256, "too many bands"

        b_site_id = self.site_id.encode("utf-8")
        b_zip_id = self.zip_id.encode("utf-8")
        b_tile_properties = json.dumps(self.tile_properties).encode("utf-8")
        b_acquisitions_metadata = self.acquisitions_metadata.write_ipc(
            file=None, compression="zstd"
        ).getvalue()

        bl.append(
            struct.pack(
                f"<B{len(b_site_id)}sB{len(b_zip_id)}sI{len(b_tile_properties)}sHB",
                len(b_site_id),
                b_site_id,
                len(b_zip_id),
                b_zip_id,
                len(b_tile_properties),
                b_tile_properties,
                n_dates,
                n_bands,
            )
        )

        for band_name, tifs_list in self.acquisitions.items():
            b_band_name = band_name.encode("utf-8")
            assert len(b_band_name) < 256, "band name too long"
            bl.append(struct.pack("<B", len(b_band_name)))
            bl.append(b_band_name)
            assert len(tifs_list) == n_dates, (
                f"expected {n_dates} dates for band {band_name}, got {len(tifs_list)}"
            )
            for tifs in tifs_list:
                b_tifs = tifs.to_bytes()
                bl.append(struct.pack("<I", len(b_tifs)))
                bl.append(b_tifs)

        bl.append(
            struct.pack("<I", len(b_acquisitions_metadata)) + b_acquisitions_metadata
        )

        return b"".join(b for b in bl)


if __name__ == "__main__":

    def get_site(site_id: str, n_dates: int, n_bands: int) -> Site:
        return Site(
            site_id=site_id,
            zip_id="s2.zip",
            tile_properties={
                "tile_id": "48RVP_03_06",
                "utm_zone": 32,
                "latitude_band": "U",
            },
            acquisitions={
                f"B{b}": [
                    Tifs(
                        dtype="<u2",
                        tile_shape=(256, 256),
                        n_tile_x=4,
                        n_tile_y=4,
                        channels=1,
                        nodata=-9999.0,
                        offsets=[i * 1000 for i in range(16)],
                        lengths=[1000 for _ in range(16)],
                        compressions=[0 for _ in range(16)],
                    )
                    for _ in range(n_dates)
                ]
                for b in range(n_bands)
            },
            acquisitions_metadata=pl.DataFrame(
                {
                    "datetime": [
                        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i)
                        for i in range(n_dates)
                    ],
                    "product_id": [f"S2A_MSIL1C_{i:04d}" for i in range(n_dates)],
                }
            ),
        )


if __name__ == "__main__" and False:
    import dataclasses
    import json
    import time

    site = get_site("test", 1000, 50)

    t1 = time.time()
    b = site.to_bytes()
    t2 = time.time()
    site2 = Site.from_bytes(b)
    t3 = time.time()
    jb = json.dumps(dataclasses.asdict(site), default=str)
    t4 = time.time()
    jbd = json.loads(jb)
    t5 = time.time()

    print(site)
    print(site2)
    print(len(b), len(jb))

    print(
        f"site to bytes: {t2 - t1:.3f}s, from bytes: {t3 - t2:.3f}s, json dumps: {t4 - t3:.3f}s, json loads: {t5 - t4:.3f}s"
    )
    exit()


@dataclass(frozen=True)
class SiteCollection:
    """
    Binary format
    -------------

    - version (2 bytes) (current version is 1)
    - n_sites (8 bytes)
    - for each site:
        - site size (8 bytes)
        - site (see Site.to_bytes)
    """

    sites: list[Site]
    """ site id -> SiteMetadata """
    site_id_to_index: dict[str, int]
    """ site_id -> index in sites list """

    def get_site(self, site_id: str) -> Site:
        return self.sites[self.site_id_to_index[site_id]]

    def get_site_ids(self) -> list[str]:
        return list(self.site_id_to_index.keys())

    @staticmethod
    def from_zips(zip_paths: list[Path]) -> SiteCollection:
        all_sites: list[Site] = []

        # make sure all zip filename are unique, because we use them as identifiers when remapping paths
        zip_filenames = [p.name for p in zip_paths]
        assert len(zip_filenames) == len(set(zip_filenames)), (
            f"zip filenames must be unique, got {zip_filenames}"
        )

        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_id = zip_path.name.removesuffix(".zip")
                try:
                    sites = _collect_metadata(zip_file, zip_id)
                except Exception as e:
                    raise RuntimeError(f"error reading {zip_path}: {e}") from e
            all_sites.extend(sites)

        return SiteCollection(
            sites=all_sites,
            site_id_to_index={s.site_id: i for i, s in enumerate(all_sites)},
        )

    @staticmethod
    def from_bytes_multiple(indexes_bytes: Iterable[bytes]) -> SiteCollection:
        all_sites: list[Site] = []
        for b in indexes_bytes:
            site = SiteCollection.from_bytes(b)
            for s in site.sites:
                all_sites.append(s)

        # make sure there are no duplicate ids
        assert len(all_sites) == len({s.site_id for s in all_sites}), (
            "duplicate site ids found"
        )

        all_site_id_to_index = {s.site_id: i for i, s in enumerate(all_sites)}
        return SiteCollection(sites=all_sites, site_id_to_index=all_site_id_to_index)

    @staticmethod
    def from_bytes(b: bytes) -> SiteCollection:
        offset = 0

        (version,) = struct.unpack("<H", b[offset : offset + 2])
        offset += 2
        assert version == 1, f"unsupported version {version}"

        (n_sites,) = struct.unpack("<Q", b[offset : offset + 8])
        offset += 8

        sites = []
        for _ in range(n_sites):
            (site_size,) = struct.unpack("<Q", b[offset : offset + 8])
            offset += 8

            site_bytes = b[offset : offset + site_size]
            site = Site.from_bytes(site_bytes)
            sites.append(site)

            offset += site_size

        assert offset == len(b), f"expected offset {offset} to be {len(b)}"

        site_id_to_index = {s.site_id: i for i, s in enumerate(sites)}

        return SiteCollection(sites=sites, site_id_to_index=site_id_to_index)

    def to_bytes(self) -> bytes:
        bl: list[bytes] = []
        bl.append(struct.pack("<H", 1))  # version
        bl.append(struct.pack("<Q", len(self.sites)))
        for site in self.sites:
            b_site = site.to_bytes()
            bl.append(struct.pack("<Q", len(b_site)))
            bl.append(b_site)
        return b"".join(b for b in bl)


if __name__ == "__main__" and False:
    import time

    def get_collection(n_sites: int, n_dates: int, n_bands: int) -> SiteCollection:
        return SiteCollection(
            sites=[get_site(f"site_{i}", n_dates, n_bands) for i in range(n_sites)],
            site_id_to_index={f"site_{i}": i for i in range(n_sites)},
        )

    collection = get_collection(1000, 90, 10)
    print("col ok")

    t1 = time.time()
    b = collection.to_bytes()
    print("to_bytes ok")
    t2 = time.time()
    collection2 = SiteCollection.from_bytes(b)
    print("from_bytes ok")
    t3 = time.time()

    print(len(b) / 1e6, "MB")

    print(f"to bytes: {t2 - t1:.3f}s, from bytes: {t3 - t2:.3f}s")
    exit()


def _decode_tile(
    compression: int,
    compressed_tile: bytes,
    dtype: str,
    tile_size: tuple[int, ...],
) -> np.ndarray:
    if compression == COMPRESSION_NONE:
        tile = compressed_tile
    elif compression == COMPRESSION_ZSTD:
        tile = zstd.decompress(compressed_tile)
    else:
        raise ValueError(f"unsupported compression {compression}")

    tile = np.frombuffer(tile, dtype=dtype)
    tile = tile.reshape(tile_size, copy=False)
    return tile


@dataclass(frozen=True)
class TifMetadata:
    path_in_zip: str
    offset_in_zip: int
    dtype: str
    shape: tuple[int, ...]
    tile_size: tuple[int, ...]
    offset: int
    length: int
    compression: tifffile.COMPRESSION
    nodata: Any

    @staticmethod
    def from_tifffile_page(
        name: str, offset_in_zip: int, page: tifffile.TiffPage
    ) -> TifMetadata:
        assert page.compression in (
            tifffile.COMPRESSION.NONE,
            tifffile.COMPRESSION.ZSTD,
        )
        assert page.predictor == 1, "only no predictor supported"
        assert page.planarconfig == tifffile.PLANARCONFIG.CONTIG
        assert len(page.dataoffsets) == 1, "only single-tile tif supported"

        tile_size = page.tile or page.shape
        return TifMetadata(
            path_in_zip=name,
            offset_in_zip=offset_in_zip,
            dtype=str(page.dtype),
            shape=page.shape,
            tile_size=tile_size,
            offset=page.dataoffsets[0],
            length=page.databytecounts[0],
            compression=page.compression,
            nodata=page.nodata,
        )


def read_tif_metadata(z: zipfile.ZipFile, name: str) -> TifMetadata:
    assert z.fp is not None

    info = z.getinfo(name)
    assert info.compress_type == zipfile.ZIP_STORED

    file_offset = info.header_offset + len(info.FileHeader()) - len(info.extra)

    with tifffile.TiffFile(z.fp, offset=file_offset, name=name) as img:
        assert len(img.pages) == 1
        page = img.pages[0]
        assert isinstance(page, tifffile.TiffPage)

        tif = TifMetadata.from_tifffile_page(
            name,
            offset_in_zip=file_offset,
            page=page,
        )

    return tif


def read_tifs_metadata(z: zipfile.ZipFile, names: list[str]) -> Tifs:
    compression_to_int = {
        tifffile.COMPRESSION.NONE: COMPRESSION_NONE,
        tifffile.COMPRESSION.ZSTD: COMPRESSION_ZSTD,
    }

    tifs = [read_tif_metadata(z, name) for name in names]
    first_tif = tifs[0]
    dtype = first_tif.dtype
    tile_shape = first_tif.tile_size
    assert len(tile_shape) == 2
    n_tile_x = max(int(t.path_in_zip.split("/")[1].split("_")[1]) for t in tifs) + 1
    n_tile_y = max(int(t.path_in_zip.split("/")[1].split("_")[0]) for t in tifs) + 1
    channels = first_tif.shape[2] if len(first_tif.shape) > 2 else 1
    nodata = first_tif.nodata
    offsets = [t.offset_in_zip + t.offset for t in tifs]
    lengths = [t.length for t in tifs]
    compressions = [compression_to_int[t.compression] for t in tifs]

    return Tifs(
        dtype=dtype,
        tile_shape=tile_shape,
        n_tile_x=n_tile_x,
        n_tile_y=n_tile_y,
        channels=channels,
        nodata=nodata,
        offsets=offsets,
        lengths=lengths,
        compressions=compressions,
    )


# TODO: this function is too slow!
# in particular, it should not seek backward, just collect info in one pass
def _collect_metadata(zip_file: zipfile.ZipFile, zip_id) -> list[Site]:
    # ex of files:
    #   48RVP_03_06/metadata.json
    #   48RVP_03_06/03_01/20200506T033529_B02.tif
    #   48RVP_03_06/03_01/20200506T033529_TCI.tif
    #   etc
    def extract_band_name(f: str) -> str:
        # f = "48RVP_03_06/03_01/20200506T033529_B02.tif"
        f = f.split("/")[-1]
        f = f.removesuffix(".tif")
        # f = "20200506T033529_B02"
        f = f.split("_")[-1]
        # f = "B02"
        return f

    files = sorted(zip_file.namelist())

    # 32ULU_10_10, 48RVP_03_06, etc
    sites = {f.split("/")[0] for f in files if f and len(f.split("/")) > 1}

    sites_metadatas = []

    for site in sorted(sites):
        print(
            f"reading metadata for site {site}; {len(sites_metadatas) + 1}/{len(sites)}"
        )
        site_files = [f for f in files if f.startswith(site + "/")]

        meta_files = [f for f in site_files if f.endswith("metadata.json")]
        assert len(meta_files) == 1, f"expected 1 metadata file, got {meta_files}"
        site_metadata = json.loads(zip_file.open(meta_files[0]).read().decode("utf-8"))
        assert site_metadata["tile"]["tile_id"] == site, (
            f"expected site id {site}, got {site_metadata['tile']['tile_id']}"
        )

        subtiles = sorted(
            {
                f.split("/")[1]
                for f in site_files
                if len(f.split("/")) > 2 and "_" in f.split("/")[1]
            }
        )

        if products := site_metadata.get("products"):
            print(f"Site {site} has {len(products)} products")

            tifs_per_band: dict[str, list[Tifs]] = {}

            for product in products:
                # in json: "2020-01-07 17:06:59.024000+00:00"
                # we want: "20200107T170659"
                dt: str = product["datetime"]
                dt: datetime.datetime = datetime.datetime.fromisoformat(dt)
                date = dt.strftime("%Y%m%dT%H%M%S")

                per_band: dict[str, list[tuple[int, int, str]]] = {}
                for subtile in subtiles:
                    y, x = map(int, subtile.split("_"))
                    subtile_files = sorted(
                        [
                            f
                            for f in site_files
                            if f.startswith(f"{site}/{subtile}/{date}_")
                            and f.endswith(".tif")
                        ]
                    )
                    assert len(subtile_files) > 0, (
                        f"expected some tif files for {site}/{subtile}/{date}, got none"
                    )

                    band_names = [extract_band_name(f) for f in subtile_files]

                    for b, f in zip(band_names, subtile_files, strict=False):
                        per_band.setdefault(b, []).append((y, x, f))

                for band_name, subt in sorted(per_band.items()):
                    subt.sort(key=lambda t: (t[0], t[1]))  # sort by y,x
                    subtile_files = [t[2] for t in subt]

                    tifs = read_tifs_metadata(zip_file, subtile_files)
                    tifs_per_band.setdefault(band_name, []).append(tifs)

            metadata = pl.from_dicts(
                products,
            ).with_columns(
                # eg "2019-05-29 18:29:29.024000+00:00"
                pl.col("datetime").str.strptime(
                    pl.Datetime(time_zone="UTC", time_unit="ms"),
                    "%Y-%m-%d %H:%M:%S%.f%z",
                ),
            )
        else:
            print(f"Site {site} has not products")
            per_band: dict[str, list[tuple[int, int, str]]] = {}
            for subtile in subtiles:
                y, x = map(int, subtile.split("_"))
                subtile_files = sorted(
                    [
                        f
                        for f in site_files
                        if f.startswith(f"{site}/{subtile}/") and f.endswith(".tif")
                    ]
                )
                assert len(subtile_files) > 0, (
                    f"expected some tif files for {site}/{subtile}/, got none"
                )

                band_names = [extract_band_name(f) for f in subtile_files]

                for b, f in zip(band_names, subtile_files, strict=False):
                    per_band.setdefault(b, []).append((y, x, f))

            tifs_per_band: dict[str, list[Tifs]] = {}
            for band_name, subt in sorted(per_band.items()):
                subt.sort(key=lambda t: (t[0], t[1]))  # sort by y,x
                subtile_files = [t[2] for t in subt]

                tifs = read_tifs_metadata(zip_file, subtile_files)
                tifs_per_band.setdefault(band_name, []).append(tifs)

            if "dem" in site_metadata:
                print(f"Site {site} has dem info")
                metadata = pl.DataFrame(site_metadata["dem"])
            else:
                print(f"Site {site} has no metadata")
                metadata = pl.DataFrame()

        site_metadata = Site(
            site_id=site,
            zip_id=zip_id,
            tile_properties=site_metadata["tile"],
            acquisitions=tifs_per_band,
            acquisitions_metadata=metadata,
        )
        sites_metadatas.append(site_metadata)

    return sites_metadatas


def sanitize_name(root: Path, p: Path) -> str:
    p = p.relative_to(root)
    return str(p.as_posix())


def cli_create_zip(root: Path, outfile: Path):
    """
    Create a zip from a dataset folder.
    The folder should have the following structure:
    root/
        site1/
            metadata.json
            00_01/  # y_x (from 00_00 to 03_03 currently)
                20191103T170451_B02.tif
                20191103T170451_B03.tif
                ...
                20191213T170711_B02.tif
                20191213T170711_B03.tif
                ...
        site2/
            ...

    This function creates a zip with optimized layout:
        - all metadata first
        - then per site, per subtiles (by y_x), per date: all bands (including KLD)
        - then per site, per subtiles (by y_x), per date: TCI band only

    This ordering is to optimize reading time of bands of a single site (more contiguous).
    """
    # NOTE: zipfile implementation doesn't look optimal, we could do a faster implementation

    root = Path(root)
    outfile = Path(outfile)

    # filter sites that don't have metadata.json
    sites = {p.parent.name for p in root.rglob("metadata.json")}
    print(f"found {len(sites)} sites with metadata.json")

    # collect all files of 'finished' sites
    files = [p for s in sites for p in root.joinpath(s).rglob("*") if p.is_file()]
    print(f"found {len(files)} files under {root} with metadata.json")

    def filename_to_order_key(p: Path) -> tuple[Any, ...]:
        parts = p.relative_to(root).parts
        site = parts[0]

        # first, metadata files, by site
        if p.name == "metadata.json":
            return (0, site)

        subtile = parts[1]  # e.g. "00_00"
        date = parts[2].split("_")[0]  # e.g. "20191103T170451"

        # then, by site + subtile, per date, have all the bands except TCI
        if not p.name.endswith("TCI.tif"):
            return (1, site, subtile, date, p.name)

        # finally, TCI files
        return (2, site, subtile, date)

    files = sorted(files, key=filename_to_order_key)
    print("sorted files")

    with zipfile.ZipFile(outfile, "w", compression=zipfile.ZIP_STORED) as z:
        for i, file in enumerate(files):
            if i % 10000 == 0:
                print(f"progress: {i}/{len(files)} files")
            z.write(file, sanitize_name(root, file))


def cli_index(out: Path, *zips: Path):
    path_zips = [Path(p) for p in zips]
    site_collection = SiteCollection.from_zips(path_zips)
    print(site_collection)
    with open(out, "wb") as f:
        f.write(site_collection.to_bytes())
    print(f"saved {out}")


def _test_deserialize(
    site_collection_path: str, site: str | None = None, zip_path: str | None = None
):
    with open(site_collection_path, "rb") as f:
        site_collection = SiteCollection.from_bytes(f.read())

    print(f"found {len(site_collection.sites)} sites")
    print(list(site_collection.site_id_to_index.keys()))

    if site:
        _test_site_collection(site_collection, site, zip_path)


def _test_site_collection(site_collection: SiteCollection, site_id: str, zip_path: str):
    import pprint

    site = site_collection.sites[site_collection.site_id_to_index[site_id]]

    roi = (20, 20, 100, 100)
    all = []
    for tif in site.acquisitions["TCI"]:
        pprint.pprint(tif)

        with open(zip_path, "rb") as f:
            bufs = []
            for offset, length in zip(tif.offsets, tif.lengths, strict=False):
                f.seek(offset)
                buf = f.read(length)
                bufs.append(OffsetedBuffer(buf, offset))
            buffer = MultiOffsetedBuffer(bufs)

        full_image = tif.decode_from_buffer(buffer, roi=roi, assert_single_tile=True)
        all.append(full_image)

    np.save("img1.npy", all)


if __name__ == "__main__":
    # usage:
    # python ziptif.py mkzip path/to/dataset bigdataset.zip
    # python ziptif.py index bigsitecollection.json bigdataset.zip anotherdataset.zip

    import fire

    fire.Fire(
        {
            "mkzip": cli_create_zip,
            "index": cli_index,
            "test-deserialize": _test_deserialize,
        }
    )
