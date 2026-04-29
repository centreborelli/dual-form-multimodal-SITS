"""
One json per modality.
Multiple zip per json (as long as they are from the same modality).
"""

from __future__ import annotations

import abc
import os
import queue
import threading
import time
from dataclasses import dataclass
from io import BufferedReader
from typing import Any, override

import numpy as np
import polars as pl

from ziptif.ziptif import (
    IBuffer,
    MultiOffsetedBuffer,
    OffsetedBuffer,
    SiteCollection,
)

type ZipId = str


@dataclass(frozen=True)
class Timeseries:
    site_id: str
    bands: dict[str, np.ndarray]
    """ dict band -> TxCxHxW array (C is probably 1, except for TCI where it would be 3 for example) """
    stac_metadata: pl.DataFrame
    """
    Per date.
    eg. datetime, ard:cloud_cover, ard:nodata_cover
    """
    tile_properties: dict[str, Any]


@dataclass(frozen=True)
class ReadRequest:
    site_collection: SiteCollection  # TODO: this is not ok

    site_id: str
    time_indices: list[int]
    """ list of indices to read """
    bands: list[str]
    roi: tuple[int, int, int, int]  # (y0, x0, h, w)

    def __hash__(self):
        return hash(
            (
                self.site_id,
                tuple(self.time_indices),
                tuple(self.bands),
                self.roi,
            )
        )


@dataclass(frozen=True)
class BundleReadRequest:
    subrequests: dict[str, ReadRequest]
    """ the key allows the user to identify the result """

    _fake: bool = False
    _verbose: bool = False

    # NOTE: maybe this should reference usable objects directly,
    # to avoid looking up the items again in the worker thread


@dataclass(frozen=True)
class BundleReadResult:
    timeseries: dict[str, Timeseries]


class ReadSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> BundleReadRequest | None:
        """
        Abstract method to select the next slice to process.
        Returns None when no more slices are available.
        """
        pass


class BuffersPlan:
    plan: dict[ZipId, list[tuple[int, int]]]
    """ zip_id -> list of (offset, length) """
    readrequest_to_zipid: dict[ReadRequest, ZipId]
    """ map each read request to the file path it will be read from """

    def __init__(self):
        self.plan = {}
        self.readrequest_to_zipid = {}

    def ingest(self, request: ReadRequest) -> None:
        site = request.site_collection.sites[
            request.site_collection.site_id_to_index[request.site_id]
        ]
        zipid = site.zip_id
        cur_slices = self.plan.get(zipid, [])

        for band in request.bands:
            imgs = site.acquisitions[band]
            for t in request.time_indices:
                tif = imgs[t]
                ranges = tif.get_read_ranges(roi=request.roi)
                cur_slices.extend(ranges)

        self.plan[zipid] = cur_slices
        self.readrequest_to_zipid[request] = zipid

    def optimize(self, gap: int = 2**20) -> None:
        """Merge overlapping and quasi-overlapping (eg. 2MB gap) slices to minimize reads.
        Sort the slices by offset.
        """
        for path, slices in self.plan.items():
            # Sort by offset
            slices = sorted(slices)

            merged_slices: list[tuple[int, int]] = []
            cur_start, cur_length = slices[0]
            cur_end = cur_start + cur_length
            for start, length in slices[1:]:
                end = start + length
                if start <= cur_end + gap:
                    # Merge
                    cur_end = max(cur_end, end)
                    cur_length = cur_end - cur_start
                else:
                    # No overlap, push current slice and start a new one
                    merged_slices.append((cur_start, cur_length))
                    cur_start = start
                    cur_length = length
                    cur_end = start + length

            # Push the last slice
            merged_slices.append((cur_start, cur_length))

            self.plan[path] = merged_slices

    def read_all(self, openedzips: OpenedZips) -> dict[ReadRequest, IBuffer]:
        all_buffers: dict[ReadRequest, IBuffer] = {}

        for zipid, slices in self.plan.items():
            bufs: list[OffsetedBuffer] = []
            for offset, length in slices:
                file = openedzips.files[zipid]
                buf = os.pread(file.fileno(), length, offset)
                bufs.append(OffsetedBuffer(buf, offset))

            if len(bufs) == 1:
                buffer = bufs[0]
            else:
                buffer = MultiOffsetedBuffer(bufs)

            for request, reqzipid in self.readrequest_to_zipid.items():
                if reqzipid == zipid:
                    all_buffers[request] = buffer

        return all_buffers


def execute_read_request(
    request: BundleReadRequest,
    mod_request: ReadRequest,
    buffer: IBuffer,
) -> Timeseries:
    if request._verbose:
        tstart = time.time()

    n_dates = len(mod_request.time_indices)
    assert n_dates > 0, "for now, we need at least one time index"

    _, _, h, w = mod_request.roi

    site_collection = mod_request.site_collection
    site = site_collection.sites[site_collection.site_id_to_index[mod_request.site_id]]

    bands: dict[str, np.ndarray] = {}

    for band in mod_request.bands:
        first = site.acquisitions[band][mod_request.time_indices[0]]
        arrays = np.empty((n_dates, first.channels, h, w), dtype=first.dtype)

        for i, t in enumerate(mod_request.time_indices):
            tif = site.acquisitions[band][t]

            if not request._fake:
                arr = tif.decode_from_buffer(buffer, mod_request.roi)
            else:
                shape = (h, w) + tif.shape[2:]
                arr = np.zeros(shape, dtype=tif.dtype)

            arrays[i] = arr.transpose(2, 0, 1)  # HWC -> CHW

        bands[band] = arrays

    stac_metadata = site.acquisitions_metadata[mod_request.time_indices]

    if request._verbose:
        print(
            f"Decoded slice {mod_request.site_id} with {n_dates} acquisitions in {time.time() - tstart:.3f}s"
        )
    return Timeseries(
        site_id=mod_request.site_id,
        bands=bands,
        stac_metadata=stac_metadata,
        tile_properties=site.tile_properties,
    )


@dataclass(frozen=True)
class FileSystemMapping:
    mapping: dict[str, str]
    """ maps zip_id to a path """

    def open_all(self) -> OpenedZips:
        return OpenedZips.from_paths(self.mapping)


@dataclass(frozen=True)
class OpenedZips:
    """
    Instances cannot be pickled.
    """

    files: dict[str, BufferedReader]
    """ zip_id -> file """

    @staticmethod
    def from_paths(paths: dict[str, str]) -> OpenedZips:
        fds = {zipid: open(path, "rb") for zipid, path in paths.items()}
        # make the file descriptors inheritable (for multiprocessing)
        # for fd in fds.values():
        # os.set_inheritable(fd, True)
        return OpenedZips(files=fds)


def execute_bundle_read_request(
    request: BundleReadRequest, openedzips: OpenedZips
) -> BundleReadResult:
    """
    The current implementation assumes that all acquisitions are stored in a single zip file.
    """
    plan = BuffersPlan()
    for typ, mod_request in request.subrequests.items():
        plan.ingest(mod_request)
    if request._verbose:
        original_number_of_slices = sum(len(slices) for slices in plan.plan.values())
    plan.optimize()

    if request._verbose:
        t = time.time()
    buffers = plan.read_all(openedzips)
    if request._verbose:
        total_bytes = sum(
            length for slices in plan.plan.values() for _, length in slices
        )
        num_slices = sum(len(slices) for slices in plan.plan.values())
        print(
            f"Read {len(request.subrequests)} subrequests, total {total_bytes / (1024**2):.2f} MB read in {num_slices} slices in {time.time() - t:.3f}s (originally {original_number_of_slices} slices)"
        )

    all_timeseries: dict[str, Timeseries] = {}
    for typ, mod_request in request.subrequests.items():
        assert mod_request in buffers
        timeseries = execute_read_request(
            request,
            mod_request,
            buffers[mod_request],
        )
        all_timeseries[typ] = timeseries
    return BundleReadResult(timeseries=all_timeseries)


class ParallelLoader:
    """
    Multi-threaded loader that processes slices of data through a bounded queue.
    """

    def __init__(
        self,
        sampler: ReadSampler,
        openedzips: OpenedZips,
        num_threads: int,
        queue_size: int,
    ):
        self.sampler = sampler
        self.openedzips = openedzips
        self.num_threads = num_threads
        self.result_queue: queue.Queue[BundleReadResult] = queue.Queue(
            maxsize=queue_size
        )
        self.request_queue: queue.Queue[BundleReadRequest] = queue.Queue(
            maxsize=queue_size
        )
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def _worker_thread(self):
        """Worker thread that processes slice requests."""
        request = None
        while not self._stop_event.is_set():
            try:
                t0 = time.time()
                request = self.request_queue.get(timeout=0.05)
                if time.time() - t0 > 0.001:
                    print(
                        f"Worker {threading.current_thread().name} waited {time.time() - t0:.4f}s for request"
                    )
                result = self._process_slice(request)
                self.result_queue.put(result)
                self.request_queue.task_done()

            except queue.Empty:
                break
            except Exception as e:
                # Log error and continue
                import traceback

                traceback.print_exc()
                print(f"Worker thread error: {e}")
                print(f"{request=}")
                self.request_queue.task_done()
                raise

    def _process_slice(self, request: BundleReadRequest) -> BundleReadResult:
        return execute_bundle_read_request(request, self.openedzips)

    def start(self):
        """Start the worker threads and begin processing."""
        self._stop_event.clear()

        # Start worker threads
        for i in range(self.num_threads):
            worker = threading.Thread(target=self._worker_thread, name=f"worker-{i}")
            worker.start()
            self._workers.append(worker)

        # Producer thread to generate requests
        def producer():
            slice_request = None
            while not self._stop_event.is_set():
                if slice_request is None:
                    slice_request = self.sampler.sample()
                    if slice_request is None:
                        break
                try:
                    self.request_queue.put(slice_request, timeout=0.1)
                    slice_request = None
                except queue.Full:
                    continue

            self._stop_event.set()

        producer_thread = threading.Thread(target=producer, name="producer")
        producer_thread.start()
        self._workers.append(producer_thread)

    def get_result(self, timeout: float | None = None) -> BundleReadResult | None:
        """Get the next result from the queue. Returns None if no more results."""
        try:
            result = self.result_queue.get(timeout=timeout)
            self.result_queue.task_done()
            return result
        except queue.Empty:
            return None

    def stop(self):
        """Stop all worker threads."""
        self._stop_event.set()

        # Wait for all threads to finish
        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test(
    site_collection_path_s2: str,
    site_collection_path_s1: str | None,
    verbose: bool = False,
):
    import pprint
    import time

    with open(site_collection_path_s2, "rb") as f:
        site_collection_s2 = SiteCollection.from_bytes(f.read())
    if site_collection_path_s1 is not None:
        with open(site_collection_path_s1, "rb") as f:
            site_collection_s1 = SiteCollection.from_bytes(f.read())
    else:
        site_collection_s1 = None
    pprint.pprint(site_collection_s2.site_id_to_index.keys())

    import random

    random.seed(0)

    class TestSampler(ReadSampler):
        def __init__(self, site_ids: list[str]):
            self.site_ids = site_ids

        @override
        def sample(self) -> BundleReadRequest | None:
            site_id = random.choice(self.site_ids)
            s2_site = site_collection_s2.sites[
                site_collection_s2.site_id_to_index[site_id]
            ]

            # TODO: how to access size of the raster for a modality?

            ox = random.randint(0, 3) * 128
            oy = random.randint(0, 3) * 128

            time_indices = list(range(len(s2_site.acquisitions)))
            time_indices = random.sample(time_indices, k=min(5, len(time_indices)))
            time_indices.sort()

            per_modality = {}
            per_modality["S2"] = ReadRequest(
                site_collection=site_collection_s2,
                site_id=site_id,
                time_indices=time_indices,
                bands=["B02", "B03", "B04", "KLD"],
                roi=(oy, ox, 128, 128),
            )
            per_modality["S2_20m"] = ReadRequest(
                site_collection=site_collection_s2,
                site_id=per_modality["S2"].site_id,
                time_indices=per_modality["S2"].time_indices,
                bands=["B05", "B06", "B8A", "B11", "B12"],
                roi=(oy // 2, ox // 2, 128 // 2, 128 // 2),
            )

            if site_collection_s1 is not None:
                if site_id in site_collection_s1.sites:
                    s1_site = site_collection_s1.sites[site_id]
                    per_modality["S1"] = ReadRequest(
                        site_collection=site_collection_s1,
                        site_id=site_id,
                        time_indices=list(range(len(s1_site.acquisitions))),
                        bands=["VV", "VH"],
                        roi=(oy, ox, 128, 128),
                    )

            return BundleReadRequest(
                subrequests=per_modality,
                _verbose=verbose,
            )

    site_ids = [sid for sid in site_collection_s2.site_id_to_index.keys()]
    print(f"Testing iterative loader with sites: {site_ids}")

    openedzips = OpenedZips(files={"dataset": os.open("dataset.zip", os.O_RDONLY)})

    sampler = TestSampler(site_ids)
    iterative_loader = ParallelLoader(
        sampler=sampler,
        openedzips=openedzips,
        num_threads=10,
        queue_size=200,
    )

    t0 = time.time()
    with iterative_loader:
        time.sleep(0.1)  # prefill the queue a bit

        all_nbbytes = 0
        avg_get_result_time = 0.0
        i = 0
        for _ in range(2000):
            t1 = time.time()
            result = iterative_loader.get_result()
            avg_get_result_time = 0.9 * avg_get_result_time + 0.1 * (time.time() - t1)

            assert result is not None

            nbbytes = sum(
                band.nbytes
                for ts in result.timeseries.values()
                for band in ts.bands.values()
            )
            all_nbbytes += nbbytes
            i += 1

    print(f"Iterative loader processed {i} sits in {time.time() - t0:.3f}s")
    print(f"Total bytes loaded: {all_nbbytes / (1024**2):.2f} MB")
    print(f"Loading speed: {all_nbbytes / (1024**2) / (time.time() - t0):.2f} MB/s")
    print(f"Average get_result time: {avg_get_result_time:.3f}s")


def test_single(site_collection_path_s2: str, site_id: str | None):
    import random

    class TestSampler(ReadSampler):
        def __init__(self, site_ids: list[str]):
            self.site_ids = site_ids

        @override
        def sample(self) -> BundleReadRequest | None:
            site_id = random.choice(self.site_ids)
            s2_site = site_collection_s2.sites[
                site_collection_s2.site_id_to_index[site_id]
            ]

            ox = random.randint(0, 3) * 128
            oy = random.randint(0, 3) * 128

            time_indices = list(range(len(s2_site.acquisitions)))
            time_indices = random.sample(time_indices, k=min(5, len(time_indices)))
            time_indices.sort()

            per_modality = {}
            per_modality["S2"] = ReadRequest(
                site_collection=site_collection_s2,
                site_id=site_id,
                time_indices=time_indices,
                bands=["B02", "B03", "B04", "KLD"],
                roi=(oy, ox, 128, 128),
            )
            per_modality["S2_20m"] = ReadRequest(
                site_collection=site_collection_s2,
                site_id=per_modality["S2"].site_id,
                time_indices=per_modality["S2"].time_indices,
                bands=["B05", "B06", "B8A", "B11", "B12"],
                roi=(oy // 2, ox // 2, 128 // 2, 128 // 2),
            )

            return BundleReadRequest(
                subrequests=per_modality,
                _verbose=True,
            )

    with open(site_collection_path_s2, "rb") as f:
        site_collection_s2 = SiteCollection.from_bytes(f.read())

    if site_id is None:
        print(site_collection_s2.site_id_to_index.keys())
        return

    openedzip = OpenedZips(files={"dataset": os.open("dataset.zip", os.O_RDONLY)})

    sampler = TestSampler([site_id])
    for _ in range(1000):
        request = sampler.sample()
        assert request is not None
        execute_bundle_read_request(request, openedzip)


if __name__ == "__main__":
    import fire

    fire.Fire({"test": test, "test-single": test_single})
