from collections.abc import Sequence

import torch

from cd_mm_sits.data.batch_class import (
    CDSupBatch,
    DeforBatch,
    MMForcBatch,
    MMSpSupBatch,
    SITSBatch,
)
from cd_mm_sits.data.dataset.brads1_defor import SampleBradS1
from cd_mm_sits.data.dataset.mm_forecast import (
    MMForcSample,
    S1ForcSITSSample,
    S2ForcSITSSample,
)
from cd_mm_sits.data.dataset.sp_mm_zarr import SPMMSampleSits
from cd_mm_sits.data.dataset.sp_s2_zarr import SPSampleSITS


def collate_fn_sp(batch: Sequence[SPSampleSITS]) -> CDSupBatch:
    """Hand-crafted function to create batch
    from my custom SampleSits dataclass
    :param batch:
    :returns:
    """
    sits = torch.nested.nested_tensor(
        [b.sits.float() for b in batch], layout=torch.jagged
    )
    time = torch.nested.nested_tensor(
        [b.time.float() for b in batch], layout=torch.jagged
    )
    label = torch.nested.nested_tensor(
        [b.label.float() for b in batch], layout=torch.jagged
    )
    if batch[0].orbites is not None:
        orbites = torch.nested.nested_tensor(
            [b.orbites.int() for b in batch], layout=torch.jagged
        )
    else:
        orbites = None
    if batch[0].mask_site is not None:
        mask_site = torch.stack([b.mask_site for b in batch])
    else:
        mask_site = None
    extracted_ids = [b.extracted_id for b in batch]
    return CDSupBatch(
        sits=sits,
        time=time,
        label=label,
        extracted_ids=extracted_ids,
        orbites=orbites,
        mask_site=mask_site,
    )


def collate_fn_sp_test(batch: Sequence[SPSampleSITS]) -> CDSupBatch:
    """Hand-crafted function to create batch
    from my custom SampleSits dataclass
    :param batch:
    :returns:
    """
    sits = torch.stack([b.sits.float() for b in batch], dim=0)
    time = torch.stack([b.time.float() for b in batch], dim=0)
    label = torch.stack([b.label.float() for b in batch], dim=0)
    if batch[0].orbites is not None:
        orbites = torch.stack([b.orbites.int() for b in batch], dim=0)
    else:
        orbites = None
    if batch[0].mask_site is not None:
        mask_site = torch.stack([b.mask_site for b in batch])
    else:
        mask_site = None
    extracted_ids = [b.extracted_id for b in batch]
    return CDSupBatch(
        sits=sits,
        time=time,
        label=label,
        extracted_ids=extracted_ids,
        orbites=orbites,
        mask_site=mask_site,
    )


def collate_fn_brads1(batch: Sequence[SampleBradS1]) -> DeforBatch:
    """Hand-crafted function to create batch
    from my custom SampleSits dataclass
    :param batch:
    :returns:
    """
    sits = torch.nested.nested_tensor([b.sits for b in batch], layout=torch.jagged)
    time = torch.nested.nested_tensor([b.time for b in batch], layout=torch.jagged)
    label = torch.stack([b.label for b in batch])  # B,H,W
    idx_bef = torch.Tensor([b.idx_bef for b in batch]).int()
    idx_after = torch.Tensor([b.idx_after for b in batch]).int()
    batch_idx = torch.Tensor([i for i in range(len(idx_bef))]).int()
    return DeforBatch(
        sits=sits,
        time=time,
        label=label,
        idx_bef=idx_bef,
        idx_after=idx_after,
        batch_idx=batch_idx,
    )


def collate_fn_brads1_test(batch: Sequence[SampleBradS1]) -> DeforBatch:
    """Hand-crafted function to create batch
    from my custom SampleSits dataclass
    :param batch:
    :returns:
    """
    sits = torch.stack([b.sits for b in batch], dim=0)
    time = torch.stack([b.time for b in batch], dim=0)
    label = torch.stack([b.label for b in batch])  # B,H,W
    idx_bef = torch.Tensor([b.idx_bef for b in batch]).int()
    idx_after = torch.Tensor([b.idx_after for b in batch]).int()
    batch_idx = torch.Tensor([i for i in range(len(idx_bef))]).int()
    return DeforBatch(
        sits=sits,
        time=time,
        label=label,
        idx_bef=idx_bef,
        idx_after=idx_after,
        batch_idx=batch_idx,
    )


def collate_fn_mm_sp(batch: Sequence[SPMMSampleSits]) -> MMSpSupBatch:
    s2 = collate_fn_sp([b.s2 for b in batch])
    s1 = collate_fn_sp([b.s1 for b in batch])
    if batch[0].mask_site is not None:
        mask_site = torch.stack([b.mask_site for b in batch], dim=0)
    else:
        mask_site = None
    return MMSpSupBatch(s2=s2, s1=s1, mask_site=mask_site)


def collate_fn_mm_sp_test(batch: Sequence[SPMMSampleSits]) -> MMSpSupBatch:
    s2 = collate_fn_sp_test([b.s2 for b in batch])
    s1 = collate_fn_sp_test([b.s1 for b in batch])
    if batch[0].mask_site is not None:
        mask_site = torch.stack([b.mask_site for b in batch], dim=0)
    else:
        mask_site = None
    return MMSpSupBatch(s2=s2, s1=s1, mask_site=mask_site)


def collate_fn_s1_forc_sample(batch: Sequence[S1ForcSITSSample]) -> SITSBatch:
    content = torch.nested.nested_tensor(
        [b.content for b in batch], layout=torch.jagged
    )
    time = torch.nested.nested_tensor([b.time for b in batch], layout=torch.jagged)
    validity_mask = torch.nested.nested_tensor(
        [b.validity_mask for b in batch], layout=torch.jagged
    )
    orbits = torch.nested.nested_tensor([b.orbits for b in batch], layout=torch.jagged)
    if batch[0].weather_var is not None:
        weather_var = torch.nested.nested_tensor(
            [b.weather_var for b in batch], layout=torch.jagged
        )
    else:
        weather_var = None
    if batch[0].angles is not None:
        angles = torch.nested.nested_tensor(
            [b.angles for b in batch], layout=torch.jagged
        )
    else:
        angles = None
    return SITSBatch(
        content=content,
        time=time,
        validity_mask=validity_mask,
        mod="s1",
        orbites=orbits,
        weather_var=weather_var,
        angles=angles,
    )


def collate_fn_s2_forc_sample(batch: Sequence[S2ForcSITSSample]) -> SITSBatch:
    content = torch.nested.nested_tensor(
        [b.content for b in batch], layout=torch.jagged
    )
    time = torch.nested.nested_tensor([b.time for b in batch], layout=torch.jagged)
    validity_mask = torch.nested.nested_tensor(
        [b.validity_mask for b in batch], layout=torch.jagged
    )
    if batch[0].weather_var is not None:
        weather_var = torch.nested.nested_tensor(
            [b.weather_var for b in batch], layout=torch.jagged
        )
    else:
        weather_var = None
    if batch[0].angles is not None:
        angles = torch.nested.nested_tensor(
            [b.angles for b in batch], layout=torch.jagged
        )
    else:
        angles = None
    return SITSBatch(
        content=content,
        time=time,
        validity_mask=validity_mask,
        mod="s2",
        weather_var=weather_var,
        orbites=None,
        angles=angles,
    )


def collate_fn_s1_forc_sample_test(batch: Sequence[S1ForcSITSSample]) -> SITSBatch:
    content = torch.stack([b.content for b in batch], dim=0)
    time = torch.stack([b.time for b in batch], dim=0)
    validity_mask = torch.stack([b.validity_mask for b in batch], dim=0)
    orbits = torch.stack([b.orbits for b in batch], dim=0)
    if batch[0].weather_var is not None:
        weather_var = torch.stack([b.weather_var for b in batch], dim=0)
    else:
        weather_var = None
    if batch[0].angles is not None:
        angles = torch.stack([b.angles for b in batch], dim=0)
    else:
        angles = None
    return SITSBatch(
        content=content,
        time=time,
        validity_mask=validity_mask,
        mod="s1",
        orbites=orbits,
        angles=angles,
        weather_var=weather_var,
    )


def collate_fn_s2_forc_sample_test(batch: Sequence[S2ForcSITSSample]) -> SITSBatch:
    content = torch.stack([b.content for b in batch], dim=0)
    time = torch.stack([b.time for b in batch], dim=0)
    validity_mask = torch.stack([b.validity_mask for b in batch], dim=0)
    if batch[0].weather_var is not None:
        weather_var = torch.stack([b.weather_var for b in batch], dim=0)
    else:
        weather_var = None
    if batch[0].angles is not None:
        angles = torch.stack([b.angles for b in batch], dim=0)
    else:
        angles = None

    return SITSBatch(
        content=content,
        time=time,
        validity_mask=validity_mask,
        weather_var=weather_var,
        mod="s2",
        orbites=None,
        angles=angles,
    )


def collate_fn_mm_forc(batch: Sequence[MMForcSample]) -> MMForcBatch:
    s1 = collate_fn_s1_forc_sample([b.s1 for b in batch])
    s2 = collate_fn_s2_forc_sample([b.s2 for b in batch])
    return MMForcBatch(s1=s1, s2=s2)


def collate_fn_mm_forc_test(batch: Sequence[MMForcSample]) -> MMForcBatch:
    s1 = collate_fn_s1_forc_sample_test([b.s1 for b in batch])
    s2 = collate_fn_s2_forc_sample_test([b.s2 for b in batch])
    return MMForcBatch(s1=s1, s2=s2)
