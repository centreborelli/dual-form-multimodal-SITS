"""File for the definition of the lightning datamodule class"""

from lightning import LightningDataModule

from cd_mm_sits.constant.dataset import SP_S2_BAND


class TemplateDataModule(LightningDataModule):
    """
    Basis class for lightning DataModule
    """

    def __init__(
        self,
        dataset_path: str,
        path_dir_csv: str,
        dataset_name="dataset",
        max_len_s2: int = 60,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        batch_size: int = 2,
        s2_band: None | list = None,
    ):
        super().__init__()
        self.path_dir_csv = path_dir_csv
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.max_len_s2 = max_len_s2
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if s2_band is None:
            s2_band = SP_S2_BAND
        self.s2_band = s2_band
