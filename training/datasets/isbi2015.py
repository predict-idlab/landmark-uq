import lightning as L  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from landmarker.data.landmark_dataset import (  # type: ignore
    HeatmapDataset,
    LandmarkDataset,
    MaskDataset,
)
from landmarker.datasets.cepha400 import get_cepha_dataset
from torch.utils.data import DataLoader


def get_isbi2015_dataset(
    path_data_dir,
    bennchmark=False,
    val_fold=2,
    test_fold=3,
    kind="landmark",
    train_transform=None,
    inference_transform=None,
    **kwargs,
):

    if kind == "landmark":
        datasetClass = LandmarkDataset
    elif kind == "heatmap":
        datasetClass = HeatmapDataset
    elif kind == "mask":
        datasetClass = MaskDataset
    else:
        raise ValueError("type must be one of 'landmark', 'heatmap', 'mask'")
    if bennchmark:
        (
            image_paths_train,
            image_paths_val,
            image_paths_test,
            landmarks_train,
            landmarks_val,
            landmarks_test,
            pixel_spacings_train,
            pixel_spacings_val,
            pixel_spacings_test,
        ) = get_cepha_dataset(path_dir=path_data_dir, junior=False, cv=False)
    else:
        assert val_fold in [0, 1, 2, 3]
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir=path_data_dir, junior=True, cv=True)

        image_paths = [image_paths_fold1, image_paths_fold2, image_paths_fold3, image_paths_fold4]
        landmarks = [landmarks_fold1, landmarks_fold2, landmarks_fold3, landmarks_fold4]
        pixel_spacings = [
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ]

        image_paths_train = np.concatenate(
            [image_paths[i] for i in range(4) if i != val_fold and i != test_fold]
        ).tolist()
        landmarks_train = np.concatenate(
            [landmarks[i] for i in range(4) if i != val_fold and i != test_fold]
        )
        pixel_spacings_train = np.concatenate(
            [pixel_spacings[i] for i in range(4) if i != val_fold and i != test_fold]
        )

        image_paths_val = image_paths[val_fold]
        landmarks_val = landmarks[val_fold]
        pixel_spacings_val = pixel_spacings[val_fold]

        image_paths_test = image_paths[test_fold]
        landmarks_test = landmarks[test_fold]
        pixel_spacings_test = pixel_spacings[test_fold]

    return (
        datasetClass(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            **kwargs,
        ),
        datasetClass(
            image_paths_val,
            landmarks_val,
            pixel_spacing=pixel_spacings_val,
            transform=inference_transform,
            **kwargs,
        ),
        datasetClass(
            image_paths_test,
            landmarks_test,
            pixel_spacing=pixel_spacings_test,
            transform=inference_transform,
            **kwargs,
        ),
    )


class ISBI2015LightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data/",
        batch_size: int = 32,
        kind="landmark",
        num_workers=0,
        val_fold=2,
        test_fold=3,
        dim_img=(512, 512),
        benchmark=False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.kind = kind
        self.num_workers = num_workers
        if self.kind not in ["landmark", "heatmap", "mask"]:
            raise ValueError("kind must be one of 'landmark', 'heatmap', 'mask'")
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.dim_img = dim_img
        self.benchmark = benchmark
        self.kwargs = kwargs

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train, self.val, self.test = get_isbi2015_dataset(
            self.data_dir,
            bennchmark=self.benchmark,
            val_fold=self.val_fold,
            test_fold=self.test_fold,
            kind=self.kind,
            dim_img=self.dim_img,
            **self.kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        pass

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass
