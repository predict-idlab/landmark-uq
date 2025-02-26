import os
from glob import glob

import lightning as L  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from landmarker.data.landmark_dataset import (  # type: ignore
    HeatmapDataset,
    LandmarkDataset,
    MaskDataset,
)
from torch.utils.data import DataLoader


def get_mml_data(
    path_data_dir,
    train_extended=False,
    original_crop=True,
    benchmark=False,
    adj_pixel_spacing=False,
):
    if train_extended:
        train_folder = "train_extended"
    else:
        train_folder = "train"
    if original_crop:
        if benchmark:
            crop_map = "MML_center_crop"
        else:
            crop_map = "MML_center_crop_shuffled"
    else:
        crop_map = "MML_filtered_crop"
    volume_train_paths = sorted(glob(f"{path_data_dir}/{crop_map}/{train_folder}/*_volume.npy"))
    volume_val_paths = sorted(glob(f"{path_data_dir}/{crop_map}/val/*_volume.npy"))
    volume_test_paths = sorted(glob(f"{path_data_dir}/{crop_map}/test/*_volume.npy"))
    landmarks_train_paths = sorted(glob(f"{path_data_dir}/{crop_map}/{train_folder}/*_label.npy"))
    landmarks_train = np.stack([np.load(path) for path in landmarks_train_paths])
    landmarks_val_paths = sorted(glob(f"{path_data_dir}/{crop_map}/val/*_label.npy"))
    landmarks_val = np.stack([np.load(path) for path in landmarks_val_paths])
    landmarks_test_paths = sorted(glob(f"{path_data_dir}/{crop_map}/test/*_label.npy"))
    landmarks_test = np.stack([np.load(path) for path in landmarks_test_paths])
    spacing_train_paths = sorted(glob(f"{path_data_dir}/{crop_map}/{train_folder}/*_spacing.npy"))
    pixel_spacings_train = np.stack([np.load(path) for path in spacing_train_paths])
    spacing_val_paths = sorted(glob(f"{path_data_dir}/{crop_map}/val/*_spacing.npy"))
    pixel_spacings_val = np.stack([np.load(path) for path in spacing_val_paths])
    spacing_test_paths = sorted(glob(f"{path_data_dir}/{crop_map}/test/*_spacing.npy"))
    pixel_spacings_test = np.stack([np.load(path) for path in spacing_test_paths])
    if original_crop and adj_pixel_spacing:
        scale_train_paths = sorted(glob(f"{path_data_dir}/{crop_map}/{train_folder}/*_scale.npy"))
        scale_val_paths = sorted(glob(f"{path_data_dir}/{crop_map}/val/*_scale.npy"))
        scale_test_paths = sorted(glob(f"{path_data_dir}/{crop_map}/test/*_scale.npy"))
        scale_train = np.stack([np.load(path) for path in scale_train_paths])
        scale_val = np.stack([np.load(path) for path in scale_val_paths])
        scale_test = np.stack([np.load(path) for path in scale_test_paths])
        pixel_spacings_train = pixel_spacings_train / scale_train
        pixel_spacings_val = pixel_spacings_val / scale_val
        pixel_spacings_test = pixel_spacings_test / scale_test
    return (
        volume_train_paths,
        landmarks_train,
        pixel_spacings_train,
        volume_val_paths,
        landmarks_val,
        pixel_spacings_val,
        volume_test_paths,
        landmarks_test,
        pixel_spacings_test,
    )


def get_mml_dataset(
    path_data_dir,
    train_extended=False,
    original_crop=True,
    benchmark=False,
    kind="landmark",
    train_transform=None,
    inference_transform=None,
    adj_pixel_spacing=False,
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
    (
        image_paths_train,
        landmarks_train,
        pixel_spacings_train,
        image_paths_val,
        landmarks_val,
        pixel_spacings_val,
        image_paths_test,
        landmarks_test,
        pixel_spacings_test,
    ) = get_mml_data(
        path_data_dir=path_data_dir,
        original_crop=original_crop,
        train_extended=train_extended,
        benchmark=benchmark,
        adj_pixel_spacing=adj_pixel_spacing,
    )
    return (
        datasetClass(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            spatial_dims=3,
            transform=train_transform,
            **kwargs,
        ),
        datasetClass(
            image_paths_val,
            landmarks_val,
            pixel_spacing=pixel_spacings_val,
            spatial_dims=3,
            transform=inference_transform,
            **kwargs,
        ),
        datasetClass(
            image_paths_test,
            landmarks_test,
            pixel_spacing=pixel_spacings_test,
            spatial_dims=3,
            transform=inference_transform,
            **kwargs,
        ),
    )


class MLLLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data/",
        train_extended: bool = False,
        original_crop: bool = True,
        benchmark: bool = True,
        batch_size: int = 4,
        kind="landmark",
        num_workers=0,
        train_transform=None,
        inference_transform=None,
        dim_img=[128, 128, 64],
        store_imgs=False,
        adj_pixel_spacing=False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_extended = train_extended
        self.original_crop = original_crop
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.kind = kind
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.inference_transform = inference_transform
        self.dim_img = dim_img
        self.store_imgs = store_imgs
        self.adj_pixel_spacing = adj_pixel_spacing
        if self.kind not in ["landmark", "heatmap", "mask"]:
            raise ValueError("kind must be one of 'landmark', 'heatmap', 'mask'")
        self.kwargs = kwargs

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.kind == "heatmap":
            self.kwargs["gamma"] = 1000
            self.kwargs["sigma"] = 5
        self.train, self.val, self.test = get_mml_dataset(
            path_data_dir=self.data_dir,
            train_extended=self.train_extended,
            original_crop=self.original_crop,
            benchmark=self.benchmark,
            kind=self.kind,
            train_transform=self.train_transform,
            inference_transform=self.inference_transform,
            dim_img=self.dim_img,
            store_imgs=self.store_imgs,
            adj_pixel_spacing=self.adj_pixel_spacing,
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