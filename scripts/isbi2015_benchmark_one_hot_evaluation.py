import sys

sys.path.append("../training")
sys.path.append("..")
import sys
from copy import deepcopy
from glob import glob

import lightning as L
import pandas as pd
import torch
from datasets.isbi2015 import get_isbi2015_dataset
from deterministic_evaluation import perform_deterministic_eval
from landmarker.losses import NLLLoss
from landmarker.models.utils import SoftmaxND
from landmarker.transforms.images import UseOnlyFirstChannel
from models.shr import StaticHeatmapRegression
from monai.transforms import Compose, RandAffine, ScaleIntensityd
from one_hot_evaluation import perform_uncertainty_eval
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader

PATH_DATA_DIR = "../data"

if __name__ == "__main__":
    nb_landmarks = 19

    one_hot_model_name = "ISBI2015_shr_unet_NLLLoss_benchmark"
    one_hot_decoder_method = "local_soft_argmax"

    one_hot_heatmap_model_ensemble = StaticHeatmapRegression(
        model=Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=nb_landmarks,
            decoder_channels=[256, 128, 64, 32, 32],
        ),
        loss=NLLLoss(),
        final_activation=SoftmaxND(spatial_dims=2),
        ensemble=True,
        decoder_method=one_hot_decoder_method,
        ensemble_ckpts=glob(
            "../results/lightning_model_checkpoints/ISBI2015_benchmark/ensemble/shr_unet/*.ckpt"
        ),
    )

    trainer = L.Trainer(accelerator="gpu")

    inference_transform = Compose(
        [UseOnlyFirstChannel(keys=["image"]), ScaleIntensityd(keys=["image"], dtype=None)]
    )
    train_dataset, val_dataset, test_dataset = get_isbi2015_dataset(
        PATH_DATA_DIR,
        val_fold=2,
        test_fold=3,
        kind="mask",
        dim_img=(512, 512),
        train_transform=inference_transform,
        inference_transform=inference_transform,
        store_imgs=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=5, shuffle=False)

    all_results = {}
    all_results["dataset"] = "ISBI2015_benchmark"
    for i, (model, model_name) in enumerate(
        [
            (one_hot_heatmap_model_ensemble, "one_hot_ensemble"),
        ]
    ):
        # if model_name contains tta or ensemble or mc, we will perform pixel based sampling and non pixel based sampling
        if model_name.split("_")[-1] in ["tta", "ensemble", "mc"]:
            pixel_based_sampling_options = [True, False]
        else:
            pixel_based_sampling_options = [True]
        for pixel_based_sampling in pixel_based_sampling_options:
            if not pixel_based_sampling:
                model_name = model_name + "_landmark_sampling"
            model.pixel_based_sampling = pixel_based_sampling
            all_results[model_name] = {}
            all_results[model_name]["det_perfomance"] = perform_deterministic_eval(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                trainer=trainer,
            )
            #
    # Save all results (pickle)
    all_results["model_filename"] = one_hot_model_name
    save_path = f"../results/isbi2015_benchmark_one_hot_evaluation_det_results.pkl"
    pd.to_pickle(all_results, save_path)
    print("All results saved")
