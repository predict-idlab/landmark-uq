import sys

sys.path.append("../training")
sys.path.append("..")
import sys
from copy import deepcopy
from glob import glob

import lightning as L
import pandas as pd
import torch
from datasets.mml import get_mml_dataset
from deterministic_evaluation import perform_deterministic_eval
from landmarker.losses import NLLLoss
from landmarker.models.utils import SoftmaxND
from landmarker.transforms.images import UseOnlyFirstChannel
from models.shr import StaticHeatmapRegression
from monai.networks.nets import FlexibleUNet
from monai.transforms import Compose, RandAffine, ScaleIntensityd
from one_hot_evaluation import perform_uncertainty_eval
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader

PATH_DATA_DIR = "../data"

if __name__ == "__main__":
    nb_landmarks = 14
    MC_inference_samples = 20
    spatial_dims = 3
    heatmap_size = (128, 128, 64)
    one_hot_model_name = "MML_shr_unet"
    one_hot_decoder_method = "local_soft_argmax"
    one_hot_heatmap_model = StaticHeatmapRegression(
        model=FlexibleUNet(
            in_channels=1,
            out_channels=14,
            backbone="efficientnet-b0",
            pretrained=True,
            decoder_channels=[128, 128, 128, 128, 128],
            spatial_dims=3,
            dropout=0.5,
            act="relu",
        ),
        loss=NLLLoss(spatial_dims=3),
        final_activation=SoftmaxND(spatial_dims=3),
        decoder_method=one_hot_decoder_method,
        spatial_dims=3,
        covariance_decoder_method="weighted_sample_covariance",
    )
    state_dict_one_hot = torch.load(
        "../results/lightning_model_checkpoints/MML/MML_shr_unet.ckpt", map_location="cpu"
    )["state_dict"]
    one_hot_heatmap_model.load_state_dict(state_dict_one_hot)
    one_hot_heatmap_model_mc = deepcopy(one_hot_heatmap_model)
    one_hot_heatmap_model_mc.MC_dropout = True
    one_hot_heatmap_model_mc.MC_inference_samples = MC_inference_samples

    ensemble_ckpts = glob("../results/lightning_model_checkpoints/MML/ensemble/*.ckpt")
    one_hot_heatmap_model_ensemble = StaticHeatmapRegression(
        model=FlexibleUNet(
            in_channels=1,
            out_channels=14,
            backbone="efficientnet-b0",
            pretrained=True,
            decoder_channels=[128, 128, 128, 128, 128],
            spatial_dims=3,
            dropout=0.5,
            act="relu",
        ),
        loss=NLLLoss(spatial_dims=3),
        final_activation=SoftmaxND(spatial_dims=3),
        decoder_method=one_hot_decoder_method,
        ensemble=True,
        ensemble_ckpts=ensemble_ckpts,
        spatial_dims=3,
        covariance_decoder_method="weighted_sample_covariance",
    )

    trainer = L.Trainer(accelerator="gpu")

    inference_transform = Compose([ScaleIntensityd(keys=["image"], dtype=None)])
    train_dataset, val_dataset, test_dataset = get_mml_dataset(
        PATH_DATA_DIR,
        train_extended=False,
        original_crop=True,
        benchmark=False,
        kind="mask",
        train_transform=inference_transform,
        inference_transform=inference_transform,
        store_imgs=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=5, shuffle=False)

    all_results = {}
    all_results["dataset"] = "MML"
    for i, (model, model_name) in enumerate(
        [
            (one_hot_heatmap_model, "one_hot"),
            (one_hot_heatmap_model_mc, "one_hot_mc"),
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
            for decoder_method in ["local_soft_argmax", "weighted_spatial_mean"]:
                model.heatmap_decoder_method = decoder_method
                df_results = perform_uncertainty_eval(
                    model=model,
                    model_name=model_name,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    trainer=trainer,
                    confidence_levels=[0.7, 0.8, 0.9, 0.95],
                    spatial_dims=spatial_dims,
                    heatmap_size=heatmap_size,
                )
                df_results["decoder_method"] = decoder_method
                df_results["model_filename"] = one_hot_model_name
                df_results["pixel_based_sampling"] = pixel_based_sampling
                save_path = f"../results/mml_one_hot_evaluation_{model_name}_{decoder_method}.csv"
                df_results.to_csv(save_path)
    # Save all results (pickle)
    all_results["model_filename"] = one_hot_model_name
    save_path = f"../results/mml_one_hot_evaluation_det_results.pkl"
    pd.to_pickle(all_results, save_path)
    print("All results saved")
