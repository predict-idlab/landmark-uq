import sys

sys.path.append("../training")
sys.path.append("..")
import sys
from copy import deepcopy
from glob import glob

import lightning as L
import pandas as pd
import torch
from datasets.bcg import get_bcg_dataset
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
    nb_landmarks = 12
    MC_inference_samples = 20
    TTA_transforms = Compose(
        [
            RandAffine(
                prob=1,
                rotate_range=(-0.1, 0.1),
                translate_range=(-10, 10),
                scale_range=(-0.1, 0.1),
            )
        ]
    )
    one_hot_model_name = "BCG_cropped_shr_unet_NLLLoss_val_fold3"
    one_hot_decoder_method = "local_soft_argmax"
    one_hot_heatmap_model = StaticHeatmapRegression(
        model=Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=nb_landmarks,
            decoder_channels=[256, 128, 64, 32, 32],
        ),
        loss=NLLLoss(),
        final_activation=SoftmaxND(spatial_dims=2),
        decoder_method=one_hot_decoder_method,
    )

    state_dict_one_hot = torch.load(
        "../results/lightning_model_checkpoints/BCG_cropped/BCG_cropped_shr_unet_NLLLoss_val_fold3.ckpt",
        map_location="cpu",
    )["state_dict"]
    one_hot_heatmap_model.load_state_dict(state_dict_one_hot)

    one_hot_heatmap_model_tta = deepcopy(one_hot_heatmap_model)
    one_hot_heatmap_model_tta.TTA = True
    one_hot_heatmap_model_tta.TTA_transforms = TTA_transforms
    one_hot_heatmap_model_tta.MC_inference_samples = MC_inference_samples

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
            "../results/lightning_model_checkpoints/BCG_cropped/ensemble/shr_unet/*.ckpt"
        ),
    )

    trainer = L.Trainer(accelerator="gpu")

    inference_transform = Compose(
        [UseOnlyFirstChannel(keys=["image"]), ScaleIntensityd(keys=["image"], dtype=None)]
    )
    train_dataset, val_dataset, test_dataset = get_bcg_dataset(
        PATH_DATA_DIR,
        val_fold=3,
        cropped_images=True,
        kind="mask",
        dim_img=(512, 512),
        train_transform=inference_transform,
        inference_transform=inference_transform,
        store_imgs=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=5, shuffle=False)

    all_results = {}
    all_results["dataset"] = "BCG_cropped"
    all_results["val_fold"] = 3
    for i, (model, model_name) in enumerate(
        [
            (one_hot_heatmap_model, "one_hot"),
            (one_hot_heatmap_model_tta, "one_hot_tta"),
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
                    spatial_dims=2,
                )
                df_results["decoder_method"] = decoder_method
                df_results["model_filename"] = one_hot_model_name
                df_results["pixel_based_sampling"] = pixel_based_sampling
                save_path = (
                    f"../results/bcg_cropped_one_hot_evaluation_{model_name}_{decoder_method}.csv"
                )
                df_results.to_csv(save_path)
    # Save all results (pickle)
    all_results["model_filename"] = one_hot_model_name
    save_path = f"../results/bcg_cropped_one_hot_evaluation_det_results.pkl"
    pd.to_pickle(all_results, save_path)
    print("All results saved")
