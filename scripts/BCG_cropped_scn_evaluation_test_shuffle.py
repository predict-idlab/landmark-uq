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
from landmarker.heatmap import GaussianHeatmapGenerator
from landmarker.losses import GaussianHeatmapL2Loss
from landmarker.models import OriginalSpatialConfigurationNet
from landmarker.transforms.images import UseOnlyFirstChannel
from models.ahr import AdaptiveHeatmapRegression
from monai.transforms import Compose, RandAffine, ScaleIntensityd
from scn_evaluation import perform_uncertainty_eval
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
    scn_model_name = "BCG_cropped_ahr_SCN_test_shuffle"
    scn_decoder_method = "local_soft_argmax"
    scn_model = AdaptiveHeatmapRegression(
        model=OriginalSpatialConfigurationNet(
            in_channels=1,
            out_channels=nb_landmarks,
        ),
        heatmap_generator=GaussianHeatmapGenerator(
            nb_landmarks=nb_landmarks, sigmas=3, gamma=100, learnable=True
        ),
        loss=GaussianHeatmapL2Loss(alpha=5),
        decoder_method=scn_decoder_method,
        final_activation=torch.nn.ReLU(),
    )
    state_dict_scn = torch.load(
        "../results/lightning_model_checkpoints/BCG_cropped_test_shuffle/BCG_cropped_ahr_SCN_test_shuffle.ckpt",
        map_location="cpu",
    )["state_dict"]
    scn_model.load_state_dict(state_dict_scn)

    scn_model_mc = deepcopy(scn_model)
    scn_model_mc.MC_dropout = True
    scn_model_mc.MC_inference_samples = MC_inference_samples

    scn_model_tta = deepcopy(scn_model)
    scn_model_tta.TTA = True
    scn_model_tta.TTA_transforms = TTA_transforms
    scn_model_tta.MC_inference_samples = MC_inference_samples

    scn_model_mc_tta = deepcopy(scn_model_tta)
    scn_model_mc_tta.MC_dropout = True

    scn_model_ensemble = AdaptiveHeatmapRegression(
        model=OriginalSpatialConfigurationNet(
            in_channels=1,
            out_channels=nb_landmarks,
        ),
        heatmap_generator=GaussianHeatmapGenerator(
            nb_landmarks=nb_landmarks, sigmas=3, gamma=100, learnable=True
        ),
        loss=GaussianHeatmapL2Loss(alpha=5),
        decoder_method=scn_decoder_method,
        ensemble=True,
        ensemble_ckpts=glob(
            "../results/lightning_model_checkpoints/BCG_cropped_test_shuffle/ensemble/ahr_SCN/*.ckpt"
        ),
        final_activation=torch.nn.ReLU(),
    )

    trainer = L.Trainer(accelerator="gpu")

    inference_transform = Compose(
        [UseOnlyFirstChannel(keys=["image"]), ScaleIntensityd(keys=["image"], dtype=None)]
    )
    train_dataset, val_dataset, test_dataset = get_bcg_dataset(
        PATH_DATA_DIR,
        val_fold=3,
        kind="landmark",
        cropped_images=True,
        dim_img=(512, 512),
        train_transform=inference_transform,
        inference_transform=inference_transform,
        store_imgs=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=5, shuffle=False)

    all_results = {}
    all_results["dataset"] = "BCG_cropped_test_shuffle"
    all_results["val_fold"] = 3

    for i, (model, model_name) in enumerate(
        [
            (scn_model, "scn"),
            (scn_model_mc, "scn_mc"),
            (scn_model_tta, "scn_tta"),
            (scn_model_mc_tta, "scn_mc_tta"),
            (scn_model_ensemble, "scn_ensemble"),
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

            for decoder_method in ["local_soft_argmax"]:
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
                df_results["model_filename"] = scn_model_name
                df_results["pixel_based_sampling"] = pixel_based_sampling
                save_path = f"../results/bcg_cropped_test_shuffle_scn_evaluation_{model_name}_{decoder_method}.csv"
                df_results.to_csv(save_path)
    # Save all results (pickle)
    all_results["model_filename"] = scn_model_name
    save_path = f"../results/bcg_cropped_test_shuffle_scn_evaluation_det_results.pkl"
    pd.to_pickle(all_results, save_path)
    print("All results saved")
