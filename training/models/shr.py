from copy import deepcopy

import lightning as L  # type: ignore
import numpy as np
import segmentation_models_pytorch
import torch
from landmarker.heatmap.decoder import (  # type: ignore
    cov_from_gaussian_ls_scipy,
    heatmap_to_coord,
    weighted_sample_cov,
    windowed_weigthed_sample_cov,
)
from landmarker.losses import NLLLoss  # type: ignore
from landmarker.metrics import point_error  # type: ignore
from landmarker.utils import pixel_to_unit  # type: ignore
from tqdm import tqdm


class StaticHeatmapRegression(L.LightningModule):
    def __init__(
        self,
        model,
        decoder_method="local_soft_argmax",
        covariance_decoder_method="windowed_weighted_sample_covariance",
        loss=NLLLoss(spatial_dims=2),
        final_activation=None,
        lr=1e-6,
        spatial_dims=2,
        pixel_based_sampling=True,  # Pixel-based sampling
        MC_dropout=False,  # Monte Carlo Dropout
        MC_inference_samples=1,  # Number of samples for Monte Carlo Dropout or Test-time augmentation
        TTA=False,  # Test-time augmentation
        TTA_transforms=None,  # Must be a MONAI Compose transform object
        gamma_heatmap=100.0,
        ensemble=False,  # Ensemble of models (only for inference, traning must be done separately)
        ensemble_ckpts=None,  # List of models checkpoint paths for ensemble models
        temperature_scaling=False,
    ):
        super().__init__()
        self.ensemble = ensemble
        if not ensemble:
            self.model = model
        else:

            self.ensemble_models = torch.nn.ModuleList(
                [deepcopy(model) for i in range(len(ensemble_ckpts))]
            )
            for i, ckpt in enumerate(ensemble_ckpts):
                state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)[
                    "state_dict"
                ]
                model_state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
                self.ensemble_models[i].load_state_dict(model_state_dict)
        self.loss = loss
        self.lr = lr
        self.heatmap_decoder_method = decoder_method
        if final_activation is not None:
            self.final_activation = final_activation
        else:
            self.final_activation = torch.nn.Identity()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_step_pe = []
        self.test_step_outputs = []
        self.test_step_pe = []
        self.spatial_dims = spatial_dims
        self.pixel_based_sampling = pixel_based_sampling
        self.MC_dropout = MC_dropout
        self.MC_inference_samples = MC_inference_samples
        self.TTA = TTA
        self.TTA_transforms = TTA_transforms
        if covariance_decoder_method == "weighted_sample_covariance":
            self.covariance_decoder_method = lambda heatmap, coords: weighted_sample_cov(
                heatmap=heatmap, coords=coords, spatial_dims=self.spatial_dims
            )
        elif covariance_decoder_method == "windowed_weighted_sample_covariance":
            self.covariance_decoder_method = lambda heatmap, coords: windowed_weigthed_sample_cov(
                heatmap=heatmap, coords=coords, spatial_dims=self.spatial_dims
            )
        elif covariance_decoder_method == "gaussian_ls":
            self.covariance_decoder_method = lambda heatmap, coords: cov_from_gaussian_ls_scipy(
                heatmap=heatmap, coords=coords, gamma=gamma_heatmap, spatial_dims=self.spatial_dims
            )
        else:
            raise ValueError(
                f"Invalid covariance_decoder_method: {covariance_decoder_method}. "
                "Valid options are 'weighted_sample_covariance', 'windowed_weighted_sample_covariance' and 'gaussian_ls'."
            )
        self.temperature_scaling = temperature_scaling
        if self.temperature_scaling:
            self.temperature = 1.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
            ],
            lr=self.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=10, cooldown=5
                ),
                "monitor": "train_avg_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        pred_heatmap = self.model(batch["image"])
        loss = self.loss(pred_heatmap, batch["mask"])
        self.training_step_outputs.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.training_step_outputs).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        pred_heatmap = self.model(batch["image"])
        if self.temperature_scaling:
            pred_heatmap = pred_heatmap / self.temperature
        loss = self.loss(pred_heatmap, batch["mask"])
        pred_heatmap = self.final_activation(pred_heatmap)
        self.validation_step_outputs.append(loss.item())
        pred_landmarks = heatmap_to_coord(
            pred_heatmap, method=self.heatmap_decoder_method, spatial_dims=self.spatial_dims
        ).float()
        pe = point_error(
            true_landmarks=batch["landmark"],
            pred_landmarks=pred_landmarks,
            dim=batch["image"].shape[-self.spatial_dims :],
            dim_orig=batch["dim_original"],
            pixel_spacing=batch["spacing"],
            padding=batch["padding"],
            reduction="none",
        )
        self.validation_step_pe.append(pe)
        return loss.item()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.validation_step_outputs).mean()
        avg_pe = torch.cat(self.validation_step_pe, dim=0).nanmean(-1).mean()
        sdr_2mm = torch.cat(self.validation_step_pe, dim=0).lt(2).float().nanmean(-1).mean()
        self.log("val_avg_loss", avg_loss, prog_bar=True)
        self.log("val_avg_pe", avg_pe, prog_bar=True)
        self.log("val_sdr_2mm", sdr_2mm, prog_bar=True)
        self.validation_step_outputs.clear()
        self.validation_step_pe.clear()

    def test_step(self, batch, batch_idx):
        if self.TTA and self.MC_dropout:
            pred_heatmap, pred_heatmaps = self.perform_TTA_MC_dropout(batch["image"])
        elif self.MC_dropout:
            pred_heatmap, pred_heatmaps = self.perform_MC_dropout(batch["image"])
        elif self.TTA:
            pred_heatmap, pred_heatmaps = self.perform_TTA(batch["image"])
        elif self.ensemble:
            pred_heatmap, pred_heatmaps = self.perform_ensemble_inference(batch["image"])
        else:
            pred_heatmap = self.model(batch["image"])
        if self.temperature_scaling:
            pred_heatmap = pred_heatmap / self.temperature
        loss = self.loss(pred_heatmap, batch["mask"])
        self.test_step_outputs.append(loss.item())
        if not self.pixel_based_sampling:
            pred_heatmaps = self.final_activation(pred_heatmaps)
            pred_landmarks = (
                heatmap_to_coord(
                    pred_heatmaps.view(
                        (
                            pred_heatmaps.shape[0] * pred_heatmaps.shape[1],
                            *list(pred_heatmaps.shape[2:]),
                        )
                    ),
                    method=self.heatmap_decoder_method,
                    spatial_dims=self.spatial_dims,
                )
                .view((pred_heatmaps.shape[0], pred_heatmaps.shape[1], pred_heatmaps.shape[2], -1))
                .float()
            )
            pred_landmarks = pred_landmarks.mean(dim=1)
        else:
            pred_heatmap = self.final_activation(pred_heatmap)
            pred_landmarks = heatmap_to_coord(
                pred_heatmap, method=self.heatmap_decoder_method, spatial_dims=self.spatial_dims
            ).float()
        pe = point_error(
            true_landmarks=batch["landmark"],
            pred_landmarks=pred_landmarks,
            dim=batch["image"].shape[-self.spatial_dims :],
            dim_orig=batch["dim_original"],
            pixel_spacing=batch["spacing"],
            padding=batch["padding"],
            reduction="none",
        )
        self.test_step_pe.append(pe)
        return loss.item()

    def on_test_epoch_end(self):
        avg_loss = torch.tensor(self.test_step_outputs).mean()
        avg_pe = torch.cat(self.test_step_pe, dim=0).nanmean(-1).mean()
        sdr_2mm = torch.cat(self.test_step_pe, dim=0).lt(2).float().nanmean(-1).mean()
        sdr_2_5mm = torch.cat(self.test_step_pe, dim=0).lt(2.5).float().nanmean(-1).mean()
        sdr_3mm = torch.cat(self.test_step_pe, dim=0).lt(3).float().nanmean(-1).mean()
        sdr_4mm = torch.cat(self.test_step_pe, dim=0).lt(4).float().nanmean(-1).mean()
        self.log("test_avg_loss", avg_loss, prog_bar=True)
        self.log("test_avg_pe", avg_pe, prog_bar=True)
        self.log("test_sdr_2mm", sdr_2mm, prog_bar=True)
        self.log("test_sdr_2.5mm", sdr_2_5mm, prog_bar=True)
        self.log("test_sdr_3mm", sdr_3mm, prog_bar=True)
        self.log("test_sdr_4mm", sdr_4mm, prog_bar=True)
        self.test_step_outputs.clear()
        self.test_step_pe.clear()

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        batch_size = images.shape[0]
        if self.TTA or self.MC_dropout or self.ensemble:
            if self.ensemble:
                heatmap, heatmaps = self.perform_ensemble_inference(images)
            elif self.TTA and self.MC_dropout:
                heatmap, heatmaps = self.perform_TTA_MC_dropout(images)
            elif self.TTA:
                heatmap, heatmaps = self.perform_TTA(images)
            else:
                heatmap, heatmaps = self.perform_MC_dropout(images)
            if self.temperature_scaling:
                heatmap = heatmap / self.temperature
            heatmap = self.final_activation(heatmap)
            if not self.pixel_based_sampling:
                heatmaps = self.final_activation(heatmaps)
                landmarks_MC = (
                    heatmap_to_coord(
                        heatmaps.view((batch_size * heatmaps.shape[1], *list(heatmaps.shape[2:]))),
                        method=self.heatmap_decoder_method,
                        spatial_dims=self.spatial_dims,
                    )
                    .view((batch_size, heatmaps.shape[1], heatmaps.shape[2], -1))
                    .float()
                )
                landmarks_MC_orig_size = pixel_to_unit(
                    landmarks=landmarks_MC,
                    dim=batch["image"].shape[-self.spatial_dims :],
                    dim_orig=batch["dim_original"],
                    pixel_spacing=None,
                    padding=batch["padding"],
                )
                covariance_MC = self.landmark_cov_mc(landmarks_MC)
                covariance_MC_orig_size = self.landmark_cov_mc(landmarks_MC_orig_size)
                landmarks = landmarks_MC.mean(dim=1)
                landmarks_orig_size = landmarks_MC_orig_size.mean(dim=1)
        else:
            heatmap = self.model(images)
            if self.temperature_scaling:
                heatmap = heatmap / self.temperature
            heatmap = self.final_activation(heatmap)
        if self.pixel_based_sampling:
            landmarks = heatmap_to_coord(
                heatmap, method=self.heatmap_decoder_method, spatial_dims=self.spatial_dims
            ).float()
            landmarks_orig_size = pixel_to_unit(
                landmarks=landmarks,
                dim=batch["image"].shape[-self.spatial_dims :],
                dim_orig=batch["dim_original"],
                pixel_spacing=None,
                padding=batch["padding"],
            )
        covariance = self.covariance_decoder_method(heatmap, landmarks)
        heatmaps_orig_size = self.transform_heatmap_to_original_size(
            heatmap,
            padding=batch["padding"],
            original_dim=batch["dim_original"],
            spatial_dims=self.spatial_dims,
        )
        covariance_orig_size = self.covariance_decoder_method(
            heatmaps_orig_size, landmarks_orig_size
        )
        if not self.pixel_based_sampling:
            return {
                "landmarks": landmarks,
                "landmarks_orig_size": landmarks_orig_size,
                "landmarks_MC": landmarks_MC,
                "landmarks_MC_orig_size": landmarks_MC_orig_size,
                "covariance": covariance,
                "covariance_orig_size": covariance_orig_size,
                "covariance_MC": covariance_MC,
                "covariance_MC_orig_size": covariance_MC_orig_size,
                "heatmaps": heatmap,
                "original_dims": batch["dim_original"],
                "padding": batch["padding"],
            }
        return {
            "landmarks": landmarks,
            "landmarks_orig_size": landmarks_orig_size,
            "covariance": covariance,
            "covariance_orig_size": covariance_orig_size,
            "heatmaps": heatmap,
            "original_dims": batch["dim_original"],
            "padding": batch["padding"],
        }

    def perform_TTA(self, images):
        assert images.shape[0] == 1, "Batch size must be 1 for Test-time augmentation"
        heatmaps_t = [
            self.model(self.TTA_transforms(images)) for _ in range(self.MC_inference_samples)
        ]
        heatmaps = []
        # inverse transforms
        for i in range(self.MC_inference_samples):
            heatmap_tta = []
            for j in range(heatmaps_t[i].shape[1]):
                heatmap_tta.append(self.TTA_transforms.inverse(heatmaps_t[i][:, j].unsqueeze(1)))
            heatmaps.append(torch.concatenate(heatmap_tta, dim=1))
        heatmaps = torch.stack(heatmaps, dim=1)
        return heatmaps.mean(dim=1), heatmaps

    def perform_MC_dropout(self, images):
        self.model.train()
        heatmaps = torch.stack(
            [self.model(images) for _ in range(self.MC_inference_samples)],
            dim=1,
        )
        self.model.eval()
        return heatmaps.mean(dim=1), heatmaps

    def perform_TTA_MC_dropout(self, images):
        assert images.shape[0] == 1, "Batch size must be 1 for Test-time augmentation"
        self.model.train()
        heatmaps_t = [
            self.model(self.TTA_transforms(images)) for _ in range(self.MC_inference_samples)
        ]
        heatmaps = []
        # inverse transforms
        for i in range(self.MC_inference_samples):
            heatmap_tta = []
            for j in range(heatmaps_t[i].shape[1]):
                heatmap_tta.append(self.TTA_transforms.inverse(heatmaps_t[i][:, j].unsqueeze(1)))
            heatmaps.append(torch.concatenate(heatmap_tta, dim=1))
        heatmaps = torch.stack(heatmaps, dim=1)
        self.model.eval()
        return heatmaps.mean(dim=1), heatmaps

    def perform_ensemble_inference(self, images):
        heatmaps = torch.stack(
            [model(images) for model in self.ensemble_models],
            dim=1,
        )
        return heatmaps.mean(dim=1), heatmaps

    def transform_heatmap_to_original_size(self, heatmaps, padding, original_dim, spatial_dims):
        if spatial_dims == 2:
            return self.transform_heatmap_to_original_size_2d(heatmaps, padding, original_dim)
        elif spatial_dims == 3:
            return self.transform_heatmap_to_original_size_3d(heatmaps, padding, original_dim)
        else:
            raise ValueError(f"Invalid spatial_dims: {spatial_dims}. Valid options are 2 and 3.")

    def transform_heatmap_to_original_size_2d(self, heatmaps, padding, original_dim):
        heatmaps_orig_size = []
        for i in range(heatmaps.shape[0]):
            # resize image to original size
            resize_dim = original_dim[i] + 2 * padding[i]
            resize_dim = (int(resize_dim[0].item()), int(resize_dim[1].item()))
            heatmap = torch.nn.functional.interpolate(
                heatmaps[i].unsqueeze(0), size=resize_dim, mode="bilinear", align_corners=False
            )
            # crop image to original size
            heatmaps_orig_size.append(
                heatmap[
                    :,
                    :,
                    int(padding[i, 0]) : int(padding[i, 0] + original_dim[i, 0].item()),
                    int(padding[i, 1]) : int(padding[i, 1] + original_dim[i, 1].item()),
                ]
            )
        return torch.cat(heatmaps_orig_size, dim=0)

    def transform_heatmap_to_original_size_3d(self, heatmaps, padding, original_dim):
        heatmaps_orig_size = []
        for i in range(heatmaps.shape[0]):
            # resize image to original size
            resize_dim = original_dim[i] + 2 * padding[i]
            resize_dim = (
                int(resize_dim[0].item()),
                int(resize_dim[1].item()),
                int(resize_dim[2].item()),
            )
            heatmap = torch.nn.functional.interpolate(
                heatmaps[i].unsqueeze(0), size=resize_dim, mode="trilinear", align_corners=False
            )
            # crop image to original size
            heatmaps_orig_size.append(
                heatmap[
                    :,
                    :,
                    int(padding[i, 0]) : int(padding[i, 0] + original_dim[i, 0].item()),
                    int(padding[i, 1]) : int(padding[i, 1] + original_dim[i, 1].item()),
                    int(padding[i, 2]) : int(padding[i, 2] + original_dim[i, 2].item()),
                ]
            )
        return torch.cat(heatmaps_orig_size, dim=0)

    def landmark_cov_mc(self, landmark):
        B, MC_samples, C, D = landmark.shape
        mean = landmark.mean(dim=1).unsqueeze(1)
        diffs = mean - landmark  # (B, MC_samples, C, D)
        # swap MC_samples and landmarks
        diffs = diffs.reshape(B * MC_samples * C, D)
        prods = torch.bmm(diffs.unsqueeze(-1), diffs.unsqueeze(-2)).reshape(B, MC_samples, C, D, D)
        landmark_cov = prods.sum(dim=1) / (MC_samples - 1)
        return landmark_cov

    def set_temperature(self, cal_loader, batch_size=4):
        """Set the temperature of the model for the NLL loss."""
        if self.ensemble:
            for model in self.ensemble_models:
                model.train()
        else:
            self.model.train()

        pred_heatmaps = []
        true_heatmaps = []
        with torch.no_grad():
            print("Get logit heatmaps for temperature scaling ...")
            for batch in tqdm(cal_loader):
                if self.TTA and self.MC_dropout:
                    pred_heatmap, _ = self.perform_TTA_MC_dropout(batch["image"].to(self.device))
                elif self.MC_dropout:
                    pred_heatmap, _ = self.perform_MC_dropout(batch["image"].to(self.device))
                elif self.TTA:
                    pred_heatmap, _ = self.perform_TTA(batch["image"].to(self.device))
                elif self.ensemble:
                    pred_heatmap, _ = self.perform_ensemble_inference(
                        batch["image"].to(self.device)
                    )
                else:
                    pred_heatmap = self.model(batch["image"].to(self.device))
                pred_heatmaps.append(pred_heatmap)
                true_heatmaps.append(batch["mask"].to(self.device))
            pred_heatmaps = torch.cat(pred_heatmaps, dim=0)
            true_heatmaps = torch.cat(true_heatmaps, dim=0)

        # Create temperature as a new parameter explicitly
        temperature = torch.nn.Parameter(torch.ones(1, device=self.device) * 1.5)

        if self.spatial_dims == 2:
            optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

            def closure():
                optimizer.zero_grad()
                loss = self.loss(pred_heatmaps / temperature, true_heatmaps)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.temperature = temperature
        else:
            print("Optimizing temperature with Adam ...")
            optimizer = torch.optim.Adam([temperature], lr=0.001)
            for _ in tqdm(range(200)):
                for b in range(0, len(pred_heatmaps), batch_size):
                    optimizer.zero_grad()
                    loss = self.loss(
                        pred_heatmaps[b : b + batch_size] / temperature,
                        true_heatmaps[b : b + batch_size],
                    )
                    loss.backward()
                    optimizer.step()
            optimizer.zero_grad()
            self.temperature = temperature
        with torch.no_grad():
            if self.spatial_dims == 2:
                begin_loss = self.loss(pred_heatmaps, true_heatmaps).item()
                after_loss = self.loss(pred_heatmaps / self.temperature, true_heatmaps).item()
            else:
                begin_loss = []
                after_loss = []
                for b in range(0, len(pred_heatmaps), batch_size):
                    begin_loss.append(
                        self.loss(
                            pred_heatmaps[b : b + batch_size], true_heatmaps[b : b + batch_size]
                        ).item()
                    )
                    after_loss.append(
                        self.loss(
                            pred_heatmaps[b : b + batch_size] / self.temperature,
                            true_heatmaps[b : b + batch_size],
                        ).item()
                    )
                begin_loss = np.mean(begin_loss)
                after_loss = np.mean(after_loss)
            print(f"Temperature: 1, Loss: {begin_loss}")
            print(f"Temperature: {self.temperature.item()}, Loss: {after_loss}")
        if self.ensemble:
            for model in self.ensemble_models:
                model.eval()
        else:
            self.model.eval()
