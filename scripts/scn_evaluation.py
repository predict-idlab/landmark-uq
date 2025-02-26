import sys

import numpy as np
import pandas as pd

sys.path.append("..")

from src.uncertainty import (
    MR2C2R,
    MR2CCP,
    ConformalRegressorBonferroni,
    ConformalRegressorMahalanobis,
    ConformalRegressorMaxNonconformity,
    ContourHuggingRegressor,
    MultivariateNormalRegressor,
)


def perform_uncertainty_eval(
    model,
    model_name,
    val_loader,
    test_loader,
    val_dataset,
    test_dataset,
    trainer,
    confidence_levels=[0.7, 0.8, 0.9, 0.95],
    spatial_dims=2,
):
    df_results = pd.DataFrame(
        columns=["model", "confidence", "approach", "idx"]
        + [f"area_{i}" for i in range(val_dataset.nb_landmarks)]
        + [f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]
        + [f"error_{i}" for i in range(val_dataset.nb_landmarks)]
    )
    val_results = trainer.predict(model, val_loader)
    test_results = trainer.predict(model, test_loader)
    true_landmarks_val = val_dataset.landmarks.numpy()
    true_landmarks_test = test_dataset.landmarks.numpy()
    pred_landmarks_val = np.concatenate([result["landmarks_orig_size"] for result in val_results])
    pred_landmarks_test = np.concatenate([result["landmarks_orig_size"] for result in test_results])
    pred_heatmap_cov_val = np.concatenate(
        [result["covariance_orig_size"] for result in val_results]
    )
    pred_heatmap_cov_test = np.concatenate(
        [result["covariance_orig_size"] for result in test_results]
    )
    pred_heatmap_val = np.concatenate([result["heatmaps"] for result in val_results])
    pred_heatmap_val = pred_heatmap_val / np.sum(pred_heatmap_val, axis=(-2, -1), keepdims=True)
    pred_heatmap_test = np.concatenate([result["heatmaps"] for result in test_results])
    pred_heatmap_test = pred_heatmap_test / np.sum(pred_heatmap_test, axis=(-2, -1), keepdims=True)
    pred_padding_val = np.concatenate([result["padding"] for result in val_results])
    pred_padding_test = np.concatenate([result["padding"] for result in test_results])
    pred_original_dims_val = np.concatenate([result["original_dims"] for result in val_results])
    pred_original_dims_test = np.concatenate([result["original_dims"] for result in test_results])

    print("=" * 100)
    # Check if val_results contains key "covariance_MC_orig_size"
    if "covariance_MC_orig_size" in val_results[0].keys():
        pred_mc_cov_val = np.concatenate(
            [result["covariance_MC_orig_size"] for result in val_results]
        )
        pred_mc_cov_test = np.concatenate(
            [result["covariance_MC_orig_size"] for result in test_results]
        )
        print("Fit and evaluate Ellipsoid CP regions sampling cov")
        cp_mahalanobis_sampling_cov = ConformalRegressorMahalanobis(
            spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
        )
        cp_mahalanobis_sampling_cov.fit(
            pred=pred_landmarks_val,
            pred_cov=pred_mc_cov_val,
            target=true_landmarks_val,
        )
        for confidence_level in confidence_levels:
            df_results_temp = pd.DataFrame(columns=df_results.columns)
            df_results_temp["idx"] = np.arange(len(test_dataset))
            df_results_temp["approach"] = "cp_ellipsoid_sampling_cov"
            df_results_temp["confidence"] = confidence_level
            df_results_temp["model"] = model_name
            in_region, area, error = cp_mahalanobis_sampling_cov.evaluate(
                pred=pred_landmarks_test,
                pred_cov=pred_mc_cov_test,
                target=true_landmarks_test,
                spacing=test_dataset.pixel_spacings.numpy(),
                confidence=confidence_level,
                return_summary=False,
            )
            df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
            df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
            df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
            df_results = df_results_temp
        print("-" * 50)

        print("Fit and evaluate Bonferroni CP regions with sampling sigma")
        cp_bonferroni_sample_sigma = ConformalRegressorBonferroni(
            spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
        )
        cp_bonferroni_sample_sigma.fit(
            pred=pred_landmarks_val,
            sigmas=pred_mc_cov_val[..., [0, 1], [0, 1]],
            target=true_landmarks_val,
        )
        for confidence_level in confidence_levels:
            df_results_temp = pd.DataFrame(columns=df_results.columns)
            df_results_temp["idx"] = np.arange(len(test_dataset))
            df_results_temp["approach"] = "cp_bonferroni_sampling_sigma"
            df_results_temp["confidence"] = confidence_level
            df_results_temp["model"] = model_name
            in_region, area, error = cp_bonferroni_sample_sigma.evaluate(
                pred=pred_landmarks_test,
                sigmas=pred_mc_cov_test[..., [0, 1], [0, 1]],
                target=true_landmarks_test,
                spacing=test_dataset.pixel_spacings.numpy(),
                confidence=confidence_level,
                return_summary=False,
            )
            df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
            df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
            df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
            df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
        print("-" * 50)

        print("Fit and evaluate Max Nonconformity CP regions sample sigma")
        cp_max_sample_sigma = ConformalRegressorMaxNonconformity(
            spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
        )
        cp_max_sample_sigma.fit(
            pred=pred_landmarks_val,
            sigmas=pred_mc_cov_val[..., [0, 1], [0, 1]],
            target=true_landmarks_val,
        )
        for confidence_level in confidence_levels:
            df_results_temp = pd.DataFrame(columns=df_results.columns)
            df_results_temp["idx"] = np.arange(len(test_dataset))
            df_results_temp["approach"] = "cp_max_score_sampling_sigma"
            df_results_temp["confidence"] = confidence_level
            df_results_temp["model"] = model_name
            in_region, area, error = cp_max_sample_sigma.evaluate(
                pred=pred_landmarks_test,
                sigmas=pred_mc_cov_test[..., [0, 1], [0, 1]],
                target=true_landmarks_test,
                spacing=test_dataset.pixel_spacings.numpy(),
                confidence=confidence_level,
                return_summary=False,
            )
            df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
            df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
            df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
            df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
        print("-" * 50)

        print("Fit and evaluate Ellipsoid Normal regions sampling cov")
        mutli_normal_sampling_cov = MultivariateNormalRegressor(
            spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
        )
        mutli_normal_sampling_cov.fit(
            pred=pred_landmarks_val,
            pred_cov=pred_mc_cov_val,
            target=true_landmarks_val,
        )
        for confidence_level in confidence_levels:
            df_results_temp = pd.DataFrame(columns=df_results.columns)
            df_results_temp["idx"] = np.arange(len(test_dataset))
            df_results_temp["approach"] = "multi_normal_sampling_cov"
            df_results_temp["confidence"] = confidence_level
            df_results_temp["model"] = model_name
            in_region, area, error = mutli_normal_sampling_cov.evaluate(
                pred=pred_landmarks_test,
                pred_cov=pred_mc_cov_test,
                target=true_landmarks_test,
                spacing=test_dataset.pixel_spacings.numpy(),
                confidence=confidence_level,
                return_summary=False,
            )
            df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
            df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
            df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
            df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
        print("-" * 50)

    print("Fit and evaluate Ellipsoid CP regions heatmap cov")
    cp_mahalanobis_sampling_cov = ConformalRegressorMahalanobis(
        spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
    )
    cp_mahalanobis_sampling_cov.fit(
        pred=pred_landmarks_val,
        pred_cov=pred_heatmap_cov_val,
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "cp_ellipsoid_heatmap_cov"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = cp_mahalanobis_sampling_cov.evaluate(
            pred=pred_landmarks_test,
            pred_cov=pred_heatmap_cov_test,
            target=true_landmarks_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        if len(df_results) == 0:
            df_results = df_results_temp
        else:
            df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate Bonferroni CP regions heatmap derived sigma")
    cp_bonferroni_heatmap_sigma = ConformalRegressorBonferroni(
        spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
    )
    cp_bonferroni_heatmap_sigma.fit(
        pred=pred_landmarks_val,
        sigmas=pred_heatmap_cov_val[..., [0, 1], [0, 1]],
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "cp_bonferroni_heatmap_sigma"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = cp_bonferroni_heatmap_sigma.evaluate(
            pred=pred_landmarks_test,
            sigmas=pred_heatmap_cov_test[..., [0, 1], [0, 1]],
            target=true_landmarks_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate Max Nonconformity CP regions heatmap sigma")
    cp_max_heatmap_sigma = ConformalRegressorMaxNonconformity(
        spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
    )
    cp_max_heatmap_sigma.fit(
        pred=pred_landmarks_val,
        sigmas=pred_heatmap_cov_val[..., [0, 1], [0, 1]],
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "cp_max_score_heatmap_sigma"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = cp_max_heatmap_sigma.evaluate(
            pred=pred_landmarks_test,
            sigmas=pred_heatmap_cov_test[..., [0, 1], [0, 1]],
            target=true_landmarks_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate MR2CCP regions")
    mr2ccp = MR2CCP(spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks)
    mr2ccp.fit(
        heatmaps=pred_heatmap_val,
        original_dims=pred_original_dims_val,
        padding=pred_padding_val,
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "mr2ccp"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = mr2ccp.evaluate(
            heatmaps=pred_heatmap_test,
            pred_landmarks=pred_landmarks_test,
            target=true_landmarks_test,
            original_dims=pred_original_dims_test,
            padding=pred_padding_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate MR2C2R regions")
    mr2c2r = MR2C2R(spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks)
    mr2c2r.fit(
        heatmaps=pred_heatmap_val,
        original_dims=pred_original_dims_val,
        padding=pred_padding_val,
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "mr2c2r"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = mr2c2r.evaluate(
            heatmaps=pred_heatmap_test,
            pred_landmarks=pred_landmarks_test,
            target=true_landmarks_test,
            original_dims=pred_original_dims_test,
            padding=pred_padding_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate MR2C2R (APS) regions")
    for confidence_level in confidence_levels:
        mr2c2r_aps = MR2C2R(
            spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks, aps=True
        )
        mr2c2r_aps.fit(
            heatmaps=pred_heatmap_val,
            original_dims=pred_original_dims_val,
            padding=pred_padding_val,
            target=true_landmarks_val,
            confidence=confidence_level,
        )
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "mr2c2r_aps"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name

        in_region, area, error = mr2c2r_aps.evaluate(
            heatmaps=pred_heatmap_test,
            pred_landmarks=pred_landmarks_test,
            target=true_landmarks_test,
            original_dims=pred_original_dims_test,
            padding=pred_padding_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate Ellipsoid Normal regions heatmap cov")
    mutli_normal_heatmap_cov = MultivariateNormalRegressor(
        spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
    )
    mutli_normal_heatmap_cov.fit(
        pred=pred_landmarks_val,
        pred_cov=pred_heatmap_cov_val,
        target=true_landmarks_val,
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "multi_normal_heatmap_cov"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name

        in_region, area, error = mutli_normal_heatmap_cov.evaluate(
            pred=pred_landmarks_test,
            pred_cov=pred_heatmap_cov_test,
            target=true_landmarks_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)

    print("Fit and evaluate contour hugging regions")
    contour_hugging = ContourHuggingRegressor(
        spatial_dims=spatial_dims, nb_landmarks=val_dataset.nb_landmarks
    )
    for confidence_level in confidence_levels:
        df_results_temp = pd.DataFrame(columns=df_results.columns)
        df_results_temp["idx"] = np.arange(len(test_dataset))
        df_results_temp["approach"] = "contour_hugging"
        df_results_temp["confidence"] = confidence_level
        df_results_temp["model"] = model_name
        in_region, area, error = contour_hugging.evaluate(
            heatmaps=pred_heatmap_test,
            pred_landmarks=pred_landmarks_test,
            target=true_landmarks_test,
            original_dims=pred_original_dims_test,
            padding=pred_padding_test,
            spacing=test_dataset.pixel_spacings.numpy(),
            confidence=confidence_level,
            return_summary=False,
        )
        df_results_temp[[f"area_{i}" for i in range(val_dataset.nb_landmarks)]] = area
        df_results_temp[[f"in_region_{i}" for i in range(val_dataset.nb_landmarks)]] = in_region
        df_results_temp[[f"error_{i}" for i in range(val_dataset.nb_landmarks)]] = error
        df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
    print("-" * 50)
    return df_results
