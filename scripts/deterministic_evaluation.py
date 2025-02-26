DECODER_APPROACHES = ["argmax", "local_soft_argmax", "weighted_spatial_mean"]


def perform_deterministic_eval(
    model,
    val_loader,
    test_loader,
    trainer,
):
    all_results = {
        "val": {decoder_name: [] for decoder_name in DECODER_APPROACHES},
        "test": {decoder_name: [] for decoder_name in DECODER_APPROACHES},
    }
    original_decoder_method = model.heatmap_decoder_method
    for loader, loader_name in zip([val_loader, test_loader], ["val", "test"]):
        for decoder_name in DECODER_APPROACHES:
            model.heatmap_decoder_method = decoder_name
            result = trainer.test(model, loader)
            all_results[loader_name][decoder_name] = result
    model.heatmap_decoder_method = original_decoder_method
    return all_results
