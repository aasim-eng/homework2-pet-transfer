# extract_features.py
# Approach 3, step 1: use the frozen pre-trained MobileNetV3-Small as a fixed
# feature extractor. For every image we take the output of
#     model.features -> model.avgpool -> flatten   (576 numbers)
# i.e. after the pooling step and before the classifier.
# No augmentation is used: train, val and test all use the weight transforms.
# Features are saved as .npy files for scikit-learn.
#
# Usage:
#     python -m src.extract_features --config configs/features.yaml
#     python -m src.extract_features --config configs/features.yaml --smoke

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data import describe_eval_transform, get_datasets
from src.models import build_model, count_parameters, extract_features, set_trainable
from src.utils import (Logger, Timer, get_device, get_environment_info,
                       load_config, save_json, set_seed, sync_device)


@torch.no_grad()
def extract_split(model, dataset, batch_size, num_workers, device):
    """Return (features, labels) as NumPy arrays for one dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats, labels = [], []
    for images, y in loader:
        f = extract_features(model, images.to(device))
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description="Extract frozen features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["extraction"]["features_dir"]
    results_dir = cfg["output"]["results_dir"]
    if args.smoke:
        out_dir = os.path.join("results", "smoke", "features", "arrays")
        results_dir = os.path.join("results", "smoke", "features")
    os.makedirs(out_dir, exist_ok=True)

    logger = Logger(os.path.join(results_dir, "extract_log.txt"))
    determinism = set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()
    logger.log(f"device: {device}")

    # Whole network frozen and in eval mode (BatchNorm uses ImageNet stats,
    # dropout is off). The 37-class head exists but is never used here.
    model = build_model()
    set_trainable(model, "features")
    model.to(device)
    model.eval()
    trainable, total = count_parameters(model)
    logger.log(f"Trainable params during extraction: {trainable} / {total:,}")

    # train_augment=False -> the training images also use the plain transform.
    ds = get_datasets(cfg, train_augment=False)
    splits = {"train": ds["train"], "val": ds["val"], "test": ds["test"]}
    if args.smoke:
        splits = {k: Subset(v, list(range(64))) for k, v in splits.items()}

    times = {}
    feature_dim = None
    for name, dataset in splits.items():
        timer = Timer()
        X, y = extract_split(model, dataset, cfg["extraction"]["batch_size"],
                             cfg["data"]["num_workers"], device)
        sync_device(device)
        times[name] = timer.elapsed()
        np.save(os.path.join(out_dir, f"X_{name}.npy"), X)
        np.save(os.path.join(out_dir, f"y_{name}.npy"), y)
        feature_dim = X.shape[1]
        logger.log(f"{name}: X {X.shape}, y {y.shape}, {times[name]:.1f}s")

    info = {
        "feature_layer": "model.features -> model.avgpool -> flatten (before classifier)",
        "feature_dim": int(feature_dim),
        "augmentation": "none (weight transforms for all splits)",
        "preprocessing": describe_eval_transform(),
        "extraction_time_s": times,
        "extraction_time_total_s": sum(times.values()),
        "trainable_params_cnn": trainable,
        "total_params_cnn": total,
        "num_images": {k: len(v) for k, v in splits.items()},
        "features_dir": out_dir,
        "determinism": determinism,
        "environment": get_environment_info(),
    }
    save_json(info, os.path.join(results_dir, "extraction_info.json"))
    logger.log(f"Feature size: {feature_dim}. Total extraction time: "
               f"{info['extraction_time_total_s']:.1f}s")


if __name__ == "__main__":
    main()
