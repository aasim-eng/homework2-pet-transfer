# data.py
# Everything about the data:
#   - loading Oxford-IIIT Pet with the 37 breed labels ("category")
#   - the stratified train/validation split of the official `trainval` split,
#     saved to splits/*.json so it can be rebuilt exactly
#   - the transforms (plain weight transforms for val/test, augmentation for train)
#   - per-breed image counts (CSV + Markdown table)
#
# Run directly to create the split and the class-count tables:
#     python -m src.data --config configs/full_finetune.yaml

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

from src.utils import load_config, load_json, save_json


WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_eval_transform():
    """The official preprocessing that belongs to the pre-trained weights.

    For MobileNet_V3_Small_Weights.IMAGENET1K_V1 this is:
    resize to 256 (bilinear) -> center crop 224 -> scale to [0, 1]
    -> normalize with ImageNet mean/std.
    Used for validation, test and feature extraction (no augmentation).
    """
    return WEIGHTS.transforms()


def describe_eval_transform():
    """Return the resize, crop, mean and std of the weight transforms."""
    t = get_eval_transform()
    return {
        "weights": str(WEIGHTS),
        "resize_size": list(t.resize_size),
        "crop_size": list(t.crop_size),
        "mean": list(t.mean),
        "std": list(t.std),
        "interpolation": str(t.interpolation),
    }


def get_train_transform(aug_cfg):
    """Training transform = light random augmentation + same normalization.

    The final image size (224) and the mean/std are taken from the weight
    transforms, so train and eval inputs have the same scale and statistics.
    """
    info = describe_eval_transform()
    crop = info["crop_size"][0]
    scale = tuple(aug_cfg["random_resized_crop_scale"])
    jitter = aug_cfg["color_jitter"]
    return T.Compose([
        T.RandomResizedCrop(crop, scale=scale, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]),
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
        T.ToTensor(),
        T.Normalize(mean=info["mean"], std=info["std"]),
    ])


def get_display_transform():
    """Resize + center crop only (no normalization), for showing images."""
    info = describe_eval_transform()
    return T.Compose([
        T.Resize(info["resize_size"][0]),
        T.CenterCrop(info["crop_size"][0]),
    ])


# ---------------------------------------------------------------------------
# Dataset and split
# ---------------------------------------------------------------------------

def load_pet(root, split, transform=None, download=True):
    """Load one official split ('trainval' or 'test') with 37 breed labels."""
    return OxfordIIITPet(
        root=root,
        split=split,
        target_types="category",
        transform=transform,
        download=download,
    )


def get_labels(dataset):
    """Return the breed label of every image, without opening the images.

    torchvision stores the labels in the attribute `_labels`. We check below
    (in `main`) that these match the labels returned by dataset[i].
    """
    return np.array(dataset._labels)


def make_or_load_split(data_cfg, seed):
    """Create (or reload) the stratified train/val split of `trainval`.

    The split is saved to:
        splits/train_idx.json, splits/val_idx.json, splits/split_info.json
    If these files already exist they are loaded, so every script and every
    approach uses exactly the same split.
    """
    splits_dir = data_cfg["splits_dir"]
    train_path = os.path.join(splits_dir, "train_idx.json")
    val_path = os.path.join(splits_dir, "val_idx.json")
    info_path = os.path.join(splits_dir, "split_info.json")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_idx = load_json(train_path)
        val_idx = load_json(val_path)
        info = load_json(info_path)
        if info["seed"] != seed or info["val_fraction"] != data_cfg["val_fraction"]:
            raise ValueError(
                "Saved split was made with different settings. "
                "Delete the splits/ files to rebuild it."
            )
        return train_idx, val_idx

    trainval = load_pet(data_cfg["root"], "trainval")
    labels = get_labels(trainval)
    all_idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=data_cfg["val_fraction"],
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    train_idx = sorted(int(i) for i in train_idx)
    val_idx = sorted(int(i) for i in val_idx)

    save_json(train_idx, train_path)
    save_json(val_idx, val_path)
    save_json({
        "seed": seed,
        "val_fraction": data_cfg["val_fraction"],
        "method": "sklearn.model_selection.train_test_split(stratify=breed labels)",
        "source_split": "OxfordIIITPet official 'trainval'",
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "indices_refer_to": "positions in OxfordIIITPet(split='trainval') in torchvision order",
    }, info_path)
    return train_idx, val_idx


def get_datasets(cfg, train_augment=True):
    """Build the datasets used for training and evaluation.

    Returns a dict with:
        train       - training subset (with augmentation if train_augment)
        train_eval  - the same training images with plain eval transform
                      (used to measure clean training accuracy)
        val         - validation subset (plain eval transform)
        test        - official test split (plain eval transform)
        classes     - list of 37 breed names
    """
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    train_idx, val_idx = make_or_load_split(data_cfg, cfg["seed"])

    eval_tf = get_eval_transform()
    if train_augment:
        train_tf = get_train_transform(cfg["augmentation"])
    else:
        train_tf = eval_tf

    # Two copies of `trainval` with different transforms. Both are indexed
    # with the same saved indices, so they contain the same images.
    trainval_train_tf = load_pet(root, "trainval", transform=train_tf)
    trainval_eval_tf = load_pet(root, "trainval", transform=eval_tf)
    test = load_pet(root, "test", transform=eval_tf)

    return {
        "train": Subset(trainval_train_tf, train_idx),
        "train_eval": Subset(trainval_eval_tf, train_idx),
        "val": Subset(trainval_eval_tf, val_idx),
        "test": test,
        "classes": list(test.classes),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def get_split_labels(cfg):
    """Return label arrays for train, val and test (no images loaded)."""
    train_idx, val_idx = make_or_load_split(cfg["data"], cfg["seed"])
    trainval = load_pet(cfg["data"]["root"], "trainval")
    test = load_pet(cfg["data"]["root"], "test")
    tv_labels = get_labels(trainval)
    return {
        "train": tv_labels[train_idx],
        "val": tv_labels[val_idx],
        "test": get_labels(test),
        "classes": list(test.classes),
    }


# ---------------------------------------------------------------------------
# Class counts
# ---------------------------------------------------------------------------

def class_count_table(cfg, out_dir="results/data"):
    """Count images per breed in train / val / test and save CSV + Markdown."""
    labels = get_split_labels(cfg)
    classes = labels["classes"]
    rows = []
    for c, name in enumerate(classes):
        rows.append({
            "class_id": c,
            "breed": name,
            "train": int((labels["train"] == c).sum()),
            "val": int((labels["val"] == c).sum()),
            "test": int((labels["test"] == c).sum()),
        })
    df = pd.DataFrame(rows)
    df["total"] = df["train"] + df["val"] + df["test"]

    total_row = pd.DataFrame([{
        "class_id": "", "breed": "TOTAL",
        "train": df["train"].sum(), "val": df["val"].sum(),
        "test": df["test"].sum(), "total": df["total"].sum(),
    }])
    df_out = pd.concat([df, total_row], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(os.path.join(out_dir, "class_counts.csv"), index=False)
    with open(os.path.join(out_dir, "class_counts.md"), "w") as f:
        f.write(df_out.to_markdown(index=False))
        f.write("\n")

    # Simple imbalance summary.
    imbalance = {}
    for split in ["train", "val", "test"]:
        counts = df[split]
        imbalance[split] = {
            "min": int(counts.min()),
            "min_breed": df.loc[counts.idxmin(), "breed"],
            "max": int(counts.max()),
            "max_breed": df.loc[counts.idxmax(), "breed"],
            "mean": round(float(counts.mean()), 2),
            "max_over_min_ratio": round(float(counts.max() / counts.min()), 3),
        }
    save_json(imbalance, os.path.join(out_dir, "class_imbalance.json"))
    return df_out, imbalance


def main():
    parser = argparse.ArgumentParser(description="Create split and class counts.")
    parser.add_argument("--config", default="configs/full_finetune.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("Weight transforms:", describe_eval_transform())
    save_json(describe_eval_transform(), "results/data/preprocessing.json")

    train_idx, val_idx = make_or_load_split(cfg["data"], cfg["seed"])
    print(f"train: {len(train_idx)} images, val: {len(val_idx)} images")
    overlap = set(train_idx) & set(val_idx)
    print("overlap between train and val indices:", len(overlap))

    # Check that the fast `_labels` lookup matches the real dataset items.
    trainval = load_pet(cfg["data"]["root"], "trainval")
    labels = get_labels(trainval)
    for i in [0, 100, 1000, len(trainval) - 1]:
        _, y = trainval[i]
        assert y == labels[i], "label mismatch"
    print("label check passed; trainval size:", len(trainval))

    df, imbalance = class_count_table(cfg)
    print(df.to_string(index=False))
    print("Imbalance summary:", imbalance)


if __name__ == "__main__":
    main()
