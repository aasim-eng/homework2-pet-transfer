# train.py
# Training loop for Approach 1 (full fine-tuning) and Approach 2 (head only).
# The validation set is used to pick the best epoch (best val macro-F1).
# The test set is NOT touched here.
#
# Usage:
#     python -m src.train --config configs/full_finetune.yaml
#     python -m src.train --config configs/head_only.yaml
#     python -m src.train --config configs/head_only.yaml --smoke   # quick test

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.data import describe_eval_transform, get_datasets
from src.models import (build_model, count_parameters, parameter_checksum,
                        predict, set_train_mode, set_trainable,
                        trainable_layer_names)
from src.utils import (Logger, Timer, get_device, get_environment_info,
                       load_config, save_json, seed_worker, set_seed,
                       sync_device)


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    """DataLoader with a fixed random generator, so shuffling is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_optimizer(model, train_cfg):
    """AdamW over the trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    if train_cfg["optimizer"] != "adamw":
        raise ValueError("Only adamw is implemented")
    return torch.optim.AdamW(params, lr=train_cfg["lr"],
                             weight_decay=train_cfg["weight_decay"])


def build_scheduler(optimizer, train_cfg):
    """Cosine learning-rate decay from lr to 0 over all epochs."""
    if train_cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"])
    if train_cfg["scheduler"] in (None, "none"):
        return None
    raise ValueError("Unknown scheduler")


def train_one_epoch(model, loader, optimizer, loss_fn, device, approach):
    """One pass over the training data. Returns (mean loss, accuracy).

    The accuracy here is measured on the *augmented* training batches while
    the weights are changing, so it is a running estimate.
    """
    set_train_mode(model, approach)
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def plot_learning_curves(history, path, title):
    """Save accuracy and loss curves (train vs validation) as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_acc"], marker="o", label="train (augmented, running)")
    axes[0].plot(history["epoch"], history["val_acc"], marker="o", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title(title + " — accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title(title + " — loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Approach 1 or 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on a small subset, outputs go to */smoke/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    approach = cfg["approach"]
    train_cfg = cfg["training"]
    results_dir = cfg["output"]["results_dir"]
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    if args.smoke:
        train_cfg["epochs"] = 1
        results_dir = os.path.join("results", "smoke", approach)
        ckpt_dir = os.path.join("checkpoints", "smoke", approach)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = Logger(os.path.join(results_dir, "train_log.txt"))
    determinism = set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()
    env = get_environment_info()
    logger.log(f"Approach: {approach}   device: {device}")
    logger.log(f"Environment: {env}")
    logger.log(f"Determinism: {determinism}")

    # ---- data ----
    ds = get_datasets(cfg, train_augment=True)
    train_set, val_set = ds["train"], ds["val"]
    if args.smoke:
        train_set = Subset(train_set, list(range(256)))
        val_set = Subset(val_set, list(range(128)))
    nw = cfg["data"]["num_workers"]
    bs = train_cfg["batch_size"]
    train_loader = make_loader(train_set, bs, True, nw, cfg["seed"])
    val_loader = make_loader(val_set, bs, False, nw, cfg["seed"])
    logger.log(f"train images: {len(train_set)}, val images: {len(val_set)}")

    # ---- model ----
    model = build_model()
    set_trainable(model, approach)
    model.to(device)
    trainable, total = count_parameters(model)
    names = trainable_layer_names(model)
    logger.log(f"Final layer: {model.classifier[3]}")
    logger.log(f"Trainable params: {trainable:,} / total: {total:,}")
    if approach == "head_only":
        logger.log(f"Trainable tensors: {names}")
    features_checksum_before = parameter_checksum(model.features)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # ---- training loop ----
    history = []
    best = {"val_macro_f1": -1.0}
    best_state = None
    timer = Timer()
    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_timer = Timer()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, approach)
        val_loss, val_pred, val_true, _ = predict(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        if scheduler is not None:
            scheduler.step()
        sync_device(device)

        row = {
            "epoch": epoch, "lr": lr_now,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
            "epoch_time_s": epoch_timer.elapsed(),
        }
        history.append(row)
        logger.log(
            f"epoch {epoch:2d} | lr {lr_now:.2e} | train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
            f"macroF1 {val_f1:.4f} | {row['epoch_time_s']:.1f}s")

        # Model selection: strictly better val macro-F1 (ties keep the earlier epoch).
        if val_f1 > best["val_macro_f1"]:
            best = dict(row)
            best_state = copy.deepcopy(model.state_dict())

    sync_device(device)
    training_time = timer.elapsed()
    features_checksum_after = parameter_checksum(model.features)

    # ---- save outputs ----
    torch.save(best_state, os.path.join(ckpt_dir, "best.pt"))
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(results_dir, "history.csv"), index=False)
    plot_learning_curves(hist_df, os.path.join(results_dir, "learning_curves.png"),
                         approach)

    config_out = {
        "approach": approach,
        "config_file": args.config,
        "smoke_test": args.smoke,
        "model": "torchvision mobilenet_v3_small",
        "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1 (DEFAULT)",
        "final_layer": str(model.classifier[3]),
        "num_classes": model.classifier[3].out_features,
        "preprocessing_eval": describe_eval_transform(),
        "augmentation_train_only": cfg["augmentation"],
        "loss": train_cfg["loss"],
        "optimizer": train_cfg["optimizer"],
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "scheduler": train_cfg["scheduler"],
        "batch_size": bs,
        "epochs": train_cfg["epochs"],
        "model_selection_rule": "checkpoint (epoch) with the highest validation macro-F1; ties keep the earlier epoch",
        "selected_epoch": best["epoch"],
        "selected_val_acc": best["val_acc"],
        "selected_val_macro_f1": best["val_macro_f1"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_tensors": names if approach == "head_only" else f"all {len(names)} parameter tensors",
        "frozen_batchnorm_in_eval_mode": approach == "head_only",
        "features_checksum_before": features_checksum_before,
        "features_checksum_after": features_checksum_after,
        "features_changed_during_training": features_checksum_before != features_checksum_after,
        "training_time_s": training_time,
        "num_train_images": len(train_set),
        "num_val_images": len(val_set),
        "checkpoint": os.path.join(ckpt_dir, "best.pt"),
        "determinism": determinism,
        "environment": env,
    }
    save_json(config_out, os.path.join(results_dir, "config.json"))
    logger.log(f"Training time: {training_time:.1f}s")
    logger.log(f"Selected epoch {best['epoch']} (val macro-F1 {best['val_macro_f1']:.4f}, "
               f"val acc {best['val_acc']:.4f})")
    logger.log(f"model.features changed during training: "
               f"{config_out['features_changed_during_training']}")


if __name__ == "__main__":
    main()
