# evaluate.py
# Final evaluation of every SELECTED model on the test set (once per model).
# Model selection already happened on the validation set (train.py and
# train_sklearn.py); here nothing is tuned or chosen.
#
# For each model it saves (in its results folder):
#   test_metrics.json        accuracy, macro-F1, bootstrap 95% CIs, train/val acc
#   per_class_report.csv     precision / recall / F1 per breed
#   confusion_matrix.png     counts (37x37, breed names)
#   confusion_matrix_normalized.png   rows sum to 1 (recall per breed)
#   confusion_matrix.csv
#   top_confused_pairs.csv   10 most confused breed pairs
#   test_predictions.csv     one row per test image
#   examples_correct_incorrect.png    grid of example predictions
# Every evaluation is also appended to results/test_access_log.csv.
#
# Usage:
#     python -m src.evaluate --all
#     python -m src.evaluate --approach head_only

import argparse
import datetime
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from src.data import get_datasets, get_display_transform, load_pet
from src.models import build_model, predict
from src.utils import (get_device, load_config, load_json, save_json,
                       set_seed)

# name -> (kind, config file, results folder)
APPROACHES = {
    "full_finetune": ("cnn", "configs/full_finetune.yaml", "results/full_finetune"),
    "head_only": ("cnn", "configs/head_only.yaml", "results/head_only"),
    "features_logreg": ("sklearn", "configs/features.yaml", "results/features/logreg"),
    "features_linear_svm": ("sklearn", "configs/features.yaml", "results/features/linear_svm"),
    "features_random_forest": ("sklearn", "configs/features.yaml", "results/features/random_forest"),
}

N_BOOTSTRAP = 2000


def predictions_cnn(cfg, results_dir):
    """Load the selected checkpoint and predict on train (clean), val, test."""
    device = get_device()
    run_cfg = load_json(os.path.join(results_dir, "config.json"))
    model = build_model()
    state = torch.load(run_cfg["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    ds = get_datasets(cfg, train_augment=False)
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    out = {}
    for split, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        loader = DataLoader(ds[key], batch_size=bs, shuffle=False, num_workers=nw)
        _, pred, true, _ = predict(model, loader, device)
        out[split] = (true, pred)
    return out, ds["classes"]


def predictions_sklearn(cfg, results_dir):
    """Load the selected sklearn pipeline and predict on saved features."""
    pipe = joblib.load(os.path.join(results_dir, "model.joblib"))
    feat_dir = cfg["extraction"]["features_dir"]
    out = {}
    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(feat_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(feat_dir, f"y_{split}.npy"))
        out[split] = (y, pipe.predict(X))
    classes = list(load_pet(cfg["data"]["root"], "test").classes)
    return out, classes


def bootstrap_ci(y_true, y_pred, seed, n=N_BOOTSTRAP):
    """95% bootstrap confidence intervals for accuracy and macro-F1.

    We resample the test images with replacement `n` times and recompute
    the metric each time; the 2.5% and 97.5% percentiles form the interval.
    This shows how much the score could move with a different test sample.
    """
    rng = np.random.default_rng(seed)
    size = len(y_true)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return {
        "n_bootstrap": n,
        "test_acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "test_macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def plot_confusion(cm, classes, path, title, normalized):
    """Draw a large, readable 37x37 confusion matrix with breed names."""
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1.0 if normalized else cm.max())
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted breed", fontsize=12)
    ax.set_ylabel("True breed", fontsize=12)
    ax.set_title(title, fontsize=14)
    threshold = (1.0 if normalized else cm.max()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value == 0:
                continue
            text = f"{value:.2f}" if normalized else str(int(value))
            ax.text(j, i, text, ha="center", va="center", fontsize=5,
                    color="white" if value > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confused_pairs(cm, classes, k=10):
    """The k breed pairs that are mixed up most often (both directions)."""
    rows = []
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            total = int(cm[i, j] + cm[j, i])
            if total > 0:
                rows.append({
                    "breed_a": classes[i], "breed_b": classes[j],
                    "total_confusions": total,
                    "a_predicted_as_b": int(cm[i, j]),
                    "b_predicted_as_a": int(cm[j, i]),
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(["total_confusions", "breed_a"], ascending=[False, True])
    return df.head(k).reset_index(drop=True)


def example_grid(cfg, y_true, y_pred, classes, path, title, seed, per_group=8):
    """Show `per_group` correct and `per_group` incorrect test predictions."""
    rng = np.random.default_rng(seed)
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    pick_c = rng.choice(correct, size=min(per_group, len(correct)), replace=False)
    pick_w = rng.choice(wrong, size=min(per_group, len(wrong)), replace=False)

    test_display = load_pet(cfg["data"]["root"], "test", transform=get_display_transform())
    cols = per_group
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 6.4))
    for row, (picks, label) in enumerate([(pick_c, "Correct"), (pick_w, "Incorrect")]):
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks):
                continue
            i = int(picks[col])
            img, _ = test_display[i]
            ax.imshow(img)
            color = "green" if label == "Correct" else "red"
            ax.set_title(f"True: {classes[y_true[i]]}\nPred: {classes[y_pred[i]]}",
                         fontsize=7, color=color)
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=12)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def log_test_access(name, n_test):
    """Append one line to results/test_access_log.csv (proof of test usage)."""
    path = "results/test_access_log.csv"
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("timestamp,approach,num_test_images,purpose\n")
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        f.write(f"{stamp},{name},{n_test},final evaluation of selected model\n")


def evaluate(name):
    """Evaluate one selected model on the test set and save all outputs."""
    kind, cfg_path, results_dir = APPROACHES[name]
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"], cfg["deterministic"])
    if kind == "cnn":
        preds, classes = predictions_cnn(cfg, results_dir)
    else:
        preds, classes = predictions_sklearn(cfg, results_dir)

    y_test, p_test = preds["test"]
    log_test_access(name, len(y_test))
    train_acc = accuracy_score(*preds["train"])
    val_acc = accuracy_score(*preds["val"])
    test_acc = accuracy_score(y_test, p_test)
    test_f1 = f1_score(y_test, p_test, average="macro")

    metrics = {
        "approach": name,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "train_acc_clean": train_acc,
        "val_acc": val_acc,
        "val_macro_f1": f1_score(*preds["val"], average="macro"),
        "generalization_gap_train_minus_test": train_acc - test_acc,
        "num_test_images": int(len(y_test)),
        "num_test_errors": int((y_test != p_test).sum()),
        "train_acc_note": "measured on training images with the plain eval transform (no augmentation), using the selected model",
    }
    metrics.update(bootstrap_ci(y_test, p_test, cfg["seed"]))
    save_json(metrics, os.path.join(results_dir, "test_metrics.json"))

    report = classification_report(y_test, p_test, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True,
                                   zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(results_dir, "per_class_report.csv"))

    cm = confusion_matrix(y_test, p_test, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(results_dir, "confusion_matrix.csv"))
    plot_confusion(cm, classes, os.path.join(results_dir, "confusion_matrix.png"),
                   f"{name} — test confusion matrix (counts)", normalized=False)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plot_confusion(cm_norm, classes,
                   os.path.join(results_dir, "confusion_matrix_normalized.png"),
                   f"{name} — test confusion matrix (row-normalized = recall)",
                   normalized=True)

    pairs = top_confused_pairs(cm, classes)
    pairs.to_csv(os.path.join(results_dir, "top_confused_pairs.csv"), index=False)

    pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "true_id": y_test, "pred_id": p_test,
        "true_breed": [classes[i] for i in y_test],
        "pred_breed": [classes[i] for i in p_test],
        "correct": y_test == p_test,
    }).to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

    example_grid(cfg, y_test, p_test, classes,
                 os.path.join(results_dir, "examples_correct_incorrect.png"),
                 f"{name}: example test predictions (top: correct, bottom: incorrect)",
                 cfg["seed"])

    print(f"{name}: test acc {test_acc:.4f} "
          f"[{metrics['test_acc_ci95'][0]:.4f}, {metrics['test_acc_ci95'][1]:.4f}], "
          f"macro-F1 {test_f1:.4f}, train acc {train_acc:.4f}, val acc {val_acc:.4f}")
    print("  top confused pairs:")
    print(pairs.head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected models on test.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--approach", choices=list(APPROACHES))
    args = parser.parse_args()
    if args.all:
        names = list(APPROACHES)
    elif args.approach:
        names = [args.approach]
    else:
        parser.error("use --all or --approach NAME")
    for name in names:
        evaluate(name)


if __name__ == "__main__":
    main()
