# make_report_assets.py
# Collect the saved results of all approaches into final tables and figures:
#   results/summary.csv / results/summary.md      one row per approach
#   results/pairwise_differences.csv / .md        paired comparison on test
#   results/figures/test_accuracy_ci.png          accuracy with 95% CI
#   results/figures/train_vs_test_accuracy.png    generalization gap
#   results/figures/val_accuracy_curves.png       val accuracy per epoch (CNNs)
#   results/figures/compute_cost.png              training time vs accuracy
# It only READS files written by train.py, train_sklearn.py and evaluate.py.
#
# Usage:
#     python -m src.make_report_assets

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.evaluate import APPROACHES
from src.utils import load_json

FIG_DIR = "results/figures"
SEED = 42
N_BOOTSTRAP = 2000


def sklearn_param_count(results_dir):
    """Number of learned numbers in a linear sklearn model (weights + biases).

    Random Forest has no fixed parameter vector; we return None and report
    the total number of tree nodes instead.
    """
    pipe = joblib.load(os.path.join(results_dir, "model.joblib"))
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size), None
    nodes = sum(tree.tree_.node_count for tree in clf.estimators_)
    return None, int(nodes)


def build_summary():
    """One row per approach with the main numbers."""
    rows = []
    for name, (kind, _, results_dir) in APPROACHES.items():
        cfg = load_json(os.path.join(results_dir, "config.json"))
        m = load_json(os.path.join(results_dir, "test_metrics.json"))
        row = {
            "approach": name,
            "test_acc": m["test_acc"],
            "test_acc_ci95_low": m["test_acc_ci95"][0],
            "test_acc_ci95_high": m["test_acc_ci95"][1],
            "test_macro_f1": m["test_macro_f1"],
            "test_macro_f1_ci95_low": m["test_macro_f1_ci95"][0],
            "test_macro_f1_ci95_high": m["test_macro_f1_ci95"][1],
            "train_acc": m["train_acc_clean"],
            "val_acc": m["val_acc"],
            "gap_train_minus_test": m["generalization_gap_train_minus_test"],
        }
        if kind == "cnn":
            row["trainable_params"] = cfg["trainable_params"]
            row["total_params"] = cfg["total_params"]
            row["feature_extraction_time_s"] = np.nan
            row["classifier_fit_time_s"] = np.nan
            row["training_time_s"] = cfg["training_time_s"]
            row["selected"] = f"epoch {cfg['selected_epoch']} of {cfg['epochs']}"
            row["note"] = ""
        else:
            params, nodes = sklearn_param_count(results_dir)
            row["trainable_params"] = params if params is not None else np.nan
            row["total_params"] = cfg["total_params_cnn"]
            row["feature_extraction_time_s"] = cfg["feature_extraction_time_s"]
            row["classifier_fit_time_s"] = cfg["classifier_fit_time_selected_s"]
            row["training_time_s"] = (cfg["feature_extraction_time_s"]
                                      + cfg["classifier_fit_time_selected_s"])
            row["selected"] = f"{cfg['tuned_hyperparameter']}={cfg['selected_value']}"
            row["note"] = ("CNN frozen (0 trainable CNN params); trainable_params = "
                           "classifier weights+biases; total_params = frozen CNN")
            if nodes is not None:
                row["note"] = (f"CNN frozen; random forest has no weight vector "
                               f"({nodes:,} tree nodes in total)")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_markdown(df):
    """Readable Markdown version of the summary table."""
    lines = [
        "| Approach | Test acc [95% CI] | Test macro-F1 [95% CI] | Train acc | Val acc | Train−test gap | Trainable params | Training time (s) | Selected |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        params = "n/a (forest)" if pd.isna(r["trainable_params"]) else f"{int(r['trainable_params']):,}"
        if pd.isna(r["feature_extraction_time_s"]):
            t = f"{r['training_time_s']:.1f}"
        else:
            t = (f"{r['training_time_s']:.1f} (extract {r['feature_extraction_time_s']:.1f} "
                 f"+ fit {r['classifier_fit_time_s']:.1f})")
        lines.append(
            f"| {r['approach']} "
            f"| {r['test_acc']:.4f} [{r['test_acc_ci95_low']:.4f}, {r['test_acc_ci95_high']:.4f}] "
            f"| {r['test_macro_f1']:.4f} [{r['test_macro_f1_ci95_low']:.4f}, {r['test_macro_f1_ci95_high']:.4f}] "
            f"| {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['gap_train_minus_test']:.4f} "
            f"| {params} | {t} | {r['selected']} |")
    return "\n".join(lines) + "\n"


def load_correct(name):
    """Boolean array: was each test image classified correctly?"""
    results_dir = APPROACHES[name][2]
    df = pd.read_csv(os.path.join(results_dir, "test_predictions.csv"))
    return df["correct"].to_numpy().astype(bool)


def pairwise_differences(df):
    """Compare every approach with the best one on the SAME test images.

    - Paired bootstrap 95% CI of the accuracy difference.
    - Exact McNemar test (binomial test on images where exactly one of the
      two models is right). A large p-value means the difference could
      easily be due to chance on this test set.
    """
    best = df.sort_values("test_acc", ascending=False).iloc[0]["approach"]
    a = load_correct(best)
    rng = np.random.default_rng(SEED)
    n = len(a)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]
    rows = []
    for name in df["approach"]:
        if name == best:
            continue
        b = load_correct(name)
        diffs = [a[idx].mean() - b[idx].mean() for idx in boot_idx]
        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
        rows.append({
            "best": best, "other": name,
            "acc_difference": a.mean() - b.mean(),
            "diff_ci95_low": float(np.percentile(diffs, 2.5)),
            "diff_ci95_high": float(np.percentile(diffs, 97.5)),
            "only_best_correct": only_a, "only_other_correct": only_b,
            "mcnemar_exact_p": p,
        })
    return pd.DataFrame(rows)


def plot_accuracy_ci(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    err = [df["test_acc"] - df["test_acc_ci95_low"], df["test_acc_ci95_high"] - df["test_acc"]]
    ax.bar(x, df["test_acc"], yerr=err, capsize=5, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(df["approach"], rotation=20, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test accuracy with bootstrap 95% CI")
    for i, v in enumerate(df["test_acc"]):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "test_accuracy_ci.png"), dpi=150)
    plt.close(fig)


def plot_train_vs_test(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["train_acc"], w, label="train (clean)", color="#F58518")
    ax.bar(x + w / 2, df["test_acc"], w, label="test", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(df["approach"], rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Train vs test accuracy (generalization gap)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "train_vs_test_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_val_curves():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["full_finetune", "head_only"]:
        h = pd.read_csv(os.path.join(APPROACHES[name][2], "history.csv"))
        line, = ax.plot(h["epoch"], h["val_acc"], marker="o", label=f"{name} val")
        ax.plot(h["epoch"], h["train_acc"], linestyle="--", color=line.get_color(),
                label=f"{name} train (augmented)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Learning curves: Approaches 1 and 2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "val_accuracy_curves.png"), dpi=150)
    plt.close(fig)


def plot_compute_cost(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in df.iterrows():
        ax.scatter(r["training_time_s"], r["test_acc"], s=60)
        ax.annotate(r["approach"], (r["training_time_s"], r["test_acc"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training time in seconds (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs computational cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "compute_cost.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = build_summary()
    df.to_csv("results/summary.csv", index=False)
    md = summary_markdown(df)
    with open("results/summary.md", "w") as f:
        f.write("# Summary of all approaches (test set, evaluated once per selected model)\n\n")
        f.write(md)
        f.write("\nTrain acc = selected model on training images without augmentation. "
                "CIs: bootstrap over test images, 2000 resamples.\n")
    print(md)

    diffs = pairwise_differences(df)
    diffs.to_csv("results/pairwise_differences.csv", index=False)
    with open("results/pairwise_differences.md", "w") as f:
        f.write(diffs.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(diffs.to_string(index=False))

    plot_accuracy_ci(df)
    plot_train_vs_test(df)
    plot_val_curves()
    plot_compute_cost(df)
    print("Figures saved in", FIG_DIR)


if __name__ == "__main__":
    main()
