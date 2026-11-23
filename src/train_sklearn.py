# train_sklearn.py
# Approach 3, step 2: train traditional classifiers on the frozen features.
#   - Logistic Regression (required)
#   - Linear SVM          (required)
#   - Random Forest       (optional)
# Each classifier is a scikit-learn Pipeline: StandardScaler -> classifier.
# The scaler is fit on the TRAINING features only (inside pipeline.fit).
# The main hyperparameter is chosen with the VALIDATION set only.
# Test features are NOT loaded in this script.
#
# Usage:
#     python -m src.train_sklearn --config configs/features.yaml
#     python -m src.train_sklearn --config configs/features.yaml --smoke

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.utils import (Logger, Timer, get_environment_info, load_config,
                       load_json, save_json, set_seed)


def make_pipeline(name, value, clf_cfg, seed):
    """Build StandardScaler -> classifier for one hyperparameter value."""
    if name == "logreg":
        clf = LogisticRegression(C=value, max_iter=clf_cfg["max_iter"],
                                 random_state=seed)
    elif name == "linear_svm":
        clf = LinearSVC(C=value, max_iter=clf_cfg["max_iter"], random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=clf_cfg["n_estimators"],
                                     max_depth=value, n_jobs=-1,
                                     random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# Which hyperparameter is tuned for each classifier.
TUNED_PARAM = {"logreg": "C", "linear_svm": "C", "random_forest": "max_depth"}


def plot_tuning(df, name, param, path):
    """Train and validation accuracy for each hyperparameter value."""
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(v) for v in df[param]]
    ax.plot(xs, df["train_acc"], marker="o", label="train")
    ax.plot(xs, df["val_acc"], marker="o", label="validation")
    ax.set_xlabel(param)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{name}: accuracy vs {param}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train sklearn classifiers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feat_dir = cfg["extraction"]["features_dir"]
    results_dir = cfg["output"]["results_dir"]
    if args.smoke:
        feat_dir = os.path.join("results", "smoke", "features", "arrays")
        results_dir = os.path.join("results", "smoke", "features")

    logger = Logger(os.path.join(results_dir, "sklearn_log.txt"))
    determinism = set_seed(cfg["seed"], cfg["deterministic"])
    seed = cfg["seed"]

    X_train = np.load(os.path.join(feat_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feat_dir, "y_train.npy"))
    X_val = np.load(os.path.join(feat_dir, "X_val.npy"))
    y_val = np.load(os.path.join(feat_dir, "y_val.npy"))
    logger.log(f"X_train {X_train.shape}, X_val {X_val.shape} (test not loaded)")
    extraction = load_json(os.path.join(results_dir, "extraction_info.json"))

    for name, clf_cfg in cfg["classifiers"].items():
        param = TUNED_PARAM[name]
        grid = clf_cfg[param]
        rows = []
        best = None
        tuning_timer = Timer()
        for value in grid:
            pipe = make_pipeline(name, value, clf_cfg, seed)
            fit_timer = Timer()
            pipe.fit(X_train, y_train)        # scaler fit on train only
            fit_time = fit_timer.elapsed()
            train_pred = pipe.predict(X_train)
            val_pred = pipe.predict(X_val)
            row = {
                param: value,
                "train_acc": accuracy_score(y_train, train_pred),
                "val_acc": accuracy_score(y_val, val_pred),
                "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
                "fit_time_s": fit_time,
            }
            rows.append(row)
            logger.log(f"{name} {param}={value}: train acc {row['train_acc']:.4f} "
                       f"val acc {row['val_acc']:.4f} val macroF1 "
                       f"{row['val_macro_f1']:.4f} ({fit_time:.1f}s)")
            # Strictly better keeps the first (smaller C / shallower) on ties.
            if best is None or row["val_macro_f1"] > best["row"]["val_macro_f1"]:
                best = {"row": row, "pipe": pipe}
        tuning_time = tuning_timer.elapsed()

        out = os.path.join(results_dir, name)
        os.makedirs(out, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "tuning.csv"), index=False)
        plot_tuning(df, name, param, os.path.join(out, "tuning_curve.png"))
        joblib.dump(best["pipe"], os.path.join(out, "model.joblib"))

        clf = best["pipe"].named_steps["clf"]
        config_out = {
            "approach": f"features_{name}",
            "config_file": args.config,
            "smoke_test": args.smoke,
            "pipeline": str(best["pipe"]),
            "classifier_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                  for k, v in clf.get_params().items()},
            "tuned_hyperparameter": param,
            "grid": grid,
            "model_selection_rule": f"value of {param} with the highest validation macro-F1; ties keep the first value in the grid",
            "selected_value": best["row"][param],
            "selected_val_acc": best["row"]["val_acc"],
            "selected_val_macro_f1": best["row"]["val_macro_f1"],
            "selected_train_acc": best["row"]["train_acc"],
            "final_model_fit_on": "train split only (same as Approaches 1 and 2)",
            "scaler_fit_on": "train features only (inside Pipeline.fit)",
            "feature_dim": int(X_train.shape[1]),
            "trainable_params_cnn": 0,
            "total_params_cnn": extraction["total_params_cnn"],
            "feature_extraction_time_s": extraction["extraction_time_total_s"],
            "feature_extraction_time_per_split_s": extraction["extraction_time_s"],
            "classifier_fit_time_selected_s": best["row"]["fit_time_s"],
            "classifier_tuning_time_total_s": tuning_time,
            "determinism": determinism,
            "environment": get_environment_info(),
        }
        save_json(config_out, os.path.join(out, "config.json"))
        logger.log(f"==> {name}: selected {param}={best['row'][param]} "
                   f"(val macro-F1 {best['row']['val_macro_f1']:.4f}); "
                   f"tuning time {tuning_time:.1f}s")


if __name__ == "__main__":
    main()
