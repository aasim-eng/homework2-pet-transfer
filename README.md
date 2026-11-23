# Homework 2 — Transfer Learning on Oxford-IIIT Pet (37 breeds) with MobileNetV3-Small

This project compares **three ways of adapting a pre-trained MobileNetV3-Small** (ImageNet-1K
weights) to classify the **37 cat and dog breeds** of the Oxford-IIIT Pet dataset:

| # | Approach | What is trained |
|---|---|---|
| 1 | **Full fine-tuning** (`full_finetune`) | every parameter of the network |
| 2 | **Classifier-head training** (`head_only`) | only `model.classifier` (`model.features` frozen) |
| 3 | **Frozen features + traditional classifier** (`features_*`) | only a scikit-learn classifier on 576-d features |

All approaches use the same data split, the same preprocessing and the same metrics.
The test set is used **once per final model**, only in `src/evaluate.py`.

## Results (test set, 3,669 images)

| Approach | Test acc [95% CI] | Test macro-F1 [95% CI] | Train acc | Val acc | Train−test gap | Trainable params | Training time (s) | Selected |
|---|---|---|---|---|---|---|---|---|
| full_finetune | 0.8490 [0.8384, 0.8605] | 0.8464 [0.8349, 0.8572] | 0.9990 | 0.8954 | 0.1500 | 1,555,781 | 251.9 | epoch 11 of 15 |
| head_only | 0.8673 [0.8566, 0.8779] | 0.8658 [0.8545, 0.8755] | 0.9915 | 0.8927 | 0.1242 | 628,773 | 83.8 | epoch 10 of 15 |
| features_logreg | 0.8569 [0.8460, 0.8681] | 0.8551 [0.8437, 0.8655] | 0.9881 | 0.8764 | 0.1312 | 21,349 | 93.1 (extract 93.0 + fit 0.1) | C=0.01 |
| features_linear_svm | 0.8335 [0.8217, 0.8457] | 0.8309 [0.8182, 0.8419] | 0.9990 | 0.8750 | 0.1655 | 21,349 | 99.6 (extract 93.0 + fit 6.6) | C=0.01 |
| features_random_forest | 0.8277 [0.8163, 0.8397] | 0.8232 [0.8112, 0.8344] | 1.0000 | 0.8614 | 0.1723 | n/a (forest) | 96.5 (extract 93.0 + fit 3.5) | max_depth=None |

- 95% CIs: bootstrap over test images (2,000 resamples). Paired comparisons: `results/pairwise_differences.md`.
- Train acc = selected model on the training images **without** augmentation.
- Feature-extraction time includes train, val **and** test (32.0 + 29.0 + 32.0 s).
- All numbers come from **one training run with seed 42**; variation across seeds was not measured.

Full table: `results/summary.md` / `results/summary.csv`.

## Environment used for the reported results

| Item | Value |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.15.0 |
| TorchVision | 0.29.2 |
| scikit-learn | 1.9.1 |
| NumPy | 2.4.6 |
| Hardware | Apple M2 (16 GB RAM), PyTorch device **MPS** (Apple GPU) |
| OS | macOS 26.6.2 (arm64) |

The code picks the device automatically: **CUDA → Apple MPS → CPU** (`src/utils.py: get_device`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned versions
```

`requirements.txt` pins the exact versions used. On a CUDA machine you may need the PyTorch
wheel for your CUDA version from https://pytorch.org.

**macOS note:** Python from python.org sometimes cannot verify HTTPS certificates
(`CERTIFICATE_VERIFY_FAILED`). The scripts handle this automatically by using the
`certifi` certificate bundle (see `fix_ssl_certificates` in `src/utils.py`).

## Dataset

- **Name:** The Oxford-IIIT Pet Dataset — 37 breeds (12 cats, 25 dogs), about 200 images per breed.
- **Source:** official page of the Visual Geometry Group, University of Oxford:
  https://www.robots.ox.ac.uk/~vgg/data/pets/
- **License (checked on the official page):** the dataset is available for commercial/research
  purposes under the **Creative Commons Attribution-ShareAlike 4.0 International License
  (CC BY-SA 4.0)**. The copyright of the images remains with the original owners.
- **Citation:**
  > O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.*
  > IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

  ```bibtex
  @InProceedings{parkhi12a,
    author    = "Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C. V.",
    title     = "Cats and Dogs",
    booktitle = "IEEE Conference on Computer Vision and Pattern Recognition",
    year      = "2012",
  }
  ```

The images are **not** included in this repository (`data/` is in `.gitignore`).

### Download

Option A (automatic): the first script you run downloads the data into `data/oxford-iiit-pet/`
through `torchvision.datasets.OxfordIIITPet(download=True)`.

Option B (manual, if the automatic download is slow): download these two files

| File | URL | MD5 |
|---|---|---|
| images.tar.gz (792 MB) | https://thor.robots.ox.ac.uk/pets/images.tar.gz | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| annotations.tar.gz (19 MB) | https://thor.robots.ox.ac.uk/pets/annotations.tar.gz | `95a8c909bbe2e81eed6a22bccdf3f68f` |

put both in `data/oxford-iiit-pet/`, then run `python -m src.data`. torchvision checks the MD5
and extracts them.

### Splits

- Labels: `target_types="category"` → **37 breeds** (not the binary cat/dog label).
- Official `trainval` (3,680 images) is split **80/20, stratified by breed**, with
  `sklearn.model_selection.train_test_split(..., stratify=labels, random_state=42)`:
  **train = 2,944**, **validation = 736**, no overlap.
- Official `test` (3,669 images) is the test set.
- The indices are saved in `splits/train_idx.json` and `splits/val_idx.json` (positions in
  `OxfordIIITPet(split="trainval")`); settings in `splits/split_info.json`. Every script loads
  these files, so all approaches use the identical split.
- Images per breed: `results/data/class_counts.md` (CSV: `class_counts.csv`). The classes are
  almost balanced: train 74–80 per breed (max/min = 1.08), val 19–20, test 88–100 (max/min = 1.14;
  smallest: Bombay 88, Staffordshire Bull Terrier 89). See `results/data/class_imbalance.json`.

## Preprocessing

Taken from `MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()` (printed and saved in
`results/data/preprocessing.json`):

| Step | Value |
|---|---|
| Resize (shorter side) | 256, bilinear |
| Center crop | 224 × 224 |
| Mean | [0.485, 0.456, 0.406] |
| Std | [0.229, 0.224, 0.225] |

- **Validation, test and feature extraction:** exactly these weight transforms, no augmentation.
- **Training set only (Approaches 1 and 2):** `RandomResizedCrop(224, scale=(0.5, 1.0))`,
  `RandomHorizontalFlip(0.5)`, `ColorJitter(0.2, 0.2, 0.2)`, then the same mean/std normalization.
- **Approach 3:** no augmentation at all (features extracted once with the weight transforms).

## Model and trainable layers

`torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)`. The original
classifier is printed by `python -m src.models`:

```
classifier.0  Linear(576 -> 1024)
classifier.1  Hardswish
classifier.2  Dropout(p=0.2)
classifier.3  Linear(1024 -> 1000)   -> replaced by Linear(1024 -> 37)
```

| Approach | Trainable layers | Trainable / total params |
|---|---|---|
| 1 full_finetune | all 142 parameter tensors (`features.*` and `classifier.*`) | 1,555,781 / 1,555,781 |
| 2 head_only | `classifier.0.weight`, `classifier.0.bias`, `classifier.3.weight`, `classifier.3.bias` (`classifier.0` keeps its ImageNet weights as a starting point, `classifier.3` is new) | 628,773 / 1,555,781 |
| 3 features_* | none in the CNN; only the scikit-learn classifier (LogReg / LinearSVM: 37 × 576 weights + 37 biases = 21,349) | 0 / 1,555,781 |

**Frozen BatchNorm in Approach 2.** `model.features` has 34 BatchNorm layers. In train mode a
BatchNorm layer normalizes with the statistics of the current mini-batch **and updates its running
mean/variance**, even when its weights have `requires_grad=False`. So `set_train_mode()` in
`src/models.py` calls `model.train()` and then `model.features.eval()`: the frozen part uses the
stored ImageNet statistics and stays exactly the pre-trained network, while Dropout in the
classifier is still active. Proof: a checksum of all `features` parameters **and** BatchNorm
running statistics is identical before and after training (7470.6387… in both cases,
`results/head_only/config.json`), while it changes for full fine-tuning (7470.64 → 7603.31).

**Approach 3 features:** `model.features → model.avgpool → flatten` = **576** numbers per image
(after pooling, before the classifier), saved to `results/features/arrays/X_{train,val,test}.npy`.

## Training setup

All settings live in `configs/*.yaml`.

| Setting | full_finetune | head_only |
|---|---|---|
| Loss | cross-entropy | cross-entropy |
| Optimizer | AdamW (trainable params only) | AdamW (trainable params only) |
| Learning rate | 3e-4 | 1e-3 |
| Weight decay | 0.01 | 0.01 |
| Scheduler | CosineAnnealingLR, T_max = 15 epochs, stepped per epoch | same |
| Batch size | 64 | 64 |
| Epochs | 15 | 15 |
| Model selection | checkpoint with the best **validation macro-F1** (ties → earlier epoch) | same |
| **Selected** | **epoch 11** (val macro-F1 0.8957, val acc 0.8954) | **epoch 10** (val macro-F1 0.8937, val acc 0.8927) |
| Training time | 251.9 s | 83.8 s |

Approach 3 (`configs/features.yaml`): each classifier is a `Pipeline(StandardScaler, classifier)`,
so the scaler is fit on **training features only**. The main hyperparameter is picked on the
**validation** set by macro-F1 (ties → first value in the grid); the final model is the one fitted
on the training split (no refit on train+val).

| Classifier | Grid | Selected | Val macro-F1 | Fit time of selected (s) |
|---|---|---|---|---|
| LogisticRegression (lbfgs, max_iter 5000) | C ∈ {0.001, 0.01, 0.1, 1, 10} | **C = 0.01** | 0.8770 | 0.13 |
| LinearSVC (max_iter 20000) | C ∈ {0.0001, 0.001, 0.01, 0.1, 1} | **C = 0.01** | 0.8739 | 6.6 |
| RandomForest (500 trees, optional) | max_depth ∈ {10, 20, None} | **max_depth = None** | 0.8616 | 3.5 |

Selected settings for every approach are saved in `results/<approach>/config.json` and
`results/features/<classifier>/config.json`.

## How to reproduce everything

Run from the project root, in this order:

```bash
python -m src.data --config configs/full_finetune.yaml       # split + class-count tables
python -m src.models                                          # print model, trainable params

# optional smoke tests (1 epoch / tiny subsets, written to results/smoke/)
python -m src.train --config configs/full_finetune.yaml --smoke
python -m src.train --config configs/head_only.yaml --smoke

python -m src.train --config configs/full_finetune.yaml       # Approach 1
python -m src.train --config configs/head_only.yaml           # Approach 2
python -m src.extract_features --config configs/features.yaml # Approach 3, step 1
python -m src.train_sklearn --config configs/features.yaml    # Approach 3, step 2
python -m src.evaluate --all                                  # test set, once per model
python -m src.make_report_assets                              # summary tables + figures
```

Running `src.evaluate` again appends new lines to `results/test_access_log.csv`; the log for the
reported results has exactly one line per model.

## Outputs

| Path | Content |
|---|---|
| `results/data/` | class counts (CSV + MD), imbalance summary, preprocessing values |
| `results/<approach>/history.csv`, `learning_curves.png` | train/val loss and accuracy per epoch (Approaches 1, 2) |
| `results/features/<clf>/tuning.csv`, `tuning_curve.png` | train/val accuracy per hyperparameter value (Approach 3) |
| `results/.../config.json` | all hyperparameters, selected epoch/value, params, time, seeds, versions |
| `results/.../test_metrics.json` | test accuracy, macro-F1, bootstrap CIs, train/val accuracy, gap |
| `results/.../per_class_report.csv` | per-breed precision / recall / F1 |
| `results/.../confusion_matrix.png`, `confusion_matrix_normalized.png`, `confusion_matrix.csv` | 37×37 confusion matrices with breed names |
| `results/.../top_confused_pairs.csv` | 10 most confused breed pairs |
| `results/.../examples_correct_incorrect.png` | 8 correct + 8 incorrect test predictions |
| `results/.../test_predictions.csv` | one row per test image |
| `results/summary.{csv,md}` | summary table of all approaches |
| `results/pairwise_differences.{csv,md}` | paired bootstrap CI + exact McNemar test vs the best model |
| `results/figures/` | accuracy with CIs, train vs test, learning curves, accuracy vs time |
| `results/test_access_log.csv` | every time the test set was evaluated |
| `checkpoints/<approach>/best.pt` | selected weights (~6 MB each) |
| `results/features/<clf>/model.joblib` | selected sklearn pipelines (the Random Forest one is 248 MB and is git-ignored; `src.train_sklearn` rebuilds it) |

## Seeds, determinism and remaining randomness

- **Seed 42** for Python `random`, NumPy, PyTorch (CPU, CUDA, MPS), the split, the DataLoader
  shuffling generator, each DataLoader worker (`seed_worker`), the scikit-learn models, the bootstrap
  and the example-image selection.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is **on**;
  `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (relevant on CUDA only).
- **Check:** training Approach 2 a second time on the same machine gave **identical** validation
  accuracy and macro-F1 at all 15 epochs.
- **Remaining sources of variation:**
  - Different hardware/backends (CUDA vs MPS vs CPU) and different library versions use different
    kernels and floating-point summation order, so numbers can differ slightly on another machine.
  - `warn_only=True` means an operation without a deterministic version would only print a warning;
    on MPS, PyTorch's determinism checks do not cover every kernel.
  - Results are from one seed. A different seed (split, head initialisation, augmentation, batch
    order) would give somewhat different numbers; this seed-to-seed variation was not measured.
  - Timings depend on the machine and on background load.

## Project structure

```
homework2-pet-transfer/
├── README.md, requirements.txt, .gitignore, Homework2_Report.pdf
├── configs/            full_finetune.yaml, head_only.yaml, features.yaml
├── splits/             train_idx.json, val_idx.json, split_info.json
├── src/                utils, data, models, train, extract_features, train_sklearn, evaluate, make_report_assets
├── results/            metrics, tables, figures
└── checkpoints/        best.pt per CNN approach
```
