# models.py
# Build MobileNetV3-Small with ImageNet weights, replace the final layer with
# a 37-class layer, freeze parts of the network, and count parameters.
#
# Run directly to print the model and the trainable parameters of each approach:
#     python -m src.models

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

NUM_CLASSES = 37
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT


def build_model(num_classes=NUM_CLASSES):
    """Load pre-trained MobileNetV3-Small and replace its final layer.

    The original classifier is:
        classifier.0  Linear(576 -> 1024)
        classifier.1  Hardswish
        classifier.2  Dropout(p=0.2)
        classifier.3  Linear(1024 -> 1000)   <- ImageNet head, replaced
    We replace classifier.3 with a new Linear(1024 -> num_classes).
    The new layer is randomly initialised; everything else is pre-trained.
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def set_trainable(model, approach):
    """Freeze/unfreeze parameters for an approach.

    full_finetune : every parameter is trainable.
    head_only     : model.features is frozen; model.classifier
                    (classifier.0 and classifier.3) is trainable.
    features      : everything is frozen (pure feature extractor).
    """
    if approach == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True
    elif approach == "head_only":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif approach == "features":
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError("Unknown approach: " + approach)
    return model


def set_train_mode(model, approach):
    """Put the model in training mode, but keep frozen BatchNorm in eval mode.

    Why: BatchNorm layers keep running estimates of the mean and variance of
    their inputs. In train mode they (1) normalise with the statistics of the
    current mini-batch and (2) update the running estimates. If the feature
    extractor is frozen, we want it to behave exactly like the pre-trained
    network, so we keep model.features in eval mode: it uses the stored
    ImageNet statistics and does not change them. Otherwise the "frozen"
    features would still drift, even though no weights get gradients.
    Dropout inside model.classifier stays active (train mode) as intended.
    """
    model.train()
    if approach == "head_only":
        model.features.eval()
    return model


def count_parameters(model):
    """Return (trainable, total) number of parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def trainable_layer_names(model):
    """Names of all parameter tensors that will be updated."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def extract_features(model, x):
    """Feature vector after the pooling step and before the classifier.

    features -> avgpool -> flatten  gives a 576-dimensional vector.
    """
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on a DataLoader (eval mode, no gradients).

    Returns (mean cross-entropy loss, predicted labels, true labels, logits)
    as NumPy arrays.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.float().cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()
    return total_loss / len(labels), preds, labels, logits


def parameter_checksum(module):
    """Sum of all parameters and BatchNorm buffers of a module.

    Used as a simple proof: if this number does not change during training,
    the module was really frozen.
    """
    total = 0.0
    for p in module.parameters():
        total += p.detach().cpu().double().sum().item()
    for name, b in module.named_buffers():
        if "running" in name:
            total += b.detach().cpu().double().sum().item()
    return total


def main():
    model = build_model()
    print(model)
    print("\nmodel.classifier:\n", model.classifier)
    print("Final layer:", model.classifier[3], "-> out_features =",
          model.classifier[3].out_features)

    for approach in ["full_finetune", "head_only", "features"]:
        set_trainable(model, approach)
        trainable, total = count_parameters(model)
        names = trainable_layer_names(model)
        print(f"\n[{approach}] trainable = {trainable:,} / total = {total:,}")
        if approach == "head_only":
            print("  trainable tensors:", names)
        else:
            print(f"  number of trainable tensors: {len(names)}")

    model.eval()
    with torch.no_grad():
        f = extract_features(model, torch.zeros(1, 3, 224, 224))
    print("\nFeature vector size:", tuple(f.shape))


if __name__ == "__main__":
    main()
