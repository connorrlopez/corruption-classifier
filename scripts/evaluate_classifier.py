from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------
# PATHS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "best_model.pth"


# -----------------------------
# DEVICE
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------

weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()


# -----------------------------
# LOAD TEST DATASET
# -----------------------------

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print(f"Classes: {test_dataset.classes}")
print(f"Class mapping: {test_dataset.class_to_idx}")
print(f"Testing images: {len(test_dataset)}")


# -----------------------------
# CREATE MODEL
# -----------------------------

model = efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Best model loaded successfully!")


# -----------------------------
# TEST MODEL
# -----------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted.cpu().tolist())


# -----------------------------
# CALCULATE METRICS
# -----------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# -----------------------------
# PRINT RESULTS
# -----------------------------

print("\n===== TEST RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)