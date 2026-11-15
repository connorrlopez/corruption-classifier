from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

from scripts.data.full_dataset_loader import FullCorruptionDataset

# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = PROJECT_DIR / "full_dataset" / "test.csv"
MODEL_PATH = PROJECT_DIR / "densenet121_full_best_model.pth"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "densenet121_full_results.txt"
CONFUSION_MATRIX_FILE = (
    RESULTS_DIR / "densenet121_full_confusion_matrix.png"
)

BATCH_SIZE = 32


# -----------------------------
# DEVICE
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}", flush=True)


# -----------------------------
# DATASET
# -----------------------------

weights = DenseNet121_Weights.DEFAULT

test_dataset = FullCorruptionDataset(
    TEST_CSV,
    transform=weights.transforms()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = test_dataset.classes

print(f"Classes: {classes}")
print(f"Testing images: {len(test_dataset)}")
print(f"Testing batches: {len(test_loader)}")


# -----------------------------
# MODEL
# -----------------------------

model = densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# EVALUATION
# -----------------------------

all_labels = []
all_predictions = []

print("\nStarting evaluation...", flush=True)

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {batch_index + 1}/{len(test_loader)} batches",
                flush=True
            )


# -----------------------------
# OVERALL METRICS
# -----------------------------

accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)


# -----------------------------
# ACCURACY BY SEVERITY
# -----------------------------

severity_results = defaultdict(
    lambda: {"correct": 0, "total": 0}
)

corruption_severity_results = defaultdict(
    lambda: {"correct": 0, "total": 0}
)

for index, record in enumerate(test_dataset.records):

    class_name = record["class"]

    if class_name == "clean":
        continue

    severity = int(record["severity"])

    correct = all_predictions[index] == all_labels[index]

    severity_results[severity]["total"] += 1
    severity_results[severity]["correct"] += int(correct)

    key = (class_name, severity)

    corruption_severity_results[key]["total"] += 1
    corruption_severity_results[key]["correct"] += int(correct)


# -----------------------------
# FORMAT RESULTS
# -----------------------------

lines = [
    "DenseNet-121 Full Dataset Results",
    "===================================",
    "",
    f"Test Images: {len(test_dataset)}",
    "",
    "Overall Test Results",
    "--------------------",
    f"Accuracy:  {accuracy:.4f}",
    f"Precision: {precision:.4f}",
    f"Recall:    {recall:.4f}",
    f"F1 Score:  {f1:.4f}",
    "",
    "Per-Class Results",
    "-----------------",
    report,
    "Confusion Matrix",
    "----------------",
    "Rows = Actual, Columns = Predicted",
    f"Class order: {classes}",
    str(cm),
    "",
    "Accuracy by Severity",
    "--------------------",
]

for severity in range(1, 6):

    results = severity_results[severity]
    correct = results["correct"]
    total = results["total"]

    score = correct / total if total else 0

    lines.append(
        f"Severity {severity}: {correct}/{total} ({score:.4f})"
    )

lines.extend([
    "",
    "Accuracy by Corruption + Severity",
    "---------------------------------",
])

for class_name in classes:

    if class_name == "clean":
        continue

    lines.append(f"\n{class_name.upper()}")

    for severity in range(1, 6):

        results = corruption_severity_results[
            (class_name, severity)
        ]

        correct = results["correct"]
        total = results["total"]

        score = correct / total if total else 0

        lines.append(
            f"  Severity {severity}: "
            f"{correct}/{total} ({score:.4f})"
        )


# -----------------------------
# SAVE AND PRINT RESULTS
# -----------------------------

results_text = "\n".join(lines)

print("\n" + results_text)

RESULTS_FILE.write_text(results_text)


# -----------------------------
# CONFUSION MATRIX IMAGE
# -----------------------------

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(cm)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

display_names = [
    name.replace("_", " ").title()
    for name in classes
]

ax.set_xticklabels(
    display_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(display_names)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("DenseNet-121 Full Dataset Confusion Matrix")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300
)

plt.close()

print(f"\nText results saved to: {RESULTS_FILE}")
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")
print("\nEvaluation complete!")
