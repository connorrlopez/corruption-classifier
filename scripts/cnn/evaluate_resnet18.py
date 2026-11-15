from pathlib import Path
import csv
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt


# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TEST_DIR = DATA_DIR / "test"
MODEL_PATH = PROJECT_DIR / "resnet18_best_model.pth"
METADATA_PATH = DATA_DIR / "corruption_metadata.csv"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "resnet18_results.txt"
CONFUSION_MATRIX_FILE = (
    RESULTS_DIR / "resnet18_confusion_matrix.png"
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

print(f"Using device: {device}")


# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()


# -----------------------------
# TEST DATASET
# -----------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = test_dataset.classes
class_to_idx = test_dataset.class_to_idx

print("\nClasses:")
print(classes)

print("\nClass mapping:")
print(class_to_idx)

print(f"\nTesting images: {len(test_dataset)}")
print(f"Testing batches: {len(test_loader)}")


# -----------------------------
# MODEL
# -----------------------------

model = resnet18(weights=None)

number_of_classes = len(classes)

model.fc = nn.Linear(
    model.fc.in_features,
    number_of_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# RUN TESTING
# -----------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# -----------------------------
# OVERALL METRICS
# -----------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

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
    target_names=classes,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# -----------------------------
# TERMINAL RESULTS
# -----------------------------

print("\n==============================")
print("OVERALL TEST RESULTS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n==============================")
print("PER-CLASS RESULTS")
print("==============================")

print(report)

print("==============================")
print("CONFUSION MATRIX")
print("==============================")

print("Rows = Actual")
print("Columns = Predicted\n")

print("Class order:")
print(classes)

print("\n", cm)


# -----------------------------
# LOAD CORRUPTION METADATA
# -----------------------------

metadata = {}

with open(
    METADATA_PATH,
    "r",
    newline=""
) as csv_file:

    reader = csv.DictReader(
        csv_file
    )

    for row in reader:

        if row["split"] == "test":

            metadata[row["filename"]] = {
                "class": row["class"],
                "severity": int(
                    row["severity"]
                )
            }


# -----------------------------
# SEVERITY RESULTS
# -----------------------------

severity_results = defaultdict(
    lambda: {
        "correct": 0,
        "total": 0
    }
)

corruption_severity_results = defaultdict(
    lambda: {
        "correct": 0,
        "total": 0
    }
)


for index, (
    image_path,
    actual_label
) in enumerate(
    test_dataset.samples
):

    filename = Path(
        image_path
    ).name

    predicted_label = (
        all_predictions[index]
    )

    actual_class = (
        classes[actual_label]
    )

    if actual_class == "clean":
        continue

    if filename not in metadata:
        continue

    severity = (
        metadata[filename][
            "severity"
        ]
    )

    is_correct = (
        predicted_label
        == actual_label
    )

    severity_results[
        severity
    ]["total"] += 1

    if is_correct:
        severity_results[
            severity
        ]["correct"] += 1

    key = (
        actual_class,
        severity
    )

    corruption_severity_results[
        key
    ]["total"] += 1

    if is_correct:
        corruption_severity_results[
            key
        ]["correct"] += 1


# -----------------------------
# PRINT SEVERITY RESULTS
# -----------------------------

print("\n==============================")
print("ACCURACY BY SEVERITY")
print("==============================")

for severity in range(1, 6):

    results = severity_results[
        severity
    ]

    correct = results["correct"]
    total = results["total"]

    severity_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"Severity {severity}: "
        f"{correct}/{total} "
        f"({severity_accuracy:.4f})"
    )


print("\n==============================")
print("ACCURACY BY CORRUPTION + SEVERITY")
print("==============================")

for class_name in classes:

    if class_name == "clean":
        continue

    print(
        f"\n{class_name.upper()}"
    )

    for severity in range(1, 6):

        key = (
            class_name,
            severity
        )

        results = (
            corruption_severity_results[
                key
            ]
        )

        correct = results["correct"]
        total = results["total"]

        severity_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"  Severity {severity}: "
            f"{correct}/{total} "
            f"({severity_accuracy:.4f})"
        )


# -----------------------------
# SAVE TEXT RESULTS
# -----------------------------

with open(
    RESULTS_FILE,
    "w"
) as file:

    file.write(
        "ResNet-18 Results\n"
    )

    file.write(
        "=======================\n\n"
    )

    file.write(
        f"Test Images: "
        f"{len(test_dataset)}\n\n"
    )

    file.write(
        "Overall Test Results\n"
    )

    file.write(
        "--------------------\n"
    )

    file.write(
        f"Accuracy:  "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Recall:    "
        f"{recall:.4f}\n"
    )

    file.write(
        f"F1 Score:  "
        f"{f1:.4f}\n\n"
    )

    file.write(
        "Per-Class Results\n"
    )

    file.write(
        "-----------------\n"
    )

    file.write(report)

    file.write(
        "\nConfusion Matrix\n"
    )

    file.write(
        "----------------\n"
    )

    file.write(
        f"Class order: "
        f"{classes}\n\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nAccuracy by Severity\n"
    )

    file.write(
        "--------------------\n"
    )

    for severity in range(1, 6):

        results = severity_results[
            severity
        ]

        correct = results["correct"]
        total = results["total"]

        severity_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        file.write(
            f"Severity {severity}: "
            f"{correct}/{total} "
            f"({severity_accuracy:.4f})\n"
        )

    file.write(
        "\nAccuracy by Corruption "
        "+ Severity\n"
    )

    file.write(
        "-------------------------\n"
    )

    for class_name in classes:

        if class_name == "clean":
            continue

        file.write(
            f"\n{class_name.upper()}\n"
        )

        for severity in range(1, 6):

            key = (
                class_name,
                severity
            )

            results = (
                corruption_severity_results[
                    key
                ]
            )

            correct = results["correct"]
            total = results["total"]

            severity_accuracy = (
                correct / total
                if total > 0
                else 0
            )

            file.write(
                f"Severity {severity}: "
                f"{correct}/{total} "
                f"({severity_accuracy:.4f})\n"
            )


# -----------------------------
# SAVE CONFUSION MATRIX IMAGE
# -----------------------------

fig, ax = plt.subplots(
    figsize=(9, 7)
)

image = ax.imshow(cm)

ax.set_xticks(
    range(len(classes))
)

ax.set_yticks(
    range(len(classes))
)

display_names = [
    name.replace("_", " ").title()
    for name in classes
]

ax.set_xticklabels(
    display_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    display_names
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_title(
    "ResNet-18 Confusion Matrix"
)

for i in range(
    len(classes)
):

    for j in range(
        len(classes)
    ):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax
)

fig.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300
)

plt.close()


# -----------------------------
# FINISHED
# -----------------------------

print("\n==============================")
print("RESULTS SAVED")
print("==============================")

print(
    f"Text results: "
    f"{RESULTS_FILE}"
)

print(
    f"Confusion matrix: "
    f"{CONFUSION_MATRIX_FILE}"
)