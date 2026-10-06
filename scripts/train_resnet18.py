from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights


# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

MODEL_PATH = PROJECT_DIR / "resnet18_best_model.pth"


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
# DATASETS
# -----------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print("\nClasses:")
print(train_dataset.classes)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# -----------------------------
# DATA LOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# -----------------------------
# MODEL
# -----------------------------

model = resnet18(weights=weights)

number_of_classes = len(train_dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    number_of_classes
)

model = model.to(device)

print(f"\nNumber of output classes: {number_of_classes}")


# -----------------------------
# LOSS AND OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# TRAINING
# -----------------------------

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # TRAIN
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # VALIDATION
    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += (
                labels.size(0)
            )

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = (
        val_running_loss
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    )


    # RESULTS
    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    # SAVE BEST MODEL
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved.")


# -----------------------------
# FINISHED
# -----------------------------

print("\nTraining complete.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best model saved to: "
    f"{MODEL_PATH}"
)