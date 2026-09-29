from pathlib import Path

import torch
from torch import nn
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader


# -----------------------------
# PATHS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


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
# LOAD DATASETS
# -----------------------------

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)


# -----------------------------
# DATASET INFORMATION
# -----------------------------

print(f"Classes: {train_dataset.classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")


# -----------------------------
# DATA LOADERS
# -----------------------------

BATCH_SIZE = 32

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Testing batches: {len(test_loader)}")


# -----------------------------
# CREATE MODEL
# -----------------------------

model = efficientnet_b0(weights=weights)

# EfficientNet-B0 was originally trained to classify 1000 classes.
# Replace its final classification layer with one that predicts
# our two classes: clean and gaussian.

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model = model.to(device)

print("\nModel created!")
print(f"Input features to classifier: {num_features}")
print(f"Output classes: {len(train_dataset.classes)}")
print(model.classifier)


# -----------------------------
# LOSS FUNCTION AND OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

LEARNING_RATE = 0.0001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nTraining setup:")
print(f"Loss function: {criterion}")
print(f"Optimizer: Adam")
print(f"Learning rate: {LEARNING_RATE}")


# -----------------------------
# TRAINING AND VALIDATION
# -----------------------------

EPOCHS = 5

# Keep track of the best validation loss we have seen
best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # =============================
    # TRAINING
    # =============================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate how wrong the predictions were
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track training loss
        running_loss += loss.item() * images.size(0)

        # Track training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate results for the entire training set
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Training Loss: {epoch_loss:.4f} | "
        f"Training Accuracy: {epoch_accuracy:.4f}"
    )


    # =============================
    # VALIDATION
    # =============================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    # Validation does not update the model
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Make predictions
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            # Track validation loss
            val_running_loss += loss.item() * images.size(0)

            # Track validation accuracy
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate results for the entire validation set
    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    print(
        f"            Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # =============================
    # SAVE BEST MODEL
    # =============================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            PROJECT_DIR / "best_model.pth"
        )

        print("            Best model saved!")