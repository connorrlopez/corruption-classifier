from pathlib import Path
import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from scripts.data.severity_dataset_loader import SeverityCorruptionDataset

# -----------------------------
# SETTINGS
# -----------------------------

DATA_DIR = Path.home() / "capstone/corrupted_datasets"
PROJECT_DIR = Path(__file__).resolve().parent.parent

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 4
RANDOM_SEED = 42


# -----------------------------
# TRAINING FUNCTION
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--severity",
        type=int,
        choices=[1, 3, 5],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    # Reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # -------------------------
    # PATHS
    # -------------------------

    severity_dir = DATA_DIR / f"severity_{args.severity}"

    train_csv = severity_dir / "train.csv"
    val_csv = severity_dir / "val.csv"

    output_dir = (
        PROJECT_DIR / "severity_results"
        / "resnet18"
        / f"severity_{args.severity}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.pth"
    history_path = output_dir / "training_history.csv"

    # -------------------------
    # DEVICE
    # -------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Severity: {args.severity}")
    print(f"Epochs: {args.epochs}")
    print(f"Training manifest: {train_csv}")
    print(f"Validation manifest: {val_csv}")

    # -------------------------
    # TRANSFORMS
    # -------------------------

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    # -------------------------
    # DATASETS
    # -------------------------

    train_dataset = SeverityCorruptionDataset(
        train_csv,
        transform=transform
    )

    val_dataset = SeverityCorruptionDataset(
        val_csv,
        transform=transform
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # -------------------------
    # DATA LOADERS
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # -------------------------
    # MODEL
    # -------------------------

    model = resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(train_dataset.classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------
    # TRAINING
    # -------------------------

    best_val_accuracy = -1.0
    history = []

    training_start = time.perf_counter()

    for epoch in range(args.epochs):

        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        # ---------------------
        # VALIDATION
        # ---------------------

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:
                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}",
            flush=True
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}",
            flush=True
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}",
            flush=True
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # ---------------------
        # SAVE BEST MODEL
        # ---------------------

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                model_path
            )

            print("Best model saved.", flush=True)

        # ---------------------
        # SAVE TRAINING HISTORY
        # ---------------------

        with history_path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

    # -------------------------
    # TRAINING TIME
    # -------------------------

    training_seconds = time.perf_counter() - training_start

    print(
        f"\nTotal training time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    print(
        f"Average time per epoch: "
        f"{training_seconds / args.epochs / 60:.2f} minutes"
    )

    # -------------------------
    # FINAL RESULTS
    # -------------------------

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()