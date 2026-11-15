from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    resnet18,
    ResNet18_Weights,
    densenet121,
    DenseNet121_Weights,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from scripts.data.severity_dataset_loader import SeverityCorruptionDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / "capstone/corrupted_datasets"
RESULTS_DIR = PROJECT_DIR / "severity_results"

BATCH_SIZE = 32
NUM_WORKERS = 4


def build_model(model_name, num_classes):
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    elif model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, weights.transforms()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["efficientnet_b0", "resnet18", "densenet121"],
        required=True
    )

    parser.add_argument(
        "--severity",
        type=int,
        choices=[1, 3, 5],
        required=True
    )

    args = parser.parse_args()

    test_csv = DATA_DIR / f"severity_{args.severity}" / "test.csv"

    output_dir = (
        RESULTS_DIR
        / args.model
        / f"severity_{args.severity}"
    )

    checkpoint_path = output_dir / "best_model.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, transform = build_model(args.model, 6)

    dataset = SeverityCorruptionDataset(
        test_csv,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    print(f"Model: {args.model}")
    print(f"Severity: {args.severity}")
    print(f"Test images: {len(dataset)}")
    print(f"Device: {device}")

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

            if batch_number % 100 == 0:
                print(
                    f"Processed {min(batch_number * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} images",
                    flush=True
                )

    labels = list(range(len(dataset.classes)))

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    results = {
        "model": args.model,
        "severity": args.severity,
        "test_images": len(dataset),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "test_metrics.json").open("w") as file:
        json.dump(results, file, indent=2)

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + dataset.classes)

        for class_name, row in zip(dataset.classes, matrix):
            writer.writerow([class_name] + row.tolist())

    with (output_dir / "test_predictions.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["path", "actual", "predicted"])

        for record, actual, predicted in zip(
            dataset.records, y_true, y_pred
        ):
            writer.writerow([
                record["path"],
                dataset.classes[actual],
                dataset.classes[predicted]
            ])

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=dataset.classes,
            zero_division=0
        )
    )

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()