from pathlib import Path
import random
import csv

import numpy as np
from PIL import Image
from imagecorruptions import corrupt


# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

RANDOM_SEED = 42
SPLITS = ["train", "val", "test"]


# -----------------------------
# SET RANDOM SEED
# -----------------------------

random.seed(RANDOM_SEED)


# -----------------------------
# CSV FOR RECORDING SEVERITIES
# -----------------------------

csv_path = DATA_DIR / "gaussian_severities.csv"

records = []


# -----------------------------
# PROCESS EACH SPLIT
# -----------------------------

for split in SPLITS:

    clean_dir = DATA_DIR / split / "clean"
    gaussian_dir = DATA_DIR / split / "gaussian"

    gaussian_dir.mkdir(parents=True, exist_ok=True)

    clean_images = sorted(clean_dir.glob("*.png"))

    print(f"\nProcessing {split}: {len(clean_images)} clean images")

    for image_path in clean_images:

        # Random severity from 1 through 5
        severity = random.randint(1, 5)

        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Add Gaussian noise
        corrupted_array = corrupt(
            image_array,
            corruption_name="gaussian_noise",
            severity=severity
        )

        # Convert back to an image
        corrupted_image = Image.fromarray(corrupted_array)

        # Keep the same filename
        output_path = gaussian_dir / image_path.name

        corrupted_image.save(output_path)

        # Record information for later analysis
        records.append({
            "split": split,
            "filename": image_path.name,
            "corruption": "gaussian_noise",
            "severity": severity
        })


# -----------------------------
# SAVE SEVERITY INFORMATION
# -----------------------------

with open(csv_path, "w", newline="") as csv_file:

    fieldnames = [
        "split",
        "filename",
        "corruption",
        "severity"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


# -----------------------------
# SUMMARY
# -----------------------------

print("\nGaussian corruption complete!")
print(f"Corrupted images created: {len(records)}")
print(f"Severity information saved to:")
print(csv_path)