from pathlib import Path
from PIL import Image
from imagecorruptions import corrupt
import numpy as np
import csv
import random

# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

RANDOM_SEED = 42

CORRUPTIONS = {
    "gaussian": "gaussian_noise",
    "shot": "shot_noise",
    "impulse": "impulse_noise",
    "motion_blur": "motion_blur",
    "defocus_blur": "defocus_blur",
}

SPLITS = ["train", "val", "test"]

random.seed(RANDOM_SEED)


# -----------------------------
# CSV RECORDS
# -----------------------------

records = []


# -----------------------------
# APPLY CORRUPTIONS
# -----------------------------

for split_name in SPLITS:

    print(f"\nProcessing {split_name}...")

    for class_name, corruption_name in CORRUPTIONS.items():

        folder = DATA_DIR / split_name / class_name

        images = sorted(folder.glob("*.png"))

        # Shuffle so severity is not tied to filename order
        random.shuffle(images)

        print(
            f"  {class_name}: "
            f"{len(images)} images"
        )

        # Assign severity levels 1-5 evenly
        for index, image_path in enumerate(images):

            severity = (index % 5) + 1

            # Open image and convert to RGB
            image = Image.open(image_path).convert("RGB")

            # Convert PIL image to NumPy array
            image_array = np.array(image)

            # Apply corruption
            corrupted_image = corrupt(
                image_array,
                corruption_name=corruption_name,
                severity=severity
            )

            # Convert back to PIL image
            corrupted_image = Image.fromarray(corrupted_image)

            # Replace the reserved clean image
            # with its corrupted version
            corrupted_image.save(image_path)

            # Save information for later analysis
            records.append({
                "split": split_name,
                "filename": image_path.name,
                "class": class_name,
                "corruption": corruption_name,
                "severity": severity,
            })


# -----------------------------
# SAVE CSV
# -----------------------------

csv_path = DATA_DIR / "corruption_metadata.csv"

with open(csv_path, "w", newline="") as csv_file:

    fieldnames = [
        "split",
        "filename",
        "class",
        "corruption",
        "severity",
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

print("\nCorruptions applied successfully!")

print(f"\nTotal corrupted images: {len(records)}")
print(f"Metadata saved to: {csv_path}")

print("\nClean images were left unchanged.")