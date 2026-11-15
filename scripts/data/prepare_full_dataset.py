from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path.home() / "Desktop/Capstone/dataset"
OUTPUT_DIR = PROJECT_DIR / "full_dataset"

RANDOM_SEED = 42

SPLITS = {
    "train": "train",
    "val": "validate",
    "test": "test",
}

CLASSES = [
    "clean",
    "gaussian",
    "shot",
    "impulse",
    "motion_blur",
    "defocus_blur",
]

SEVERITIES = [1, 2, 3, 4, 5]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# PROCESS EACH SPLIT
# -----------------------------

for split_name, source_name in SPLITS.items():

    source_folder = SOURCE_DIR / source_name

    if not source_folder.exists():
        raise FileNotFoundError(source_folder)

    print(f"\nProcessing {split_name}...")

    images = sorted(source_folder.rglob("*.png"))

    print(f"Found {len(images)} images.")

    # Group images by frame, including incomplete frames
    frames = defaultdict(list)

    for image_path in images:

        frame_number = image_path.stem.split("_camera")[0]

        frame_id = (
            str(image_path.parent),
            frame_number,
        )

        frames[frame_id].append(image_path)

    # Include every frame, regardless of camera count
    all_frames = [
        sorted(frame_images)
        for frame_images in frames.values()
    ]

    print(f"Total frames: {len(all_frames)}")

    # Shuffle frames reproducibly
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_frames)

    records = []

    # Assign one class per frame
    for index, frame_images in enumerate(all_frames):

        class_name = CLASSES[index % len(CLASSES)]

        if class_name == "clean":
            severity = 0
        else:
            severity = SEVERITIES[
                (index // len(CLASSES)) % len(SEVERITIES)
            ]

        for image_path in frame_images:

            records.append({
                "path": str(image_path),
                "class": class_name,
                "severity": severity,
            })

    # Save CSV manifest
    csv_path = OUTPUT_DIR / f"{split_name}.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["path", "class", "severity"]
        )

        writer.writeheader()
        writer.writerows(records)

    # Print summary
    counts = Counter(
        record["class"] for record in records
    )

    print(f"Saved: {csv_path}")
    print(f"Total assigned images: {len(records)}")

    for class_name in CLASSES:
        print(f"  {class_name}: {counts[class_name]}")

print("\nFull dataset manifests created successfully!")
