
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from imagecorruptions import corrupt

CLASSES = [
    "clean",
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "motion_blur",
    "defocus_blur",
]


def frame_key(path):
    # Group camera views from the same frame.
    frame_id = path.stem.split("_camera")[0]
    return (str(path.parent), frame_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--images-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    dataset = Path(args.dataset)
    output = Path(args.output)

    # Select one camera image per frame.
    unique_frames = {}

    for path in sorted(dataset.rglob("*.png")):
        key = frame_key(path)
        if key not in unique_frames:
            unique_frames[key] = path

    candidates = list(unique_frames.values())
    required = len(CLASSES) * args.images_per_class

    if len(candidates) < required:
        raise ValueError(
            f"Need {required} unique frames, found {len(candidates)}."
        )

    selected = rng.sample(candidates, required)

    output.mkdir(parents=True, exist_ok=True)
    records = []

    for class_index, class_name in enumerate(CLASSES):
        start = class_index * args.images_per_class
        end = start + args.images_per_class

        class_dir = output / class_name
        class_dir.mkdir(exist_ok=True)

        for index, source_path in enumerate(selected[start:end]):
            with Image.open(source_path) as source:
                image = source.convert("RGB")

            if class_name != "clean":
                corrupted = corrupt(
                    np.array(image),
                    corruption_name=class_name,
                    severity=args.severity,
                )
                image = Image.fromarray(corrupted)

            destination = class_dir / f"{index:04d}.png"
            image.save(destination)

            records.append({
                "image": str(destination),
                "source": str(source_path),
                "actual": class_name,
                "severity": 0 if class_name == "clean" else args.severity,
            })

            print(f"Saved: {destination}")

    manifest = output / "manifest.csv"

    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "source", "actual", "severity"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nGenerated {len(records)} images.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
