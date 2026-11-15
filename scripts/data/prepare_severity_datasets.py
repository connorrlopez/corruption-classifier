from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import os
import random
import shutil

import numpy as np
from PIL import Image
from imagecorruptions import corrupt


# -----------------------------
# SETTINGS
# -----------------------------

SOURCE_DIR = Path.home() / "capstone/Dataset"
OUTPUT_DIR = Path.home() / "capstone/corrupted_datasets"

RANDOM_SEED = 42
SEVERITIES = [1, 3, 5]
NUM_WORKERS = 8

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

CORRUPTIONS = {
    "gaussian": "gaussian_noise",
    "shot": "shot_noise",
    "impulse": "impulse_noise",
    "motion_blur": "motion_blur",
    "defocus_blur": "defocus_blur",
}


# -----------------------------
# VALIDATE EXISTING IMAGE
# -----------------------------

def valid_image(path):
    if not path.is_file():
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


# -----------------------------
# GENERATE AND SAVE IMAGE
# -----------------------------

def process_image(task):
    source_path, destination, class_name, severity = task

    source_path = Path(source_path)
    destination = Path(destination)

    # Resume safely if this image is already complete
    if valid_image(destination):
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Each worker uses a unique temporary filename
    temporary = destination.with_name(
        f"{destination.stem}.{os.getpid()}.tmp.png"
    )

    try:
        if class_name == "clean":
            # Preserve the original clean PNG exactly
            shutil.copyfile(source_path, temporary)

        else:
            with Image.open(source_path) as source:
                image_array = np.asarray(
                    source.convert("RGB")
                ).copy()

            corruption_name = CORRUPTIONS[class_name]

            # Deterministic random seed per image
            seed_text = (
                f"{source_path}|{class_name}|{severity}"
            )

            seed = int.from_bytes(
                hashlib.sha256(
                    seed_text.encode("utf-8")
                ).digest()[:4],
                "big"
            )

            np.random.seed(seed)

            corrupted_array = corrupt(
                image_array,
                corruption_name=corruption_name,
                severity=severity
            )

            Image.fromarray(
                np.asarray(corrupted_array, dtype=np.uint8)
            ).save(temporary, format="PNG")

        if not valid_image(temporary):
            raise RuntimeError(
                f"Generated image is invalid: {temporary}"
            )

        # Atomic replacement after successful generation
        temporary.replace(destination)

        return "saved"

    finally:
        if temporary.exists():
            temporary.unlink()


# -----------------------------
# ASSIGN CLASSES TO FRAMES
# -----------------------------

def assign_frames(source_folder):
    images = sorted(source_folder.rglob("*.png"))

    print(
        f"Found {len(images)} images in "
        f"{source_folder.name}",
        flush=True
    )

    if not images:
        raise RuntimeError(
            f"No PNG images found in {source_folder}"
        )

    # Keep all cameras from one original frame together
    frames = defaultdict(list)

    for image_path in images:
        frame_number = (
            image_path.stem.split("_camera")[0]
        )

        frame_id = (
            str(image_path.parent),
            frame_number
        )

        frames[frame_id].append(image_path)

    all_frames = list(frames.values())

    # Reproducible assignment
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_frames)

    assignments = []

    for index, frame_images in enumerate(all_frames):
        class_name = CLASSES[index % len(CLASSES)]

        for image_path in frame_images:
            assignments.append(
                (image_path, class_name)
            )

    counts = Counter(
        class_name
        for _, class_name in assignments
    )

    print(
        f"Total frames: {len(all_frames)}",
        flush=True
    )

    for class_name in CLASSES:
        print(
            f"  {class_name}: {counts[class_name]} images",
            flush=True
        )

    return assignments


# -----------------------------
# LIMIT FRAMES FOR TESTING
# -----------------------------

def limit_assignments(assignments, max_frames):
    if max_frames is None:
        return assignments

    selected = []
    seen_frames = set()

    for image_path, class_name in assignments:
        frame_id = (
            str(image_path.parent),
            image_path.stem.split("_camera")[0]
        )

        if frame_id not in seen_frames:
            if len(seen_frames) >= max_frames:
                break

            seen_frames.add(frame_id)

        selected.append((image_path, class_name))

    return selected


# -----------------------------
# WRITE CSV MANIFEST
# -----------------------------

def write_manifest(manifest_path, records):
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = manifest_path.with_suffix(".csv.tmp")

    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "source",
                "class",
                "severity",
                "split"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    temporary.replace(manifest_path)


# -----------------------------
# GENERATE ONE SPLIT
# -----------------------------

def generate_split(
    assignments,
    source_folder,
    output_root,
    split_name,
    severity
):
    output_folder = (
        output_root
        / f"severity_{severity}"
        / split_name
    )

    tasks = []
    records = []

    for source_path, class_name in assignments:
        relative_path = (
            source_path.relative_to(source_folder)
        )

        destination = (
            output_folder
            / class_name
            / relative_path
        )

        tasks.append((
            str(source_path),
            str(destination),
            class_name,
            severity
        ))

        records.append({
            "path": str(destination),
            "source": str(source_path),
            "class": class_name,
            "severity": (
                0 if class_name == "clean"
                else severity
            ),
            "split": split_name
        })

    print(
        f"\nGenerating {split_name}, "
        f"severity {severity}: {len(tasks)} images",
        flush=True
    )

    saved = 0
    skipped = 0

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        results = executor.map(
            process_image,
            tasks,
            chunksize=8
        )

        for index, result in enumerate(results, start=1):
            if result == "saved":
                saved += 1
            else:
                skipped += 1

            if index % 1000 == 0:
                print(
                    f"  {index}/{len(tasks)} processed",
                    flush=True
                )

    # Only write the manifest after successful completion
    manifest_path = (
        output_root
        / f"severity_{severity}"
        / f"{split_name}.csv"
    )

    write_manifest(manifest_path, records)

    print(
        f"Completed: {saved} saved, "
        f"{skipped} skipped",
        flush=True
    )

    print(
        f"Manifest: {manifest_path}",
        flush=True
    )


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate severity-specific OPV2V "
            "corruption datasets."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of original frames "
            "per split for a small test"
        )
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error(
            "--limit must be at least 1"
        )

    # Keep test outputs separate from full datasets
    if args.limit is not None:
        output_root = (
            OUTPUT_DIR / f"test_limit_{args.limit}"
        )
        print(
            "\nTEST MODE: using separate output folder",
            flush=True
        )
    else:
        output_root = OUTPUT_DIR
        print(
            "\nFULL DATASET GENERATION",
            flush=True
        )

    print(f"Source: {SOURCE_DIR}", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Workers: {NUM_WORKERS}", flush=True)
    print(f"Severities: {SEVERITIES}", flush=True)

    for split_name, source_name in SPLITS.items():
        source_folder = SOURCE_DIR / source_name

        if not source_folder.is_dir():
            raise FileNotFoundError(
                f"Missing source folder: {source_folder}"
            )

        print(
            f"\nPreparing {split_name}...",
            flush=True
        )

        # Assign original frames only once per split
        assignments = assign_frames(source_folder)

        assignments = limit_assignments(
            assignments,
            args.limit
        )

        for severity in SEVERITIES:
            generate_split(
                assignments=assignments,
                source_folder=source_folder,
                output_root=output_root,
                split_name=split_name,
                severity=severity
            )

    print(
        "\nAll requested datasets generated!",
        flush=True
    )


if __name__ == "__main__":
    main()