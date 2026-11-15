from pathlib import Path
import random
import shutil
from collections import defaultdict

SOURCE_DIR = Path("/Users/connorlopez/Desktop/Capstone/dataset/test")
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

RANDOM_SEED = 42

# 200 frames per class
TRAIN_FRAMES_PER_CLASS = 160
VAL_FRAMES_PER_CLASS = 20
TEST_FRAMES_PER_CLASS = 20

CLASSES = [
    "clean",
    "gaussian",
    "shot",
    "impulse",
    "motion_blur",
    "defocus_blur",
]

FRAMES_PER_CLASS = (
    TRAIN_FRAMES_PER_CLASS
    + VAL_FRAMES_PER_CLASS
    + TEST_FRAMES_PER_CLASS
)

TOTAL_FRAMES = FRAMES_PER_CLASS * len(CLASSES)


# -----------------------------
# FIND ALL PNG IMAGES
# -----------------------------

all_images = list(
    SOURCE_DIR.rglob("*.png")
)

print(
    f"Found {len(all_images)} PNG images."
)


# -----------------------------
# GROUP IMAGES BY FRAME
# -----------------------------

frames = defaultdict(list)

for image_path in all_images:

    frame_number = (
        image_path.stem.split("_camera")[0]
    )

    frame_id = (
        image_path.parent.parent.name,
        image_path.parent.name,
        frame_number,
    )

    frames[frame_id].append(
        image_path
    )


# -----------------------------
# KEEP ONLY COMPLETE
# 4-CAMERA FRAMES
# -----------------------------

complete_frames = {
    frame_id: images
    for frame_id, images in frames.items()
    if len(images) == 4
}

print(
    f"Found {len(complete_frames)} "
    f"complete 4-camera frames."
)


# -----------------------------
# CHECK THAT WE HAVE ENOUGH
# -----------------------------

if len(complete_frames) < TOTAL_FRAMES:

    raise ValueError(
        "Not enough complete frames."
    )


# -----------------------------
# RANDOMLY SELECT FRAMES
# -----------------------------

random.seed(RANDOM_SEED)

selected_frame_ids = random.sample(
    list(complete_frames.keys()),
    TOTAL_FRAMES
)

random.shuffle(
    selected_frame_ids
)


# -----------------------------
# ASSIGN UNIQUE FRAMES
# TO EACH CLASS
# -----------------------------

class_frames = {}

start = 0

for class_name in CLASSES:

    end = start + FRAMES_PER_CLASS

    class_frames[class_name] = (
        selected_frame_ids[start:end]
    )

    start = end


# -----------------------------
# SPLIT EACH CLASS INTO
# TRAIN / VAL / TEST
# -----------------------------

dataset_splits = {}

for class_name in CLASSES:

    frame_ids = class_frames[class_name]

    train_frames = (
        frame_ids[
            :TRAIN_FRAMES_PER_CLASS
        ]
    )

    val_start = (
        TRAIN_FRAMES_PER_CLASS
    )

    val_end = (
        TRAIN_FRAMES_PER_CLASS
        + VAL_FRAMES_PER_CLASS
    )

    val_frames = (
        frame_ids[
            val_start:val_end
        ]
    )

    test_frames = (
        frame_ids[val_end:]
    )

    dataset_splits[class_name] = {
        "train": train_frames,
        "val": val_frames,
        "test": test_frames,
    }


# -----------------------------
# DELETE OLD DATA FOLDER
# -----------------------------

if DATA_DIR.exists():

    shutil.rmtree(
        DATA_DIR
    )


# -----------------------------
# CREATE NEW DATASET
# -----------------------------

for class_name in CLASSES:

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        output_dir = (
            DATA_DIR
            / split_name
            / class_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        frame_ids = (
            dataset_splits[
                class_name
            ][split_name]
        )

        image_counter = 0

        for frame_id in frame_ids:

            images = sorted(
                complete_frames[
                    frame_id
                ]
            )

            scene = frame_id[0]
            vehicle = frame_id[1]

            for image_path in images:

                new_name = (
                    f"{scene}_"
                    f"{vehicle}_"
                    f"{image_path.name}"
                )

                shutil.copy2(
                    image_path,
                    output_dir
                    / new_name
                )

                image_counter += 1

        print(
            f"{split_name}/"
            f"{class_name}: "
            f"{len(frame_ids)} frames, "
            f"{image_counter} images"
        )


# -----------------------------
# VERIFY NO FRAME REUSE
# -----------------------------

all_used_frames = []

for class_name in CLASSES:

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        all_used_frames.extend(
            dataset_splits[
                class_name
            ][split_name]
        )

assert (
    len(all_used_frames)
    ==
    len(set(all_used_frames))
)


# -----------------------------
# SUMMARY
# -----------------------------

print(
    "\nDataset created successfully!"
)

print(
    f"\nClasses: {len(CLASSES)}"
)

print(
    f"Frames per class: "
    f"{FRAMES_PER_CLASS}"
)

print(
    f"Images per class: "
    f"{FRAMES_PER_CLASS * 4}"
)

print(
    "\nTraining images:   "
    f"{TRAIN_FRAMES_PER_CLASS * 4 * len(CLASSES)}"
)

print(
    "Validation images: "
    f"{VAL_FRAMES_PER_CLASS * 4 * len(CLASSES)}"
)

print(
    "Testing images:    "
    f"{TEST_FRAMES_PER_CLASS * 4 * len(CLASSES)}"
)

print(
    "Total images:      "
    f"{TOTAL_FRAMES * 4}"
)

print(
    "\nVerified: "
    "Every source frame is used only once."
)