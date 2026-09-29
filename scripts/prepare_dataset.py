from pathlib import Path
import random
import shutil
from collections import defaultdict

# -----------------------------
# SETTINGS
# -----------------------------

# CHANGE THIS to your actual OPV2V test folder
SOURCE_DIR = Path("/Users/connorlopez/Desktop/Capstone/dataset/test")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

RANDOM_SEED = 42

TRAIN_FRAMES = 200
VAL_FRAMES = 25
TEST_FRAMES = 25

TOTAL_FRAMES = TRAIN_FRAMES + VAL_FRAMES + TEST_FRAMES


# -----------------------------
# FIND AND GROUP IMAGES BY FRAME
# -----------------------------

all_images = list(SOURCE_DIR.rglob("*.png"))

print(f"Found {len(all_images)} PNG images.")

frames = defaultdict(list)

for image_path in all_images:
    # Example:
    # 000242_camera2.png -> 000242
    frame_number = image_path.stem.split("_camera")[0]

    # Include the parent folders because frame numbers can repeat
    # in different OPV2V scenes/vehicles.
    frame_id = (
        image_path.parent.parent.name,
        image_path.parent.name,
        frame_number,
    )

    frames[frame_id].append(image_path)


# -----------------------------
# KEEP COMPLETE 4-CAMERA FRAMES
# -----------------------------

complete_frames = {
    frame_id: images
    for frame_id, images in frames.items()
    if len(images) == 4
}

print(f"Found {len(complete_frames)} complete 4-camera frames.")

if len(complete_frames) < TOTAL_FRAMES:
    raise ValueError("Not enough complete frames.")


# -----------------------------
# RANDOMLY SELECT 250 FRAMES
# -----------------------------

random.seed(RANDOM_SEED)

selected_frame_ids = random.sample(
    list(complete_frames.keys()),
    TOTAL_FRAMES
)

random.shuffle(selected_frame_ids)

train_frames = selected_frame_ids[:TRAIN_FRAMES]

val_frames = selected_frame_ids[
    TRAIN_FRAMES:
    TRAIN_FRAMES + VAL_FRAMES
]

test_frames = selected_frame_ids[
    TRAIN_FRAMES + VAL_FRAMES:
]


# -----------------------------
# CLEAR OLD DATA
# -----------------------------

if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)


# -----------------------------
# COPY COMPLETE FRAMES
# -----------------------------

splits = {
    "train": train_frames,
    "val": val_frames,
    "test": test_frames,
}

for split_name, frame_ids in splits.items():

    output_dir = DATA_DIR / split_name / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_counter = 0

    for frame_id in frame_ids:

        images = sorted(complete_frames[frame_id])

        scene, vehicle, frame_number = frame_id

        for image_path in images:

            # Preserve enough information to identify the source
            new_name = (
                f"{scene}_{vehicle}_"
                f"{image_path.name}"
            )

            shutil.copy2(
                image_path,
                output_dir / new_name
            )

            image_counter += 1

    print(
        f"{split_name}: "
        f"{len(frame_ids)} frames, "
        f"{image_counter} images"
    )


# -----------------------------
# VERIFY NO FRAME OVERLAP
# -----------------------------

train_set = set(train_frames)
val_set = set(val_frames)
test_set = set(test_frames)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)


# -----------------------------
# SUMMARY
# -----------------------------

print("\nDataset created successfully!")
print(f"Training:   {TRAIN_FRAMES} frames / {TRAIN_FRAMES * 4} images")
print(f"Validation: {VAL_FRAMES} frames / {VAL_FRAMES * 4} images")
print(f"Testing:    {TEST_FRAMES} frames / {TEST_FRAMES * 4} images")
print(f"Total:      {TOTAL_FRAMES} frames / {TOTAL_FRAMES * 4} images")
print("\nVerified: No frame overlap between splits.")