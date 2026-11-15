
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = PROJECT_DIR / "full_dataset" / "test.csv"
MODEL_PATH = PROJECT_DIR / "efficientnet_full_best_model.pth"

SAMPLE_COUNT = 100
SEED = 42

DENSITIES = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

CLASSES = [
    "clean",
    "gaussian",
    "shot",
    "impulse",
    "motion_blur",
    "defocus_blur",
]

# -----------------------------
# DEVICE
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------

weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASSES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# LOAD TEST IMAGES
# -----------------------------

with open(TEST_CSV, newline="") as file:
    records = list(csv.DictReader(file))

paths = sorted({record["path"] for record in records})

random.Random(SEED).shuffle(paths)

paths = paths[:SAMPLE_COUNT]

print(f"Testing {len(paths)} original images.")

# -----------------------------
# TEST IMPULSE NOISE DENSITIES
# -----------------------------

print("\nImpulse Noise Density Results")
print("=============================")

for density in DENSITIES:

    correct = 0
    prediction_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    for index, path in enumerate(paths):

        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.array(image).copy()

        rng = np.random.default_rng(SEED + index)

        mask = rng.random(array.shape[:2])

        array[mask < density / 2] = 0
        array[
            (mask >= density / 2) &
            (mask < density)
        ] = 255

        corrupted_image = Image.fromarray(array)

        tensor = transform(corrupted_image)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prediction = output.argmax(dim=1).item()
            predicted_class = CLASSES[prediction]
            prediction_counts[predicted_class] += 1

        if prediction == CLASSES.index("impulse"):
            correct += 1

    accuracy = correct / len(paths)

    print(
        f"Density {density * 100:.0f}%: "
        f"{correct}/{len(paths)} "
        f"({accuracy:.4f})"
    )
    for class_name, count in prediction_counts.items():
        print(f"  {class_name}: {count}")

from torchvision.transforms.functional import to_pil_image

output_dir = PROJECT_DIR / "results" / "impulse_debug"
output_dir.mkdir(parents=True, exist_ok=True)

with Image.open(paths[0]) as image:
    original = image.convert("RGB")

for density in [0.0, 0.01, 0.20]:

    array = np.array(original).copy()

    if density > 0:
        rng = np.random.default_rng(SEED)
        mask = rng.random(array.shape[:2])

        array[mask < density / 2] = 0
        array[
            (mask >= density / 2) &
            (mask < density)
        ] = 255

    image = Image.fromarray(array)

    name = f"density_{int(density * 100)}"

    image.save(output_dir / f"{name}_before.png")

    transformed = transform(image)

    # Undo ImageNet normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    visual_tensor = (transformed.cpu() * std + mean).clamp(0, 1)

    to_pil_image(visual_tensor).save(
        output_dir / f"{name}_after.png"
    )

print(f"Debug images saved to: {output_dir}")

print("\nTesting complete!")
