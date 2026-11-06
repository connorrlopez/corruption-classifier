
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# SETTINGS
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_DIR / "full_dataset" / "test.csv"
MODEL_PATH = PROJECT_DIR / "efficientnet_full_best_model.pth"

SAMPLE_COUNT = 100
BATCH_SIZE = 32
SEED = 42

CLASSES = [
    "clean",
    "gaussian",
    "shot",
    "impulse",
    "motion_blur",
    "defocus_blur",
]

# -----------------------------
# ALTERNATIVE CORRUPTIONS
# -----------------------------

def apply_corruption(image, corruption, rng):
    arr = np.array(image).astype(np.float32)

    if corruption == "clean":
        return image

    elif corruption == "gaussian":
        noise = rng.normal(0, 12, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    elif corruption == "shot":
        scale = 25
        arr = rng.poisson(arr / 255 * scale) / scale * 255
        arr = np.clip(arr, 0, 255)

    elif corruption == "impulse":
        mask = rng.random(arr.shape[:2])
        arr[mask < 0.015] = 0
        arr[(mask >= 0.015) & (mask < 0.03)] = 255

    elif corruption == "motion_blur":
        # Horizontal directional blur
        from scipy.ndimage import convolve
        kernel = np.zeros((7, 7), dtype=np.float32)
        kernel[3, :] = 1 / 7
        arr = np.stack([
            convolve(arr[:, :, c], kernel, mode="reflect")
            for c in range(3)
        ], axis=-1)

    elif corruption == "defocus_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))

    return Image.fromarray(arr.astype(np.uint8))


# -----------------------------
# DATASET
# -----------------------------

class AlternativeDataset(Dataset):
    def __init__(self, csv_path, transform):
        with open(csv_path, newline="") as f:
            records = list(csv.DictReader(f))

        # Select distinct original image paths.
        paths = sorted({r["path"] for r in records})
        random.Random(SEED).shuffle(paths)
        paths = paths[:SAMPLE_COUNT]

        self.samples = [
            (path, cls)
            for path in paths
            for cls in CLASSES
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, cls = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")

        # Reproducible noise for each sample
        rng = np.random.default_rng(SEED + index)
        img = apply_corruption(img, cls, rng)

        return self.transform(img), CLASSES.index(cls)


# -----------------------------
# LOAD MODEL
# -----------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASSES)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model = model.to(device)
model.eval()

# -----------------------------
# EVALUATE
# -----------------------------

dataset = AlternativeDataset(TEST_CSV, weights.transforms())
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu().tolist()
        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

print("\nAlternative Corruption Results")
print("==============================")
print(f"Test images: {len(dataset)}")
print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")

print("\nPer-Class Results:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=CLASSES,
    digits=4,
    zero_division=0
))
