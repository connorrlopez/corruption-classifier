from pathlib import Path
import csv
import hashlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from imagecorruptions import corrupt


# -----------------------------
# CLASS SETTINGS
# -----------------------------

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
# CUSTOM DATASET
# -----------------------------

class FullCorruptionDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {
            name: index
            for index, name in enumerate(CLASSES)
        }

        with open(csv_path, newline="") as file:
            self.records = list(csv.DictReader(file))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = Path(record["path"])
        class_name = record["class"]
        severity = int(record["severity"])

        # Load the original image
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_array = np.array(image)

        # Apply corruption in memory
        if class_name != "clean":

            corruption_name = CORRUPTIONS[class_name]

            # Reproducible seed for each image
            seed_text = (
                f"{image_path}|{class_name}|{severity}"
            )

            seed = int.from_bytes(
                hashlib.sha256(
                    seed_text.encode()
                ).digest()[:4],
                "big"
            )

            # Preserve NumPy's random state
            previous_state = np.random.get_state()

            try:
                np.random.seed(seed)

                image_array = corrupt(
                    image_array,
                    corruption_name=corruption_name,
                    severity=severity
                )
            finally:
                np.random.set_state(previous_state)

        image = Image.fromarray(image_array)

        # Apply model preprocessing
        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[class_name]

        return image, label
