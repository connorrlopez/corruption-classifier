from pathlib import Path
import csv

from PIL import Image
from torch.utils.data import Dataset


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


# -----------------------------
# PRE-GENERATED IMAGE DATASET
# -----------------------------

class SeverityCorruptionDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.transform = transform

        self.classes = CLASSES
        self.class_to_idx = {
            name: index
            for index, name in enumerate(CLASSES)
        }

        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {csv_path}"
            )

        with csv_path.open(newline="") as file:
            self.records = list(csv.DictReader(file))

        if not self.records:
            raise ValueError(
                f"Manifest contains no images: {csv_path}"
            )

        for record in self.records:
            if record["class"] not in self.class_to_idx:
                raise ValueError(
                    f"Unknown class: {record['class']}"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Load the already-generated PNG
        image_path = Path(record["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = self.class_to_idx[record["class"]]

        return image, label