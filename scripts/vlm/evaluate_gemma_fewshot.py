import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"

CLASSES = [
    "clean",
    "gaussian",
    "shot",
    "impulse",
    "motion_blur",
    "defocus_blur",
]

DATA_ROOT = Path.home() / "capstone/corrupted_datasets"


def read_manifest(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def select_examples(train_rows, seed=42):
    rng = random.Random(seed)
    examples = []

    for label in CLASSES:
        candidates = [
            row for row in train_rows
            if row["class"] == label
        ]
        if not candidates:
            raise ValueError(f"No training images for {label}")
        examples.append((label, rng.choice(candidates)))

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity", type=int, default=1,
                        choices=[1, 3, 5])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    root = DATA_ROOT / f"severity_{args.severity}"
    train_rows = read_manifest(root / "train.csv")
    test_rows = read_manifest(root / "test.csv")

    examples = select_examples(train_rows)
    test_row = test_rows[args.index]

    print("Reference examples:")
    for label, row in examples:
        print(f"  {label}: {row['path']}")

    print(f"\nActual test label: {test_row['class']}")

    print("\nLoading Gemma processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Gemma model...", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    content = [
        {
            "type": "text",
            "text": (
                "You are classifying image corruption. "
                "Study these six labeled reference images. "
                "Each image demonstrates one category."
            ),
        }
    ]

    for label, row in examples:
        content.append({
            "type": "text",
            "text": f"Reference image. Correct label: {label}"
        })
        content.append({
            "type": "image",
            "image": load_image(row["path"]),
        })

    content.append({
        "type": "text",
        "text": (
            "Now classify the following test image. "
            "Choose exactly one label from: "
            + ", ".join(CLASSES)
            + ". Return only the label."
        ),
    })
    content.append({
        "type": "image",
        "image": load_image(test_row["path"]),
    })

    messages = [{"role": "user", "content": content}]

    print("\nProcessing seven images...", flush=True)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("Generating prediction...", flush=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    prediction = processor.decode(
        generated, skip_special_tokens=True
    ).strip().lower()

    print("\n===== GEMMA FEW-SHOT RESULT =====")
    print("Severity:", args.severity)
    print("Actual:", test_row["class"])
    print("Predicted:", prediction)
    print("Correct:", prediction == test_row["class"])


if __name__ == "__main__":
    main()