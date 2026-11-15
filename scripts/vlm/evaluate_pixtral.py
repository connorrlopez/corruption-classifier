
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "mistral-experimental/pixtral-12b"

CLASSES = [
    "clean",
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "motion_blur",
    "defocus_blur",
]

PROMPT = """
Examine the image for visual corruption.

Choose exactly one category:
clean
gaussian_noise
shot_noise
impulse_noise
motion_blur
defocus_blur

Return only the category name.
"""


def classify_image(model, processor, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # Match the image tensor's dtype to the model's BF16 weights.
    inputs["pixel_values"] = inputs["pixel_values"].to(
        dtype=model.dtype
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="results/vlm/pixtral_diagnostic.csv",
    )
    args = parser.parse_args()

    manifest_path = Path(args.dataset) / "manifest.csv"

    with manifest_path.open(newline="") as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError("The evaluation manifest is empty.")

    print("Loading Pixtral processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Pixtral 12B...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()

    results = []

    for record in records:
        image_path = Path(record["image"])
        actual = record["actual"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        prediction = classify_image(model, processor, image)

        if prediction not in CLASSES:
            prediction = "invalid"

        results.append({
            "image": str(image_path),
            "severity": int(record["severity"]),
            "actual": actual,
            "predicted": prediction,
        })

        print(f"{actual} -> {prediction}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "severity", "actual", "predicted"],
        )
        writer.writeheader()
        writer.writerows(results)

    actual = [row["actual"] for row in results]
    predicted = [row["predicted"] for row in results]

    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    print("\nPixtral diagnostic results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
