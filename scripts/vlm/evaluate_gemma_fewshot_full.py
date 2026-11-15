import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"
CLASSES = [
    "clean", "gaussian", "shot", "impulse",
    "motion_blur", "defocus_blur",
]
ROOT = Path.home() / "capstone/corrupted_datasets"
FIELDNAMES = [
    "index", "image", "actual", "predicted",
    "raw_response", "seconds",
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def image_at(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def select_references(rows, seed):
    rng = random.Random(seed)
    references = []

    for label in CLASSES:
        choices = [
            row for row in rows if row["class"] == label
        ]
        if not choices:
            raise ValueError(f"Missing training class: {label}")
        references.append((label, rng.choice(choices)))

    return references


def build_content(references, test_path):
    content = [{
        "type": "text",
        "text": (
            "You are classifying image corruption. "
            "Study these six labeled reference images. "
            "Each image demonstrates one category."
        ),
    }]

    for label, row in references:
        content.append({
            "type": "text",
            "text": f"Reference image. Correct label: {label}",
        })
        content.append({
            "type": "image",
            "image": image_at(row["path"]),
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
        "image": image_at(test_path),
    })

    return content


def load_existing(path, test_rows):
    if not path.exists():
        return []

    rows = read_csv(path)

    for i, row in enumerate(rows):
        if (
            int(row["index"]) != i
            or row["image"] != test_rows[i]["path"]
            or row["actual"] != test_rows[i]["class"]
        ):
            raise ValueError(
                "Existing predictions do not match the test manifest."
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--severity", type=int, choices=[1, 3, 5], default=1
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="0 means all test images"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.limit < 0 or args.progress_every < 1:
        parser.error("Invalid limit or progress interval.")

    root = ROOT / f"severity_{args.severity}"
    train_rows = read_csv(root / "train.csv")
    test_rows = read_csv(root / "test.csv")

    total = (
        len(test_rows)
        if args.limit == 0
        else min(args.limit, len(test_rows))
    )

    references = select_references(train_rows, args.seed)

    output_dir = (
        Path("severity_results")
        / "gemma_3_4b_fewshot"
        / f"severity_{args.severity}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "test_predictions.csv"
    config_path = output_dir / "evaluation_config.json"

    config = {
        "model": MODEL_ID,
        "severity": args.severity,
        "seed": args.seed,
        "classes": CLASSES,
        "prompt_version": "six_references_v1",
        "references": [
            {
                "class": label,
                "path": row["path"],
                "source": row["source"],
            }
            for label, row in references
        ],
    }

    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(
                "Saved configuration differs; refusing to resume."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))

    existing = load_existing(predictions_path, test_rows)

    if len(existing) > total:
        raise ValueError(
            "Existing results exceed the requested limit."
        )

    print(
        f"Severity: {args.severity}\n"
        f"Test images: {len(test_rows)}\n"
        f"Target: {total}\n"
        f"Already saved: {len(existing)}",
        flush=True,
    )

    for label, row in references:
        print(f"Reference {label}: {row['path']}", flush=True)

    if len(existing) < total:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        print("Loading Gemma processor and model...", flush=True)

        processor = AutoProcessor.from_pretrained(
            MODEL_ID, use_fast=False
        )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        with predictions_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

            if not existing:
                writer.writeheader()
                f.flush()

            for i in range(len(existing), total):
                row = test_rows[i]
                start = time.perf_counter()

                messages = [{
                    "role": "user",
                    "content": build_content(
                        references, row["path"]
                    ),
                }]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                    )

                generated = output[0][
                    inputs["input_ids"].shape[-1]:
                ]

                raw = processor.decode(
                    generated, skip_special_tokens=True
                ).strip()

                cleaned = raw.lower().replace(
                    "<end_of_turn>", ""
                ).strip()

                predicted = (
                    cleaned if cleaned in CLASSES else "invalid"
                )

                elapsed = time.perf_counter() - start

                writer.writerow({
                    "index": i,
                    "image": row["path"],
                    "actual": row["class"],
                    "predicted": predicted,
                    "raw_response": raw,
                    "seconds": f"{elapsed:.4f}",
                })
                f.flush()

                if (
                    (i + 1) % args.progress_every == 0
                    or i + 1 == total
                ):
                    print(
                        f"Processed {i + 1}/{total}; "
                        f"latest: {row['class']} -> {predicted}; "
                        f"{elapsed:.2f}s",
                        flush=True,
                    )

                del inputs, output, generated

    results = read_csv(predictions_path)[:total]

    actual = [r["actual"] for r in results]
    predicted = [r["predicted"] for r in results]

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "model": MODEL_ID,
        "severity": args.severity,
        "evaluated": len(results),
        "total_test_images": len(test_rows),
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "invalid_predictions": Counter(predicted)["invalid"],
        "total_inference_seconds": sum(
            float(r["seconds"]) for r in results
        ),
    }

    is_full = total == len(test_rows)

    metrics_path = output_dir / (
        "test_metrics.json"
        if is_full else "diagnostic_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(
        actual,
        predicted,
        labels=CLASSES + ["invalid"],
    )

    cm_path = output_dir / (
        "confusion_matrix.csv"
        if is_full else "diagnostic_confusion_matrix.csv"
    )

    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted"] + CLASSES + ["invalid"])

        for label, counts in zip(CLASSES + ["invalid"], cm):
            writer.writerow([label] + list(map(int, counts)))

    print("\n===== GEMMA FEW-SHOT RESULTS =====")
    print(f"Images: {len(results)} / {len(test_rows)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Invalid predictions: {metrics['invalid_predictions']}")
    print(
        "Total inference time: "
        f"{metrics['total_inference_seconds'] / 60:.2f} minutes"
    )
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()