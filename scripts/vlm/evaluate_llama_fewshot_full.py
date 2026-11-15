import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
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
    messages = []

    for label, row in references:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Classify this image corruption. "
                        "Return only the category name."
                    ),
                },
            ],
        })

        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": label}
            ],
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Classify this image corruption. "
                    "Choose exactly one category: "
                    + ", ".join(CLASSES)
                    + ". Return only the category name."
                ),
            },
        ],
    })

    images = [
        image_at(row["path"])
        for _, row in references
    ]
    images.append(image_at(test_path))

    return messages, images


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
    parser.add_argument(
        "--balanced-test",
        action="store_true",
        help="Evaluate one test image from each class"
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.limit < 0 or args.progress_every < 1:
        parser.error("Invalid limit or progress interval.")

    root = ROOT / f"severity_{args.severity}"
    train_rows = read_csv(root / "train.csv")
    test_rows = read_csv(root / "test.csv")
    if args.balanced_test:
        test_rows = [
            next(row for row in test_rows if row["class"] == label)
            for label in CLASSES
        ]

    total = (
        len(test_rows)
        if args.limit == 0
        else min(args.limit, len(test_rows))
    )

    references = select_references(train_rows, args.seed)

    output_dir = (
        Path("severity_results")
        / "llama_3_2_11b_fewshot_v2"
        / f"severity_{args.severity}"
        / ("balanced_diagnostic" if args.balanced_test else "full")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "test_predictions.csv"
    config_path = output_dir / "evaluation_config.json"

    config = {
        "model": MODEL_ID,
        "severity": args.severity,
        "seed": args.seed,
        "classes": CLASSES,
        "prompt_version": "six_references_conversation_v2",
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

        print("Loading Llama processor and model...", flush=True)

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
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

                messages, images = build_content(
                    references, row["path"]
                )

                prompt = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                inputs = processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                ).to(model.device)

                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(
                        dtype=model.dtype
                    )

                if i == len(existing):
                    print("\n===== INPUT DIAGNOSTICS =====", flush=True)
                    print("Input keys:", list(inputs.keys()), flush=True)

                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(
                                f"{key}: shape={tuple(value.shape)}, "
                                f"dtype={value.dtype}",
                                flush=True,
                            )

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
                ).strip().strip(".,!?;: \n")

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

    print("\n===== LLAMA FEW-SHOT RESULTS =====")
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