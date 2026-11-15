import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"

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


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/vlm/test_gemma.py IMAGE_PATH")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Gemma 3 4B...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    print("Classifying image...")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    prediction = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print(f"\nPrediction: {prediction}")


if __name__ == "__main__":
    main()