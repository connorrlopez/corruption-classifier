
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "mistral-experimental/pixtral-12b"

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
        print("Usage: python scripts/vlm/test_pixtral.py IMAGE_PATH")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

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

    image = Image.open(image_path).convert("RGB")

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

    inputs["pixel_values"] = inputs["pixel_values"].to(
        dtype=model.dtype
    )

    print("Classifying image...")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]

    prediction = processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    print(f"\nPrediction: {prediction}")


if __name__ == "__main__":
    main()
