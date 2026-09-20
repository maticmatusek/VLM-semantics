# pip install accelerate bitsandbytes transformers pillow

import json
import os
import torch
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image  # kept from your original code
####################################################################################################

model_id = "google/gemma-3-4b-it"

#model_id = "GaMS-Beta/SVILA-1-4B"

####################################################################################################
# 4-bit NF4 quantization config (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # change to torch.float32 if you get CUDA issues
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,      # compute dtype; try float32 if needed
    #quantization_config=bnb_config,  # <-------------------------------------------------------------- quantization
).eval()

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("hf_device_map:", getattr(model, "hf_device_map", None))

processor = AutoProcessor.from_pretrained(model_id)

# ---------- helper: ask Gemma one question about one local image ----------

def ask_gemma(image_path: str, question: str) -> str:
    # load local image
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # move inputs to first visible CUDA device
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded.strip()

# ---------- main: load JSON, run Gemma, save new JSON ----------

input_json_path = "data/golden_final_test.json"
output_json_path = f"out_jsons/{model_id}_golden_final_test.json"

# Load original data
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Iterate over all items and all SLO/F*/I*/A* questions
for idx, item in enumerate(data):
    # use local image path from "image_path"
    raw_image = item.get("image_path", None)
    if raw_image is None:
        continue

    # normalize slashes
    image_path = "data/" + raw_image.replace("\\", "/")

    if not os.path.exists(image_path):
        print(f"[{idx+1}/{len(data)}] WARNING: image not found: {image_path}")
        continue

    print(f"[{idx+1}/{len(data)}] Processing id={item.get('image_id', item.get('id'))} with image {image_path} ...")

    # Go through all keys, find question fields SLO/F*/I*/A* (but not Answer...)
    for key, value in list(item.items()):
        if (
            isinstance(value, str)
            and (
                key == "SLO"
                or key.startswith("F")
                or key.startswith("I")
                or key.startswith("A")
             )
            and not key.startswith("Answer") and "suitable" not in key
        ):
            q_key = key
            a_key = "Answer" + q_key  # e.g. F1 -> AnswerF1, SLO -> AnswerSLO

            print(f"  - Asking: {q_key} ({value[:60]}...)")
            answer = ask_gemma(image_path, value)
            print(f"    ANSWER: {answer}")

            # overwrite or create Answer* field with Gemma's answer
            item[a_key] = answer

# Save new JSON with same structure (questions + Answer* fields)
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. Saved Gemma answers to: {output_json_path}")
