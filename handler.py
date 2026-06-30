"""
RunPod Serverless handler — Qwen2-VL-7B with model caching.
Accepts image (base64) + question, returns vision analysis.

Deploy with:
  - Model caching: Qwen/Qwen2-VL-7B-Instruct
  - GPU: 24GB+ (A5000, L4, 3090)
  - Container disk: 30GB
"""

import os
import base64
import io
import runpod
import torch
from PIL import Image

# Force offline — model must be pre-cached by RunPod
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
HF_CACHE = "/runpod-volume/huggingface-cache/hub"

model = None
processor = None


def resolve_snapshot_path(model_id: str) -> str:
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    # Fallback: first available snapshot
    versions = sorted(
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    )
    if not versions:
        raise RuntimeError(f"No snapshots found in {snapshots_dir}")
    return os.path.join(snapshots_dir, versions[0])


def load_model():
    global model, processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    local_path = resolve_snapshot_path(MODEL_ID)
    print(f"[Vision] Loading from: {local_path}", flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    print(f"[Vision] Loaded on {torch.cuda.get_device_name(0)}", flush=True)


def handler(job):
    global model, processor

    if model is None:
        load_model()

    job_input = job.get("input", {}) or {}
    image_b64 = job_input.get("image", "")
    question = job_input.get("question", "Describe this image in detail.")
    max_tokens = int(job_input.get("max_tokens", 512))
    temperature = float(job_input.get("temperature", 0.0))

    if not image_b64:
        return {"status": "error", "error": "No 'image' (base64) provided"}

    try:
        # Decode image
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        from qwen_vl_utils import process_vision_info

        # Build messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "status": "success",
            "output": output_text,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
