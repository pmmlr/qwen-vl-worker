import runpod
import os, base64, io, torch
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
HF_CACHE = "/runpod-volume/huggingface-cache/hub"
model = None
processor = None


def resolve_snapshot_path(model_id):
    org, name = model_id.split("/", 1)
    root = os.path.join(HF_CACHE, f"models--{org}--{name}")
    refs = os.path.join(root, "refs", "main")
    snaps = os.path.join(root, "snapshots")
    if os.path.isfile(refs):
        with open(refs) as f:
            h = f.read().strip()
        c = os.path.join(snaps, h)
        if os.path.isdir(c):
            return c
    vers = sorted(d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d)))
    return os.path.join(snaps, vers[0])


def load_model():
    global model, processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    path = resolve_snapshot_path(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)


def handler(job):
    global model, processor
    if model is None:
        load_model()

    inp = job.get("input", {}) or {}
    img_b64 = inp.get("image", "")
    question = inp.get("question", "Describe this image.")
    max_tokens = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.0))

    if not img_b64:
        return {"status": "error", "error": "No 'image' (base64) provided"}

    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_tokens,
                         temperature=temperature if temperature > 0 else None,
                         do_sample=temperature > 0)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"status": "success", "output": out}


runpod.serverless.start({"handler": handler})
