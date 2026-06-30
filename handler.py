import runpod
import os, base64, io, torch, tempfile
from PIL import Image

os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "flash-attn"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/TRELLIS-image-large")
HF_CACHE = "/runpod-volume/huggingface-cache/hub"
pipeline = None

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

def load_pipeline():
    global pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    path = resolve_snapshot_path(MODEL_ID)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(path)
    pipeline.cuda()

def handler(job):
    global pipeline
    if pipeline is None:
        load_pipeline()

    from trellis.utils import postprocessing_utils
    
    inp = job.get("input", {}) or {}
    img_b64 = inp.get("image", "")
    seed = int(inp.get("seed", 1))
    simplify = float(inp.get("simplify", 0.95))
    texture_size = int(inp.get("texture_size", 1024))

    if not img_b64:
        return {"status": "error", "error": "No 'image' (base64) provided"}

    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    outputs = pipeline.run(img, seed=seed)

    gs = outputs.get("gaussian", [None])[0]
    mesh = outputs.get("mesh", [None])[0]

    if mesh is None:
        return {"status": "error", "error": "No mesh generated"}

    glb = postprocessing_utils.to_glb(gs, mesh, simplify=simplify, texture_size=texture_size)
    
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb.export(tmp.name)
        with open(tmp.name, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp.name)

    return {
        "status": "success",
        "output": glb_b64,
        "format": "glb",
        "seed": seed,
    }

runpod.serverless.start({"handler": handler})
