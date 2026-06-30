import runpod
import os, base64, io, torch, tempfile
from PIL import Image

os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "flash-attn"

MODEL_ID = os.environ.get("MODEL_ID", os.environ.get("MODEL_NAME", "microsoft/TRELLIS-image-large"))
pipeline = None

def load_pipeline():
    global pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
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

    return {"status": "success", "output": glb_b64, "format": "glb", "seed": seed}

runpod.serverless.start({"handler": handler})
