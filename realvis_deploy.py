#!/usr/bin/env python3
"""
RealVisXL_V4.0 Deployment Script for Kaggle GPU.
SDXL-based photorealistic image generation.

Two-phase loading: download files first, then load from disk.
FastAPI server with /generate and /img2img + Cloudflare tunnel.

Model: SG161222/RealVisXL_V4.0 (~6.5 GB)
"""

import subprocess, sys, time, os, re, json, io, base64
from pathlib import Path

MODEL_REPO = "SG161222/RealVisXL_V4.0"
MODEL_DIR = "/kaggle/working"
PORT = 8000
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.0
DEFAULT_SIZE = 1024

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("RealVisXL_V4.0 — Kaggle GPU Deployment v4")
log(f"Model: {MODEL_REPO}")
log("=" * 60)

# ── Step 1: Install CUDA PyTorch first ─────────────────────────────
log("\n[1/7] Installing CUDA PyTorch (for GPU)...")
t0 = time.time()
# Kaggle default PyTorch is CPU-only — replace with CUDA build
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
], timeout=300)
log(f"  Done ({time.time()-t0:.0f}s)")

import torch
log(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_capability()})")

# ── Step 2: Install other deps ─────────────────────────────────────
log("\n[2/7] Installing other dependencies...")
t0 = time.time()
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "diffusers", "transformers", "accelerate", "fastapi", "uvicorn",
    "pillow", "safetensors", "huggingface_hub",
], timeout=300)
log(f"  Done ({time.time()-t0:.0f}s)")

# ── Step 2: Download model files (no loading) ───────────────────────
log(f"\n[3/7] Downloading model files (no loading)...")
t0 = time.time()
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    MODEL_REPO,
    cache_dir=MODEL_DIR,
    ignore_patterns=["*.md", "*.msgpack", "*.safetensors.index.json"],
    resume_download=True,
)
log(f"  Path: {local_path}")
total_gb = sum(p.stat().st_size for p in Path(local_path).rglob('*') if p.is_file()) / 1e9
log(f"  Size: {total_gb:.1f} GB ({time.time()-t0:.0f}s)")

# ── Step 3: Load pipeline from local files ──────────────────────────
log(f"\n[4/7] Loading SDXL pipeline from disk...")
t0 = time.time()
from diffusers import StableDiffusionXLPipeline

# Load to CPU first, fp32 → fp16 conversion later
pipe = StableDiffusionXLPipeline.from_pretrained(
    local_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
log(f"  Pipeline loaded ({time.time()-t0:.0f}s)")

# ── Step 4: Try GPU, fall back to CPU ───────────────────────────────
log(f"\n[5/7] Device setup...")
t0 = time.time()
device = "cpu"

if torch.cuda.is_available():
    try:
        log("  Trying CUDA...")
        pipe = pipe.to("cuda")
        # Test with tiny generation
        _ = pipe("test", num_inference_steps=1, width=64, height=64)
        device = "cuda"
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            log("  xformers: on")
        except:
            log("  xformers: N/A")
        vram = torch.cuda.memory_allocated()/1e9
        log(f"  GPU OK — VRAM: {vram:.1f} GB ({time.time()-t0:.0f}s)")
    except Exception as e:
        log(f"  GPU failed: {e}")
        pipe = pipe.to("cpu")
        device = "cpu"

if device == "cpu":
    log("  CPU mode — enabling optimizations...")
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    log(f"  CPU ready ({time.time()-t0:.0f}s)")

# ── Step 5: FastAPI server ──────────────────────────────────────────
log(f"\n[6/7] FastAPI on port {PORT}...")
subprocess.run("fuser -k 8000/tcp 2>/dev/null", shell=True)
time.sleep(1)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn, threading

app = FastAPI(title="RealVisXL_V4.0 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_params = {
    "steps": DEFAULT_STEPS, "cfg_scale": DEFAULT_CFG,
    "width": DEFAULT_SIZE, "height": DEFAULT_SIZE, "seed": -1,
    "negative_prompt": "",
}

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = DEFAULT_STEPS
    cfg_scale: Optional[float] = DEFAULT_CFG
    width: Optional[int] = DEFAULT_SIZE
    height: Optional[int] = DEFAULT_SIZE
    seed: Optional[int] = -1

class Img2ImgRequest(BaseModel):
    prompt: str
    image_b64: str
    strength: Optional[float] = 0.75
    steps: Optional[int] = DEFAULT_STEPS
    cfg_scale: Optional[float] = DEFAULT_CFG
    seed: Optional[int] = -1

@app.get("/health")
def health():
    vram = torch.cuda.memory_allocated()/1e9 if device=="cuda" else 0
    return {"status":"ok","model":MODEL_REPO,"device":device,"vram_used_gb":round(vram,2),"default_params":current_params}

@app.get("/params")
def get_params():
    return current_params

@app.post("/params")
def set_params(data: dict):
    for k in ["steps","cfg_scale","width","height","seed","negative_prompt"]:
        if k in data: current_params[k] = data[k]
    return {"ok":True,"params":current_params}

@app.post("/generate")
def generate(req: GenerateRequest):
    gen_device = device
    seed = req.seed if req.seed>=0 else int(torch.randint(0,2**31,(1,)).item())
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or current_params["negative_prompt"],
        num_inference_steps=req.steps or current_params["steps"],
        guidance_scale=req.cfg_scale or current_params["cfg_scale"],
        width=req.width or current_params["width"],
        height=req.height or current_params["height"],
        generator=generator,
    ).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"ok":True,"prompt":req.prompt,"seed":seed,"image_b64":b64,"size_bytes":len(buf.getvalue()),"device":device}

@app.post("/img2img")
def img2img(req: Img2ImgRequest):
    from PIL import Image
    gen_device = device
    seed = req.seed if req.seed>=0 else int(torch.randint(0,2**31,(1,)).item())
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    img_bytes = base64.b64decode(req.image_b64)
    init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = pipe(
        prompt=req.prompt, image=init_image, strength=req.strength,
        num_inference_steps=req.steps or current_params["steps"],
        guidance_scale=req.cfg_scale or current_params["cfg_scale"],
        generator=generator,
    ).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"ok":True,"prompt":req.prompt,"strength":req.strength,"seed":seed,"image_b64":b64,"size_bytes":len(buf.getvalue()),"device":device}

# Start server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

threading.Thread(target=run_server, daemon=True).start()
time.sleep(5)
log("  Server: OK")

# ── Step 6: Cloudflare tunnel ───────────────────────────────────────
log(f"\n[7/7] Cloudflare tunnel...")
cf_bin = f"{MODEL_DIR}/cloudflared"
if not os.path.exists(cf_bin):
    subprocess.run(
        f"curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o {cf_bin} && chmod +x {cf_bin}",
        shell=True, timeout=60
    )

tlog = f"{MODEL_DIR}/tunnel_realvis.log"
tp = subprocess.Popen(
    [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=open(tlog, "w"), stderr=subprocess.STDOUT,
)
time.sleep(15)

log_text = open(tlog).read()
urls = re.findall(r'https://[^ ]*trycloudflare\.com', log_text)
if urls:
    api_url = urls[0]
    log(f"\n{'='*60}")
    log(f"API URL: {api_url}")
    log(f"Device:  {device}")
    log(f"Health:  curl {api_url}/health")
    log(f"{'='*60}")
    with open(f"{MODEL_DIR}/api_url_realvis.txt", "w") as f:
        f.write(api_url)
    log("Ready. Keep-alive active.")
    while True:
        if tp.poll() is not None:
            log("Tunnel restarting...")
            tp = subprocess.Popen(
                [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
                stdout=open(tlog, "a"), stderr=subprocess.STDOUT,
            )
        time.sleep(30)
else:
    log(f"Tunnel failed. Log: {open(tlog).read()[-500:]}")
