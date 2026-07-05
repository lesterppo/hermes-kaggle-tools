#!/usr/bin/env python3
"""
MedGemma 1.5 4B Deployment Script for Kaggle GPU (T4 x2).
Supports multimodal image analysis via mmproj-F16.gguf.

Single-cell deployment: installs llama-cpp-python, downloads model + mmproj,
starts OpenAI-compatible API server with vision, sets up Cloudflare tunnel.

Model: unsloth/medgemma-1.5-4b-it-GGUF (Q4_K_M, 2.49 GB + mmproj-F16 851 MB)
Chat format: gemma (Gemma 3 instruct format)
Context: 8192 tokens (GPU)
Vision: mmproj-F16.gguf enables image understanding (X-ray, CT, clinical photos)
"""

import subprocess, sys, time, os, re, json

# ── Config ──────────────────────────────────────────────────────────
MODEL_REPO = "unsloth/medgemma-1.5-4b-it-GGUF"
MODEL_FILE = "medgemma-1.5-4b-it-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"    # 851 MB — multimodal projection for vision
MODEL_DIR = "/kaggle/working"
PORT = 8000
N_GPU_LAYERS = 99       # all layers on GPU for 4B model
N_CTX = 8192
CHAT_FORMAT = "gemma"   # Gemma 3 instruct format

print("=" * 60)
print("MedGemma 1.5 4B — Kaggle GPU Deployment (Multimodal)")
print(f"Model: {MODEL_REPO} ({MODEL_FILE})")
print(f"Vision: {MMPROJ_FILE} (851 MB)")
print(f"GPU layers: {N_GPU_LAYERS}, context: {N_CTX}")
print("=" * 60)

# ── Step 1: Install llama-cpp-python with CUDA ─────────────────────
print("\n[1/5] Installing llama-cpp-python with CUDA support...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "llama-cpp-python[server]",
    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu122",
    "-q"
], timeout=180)
print("  Done.")

# ── Step 2: Download model + mmproj ─────────────────────────────────
print(f"\n[2/5] Downloading {MODEL_FILE} (~2.5 GB)...")
from huggingface_hub import hf_hub_download
mp = hf_hub_download(
    repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODEL_DIR,
)
print(f"  Model: {os.path.getsize(mp)/1e9:.2f} GB")

print(f"\n[3/5] Downloading {MMPROJ_FILE} (~851 MB)...")
mmp = hf_hub_download(
    repo_id=MODEL_REPO, filename=MMPROJ_FILE, local_dir=MODEL_DIR,
)
print(f"  mmproj: {os.path.getsize(mmp)/1e6:.0f} MB")

# ── Step 4: Start API server with vision ────────────────────────────
print(f"\n[4/5] Starting llama.cpp server with vision on port {PORT}...")
subprocess.run("fuser -k 8000/tcp 2>/dev/null", shell=True)
time.sleep(1)

# Save config
config = {
    "model": MODEL_FILE, "repo": MODEL_REPO,
    "mmproj": MMPROJ_FILE, "quantization": "Q4_K_M",
    "gpu_layers": N_GPU_LAYERS, "context": N_CTX,
    "chat_format": CHAT_FORMAT, "vision": True,
}
with open(f"{MODEL_DIR}/deploy_config.json", "w") as f:
    json.dump(config, f, indent=2)

sp = subprocess.Popen([
    sys.executable, "-m", "llama_cpp.server",
    "--model", mp,
    "--clip_model_path", mmp,         # ← multimodal projector
    "--n_gpu_layers", str(N_GPU_LAYERS),
    "--n_ctx", str(N_CTX),
    # Don't set --chat_format — let server use model's built-in template
    # which has image handling (<start_of_image> tokens)
    "--host", "127.0.0.1",
    "--port", str(PORT),
])
time.sleep(25)   # extra time for mmproj loading

if sp.poll() is None:
    print("  Server: OK (PID {}, vision enabled)".format(sp.pid))
else:
    print("  Server: FAILED")
    sys.exit(1)

# ── Step 5: Cloudflare tunnel ───────────────────────────────────────
print(f"\n[5/5] Starting Cloudflare tunnel...")
cf_bin = f"{MODEL_DIR}/cloudflared"
if not os.path.exists(cf_bin):
    subprocess.run(
        f"curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o {cf_bin} && chmod +x {cf_bin}",
        shell=True
    )

tlog = f"{MODEL_DIR}/tunnel.log"
tp = subprocess.Popen(
    [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=open(tlog, "w"), stderr=subprocess.STDOUT,
)
time.sleep(15)

log_text = open(tlog).read()
urls = re.findall(r'https://[^ ]*trycloudflare\.com', log_text)
if urls:
    api_url = urls[0] + "/v1"
    print(f"\n{'=' * 60}")
    print(f"API URL: {api_url}")
    print(f"Test: curl {api_url}/models")
    print(f"Vision: curl {api_url}/chat/completions -d '{{\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/jpeg;base64,...\"}}}},{{\"type\":\"text\",\"text\":\"Describe this image\"}}]}}]}}'")
    print(f"{'=' * 60}")

    with open(f"{MODEL_DIR}/api_url.txt", "w") as f:
        f.write(api_url)

    print("\nDeployment complete. Server running with vision (Ctrl+C to stop)...")
    sys.stdout.flush()

    while True:
        if sp.poll() is not None:
            print("Server died unexpectedly, exiting.")
            break
        if tp.poll() is not None:
            print("Tunnel died, restarting...")
            tp = subprocess.Popen(
                [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
                stdout=open(tlog, "a"), stderr=subprocess.STDOUT,
            )
        time.sleep(30)
else:
    print(f"\nTunnel URL not found. Check {tlog}")
    print("--- tunnel log ---")
    print(open(tlog).read()[-500:])
    print("--- end ---")
