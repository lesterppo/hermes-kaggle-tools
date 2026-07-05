#!/usr/bin/env python3
"""
MedGemma 1.5 4B Deployment Script for Kaggle GPU (T4 x2).

Single-cell deployment: installs llama-cpp-python, downloads model,
starts OpenAI-compatible API server, sets up Cloudflare tunnel.

Paste this entire script into a Kaggle notebook cell and run.
Or: use kg.py to push + run with GPU.

Model: unsloth/medgemma-1.5-4b-it-GGUF (Q4_K_M, 2.49 GB)
Chat format: gemma (Gemma 3 instruct format)
Context: 8192 tokens (GPU)
"""

import subprocess, sys, time, os, re, json

# ── Config ──────────────────────────────────────────────────────────
MODEL_REPO = "unsloth/medgemma-1.5-4b-it-GGUF"
MODEL_FILE = "medgemma-1.5-4b-it-Q4_K_M.gguf"
MODEL_DIR = "/kaggle/working"
PORT = 8000
N_GPU_LAYERS = 99       # all layers on GPU for 4B model
N_CTX = 8192
CHAT_FORMAT = "gemma"   # Gemma 3 instruct format (same as <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...)

print("=" * 60)
print("MedGemma 1.5 4B — Kaggle GPU Deployment")
print(f"Model: {MODEL_REPO} ({MODEL_FILE})")
print(f"GPU layers: {N_GPU_LAYERS}, context: {N_CTX}")
print("=" * 60)

# ── Step 1: Install llama-cpp-python with CUDA ─────────────────────
print("\n[1/4] Installing llama-cpp-python with CUDA support...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "llama-cpp-python[server]",
    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu122",
    "-q"
], timeout=180)
print("  Done.")

# ── Step 2: Download model ──────────────────────────────────────────
print(f"\n[2/4] Downloading {MODEL_FILE} (~2.5 GB)...")
from huggingface_hub import hf_hub_download
mp = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=MODEL_DIR,
)
size_gb = os.path.getsize(mp) / 1e9
print(f"  Model: {size_gb:.2f} GB at {mp}")

# ── Step 3: Start API server ────────────────────────────────────────
print(f"\n[3/4] Starting llama.cpp server on port {PORT}...")
subprocess.run("fuser -k 8000/tcp 2>/dev/null", shell=True)
time.sleep(1)

# Save server config
config = {
    "model": MODEL_FILE,
    "repo": MODEL_REPO,
    "quantization": "Q4_K_M",
    "gpu_layers": N_GPU_LAYERS,
    "context": N_CTX,
    "chat_format": CHAT_FORMAT,
}
with open(f"{MODEL_DIR}/deploy_config.json", "w") as f:
    json.dump(config, f, indent=2)

sp = subprocess.Popen([
    sys.executable, "-m", "llama_cpp.server",
    "--model", mp,
    "--n_gpu_layers", str(N_GPU_LAYERS),
    "--n_ctx", str(N_CTX),
    "--chat_format", CHAT_FORMAT,
    "--host", "127.0.0.1",
    "--port", str(PORT),
])
time.sleep(20)

if sp.poll() is None:
    print("  Server: OK (PID {})".format(sp.pid))
else:
    print("  Server: FAILED")
    sys.exit(1)

# ── Step 4: Cloudflare tunnel ───────────────────────────────────────
print(f"\n[4/4] Starting Cloudflare tunnel...")
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

# Extract URL
log_text = open(tlog).read()
import re
urls = re.findall(r'https://[^ ]*trycloudflare\.com', log_text)
if urls:
    api_url = urls[0] + "/v1"
    print(f"\n{'=' * 60}")
    print(f"API URL: {api_url}")
    print(f"Test: curl {api_url}/models")
    print(f"{'=' * 60}")

    # Save for easy retrieval
    with open(f"{MODEL_DIR}/api_url.txt", "w") as f:
        f.write(api_url)

    # Keep the script alive — Kaggle kills child processes when the script exits.
    # Block on the server process so the API stays reachable.
    print("\nDeployment complete. Server running (Ctrl+C to stop)...")
    sys.stdout.flush()
    
    # Monitor server + tunnel, restart tunnel if it dies
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
    # Dump log for debugging
    print("--- tunnel log ---")
    print(open(tlog).read()[-500:])
    print("--- end ---")
