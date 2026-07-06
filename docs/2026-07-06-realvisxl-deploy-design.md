# RealVisXL_V4.0 Kaggle Deployment + Local Chat CLI

## Architecture

Two components:
1. **Kaggle deploy script** (`realvis_deploy.py`) — single notebook cell: install deps, download model, start FastAPI server with Cloudflare tunnel
2. **Local chat CLI** (`realvis_chat.py`) — Rich TUI with interactive prompt loop, img2img, parameter tuning, history

## Kaggle Side (`realvis_deploy.py`)

FastAPI server on port 8000 behind Cloudflare tunnel:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Model status, VRAM, uptime |
| `/generate` | POST | text→image |
| `/img2img` | POST | image→image |
| `/params` | GET | Current defaults |

Steps: install diffusers[torch] + transformers + accelerate → hf_hub_download RealVisXL_V4.0 → load SDXL pipeline (fp16, cuda) → start uvicorn → Cloudflare tunnel → keep-alive.

## Local CLI (`realvis_chat.py`)

Commands:
- `prompt text` — generate image (--steps, --cfg, --w, --h, --seed, --neg)
- `/img2img <path> prompt` — image-to-image with strength
- `/params` — show/set defaults
- `/history` — last N prompts + thumbnails
- `/open [N]` — open in system viewer
- `/save <path>` — save to path
- `/retry` — re-generate last prompt
- `/health` — check server
- `/url <url>` — set/change API URL
- `/help` — all commands

Images saved to `~/.hermes/kaggle_output/realvis/<timestamp>_<prompt_slug>.png`

## Model

SG161222/RealVisXL_V4.0 — SDXL-based photorealism model, ~6.5GB, fp16 on T4 x2. Default: 25 steps, CFG 7, 1024x1024.
