---
name: kaggle-cli
description: Operate Kaggle notebooks via agent-native CLI.
version: 2.1.0
author: Peter (lesterppo)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [kaggle, notebook, gpu, cloud, ml, vision, multimodal]
    category: devops
    config:
      kaggle.session_name:
        description: Default Kaggle session name
        type: string
        default: ""
---

# Kaggle CLI Skill v2.1

Operate Kaggle notebooks from the terminal with an AI-agent-native, token-efficient CLI.
20 commands covering the full notebook lifecycle: list, create, run, stop, monitor,
download output, check GPU quota. Wraps the kagglesdk OAuth REST API.

Also covers deploying multimodal LLMs (MedGemma) on Kaggle GPU with vision support.

## When to Use

- User needs to list, inspect, run, or manage Kaggle notebooks
- User wants to start/stop notebook sessions (GPU/TPU) from the terminal
- User wants to deploy a GGUF model with vision (mmproj) on Kaggle GPU
- User wants to download output or check GPU quota

## Prerequisites

### 1. Install

```bash
git clone https://github.com/lesterppo/hermes-kaggle-tools.git
cd hermes-kaggle-tools
./install.sh
```

### 2. Authenticate (one-time OAuth)

```bash
python3 ~/.hermes/scripts/kaggle/oauth_login.py
# → prints URL → open in browser → sign in → paste verification code
```

## How to Run

```bash
KG=~/.hermes/scripts/kaggle/kg.py

python3 $KG whoami
python3 $KG list --mine --full --limit 10
python3 $KG get owner/slug
python3 $KG run owner/slug --gpu --internet --wait
python3 $KG stop owner/slug
python3 $KG sessions
python3 $KG url owner/slug
python3 $KG health owner/slug
python3 $KG output owner/slug --path /tmp/output
python3 $KG quota
```

## Command Reference (20 commands)

All output is JSON on stdout: `{"ok": true, "summary": "...", ...}`.
Large data goes to `~/.hermes/kaggle_output/` with a `"file"` pointer.

| Command | Description |
|---------|-------------|
| `whoami` | Show authenticated user + API key status |
| `list [--mine] [--full] [--limit N]` | List notebooks. `--full` enriches with parallel get_kernel |
| `get owner/slug` | Full notebook metadata |
| `status owner/slug` | Session status (idle/queued/running/complete/error) |
| `run owner/slug [--gpu] [--tpu] [--internet] [--wait]` | Start session. `--wait` blocks until done |
| `stop owner/slug` | Cancel running session (needs session_id from saved state) |
| `sessions` | List all active sessions across kernels |
| `logs owner/slug [--lines N]` | Get session logs |
| `url owner/slug` | Get/discover tunnel API URL |
| `health owner/slug` | Probe tunnel API health (models + inference + vision) |
| `output owner/slug [--path DIR]` | Download session output files |
| `files owner/slug` | List kernel source files |
| `push --folder DIR` | Push kernel from local folder (needs kaggle.json) |
| `pull owner/slug [--path DIR]` | Pull kernel to local folder (needs kaggle.json) |
| `delete owner/slug` | Delete a notebook |
| `quota` | Check GPU accelerator quota |
| `init [--folder DIR]` | Initialize new kernel folder (needs kaggle.json) |

## Architecture

Dual auth:
1. **OAuth** (`kagglesdk.KaggleCredentials` → `KaggleClient(api_token=...)`) — for all kernel operations.
2. **Legacy API key** (`kaggle.api.KaggleApi`) — only for push, pull, init.

Kernel refs use `owner/slug` format everywhere. The CLI normalizes slug format per-API
(save_kernel uses empty slug, create_kernel_session uses owner/slug).

## GGUF Model Deployment (Multimodal)

Deploy any GGUF model on Kaggle GPU. The deployment script pattern is at
`medgemma_deploy.py` — it handles CUDA llama-cpp install, model + mmproj download,
server startup, Cloudflare tunnel, and keep-alive.

**Critical flags for vision:**
- `--clip_model_path <mmproj.gguf>` — NOT `--mmproj`
- Do NOT set `--chat_format` — let server use model's built-in template
- Setting `--chat_format gemma` breaks Gemma 3 image processing

**MedGemma vision prompt engineering:**
- The word "medical" triggers safety refusal — use "radiograph", "X-ray", "scan" instead
- The word "findings" triggers refusal — rewrite to "observations"
- Proven template: `"This is a radiograph. Describe all visible structures, their positions, shapes, and densities. Focus on: {user_question}"`
- MedGemma only analyzes medical images; refuses general photos

## Pitfalls

1. **Legacy API key insufficient.** Returns 401 for all kernel operations. Run `oauth_login.py`.
2. **Script kernels exit on completion.** Add keep-alive loop after server startup.
3. **SAVE_AND_RUN_ALL auto-starts session.** Session ID not captured. Prefer explicit `kg.py run`.
4. **`stop` needs session_id from saved state.** Only `kg.py run` saves it.
5. **GPU quota API returns zero for free tier.** Tracked server-side (~30h/week).
6. **save_kernel slug must be empty or owner/slug.** Bare slug → "Invalid slug".
7. **create_kernel_session requires owner/slug.** Bare slug → 400.
8. **Cannot change kernel type after creation.** Create new kernel for different type.
9. **Response ref has `/code/` prefix.** Strip it for URLs.
10. **Tunnel URL changes per restart.** Use `kg.py url` to retrieve.
11. **Push/Pull/Init need kaggle.json API key.** Both auth files must exist.
12. **Vision: `--clip_model_path` not `--mmproj`.** The old flag is removed.
13. **Vision: never set `--chat_format`.** Breaks image processing.
14. **Vision: avoid "medical" and "findings" in prompts.** Triggers safety refusal.

## Verification

```bash
python3 ~/.hermes/scripts/kaggle/kg.py whoami
python3 ~/.hermes/scripts/kaggle/kg.py list --mine
python3 ~/.hermes/scripts/kaggle/kg.py quota
```
