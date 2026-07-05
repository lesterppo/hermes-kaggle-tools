---
name: kaggle-cli
description: Operate Kaggle notebooks via agent-native CLI.
version: 2.0.0
author: Peter (lesterppo)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [kaggle, notebook, gpu, cloud, ml]
    category: devops
    config:
      kaggle.session_name:
        description: Default Kaggle session name
        type: string
        default: ""
---

# Kaggle CLI Skill v2

Operate Kaggle notebooks from the terminal with an AI-agent-native, token-efficient CLI.
20 commands covering the full notebook lifecycle: list, create, run, stop, monitor,
download output, check GPU quota. Wraps the kagglesdk OAuth REST API — no browser
automation needed.

## When to Use

- User needs to list, inspect, run, or manage Kaggle notebooks
- User wants to start/stop notebook sessions (GPU/TPU) from the terminal
- User asks to "check my Kaggle notebooks", "run my Kaggle GPU notebook"
- User needs to download output files from a completed notebook session
- User wants to check GPU quota usage or find active sessions
- User wants to deploy a GGUF model on Kaggle GPU with auto-tunneled API

## Prerequisites

### 1. Install

```bash
git clone https://github.com/lesterppo/hermes-kaggle-tools.git
cd hermes-kaggle-tools
./install.sh
```

Or with uv:
```bash
uv pip install kaggle kagglehub
mkdir -p ~/.hermes/scripts/kaggle ~/.hermes/skills/devops/kaggle-cli
cp kg.py oauth_login.py ~/.hermes/scripts/kaggle/
cp SKILL.md ~/.hermes/skills/devops/kaggle-cli/
chmod +x ~/.hermes/scripts/kaggle/kg.py
```

### 2. Authenticate (one-time OAuth)

The legacy `~/.kaggle/kaggle.json` API key (`KGAT_*`) does NOT work for kernel
operations (returns 401). OAuth is required.

```bash
python3 ~/.hermes/scripts/kaggle/oauth_login.py
# → prints a URL → open in browser → sign in → paste verification code
```

Credentials are saved to `~/.kaggle/credentials.json` and auto-loaded. No further auth needed.

## How to Run

All commands use the `kg.py` script.

```bash
KG=~/.hermes/scripts/kaggle/kg.py

# Check auth
python3 $KG whoami

# List my notebooks (--full for complete metadata)
python3 $KG list --mine --full --limit 10

# Full kernel metadata
python3 $KG get owner/slug

# Start GPU session + wait for completion + extract tunnel URL
python3 $KG run owner/slug --gpu --internet --wait

# Cancel a running session
python3 $KG stop owner/slug

# List active sessions
python3 $KG sessions

# Get tunnel API URL and probe health
python3 $KG url owner/slug
python3 $KG health owner/slug

# Download output files
python3 $KG output owner/slug --path /tmp/output

# Check GPU quota
python3 $KG quota
```

## Command Reference (20 commands)

All output is JSON on stdout: `{"ok": true, "summary": "...", ...}`.
Large data goes to `~/.hermes/kaggle_output/` with a `"file"` pointer.

| Command | Description |
|---------|-------------|
| `whoami` | Show authenticated user + key status |
| `list [--mine] [--full] [--limit N]` | List notebooks. `--full` enriches with parallel get_kernel |
| `get owner/slug` | Full notebook metadata |
| `status owner/slug` | Session status (idle/queued/running/complete/error) |
| `run owner/slug [--gpu] [--tpu] [--internet] [--wait]` | Start session. `--wait` blocks until done |
| `stop owner/slug` | Cancel running session |
| `sessions` | List all active sessions across kernels |
| `logs owner/slug [--lines N]` | Get session logs |
| `url owner/slug` | Get/discover tunnel API URL |
| `health owner/slug` | Probe tunnel API health (models + inference) |
| `output owner/slug [--path DIR]` | Download session output files |
| `files owner/slug` | List kernel source files |
| `push --folder DIR` | Push kernel from local folder (needs kaggle.json) |
| `pull owner/slug [--path DIR]` | Pull kernel to local folder (needs kaggle.json) |
| `delete owner/slug` | Delete a notebook |
| `quota` | Check GPU accelerator quota |
| `init [--folder DIR]` | Initialize new kernel folder (needs kaggle.json) |

## Procedure

### Pattern 1: Discover + Inspect

```
kg.py list --mine --full     → all kernels with full metadata
kg.py get owner/slug         → one kernel in detail
kg.py sessions               → which ones are running
```

### Pattern 2: Run + Wait + Retrieve

```
kg.py run owner/slug --gpu --wait
                              → starts GPU session, blocks until COMPLETE/ERROR
                              → auto-saves tunnel URL from logs
kg.py url owner/slug          → retrieve the tunnel URL
kg.py health owner/slug       → probe the API
kg.py output owner/slug       → download output files
kg.py stop owner/slug         → cancel if still running
```

### Pattern 3: Deploy GGUF Model

Use the deployment template at `medgemma_deploy.py`:

```python
# 1. Create script kernel with GPU
from kagglesdk import KaggleClient, KaggleCredentials
# ... save_kernel with text=deploy_code, kernel_type="script", enable_gpu=True

# 2. Run + wait
kg.py run owner/slug --gpu --wait

# 3. Get API URL
kg.py url owner/slug
# → https://xxx.trycloudflare.com/v1
```

## Architecture

The CLI uses two auth paths:

1. **OAuth** (`kagglesdk.KaggleCredentials` → `KaggleClient(api_token=...)`) —
   for all kernel operations. This is the primary path. Credentials file at
   `~/.kaggle/credentials.json`.

2. **Legacy API key** (`kaggle.api.KaggleApi`) — only for push, pull, init
   (file upload/download using the old API). Key file at `~/.kaggle/kaggle.json`.

Kernel refs use `owner/slug` format everywhere. Bare slugs are auto-resolved
to `<current_user>/slug`. Internally, the CLI normalizes format per-API
(save_kernel uses empty slug, create_kernel_session uses owner/slug, etc.).

## Pitfalls

1. **Legacy API key insufficient.** The `kaggle.json` key works for
   datasets/competitions but returns 401 for all kernel operations.
   Run `oauth_login.py` once.

2. **Script kernels exit on completion.** Kaggle kills all child processes
   when the parent Python script exits. Add an infinite loop after starting
   server + tunnel. The deployment template handles this.

3. **SAVE_AND_RUN_ALL auto-starts session.** Use with caution — the session
   ID is NOT captured automatically. Prefer explicit `kg.py run` for
   lifecycle control.

4. **`stop` needs session_id from saved state.** Only `kg.py run` saves the
   session ID. Sessions started via `SAVE_AND_RUN_ALL` or the Kaggle web UI
   must be stopped manually.

5. **GPU quota API returns zero for free tier.** The endpoint may return
   `0h/0h`. GPU usage is tracked and enforced server-side (~30h/week typical).

6. **Session timeout ~60 min idle.** Kaggle kills idle GPU sessions.

7. **Push/Pull/Init need both auth files.** These use the legacy API key.

8. **Tunnel URL changes per restart.** Every session gets a new
   `*.trycloudflare.com` URL. Use `kg.py url` to retrieve and `kg.py health`
   to verify it's alive.

## Verification

```bash
# Auth
python3 ~/.hermes/scripts/kaggle/kg.py whoami
# → {"ok": true, "summary": "Authenticated as <user>", "has_legacy_api_key": true}

# List
python3 ~/.hermes/scripts/kaggle/kg.py list --mine
# → {"ok": true, "summary": "N kernel(s)", "kernels": [...]}

# Quota
python3 ~/.hermes/scripts/kaggle/kg.py quota
# → {"ok": true, "summary": "GPU quota: unknown (free tier — tracked server-side)", ...}
```
