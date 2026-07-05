# AGENTS.md — Kaggle Notebook Tools for AI Coding Agents

## Overview

This repo provides AI-agent-native tools for managing Kaggle notebooks through the Kaggle OAuth REST API. A coding agent can integrate these tools into any agent framework (Hermes, Claude Code, custom) for programmatic notebook lifecycle management.

## Architecture

```
┌──────────────────────────────────┐
│ kg.py (CLI, 20 commands)         │
│   ├── OAuth via KaggleCredentials│
│   ├── Normalized slug resolution │
│   ├── Session state persistence  │
│   └── JSON pointer output        │
├──────────────────────────────────┤
│ kagglesdk.KaggleClient           │  ← OAuth REST API
│   ├── kernels_api_client         │
│   │   ├── save_kernel            │
│   │   ├── create_kernel_session  │
│   │   ├── cancel_kernel_session  │
│   │   ├── get_kernel             │
│   │   ├── list_kernels           │
│   │   ├── get_kernel_session_status│
│   │   ├── get_kernel_session_logs │
│   │   └── ...                    │
├──────────────────────────────────┤
│ kaggle.api.KaggleApi             │  ← Legacy API key (push/pull/init only)
└──────────────────────────────────┘
```

## Key Design Patterns

### 1. Dual Auth

- **OAuth** (kagglesdk) — for all kernel operations: list, get, status, run, stop, logs, sessions, url, health, delete, quota. Uses `~/.kaggle/credentials.json`.
- **Legacy API key** (kaggle.api) — only for push, pull, init. Uses `~/.kaggle/kaggle.json`.

### 2. Slug Normalization

The API has inconsistent slug requirements. The CLI normalizes transparently:

| User input | save_kernel | create_kernel_session | status/logs |
|------------|-------------|-----------------------|-------------|
| `owner/slug` | empty slug field (auto-gen from title) | `"owner/slug"` | owner + slug separately |
| `slug` (bare) | not supported | not supported | resolves with current user |

Never pass bare slugs to the API — always use `owner/slug` or empty.

### 3. Script vs Notebook Kernels

- **`kernel_type="script"`** — code is raw Python text. Simpler, but Kaggle kills child processes when the parent script exits. MUST add a keep-alive loop at the end.
- **`kernel_type="notebook"`** — code must be `.ipynb` JSON format. Better for multi-cell workflows. Cannot change kernel type after creation.

### 4. Session State

Session IDs are persisted at `~/.hermes/kaggle_output/state/<owner>_<slug>.json`. The `run` command auto-saves the session ID; `stop` reads it back. `SAVE_AND_RUN_ALL` auto-starts but does NOT save the session ID — use `run` explicitly for lifecycle control.

### 5. Token-Efficient Output

Every command outputs a single JSON line:
```json
{"ok": true, "summary": "3 kernel(s)", "count": 3, "kernels": [...]}
```

Large data (logs, file lists) is written to `~/.hermes/kaggle_output/` with a `"file"` pointer in the response. The agent reads the pointer, not the full data.

## API Pitfalls

1. **save_kernel slug** — Must be empty string (auto-generates from title) or full `owner/slug`. Bare slug → `"Invalid slug"`.
2. **create_kernel_session slug** — Must be `owner/slug`. Bare slug → 400. Empty → fails.
3. **kernel type immutable** — Cannot change script↔notebook after creation.
4. **SAVE_AND_RUN_ALL** — Auto-starts session. Subsequent `create_kernel_session` → 409 Conflict.
5. **Response ref** — `save_kernel` returns ref with `/code/` prefix. Strip it.
6. **Legacy API key** — kaggle.json returns 401 for all kernel operations. OAuth only.
7. **GPU quota API** — Returns 0/0 for free tier. Quota enforced server-side.

## Integration Guide

### For Hermes Agent

Load `SKILL.md` as a skill:
```bash
# Install
git clone https://github.com/lesterppo/hermes-kaggle-tools.git ~/.hermes/skills/devops/kaggle-cli
cp kg.py oauth_login.py medgemma_deploy.py ~/.hermes/scripts/kaggle/

# Use from Hermes
skill_view("kaggle-cli")  # loads SKILL.md
terminal("python3 ~/.hermes/scripts/kaggle/kg.py list --mine")
```

### For Standalone Use

```python
import subprocess, json

def kg(*args):
    r = subprocess.run(["python3", "kg.py"] + list(args), capture_output=True, text=True)
    return json.loads(r.stdout)

result = kg("list", "--mine", "--limit", "5")
for k in result["kernels"]:
    print(k["ref"], k["title"])
```

### For Other Agent Frameworks

The CLI is framework-agnostic. Any agent that can execute shell commands and parse JSON can use it. Key contract:

1. Setup: `python3 oauth_login.py` once
2. Discover: `kg.py list --mine`
3. Inspect: `kg.py get owner/slug`
4. Run: `kg.py run owner/slug --gpu --wait`
5. Monitor: `kg.py sessions`

## Extending

To add a new command:
1. Add the handler function in `kg.py` following the `cmd_<name>(args)` pattern
2. Add to the `COMMANDS` dict
3. Add argparse subparser in `main()`
4. Follow the `_pointer` / `_dump` output contract
