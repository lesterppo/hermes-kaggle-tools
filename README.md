# Hermes Kaggle Tools

AI-agent-native, token-efficient tools for operating Kaggle notebooks — list, create, run, stop, monitor, and manage kernel sessions entirely from the CLI.

Built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) but works standalone — any AI agent can consume the compact JSON output.

## Quick Start

```bash
git clone https://github.com/lesterppo/hermes-kaggle-tools.git
cd hermes-kaggle-tools
./install.sh
```

## What's Included

| File | Purpose |
|------|---------|
| `kg.py` | Main CLI — 20 commands for notebook lifecycle management |
| `oauth_login.py` | One-time OAuth authentication helper |
| `medgemma_deploy.py` | Deploy template for GGUF models on Kaggle GPU |
| `SKILL.md` | AI agent skill — load this in Hermes for guided workflows |
| `AGENTS.md` | AI coding agent guide — design patterns, API internals |

## Commands

```bash
kg.py whoami                     # Check auth
kg.py list --mine --full         # List notebooks with full metadata
kg.py get owner/slug             # Full kernel metadata
kg.py status owner/slug          # Session status (idle/queued/running/complete/error)
kg.py run owner/slug --gpu --wait  # Start + wait for completion
kg.py stop owner/slug            # Cancel running session
kg.py sessions                   # List active sessions
kg.py logs owner/slug            # Get session logs
kg.py url owner/slug             # Discover tunnel API URL
kg.py health owner/slug          # Probe tunnel API health
kg.py output owner/slug          # Download output files
kg.py quota                      # Check GPU quota
```

All commands output JSON on stdout: `{"ok": true, "summary": "..."}`. Large outputs (logs, files) go to `~/.hermes/kaggle_output/`.

## Requirements

- Python 3.10+
- Kaggle account with phone verification (for GPU)
- One-time OAuth setup (see [Authentication](#authentication))

## Authentication

```bash
python3 oauth_login.py
# → prints URL → open in browser → sign in → paste code
```

After OAuth, credentials are stored at `~/.kaggle/credentials.json` and auto-refresh.

## Design

Token-efficient by design: each command returns 30-100 tokens of JSON. Full data (logs, file lists) is written to disk with a file path pointer in the response. No browser automation — everything goes through the Kaggle OAuth REST API.

## License

MIT
