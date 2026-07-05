#!/usr/bin/env bash
# install.sh — Hermes Kaggle Tools installer
# Installs dependencies and copies scripts to ~/.hermes/scripts/kaggle/
set -euo pipefail

HERMES_SCRIPTS="${HOME}/.hermes/scripts/kaggle"
DEST_SKILL="${HOME}/.hermes/skills/devops/kaggle-cli"

echo "=== Hermes Kaggle Tools Installer ==="
echo ""

# ── Dependencies ──────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."
if command -v uv &>/dev/null; then
    uv pip install kaggle kagglehub 2>&1 | tail -1
else
    pip install --break-system-packages kaggle kagglehub 2>&1 | tail -1
fi
echo "  Done."

# ── Copy scripts ──────────────────────────────────────────────────
echo "[2/3] Installing scripts to ${HERMES_SCRIPTS}..."
mkdir -p "${HERMES_SCRIPTS}"
cp kg.py oauth_login.py medgemma_deploy.py "${HERMES_SCRIPTS}/"
chmod +x "${HERMES_SCRIPTS}/kg.py" "${HERMES_SCRIPTS}/oauth_login.py" "${HERMES_SCRIPTS}/medgemma_deploy.py"
echo "  Done."

# ── Install skill ─────────────────────────────────────────────────
echo "[3/3] Installing skill to ${DEST_SKILL}..."
mkdir -p "${DEST_SKILL}"
cp SKILL.md "${DEST_SKILL}/SKILL.md"
echo "  Done."

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Authenticate:   python3 ${HERMES_SCRIPTS}/oauth_login.py"
echo "  2. Test:           python3 ${HERMES_SCRIPTS}/kg.py whoami"
echo "  3. List notebooks: python3 ${HERMES_SCRIPTS}/kg.py list --mine"
echo ""
echo "For Hermes Agent:    skill_view('kaggle-cli')"
