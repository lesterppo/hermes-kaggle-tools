#!/usr/bin/env python3
"""
Kaggle OAuth login helper. Two-step flow:
  1. Run without args → prints URL, saves state to /tmp/kaggle_oauth_state.json
  2. Run with CODE arg  → exchanges code, saves credentials to ~/.kaggle/credentials.json
"""
import json
import os
import sys
from pathlib import Path

STATE_FILE = "/tmp/kaggle_oauth_state.json"
CREDS_FILE = os.path.expanduser("~/.kaggle/credentials.json")
SCOPES = ["resources.admin:*"]


def step1_print_url():
    """Generate OAuth URL and persist state."""
    from kagglesdk import KaggleClient, KaggleOAuth
    from kagglesdk.kaggle_env import KaggleEnv

    client = KaggleClient(env=KaggleEnv.PROD)
    oauth = KaggleOAuth(client)
    state = oauth.OAuthState()

    redirect_uri = oauth._http_client.get_oauth_default_redirect_url()
    auth_url = oauth._http_client.build_start_oauth_url(
        client_id=KaggleOAuth.OAUTH_CLIENT_ID,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        state=state.state,
        code_challenge=state.code_challenge,
    )

    # Persist the OAuth state so step2 can pick it up
    saved = {
        "state": state.state,
        "code_verifier": state.code_verifier,
        "code_challenge": state.code_challenge,
    }
    Path(STATE_FILE).write_text(json.dumps(saved, indent=2))
    os.chmod(STATE_FILE, 0o600)

    print(auth_url)
    return 0


def step2_exchange(code: str):
    """Exchange auth code for tokens and save credentials."""
    from kagglesdk import KaggleClient, KaggleOAuth
    from kagglesdk.kaggle_env import KaggleEnv

    if not os.path.exists(STATE_FILE):
        print("ERROR: No state file found. Run without arguments first.", file=sys.stderr)
        return 1

    saved = json.loads(Path(STATE_FILE).read_text())

    client = KaggleClient(env=KaggleEnv.PROD)
    oauth = KaggleOAuth(client)

    # Reconstruct the state object
    oauth_state = oauth.OAuthState()
    oauth_state.state = saved["state"]
    oauth_state.code_verifier = saved["code_verifier"]
    oauth_state.code_challenge = saved["code_challenge"]

    # Exchange code for tokens
    oauth._exchange_oauth_token(code, SCOPES, oauth_state)

    # Validate and save
    creds = oauth._creds
    if not creds:
        print("ERROR: Authentication failed — no credentials returned.", file=sys.stderr)
        return 1

    username = oauth._ensure_creds_valid(creds)
    creds.save()

    # Clean up state file
    Path(STATE_FILE).unlink(missing_ok=True)

    print(f"OK: Logged in as {username}")
    print(f"Credentials saved to {CREDS_FILE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(step2_exchange(sys.argv[1]))
    else:
        sys.exit(step1_print_url())
