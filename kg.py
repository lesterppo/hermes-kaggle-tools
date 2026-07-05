#!/usr/bin/env python3
"""
Kaggle Notebook CLI v2 — AI-agent-native, token-efficient.
OAuth-based via ~/.kaggle/credentials.json (run oauth_login.py first).

Commands:
  kg.py whoami                           Show authenticated user
  kg.py list [--mine] [--full] [--limit N] [--search T]   List notebooks
  kg.py get owner/slug                   Full kernel metadata
  kg.py status owner/slug                Session status
  kg.py run owner/slug [--gpu] [--tpu] [--internet] [--wait]  Start + optionally wait
  kg.py stop owner/slug                  Cancel running session
  kg.py logs owner/slug [--follow] [--lines N]   Get/stream session logs
  kg.py output owner/slug [--path DIR]   Download output files
  kg.py files owner/slug                 List kernel source files
  kg.py url owner/slug                   Get/save tunnel API URL from logs
  kg.py health owner/slug                Probe tunnel health
  kg.py push --folder DIR                Push kernel from local folder
  kg.py pull owner/slug [--path DIR]     Pull kernel to local folder
  kg.py delete owner/slug                Delete a kernel
  kg.py quota                            Check GPU/accelerator quota
  kg.py sessions                         List active sessions
  kg.py init [--folder DIR]              Initialize new kernel folder

All commands output JSON on stdout: {"ok": true, "summary": "...", ...}
Large outputs (logs, files) go to ~/.hermes/kaggle_output/.
"""

import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
DISK_DIR = Path.home() / ".hermes" / "kaggle_output"
DISK_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = DISK_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
CREDS_FILE = Path.home() / ".kaggle" / "credentials.json"
APIKEY_FILE = Path.home() / ".kaggle" / "kaggle.json"

_client_cache = None
_user_cache = None


# ══════════════════════════════════════════════════════════════════════
# Auth
# ══════════════════════════════════════════════════════════════════════

def _get_client():
    """OAuth-authenticated KaggleClient. Cached per process."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    from kagglesdk import KaggleClient, KaggleCredentials
    from kagglesdk.kaggle_env import KaggleEnv

    if not CREDS_FILE.exists():
        _die("not_authenticated",
             "No credentials found. Run: python3 ~/.hermes/scripts/kaggle/oauth_login.py\n"
             "Then open the URL in your browser and paste the code.")

    bootstrap = KaggleClient(env=KaggleEnv.PROD)
    creds = KaggleCredentials.load(client=bootstrap)
    if creds:
        try:
            token = creds.get_access_token()
            if token:
                global _user_cache
                _user_cache = creds.get_username()
                _client_cache = KaggleClient(env=KaggleEnv.PROD, api_token=token)
                return _client_cache
        except Exception as e:
            _die("auth_expired",
                 f"Credentials expired or invalid: {e}\n"
                 "Re-run: python3 ~/.hermes/scripts/kaggle/oauth_login.py")

    from kagglesdk import get_access_token_from_env
    token, _ = get_access_token_from_env()
    if token:
        _client_cache = KaggleClient(env=KaggleEnv.PROD, api_token=token)
        return _client_cache

    _die("not_authenticated",
         "Not authenticated. Run: python3 ~/.hermes/scripts/kaggle/oauth_login.py")


def _get_user():
    global _user_cache
    if _user_cache:
        return _user_cache
    try:
        _get_client()
    except SystemExit:
        pass
    return _user_cache or "unknown"


def _check_legacy_api_key():
    """Return True if kaggle.json exists (needed for push/pull/init)."""
    return APIKEY_FILE.exists()


# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════

def _pointer(ok, summary, path=None, extra=None, warning=None):
    out = {"ok": ok, "summary": summary}
    if path:
        out["file"] = str(path)
    if extra:
        out.update(extra)
    if warning:
        out["warning"] = warning
    return out


def _dump(data, disk_file=None):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    if disk_file and data.get("ok"):
        p = Path(disk_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        val = data.get("full", data)
        with open(p, "w") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)


def _die(code, msg):
    _dump({"ok": False, "error": code, "summary": msg})
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# Slug normalization — handles all formats transparently
# ══════════════════════════════════════════════════════════════════════

def _resolve_ref(kernel: str):
    """Normalize kernel ref to (owner, slug). Accepts:
    - owner/slug  → (owner, slug)
    - slug        → (current_user, slug)"""
    kernel = kernel.strip("/").lstrip("code/")
    parts = kernel.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    # Bare slug → assume current user
    user = _get_user()
    if user == "unknown":
        raise ValueError(f"Cannot resolve bare slug '{kernel}' — unknown user. Use owner/slug format.")
    return user, kernel


def _ref_str(kernel: str) -> str:
    """Normalize to 'owner/slug' string."""
    o, s = _resolve_ref(kernel)
    return f"{o}/{s}"


def _api_slug(kernel: str) -> str:
    """Return just the slug part (for save_kernel API)."""
    return _resolve_ref(kernel)[1]


def _full_ref(kernel: str) -> str:
    """Return 'owner/slug' for create_kernel_session API."""
    return _ref_str(kernel)


# ══════════════════════════════════════════════════════════════════════
# Session state persistence
# ══════════════════════════════════════════════════════════════════════

def _session_state_path(kernel: str) -> Path:
    o, s = _resolve_ref(kernel)
    return STATE_DIR / f"{o}_{s}.json"


def _load_session_state(kernel: str) -> dict:
    p = _session_state_path(kernel)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_session_state(kernel: str, data: dict):
    p = _session_state_path(kernel)
    existing = _load_session_state(kernel)
    existing.update(data)
    p.write_text(json.dumps(existing, indent=2))


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _fmt_seconds(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _extract_tunnel_url(log_text: str):
    """Extract Cloudflare tunnel URL from log output."""
    urls = re.findall(r'https://[^ ]*trycloudflare\.com(?:/v\d+)?', log_text)
    if urls:
        url = urls[-1]
        if not url.endswith("/v1"):
            url += "/v1"
        return url
    return None


def _extract_session_id(op_name: str):
    """Extract numeric session ID from operation name like 'operations/create-kernel-session/332872369'."""
    m = re.search(r'/(\d+)$', op_name)
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════

def cmd_whoami(args):
    try:
        client = _get_client()
        user = _get_user()
        has_apikey = _check_legacy_api_key()
        w = None if has_apikey else "kaggle.json API key not found — push/pull/init will fail. Create at kaggle.com → Settings → API."
        _dump(_pointer(True, f"Authenticated as {user}",
                       extra={"user": user, "has_legacy_api_key": has_apikey},
                       warning=w))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_list(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest, ApiGetKernelRequest
        from kagglesdk.kernels.types.kernels_enums import KernelsListViewType, KernelsListSortType

        client = _get_client()
        req = ApiListKernelsRequest()
        req.page = 1
        req.page_size = min(args.limit or 20, 50)
        req.sort_by = KernelsListSortType.HOTNESS

        if args.mine:
            req.group = KernelsListViewType.PUBLIC_AND_USERS_PRIVATE
            req.user = _get_user()
        elif args.user:
            req.group = KernelsListViewType.PUBLIC_AND_USERS_PRIVATE
            req.user = args.user
        else:
            req.group = KernelsListViewType.EVERYONE
            if args.search:
                req.search = args.search

        resp = client.kernels.kernels_api_client.list_kernels(req)
        kernels = []
        refs_to_enrich = []
        if resp and resp.kernels:
            for k in resp.kernels:
                if k is None:
                    continue
                ref = getattr(k, "ref", "")
                slug = getattr(k, "slug", "") or (ref.split("/")[-1] if "/" in ref else ref)
                kernels.append({
                    "ref": ref, "title": getattr(k, "title", ""),
                    "author": getattr(k, "author", ""), "slug": slug,
                    "language": getattr(k, "language", "") or "unknown",
                    "kernel_type": getattr(k, "kernel_type", "") or "unknown",
                    "last_run_time": str(getattr(k, "last_run_time", "")),
                    "is_private": getattr(k, "is_private", False),
                    "gpu": getattr(k, "enable_gpu", False),
                    "internet": getattr(k, "enable_internet", False),
                    "machine_shape": getattr(k, "machine_shape", ""),
                    "url": f"https://www.kaggle.com/code/{ref}",
                })
                refs_to_enrich.append(ref)

        # --full: enrich with get_kernel in parallel (up to 5 concurrent)
        if args.full and refs_to_enrich:
            def _enrich(ref):
                try:
                    o, s = ref.split("/", 1)
                    greq = ApiGetKernelRequest()
                    greq.user_name = o; greq.kernel_slug = s
                    gresp = client.kernels.kernels_api_client.get_kernel(greq)
                    m = gresp.metadata
                    return {
                        "ref": ref,
                        "language": m.language,
                        "kernel_type": m.kernel_type,
                        "gpu": m.enable_gpu,
                        "tpu": getattr(m, "enable_tpu", False),
                        "internet": m.enable_internet,
                        "machine_shape": m.machine_shape,
                        "private": m.is_private,
                        "version": m.current_version_number,
                    }
                except Exception:
                    return {"ref": ref}

            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(_enrich, r): r for r in refs_to_enrich}
                enrich_map = {}
                for fut in as_completed(futures):
                    result = fut.result()
                    enrich_map[result["ref"]] = result

            for k in kernels:
                enrich = enrich_map.get(k["ref"], {})
                for key in ("language", "kernel_type", "gpu", "tpu", "internet",
                            "machine_shape", "private", "version"):
                    if key in enrich and enrich[key] not in (None, "", "unknown"):
                        k[key] = enrich[key]

        _dump(_pointer(True, f"{len(kernels)} kernel(s)",
                       extra={"count": len(kernels), "kernels": kernels}))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_get(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()

        req = ApiGetKernelRequest()
        req.user_name = owner; req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.get_kernel(req)
        m = resp.metadata

        # Also load session state for tunnel URL
        state = _load_session_state(args.kernel)

        _dump(_pointer(True, f"Kernel: {m.title}", extra={
            "kernel": ref, "title": m.title,
            "language": m.language, "kernel_type": m.kernel_type,
            "gpu": m.enable_gpu, "tpu": getattr(m, "enable_tpu", False),
            "internet": m.enable_internet, "machine_shape": m.machine_shape,
            "private": m.is_private, "version": m.current_version_number,
            "url": f"https://www.kaggle.com/code/{m.ref}",
            "tunnel_url": state.get("tunnel_url"),
            "session_id": state.get("session_id"),
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_status(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()

        req = ApiGetKernelSessionStatusRequest()
        req.user_name = owner; req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.get_kernel_session_status(req)
        st = resp.status
        st_name = st.name if hasattr(st, "name") else str(st)

        _dump(_pointer(True, f"Status: {st_name}", extra={
            "status": st_name.lower(), "failure_message": resp.failure_message or "",
            "kernel": ref,
        }))
    except SystemExit:
        raise
    except Exception as e:
        msg = str(e)
        if "404" in msg:
            _dump(_pointer(True, "No active session", extra={"status": "idle", "kernel": _ref_str(args.kernel)}))
        else:
            _dump(_pointer(False, msg))


def _stream_logs_to_file(owner, slug, max_lines=500, timeout=300):
    """Stream logs and save to disk. Returns (lines, log_file_path, tunnel_url)."""
    import requests as req_lib
    client = _get_client()
    http = client._http_client
    http._init_session()
    from kagglesdk.kaggle_env import KaggleEnv

    base = http._endpoint
    env = getattr(http, "_env", None)
    if env is not None and hasattr(env, "name") and env.name != "PROD":
        base = f"{http._endpoint}/api"

    url = f"{base}/v1/kernels/logs/stream/{owner}/{slug}"
    headers = dict(http._session.headers)
    headers["Accept"] = "text/event-stream, */*"
    headers.pop("Content-Type", None)

    r = http._session.get(url, stream=True, headers=headers, auth=http._session.auth,
                          timeout=min(timeout, 300))
    r.raise_for_status()

    lines = []
    tunnel_url = None
    ct = (r.headers.get("Content-Type") or "").lower()

    if "text/event-stream" in ct:
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = raw_line[len("data:"):].strip()
            if payload == "END_OF_LOG":
                break
            try:
                evt = json.loads(payload)
                data = evt.get("data", "")
            except json.JSONDecodeError:
                data = payload
            lines.append(data)
            if not tunnel_url:
                tunnel_url = _extract_tunnel_url(data)
            if len(lines) >= max_lines:
                break
    else:
        body = r.text
        try:
            for evt in json.loads(body):
                data = evt.get("data", "")
                lines.append(data)
                if not tunnel_url:
                    tunnel_url = _extract_tunnel_url(data)
        except (json.JSONDecodeError, ValueError):
            lines = body.split("\n")
    r.close()

    log_file = DISK_DIR / f"logs_{owner}_{slug}.txt"
    log_file.write_text("\n".join(lines))
    return lines, log_file, tunnel_url


def cmd_run(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiCreateKernelSessionRequest
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()

        req = ApiCreateKernelSessionRequest()
        req.slug = ref  # must be owner/slug format

        if args.gpu:
            req.machine_shape = "GPU"
        elif args.tpu:
            req.machine_shape = "TPU"
        if args.internet:
            req.enable_internet = True

        op = client.kernels.kernels_api_client.create_kernel_session(req)
        op_name = getattr(op, "name", "")
        session_id = _extract_session_id(op_name)

        _save_session_state(args.kernel, {
            "session_id": session_id,
            "operation": op_name,
            "started_at": time.time(),
        })

        if args.wait:
            print(json.dumps({"ok": True, "summary": "Session started, waiting for completion...",
                              "kernel": ref, "session_id": session_id}))
            sys.stdout.flush()

            # Poll loop
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
            poll_interval = 10
            max_wait = getattr(args, 'wait_timeout', 600) or 600
            elapsed = 0

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                status_req = ApiGetKernelSessionStatusRequest()
                status_req.user_name = owner; status_req.kernel_slug = slug
                try:
                    st_resp = client.kernels.kernels_api_client.get_kernel_session_status(status_req)
                    st = st_resp.status
                    st_name = st.name if hasattr(st, "name") else str(st)
                except Exception:
                    continue

                if st_name in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
                    # Grab logs
                    lines, log_file, tunnel_url = _stream_logs_to_file(owner, slug)
                    if tunnel_url:
                        _save_session_state(args.kernel, {"tunnel_url": tunnel_url})

                    extra = {
                        "kernel": ref, "session_id": session_id,
                        "status": st_name.lower(),
                        "failure_message": getattr(st_resp, "failure_message", "") or "",
                        "log_lines": len(lines), "log_file": str(log_file),
                        "tunnel_url": tunnel_url,
                    }
                    _dump(_pointer(True, f"Session {st_name.lower()}", path=str(log_file), extra=extra))
                    return

            # Timeout
            _dump(_pointer(True, "Session still running (wait timeout)", extra={
                "kernel": ref, "session_id": session_id, "status": "running",
            }))
        else:
            _dump(_pointer(True, "Session started", extra={
                "kernel": ref, "operation": op_name, "session_id": session_id,
                "done": getattr(op, "done", False),
                "gpu": args.gpu, "tpu": args.tpu, "internet": args.internet,
            }))

    except SystemExit:
        raise
    except Exception as e:
        msg = str(e)
        if "409" in msg or "Conflict" in msg:
            _dump(_pointer(False, "GPU quota exhausted or session already running. Check: kg.py quota; kg.py status owner/slug"))
        else:
            _dump(_pointer(False, msg))


def cmd_stop(args):
    """Cancel a running kernel session."""
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest
        ref = _full_ref(args.kernel)
        client = _get_client()

        # Get session_id from saved state
        state = _load_session_state(args.kernel)
        session_id = state.get("session_id")
        if not session_id:
            _dump(_pointer(False, "No session_id found in saved state. Use kg.py status owner/slug first, or stop via Kaggle web UI."))
            return

        req = ApiCancelKernelSessionRequest()
        req.kernel_session_id = int(session_id)

        client.kernels.kernels_api_client.cancel_kernel_session(req)

        _save_session_state(args.kernel, {"session_id": None})
        _dump(_pointer(True, f"Session cancelled: {ref}", extra={"kernel": ref, "session_id": session_id}))

    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_logs(args):
    try:
        owner, slug = _resolve_ref(args.kernel)
        lines, log_file, tunnel_url = _stream_logs_to_file(owner, slug, max_lines=args.lines or 200)

        if tunnel_url:
            _save_session_state(args.kernel, {"tunnel_url": tunnel_url})

        preview = "\n".join(lines[-20:]) if lines else "(empty)"
        _dump(_pointer(True, f"{len(lines)} log line(s)", path=str(log_file), extra={
            "kernel": _ref_str(args.kernel), "lines": len(lines),
            "preview_last_20": preview, "tunnel_url": tunnel_url,
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_url(args):
    """Get or discover the Cloudflare tunnel URL for a kernel."""
    ref = _ref_str(args.kernel)
    owner, slug = _resolve_ref(args.kernel)

    # Check saved state first
    state = _load_session_state(args.kernel)
    tunnel_url = state.get("tunnel_url")

    if not tunnel_url and not args.no_scan:
        # Try to extract from recent logs
        try:
            lines, _, found = _stream_logs_to_file(owner, slug, max_lines=300)
            tunnel_url = found
            if tunnel_url:
                _save_session_state(args.kernel, {"tunnel_url": tunnel_url})
        except Exception:
            pass

    if tunnel_url:
        _dump(_pointer(True, f"Tunnel URL: {tunnel_url}", extra={
            "kernel": ref, "tunnel_url": tunnel_url,
        }))
    else:
        _dump(_pointer(False, "No tunnel URL found. Start a session first with: kg.py run owner/slug"))


def cmd_health(args):
    """Probe the tunnel API health."""
    import requests as req_lib
    ref = _ref_str(args.kernel)
    state = _load_session_state(args.kernel)
    tunnel_url = state.get("tunnel_url")

    if not tunnel_url:
        # Try to auto-discover
        owner, slug = _resolve_ref(args.kernel)
        try:
            _, _, found = _stream_logs_to_file(owner, slug, max_lines=300)
            if found:
                tunnel_url = found
                _save_session_state(args.kernel, {"tunnel_url": tunnel_url})
        except Exception:
            pass

    if not tunnel_url:
        _dump(_pointer(False, "No tunnel URL. Start session with: kg.py run --wait owner/slug"))
        return

    results = {"kernel": ref, "tunnel_url": tunnel_url}

    # Test /models
    try:
        r = req_lib.get(f"{tunnel_url}/models", timeout=10)
        if r.ok:
            data = r.json()
            models = data.get("data", [])
            results["models"] = len(models)
            results["model_ids"] = [m.get("id", "?") for m in models]
        else:
            results["models_error"] = f"HTTP {r.status_code}"
    except Exception as e:
        results["models_error"] = str(e)

    # Test /chat/completions
    try:
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10, "stream": False}
        r = req_lib.post(f"{tunnel_url}/chat/completions", json=body,
                         headers={"Authorization": "Bearer no-key"}, timeout=30)
        if r.ok:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            results["inference"] = "ok"
            results["inference_chars"] = len(content)
        else:
            results["inference_error"] = f"HTTP {r.status_code}"
    except Exception as e:
        results["inference_error"] = str(e)

    healthy = "models" in results and "inference" in results
    _dump(_pointer(healthy, "healthy" if healthy else "unhealthy", extra=results))


def cmd_sessions(args):
    """List kernels with active sessions."""
    try:
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiListKernelsRequest, ApiGetKernelSessionStatusRequest,
        )
        from kagglesdk.kernels.types.kernels_enums import KernelsListViewType, KernelsListSortType

        client = _get_client()
        req = ApiListKernelsRequest()
        req.page = 1; req.page_size = 30; req.sort_by = KernelsListSortType.DATE_RUN
        req.group = KernelsListViewType.PUBLIC_AND_USERS_PRIVATE
        req.user = _get_user()
        resp = client.kernels.kernels_api_client.list_kernels(req)

        active = []
        if resp and resp.kernels:
            for k in resp.kernels:
                if k is None:
                    continue
                ref = getattr(k, "ref", "")
                if "/" not in ref:
                    continue
                owner, slug = ref.split("/", 1)
                try:
                    sr = ApiGetKernelSessionStatusRequest()
                    sr.user_name = owner; sr.kernel_slug = slug
                    st_resp = client.kernels.kernels_api_client.get_kernel_session_status(sr)
                    st = st_resp.status
                    st_name = st.name if hasattr(st, "name") else str(st)
                    if st_name not in ("QUEUED", "RUNNING"):
                        continue
                except Exception:
                    continue

                # Load tunnel URL from saved state
                state = _load_session_state(ref)
                active.append({
                    "ref": ref, "title": getattr(k, "title", ""),
                    "status": st_name.lower(),
                    "tunnel_url": state.get("tunnel_url"),
                    "session_id": state.get("session_id"),
                })

        _dump(_pointer(True, f"{len(active)} active session(s)", extra={"sessions": active}))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_output(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiListKernelSessionOutputRequest, ApiDownloadKernelOutputRequest,
        )
        import requests as req_lib
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()
        dest = Path(args.path or str(DISK_DIR / f"output_{owner}_{slug}"))
        dest.mkdir(parents=True, exist_ok=True)

        list_req = ApiListKernelSessionOutputRequest()
        list_req.user_name = owner; list_req.kernel_slug = slug
        list_resp = client.kernels.kernels_api_client.list_kernel_session_output(list_req)

        downloaded = []
        if list_resp and list_resp.files:
            http = client._http_client
            http._init_session()
            for f in list_resp.files:
                fname = f.name; fpath = dest / fname
                dl_req = ApiDownloadKernelOutputRequest()
                dl_req.owner_slug = owner; dl_req.kernel_slug = slug
                dl_req.file_path = fname
                redirect = client.kernels.kernels_api_client.download_kernel_output(dl_req)
                url = redirect.url if hasattr(redirect, "url") else str(redirect)
                r = http._session.get(url, headers=http._session.headers)
                r.raise_for_status()
                fpath.write_bytes(r.content)
                downloaded.append({"name": fname, "size": len(r.content), "path": str(fpath)})

        _dump(_pointer(True, f"{len(downloaded)} file(s) downloaded", extra={
            "kernel": ref, "files": downloaded, "output_dir": str(dest),
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_files(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelFilesRequest
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()
        req = ApiListKernelFilesRequest()
        req.user_name = owner; req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.list_kernel_files(req)

        files = []
        if resp and resp.files:
            for f in resp.files:
                files.append({"name": getattr(f, "name", ""), "size": getattr(f, "size", 0),
                              "creation_date": str(getattr(f, "creation_date", ""))})

        _dump(_pointer(True, f"{len(files)} file(s)", extra={"kernel": ref, "files": files}))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_push(args):
    try:
        if not _check_legacy_api_key():
            _dump(_pointer(False,
                           "kaggle.json API key required for push. Create at kaggle.com → Settings → API → Create New Token.\n"
                           "Save the downloaded kaggle.json to ~/.kaggle/kaggle.json"))
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        folder = args.folder
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        result = api.kernels_push(folder)
        _dump(_pointer(True, f"Kernel pushed from {folder}", extra={
            "folder": str(Path(folder).resolve()),
            "ref": getattr(result, "ref", "") if result else "",
            "version": getattr(result, "version_number", 0) if result else 0,
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_pull(args):
    try:
        if not _check_legacy_api_key():
            _dump(_pointer(False,
                           "kaggle.json API key required for pull. Create at kaggle.com → Settings → API."))
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        ref = _ref_str(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        dest = args.path or str(Path.cwd() / slug)
        api.kernels_pull(ref, path=dest)
        _dump(_pointer(True, f"Kernel pulled to {dest}", extra={
            "kernel": ref, "destination": str(Path(dest).resolve()),
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_delete(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest, ApiGetKernelRequest
        ref = _full_ref(args.kernel)
        owner, slug = _resolve_ref(args.kernel)
        client = _get_client()
        greq = ApiGetKernelRequest(); greq.user_name = owner; greq.kernel_slug = slug
        k_info = client.kernels.kernels_api_client.get_kernel(greq)
        kid = k_info.metadata.id
        dreq = ApiDeleteKernelRequest(); dreq.id = kid; dreq.user_name = owner; dreq.kernel_slug = slug
        client.kernels.kernels_api_client.delete_kernel(dreq)
        _dump(_pointer(True, f"Kernel deleted: {ref}", extra={"kernel": ref}))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_quota(args):
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
        client = _get_client()
        req = ApiGetAcceleratorQuotaStatisticsRequest()
        resp = client.kernels.kernels_api_client.get_accelerator_quota_statistics(req)

        quotas = []
        if resp and hasattr(resp, "quotas"):
            for q in resp.quotas or []:
                quotas.append({
                    "type": getattr(q, "accelerator_type", "unknown"),
                    "used_hours": round(getattr(q, "time_used", timedelta()).total_seconds() / 3600, 2),
                    "reserved_hours": round(getattr(q, "time_reserved", timedelta()).total_seconds() / 3600, 2),
                    "total_hours": round(getattr(q, "total_time_allowed", timedelta()).total_seconds() / 3600, 2),
                })

        total_used = sum(q["used_hours"] for q in quotas)
        total_allowed = sum(q["total_hours"] for q in quotas)

        if total_allowed == 0:
            _dump(_pointer(True, "GPU quota: unknown (free tier — tracked server-side)",
                           extra={"note": "Free-tier quota not exposed via API. ~30h/week typical."}))
        else:
            pct = round(total_used / total_allowed * 100, 1)
            _dump(_pointer(True, f"GPU quota: {total_used:.1f}h / {total_allowed:.1f}h ({pct}%)",
                           extra={"used_hours": total_used, "total_hours": total_allowed,
                                  "percent": pct, "details": quotas}))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


def cmd_init(args):
    try:
        if not _check_legacy_api_key():
            _dump(_pointer(False,
                           "kaggle.json API key required for init. Create at kaggle.com → Settings → API."))
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        folder = args.folder or str(Path.cwd())
        s = api.kernels_initialize(folder)
        _dump(_pointer(True, f"Kernel initialized: {s}", extra={
            "slug": s, "folder": str(Path(folder).resolve()),
        }))
    except SystemExit:
        raise
    except Exception as e:
        _dump(_pointer(False, str(e)))


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

COMMANDS = {
    "whoami": cmd_whoami,
    "list": cmd_list, "ls": cmd_list,
    "get": cmd_get, "info": cmd_get,
    "status": cmd_status,
    "run": cmd_run, "start": cmd_run,
    "stop": cmd_stop, "cancel": cmd_stop,
    "logs": cmd_logs,
    "url": cmd_url,
    "health": cmd_health,
    "sessions": cmd_sessions,
    "output": cmd_output,
    "files": cmd_files,
    "push": cmd_push,
    "pull": cmd_pull,
    "delete": cmd_delete, "rm": cmd_delete,
    "quota": cmd_quota,
    "init": cmd_init,
}


def main():
    parser = argparse.ArgumentParser(description="Kaggle Notebook CLI v2")
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("whoami", help="Show authenticated user + API key status")

    p = subs.add_parser("list", aliases=["ls"], help="List notebooks")
    p.add_argument("--mine", action="store_true")
    p.add_argument("--user", type=str)
    p.add_argument("--search", type=str)
    p.add_argument("--full", action="store_true", help="Enrich with full metadata (parallel)")
    p.add_argument("--limit", type=int, default=20)

    p = subs.add_parser("get", aliases=["info"], help="Full kernel metadata")
    p.add_argument("kernel")

    p = subs.add_parser("status", help="Get session status (returns idle if 404)")
    p.add_argument("kernel")

    p = subs.add_parser("run", aliases=["start"], help="Start a kernel session")
    p.add_argument("kernel")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--tpu", action="store_true")
    p.add_argument("--internet", action="store_true")
    p.add_argument("--wait", action="store_true", help="Block until COMPLETE/ERROR, stream logs")
    p.add_argument("--wait-timeout", type=int, default=600, help="Max seconds for --wait")

    p = subs.add_parser("stop", aliases=["cancel"], help="Cancel a running session")
    p.add_argument("kernel")

    p = subs.add_parser("logs", help="Get session logs")
    p.add_argument("kernel")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--lines", type=int, default=200)

    p = subs.add_parser("url", help="Get/discover tunnel API URL")
    p.add_argument("kernel")
    p.add_argument("--no-scan", action="store_true", help="Don't scan logs, just return saved")

    p = subs.add_parser("health", help="Probe tunnel API health")
    p.add_argument("kernel")

    subs.add_parser("sessions", help="List kernels with active sessions")

    p = subs.add_parser("output", help="Download output files")
    p.add_argument("kernel")
    p.add_argument("--path", type=str)

    p = subs.add_parser("files", help="List kernel source files")
    p.add_argument("kernel")

    p = subs.add_parser("push", help="Push kernel from local folder (needs kaggle.json)")
    p.add_argument("--folder", "-f", type=str, required=True)

    p = subs.add_parser("pull", help="Pull kernel to local folder (needs kaggle.json)")
    p.add_argument("kernel")
    p.add_argument("--path", type=str)

    p = subs.add_parser("delete", aliases=["rm"], help="Delete a kernel")
    p.add_argument("kernel")

    subs.add_parser("quota", help="Check GPU/accelerator quota")

    p = subs.add_parser("init", help="Initialize new kernel folder (needs kaggle.json)")
    p.add_argument("--folder", "-f", type=str)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    fn = COMMANDS.get(args.command)
    if not fn:
        _dump(_pointer(False, f"Unknown command: {args.command}"))
        sys.exit(1)

    fn(args)


if __name__ == "__main__":
    main()
