#!/usr/bin/env python3
"""
RealVisXL Chat CLI — Interactive image generation client.
Sends prompts to Kaggle-deployed RealVisXL_V4.0 API via Cloudflare tunnel.

Commands:
  <prompt text>          Generate image from text
  /img2img <path> <txt>  Image-to-image with strength
  /params [k=v ...]      Show or set default params
  /history [N]           Show last N prompts
  /open [N]              Open image N in system viewer
  /save <path>           Save last image to path
  /retry                 Re-generate last prompt
  /health                Check server status
  /url <url>             Set/get API URL
  /help                  Show this help
  /quit                  Exit

Images saved to ~/.hermes/kaggle_output/realvis/
"""

import base64, io, json, os, re, shlex, sys, textwrap, time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Paths ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path.home() / ".hermes" / "kaggle_output" / "realvis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
URL_FILE = OUTPUT_DIR / "api_url.txt"
HISTORY_FILE = OUTPUT_DIR / "history.json"

# ── State ───────────────────────────────────────────────────────────
api_url: Optional[str] = None
history: list = []
last_image_path: Optional[str] = None
last_request: Optional[dict] = None

# Default params
params = {
    "steps": 25,
    "cfg_scale": 7.0,
    "width": 1024,
    "height": 1024,
    "seed": -1,
    "negative_prompt": "",
    "img2img_strength": 0.75,
}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _die(msg: str, code: int = 1):
    print(f"\033[31mERROR:\033[0m {msg}")
    sys.exit(code)


def _ok(msg: str):
    print(f"  \033[32m✓\033[0m {msg}")


def _info(msg: str):
    print(f"  \033[36mℹ\033[0m {msg}")


def _warn(msg: str):
    print(f"  \033[33m⚠\033[0m {msg}")


def _api(endpoint: str, data: Optional[dict] = None, method: str = "POST") -> dict:
    """Call Kaggle API endpoint."""
    if not api_url:
        _die("No API URL set. Use /url <url> or /health to discover.")
    url = f"{api_url.rstrip('/')}{endpoint}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except URLError as e:
        _die(f"API call failed: {e}")
    except json.JSONDecodeError:
        _die("API returned invalid JSON")


def _slug(text: str, max_len: int = 40) -> str:
    """Safe filename slug from prompt text."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower().strip())[:max_len].strip('_')
    return slug or "image"


def _save_image(b64: str, prompt: str, seed: int) -> str:
    """Decode base64, save PNG, return path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{ts}_{_slug(prompt)}_s{seed}.png"
    path.write_bytes(base64.b64decode(b64))
    return str(path)


def _save_history(entry: dict):
    history.append(entry)
    HISTORY_FILE.write_text(json.dumps(history[-50:], indent=2))  # keep last 50


def _load_history():
    global history
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            history = []


def _notify(title: str, body: str):
    """Desktop notification (Linux notify-send, Windows fallback)."""
    try:
        subprocess = __import__('subprocess')
        subprocess.run(["notify-send", title, body], timeout=2)
    except:
        pass  # no notify-send available


def _open_image(path: str):
    """Open image in system viewer."""
    img_path = str(path)
    if sys.platform == "win32":
        os.startfile(img_path)
    elif sys.platform == "darwin":
        __import__('subprocess').run(["open", img_path])
    else:
        # WSL / Linux — prefer cmd.exe on WSL, else xdg-open
        if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
            # Convert to Windows path
            win_path = str(path).replace("/home/peter", "C:\\Users\\Peter").replace("/", "\\")
            try:
                __import__('subprocess').run(
                    ["cmd.exe", "/c", "start", "", win_path], timeout=5)
                return
            except:
                pass
        __import__('subprocess').run(["xdg-open", img_path], timeout=5)


def _color(text: str, code: str) -> str:
    colors = {"red": "31", "green": "32", "yellow": "33", "blue": "34",
              "magenta": "35", "cyan": "36", "white": "37", "bold": "1"}
    c = colors.get(code, "0")
    return f"\033[{c}m{text}\033[0m"


# ══════════════════════════════════════════════════════════════════════
# Command handlers
# ══════════════════════════════════════════════════════════════════════

def cmd_generate(prompt: str):
    """Send /generate request to API."""
    global last_image_path, last_request

    steps = int(params["steps"])
    seed = int(params["seed"]) if params["seed"] >= 0 else -1

    print(f"\n{_color('GENERATING', 'bold')} — \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
    print(f"  {params['width']}x{params['height']}, {steps} steps, CFG {params['cfg_scale']}, seed={'auto' if seed < 0 else seed}")

    t0 = time.time()
    result = _api("/generate", {
        "prompt": prompt,
        "negative_prompt": params["negative_prompt"],
        "steps": steps,
        "cfg_scale": params["cfg_scale"],
        "width": int(params["width"]),
        "height": int(params["height"]),
        "seed": seed,
    })
    elapsed = time.time() - t0

    if not result.get("ok"):
        _die(f"Generation failed: {result}")

    actual_seed = result.get("seed", seed)
    path = _save_image(result["image_b64"], prompt, actual_seed)
    last_image_path = path
    last_request = {"type": "generate", "prompt": prompt, "seed": actual_seed,
                    "steps": result.get("steps", steps), "cfg": result.get("cfg_scale", params["cfg_scale"])}

    size_kb = result.get("size_bytes", 0) / 1024
    _ok(f"Done in {elapsed:.1f}s — {size_kb:.0f} KB — seed={actual_seed}")
    _info(f"Saved: {path}")

    _save_history({"ts": datetime.now().isoformat(), "type": "generate",
                    "prompt": prompt, "seed": actual_seed, "file": path})

    _open_image(path)
    _notify("RealVisXL", f"Generated: {prompt[:60]}...")


def cmd_img2img(image_path: str, prompt: str):
    """Send /img2img request to API."""
    global last_image_path, last_request

    ip = Path(image_path)
    if not ip.exists():
        _die(f"File not found: {image_path}")

    b64 = base64.b64encode(ip.read_bytes()).decode()
    strength = float(params["img2img_strength"])
    seed = int(params["seed"]) if params["seed"] >= 0 else -1

    print(f"\n{_color('IMG2IMG', 'bold')} — \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
    print(f"  Source: {ip.name}, strength={strength}, seed={'auto' if seed < 0 else seed}")

    t0 = time.time()
    result = _api("/img2img", {
        "prompt": prompt,
        "image_b64": b64,
        "strength": strength,
        "steps": int(params["steps"]),
        "cfg_scale": params["cfg_scale"],
        "seed": seed,
    })
    elapsed = time.time() - t0

    if not result.get("ok"):
        _die(f"Img2img failed: {result}")

    actual_seed = result.get("seed", seed)
    path = _save_image(result["image_b64"], f"i2i_{prompt}", actual_seed)
    last_image_path = path
    last_request = {"type": "img2img", "prompt": prompt, "seed": actual_seed,
                    "source": str(ip), "strength": strength}

    _ok(f"Done in {elapsed:.1f}s — seed={actual_seed}")
    _info(f"Saved: {path}")

    _save_history({"ts": datetime.now().isoformat(), "type": "img2img",
                    "prompt": prompt, "source": str(ip), "seed": actual_seed, "file": path})

    _open_image(path)
    _notify("RealVisXL img2img", f"Generated: {prompt[:60]}...")


def cmd_params(args: list):
    """Show or set default parameters."""
    global params

    if not args:
        print(f"\n{_color('CURRENT PARAMS', 'bold')}:")
        for k, v in params.items():
            print(f"  {k:20s} = {v}")
        return

    for arg in args:
        if "=" not in arg:
            print(f"  Invalid: {arg} (use key=value)")
            continue
        k, v = arg.split("=", 1)
        if k not in params:
            print(f"  Unknown param: {k}")
            continue
        # Coerce type
        old = params[k]
        if isinstance(old, float):
            params[k] = float(v)
        elif isinstance(old, int):
            params[k] = int(v)
        elif isinstance(old, str):
            params[k] = v
        print(f"  {k}: {old} → {params[k]}")

    # Sync with server
    if api_url:
        try:
            _api("/params", {
                "steps": int(params["steps"]),
                "cfg_scale": params["cfg_scale"],
                "width": int(params["width"]),
                "height": int(params["height"]),
                "seed": int(params["seed"]),
                "negative_prompt": params["negative_prompt"],
            }, method="POST")
        except:
            pass  # server /params endpoint optional


def cmd_history(n: int = 10):
    """Show recent history."""
    items = history[-n:]
    if not items:
        _info("No history yet.")
        return

    print(f"\n{_color('HISTORY', 'bold')} (last {len(items)}):")
    for i, h in enumerate(items):
        idx = len(history) - len(items) + i + 1
        ts = h["ts"][:19].replace("T", " ")
        ptype = h.get("type", "gen")
        prompt = h["prompt"][:60]
        seed = h.get("seed", "?")
        exists = " 📁" if Path(h["file"]).exists() else " ❌"
        print(f"  [{idx}] {ts} {ptype:7s} s={seed} \"{prompt}\"{exists}")


def cmd_open(idx: Optional[int] = None):
    """Open image from history."""
    if idx is None:
        if last_image_path:
            _open_image(last_image_path)
            _info(f"Opened: {last_image_path}")
        else:
            _warn("No image to open.")
    elif 1 <= idx <= len(history):
        path = history[idx - 1]["file"]
        if Path(path).exists():
            _open_image(path)
            _info(f"Opened: {path}")
        else:
            _warn(f"File not found: {path}")
    else:
        _warn(f"Invalid index: {idx}")


def cmd_save(path: str):
    """Save last image to specified path."""
    if not last_image_path or not Path(last_image_path).exists():
        _warn("No image to save.")
        return
    dest = Path(path)
    dest.write_bytes(Path(last_image_path).read_bytes())
    _ok(f"Saved: {dest}")


def cmd_retry():
    """Re-generate last prompt."""
    if not last_request:
        _warn("Nothing to retry — generate an image first.")
        return
    if last_request["type"] == "generate":
        cmd_generate(last_request["prompt"])
    elif last_request["type"] == "img2img":
        cmd_img2img(last_request["source"], last_request["prompt"])


def cmd_health():
    """Check API server status."""
    global api_url
    if not api_url:
        _warn("No API URL set.")
        return

    print(f"\n{_color('CHECKING', 'bold')} {api_url}/health ...")
    try:
        result = _api("/health", method="GET")
        print(f"  Status:    {_color('ONLINE', 'green')}")
        print(f"  Model:     {result.get('model', '?')}")
        print(f"  VRAM:      {result.get('vram_used_gb', '?')} GB")
        print(f"  Device:    {result.get('device', '?')}")
        server_params = result.get("default_params", {})
        if server_params:
            print(f"  Server params: {json.dumps(server_params)}")
    except:
        _warn(f"Server unreachable at {api_url}")


def cmd_url(url: Optional[str] = None):
    """Set or show API URL."""
    global api_url
    if url:
        api_url = url.rstrip("/")
        URL_FILE.write_text(api_url)
        _ok(f"API URL set: {api_url}")
    else:
        if api_url:
            print(f"  API URL: {api_url}")
        else:
            _warn("No API URL set. Use /url <url> or paste a trycloudflare.com URL.")


def cmd_help():
    """Show help."""
    print(f"""
{_color('RealVisXL Chat CLI', 'bold')} — SDXL Image Generation Client

{_color('COMMANDS:', 'bold')}
  {_color('<prompt text>', 'cyan')}          Generate image from text
  {_color('/img2img <path> <text>', 'cyan')} Image-to-image with strength from /params
  {_color('/params [k=v ...]', 'cyan')}      Show or set params (steps, cfg_scale, width,
                             height, seed, negative_prompt, img2img_strength)
  {_color('/history [N]', 'cyan')}           Show last N prompts (default 10)
  {_color('/open [N]', 'cyan')}              Open image N in system viewer (default last)
  {_color('/save <path>', 'cyan')}           Save last image to path
  {_color('/retry', 'cyan')}                 Re-generate last prompt with same seed
  {_color('/health', 'cyan')}                Check server status
  {_color('/url [url]', 'cyan')}             Set or show API URL
  {_color('/help', 'cyan')}                  Show this help
  {_color('/quit', 'cyan')}                  Exit

{_color('EXAMPLES:', 'bold')}
  a photorealistic cat wearing a top hat, 8k, highly detailed
  /params steps=30 cfg_scale=7.5 width=768 height=1024
  /img2img input.jpg make it look like an oil painting
  /history 5
  /open 3

{_color('FILES:', 'bold')}
  Images:    {OUTPUT_DIR}/
  API URL:   {URL_FILE}
""")

    # Show current params
    print(f"{_color('CURRENT PARAMS:', 'bold')}")
    for k, v in params.items():
        print(f"  {k} = {v}")


# ══════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════

def main():
    global api_url

    _load_history()

    # Load saved API URL
    if URL_FILE.exists():
        api_url = URL_FILE.read_text().strip()
        print(f"API URL loaded: {api_url}")

    print(f"{_color('RealVisXL_V4.0 Chat CLI', 'bold')}")
    print(f"Type /help for commands, /quit to exit.\n")

    # Try to auto-discover URL from Kaggle output
    if not api_url:
        kaggle_url_file = Path.home() / ".hermes" / "kaggle_output" / "realvis" / "api_url.txt"
        # Also check medial gemma pattern
        alt_url = Path("/kaggle/working/api_url_realvis.txt")
        # Check kg.py url
        try:
            result = __import__('subprocess').run(
                ["python3", str(Path.home() / ".hermes" / "scripts" / "kaggle" / "kg.py"),
                 "url", "lesteryannes/realvisxl-deploy"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get("url"):
                    api_url = data["url"]
                    URL_FILE.write_text(api_url)
                    _ok(f"Discovered API URL via kg.py: {api_url}")
        except:
            pass

    if not api_url:
        _warn("No API URL found. Set with /url <trycloudflare-url>")

    while True:
        try:
            raw = input(f"\n{_color('realvis>', 'green')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # Parse: /command args...  or  plain prompt text
        if raw.startswith("/"):
            parts = shlex.split(raw)
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                break
            elif cmd == "/help" or cmd == "/h":
                cmd_help()
            elif cmd == "/health":
                cmd_health()
            elif cmd == "/url":
                cmd_url(args[0] if args else None)
            elif cmd == "/params":
                cmd_params(args)
            elif cmd == "/history":
                n = int(args[0]) if args else 10
                cmd_history(n)
            elif cmd == "/open":
                idx = int(args[0]) if args else None
                cmd_open(idx)
            elif cmd == "/save":
                if args:
                    cmd_save(args[0])
                else:
                    _warn("Usage: /save <path>")
            elif cmd == "/retry":
                cmd_retry()
            elif cmd == "/img2img":
                if len(args) < 2:
                    _warn("Usage: /img2img <image_path> <prompt>")
                else:
                    cmd_img2img(args[0], " ".join(args[1:]))
            else:
                _warn(f"Unknown command: {cmd}. Type /help for commands.")
        else:
            # Plain text = generate prompt
            cmd_generate(raw)

    print(f"\n{_color('Bye!', 'bold')} Generated {len([h for h in history if 'file' in h])} images this session.")


if __name__ == "__main__":
    main()
