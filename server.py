"""
Local Fish Audio S2-Pro API server manager.

The server loads the LLaMA + VQGAN models once and keeps them in memory.
It auto-selects MPS (Apple Silicon), CUDA, or CPU as available.

Usage:
  uv run python server.py          # start in foreground (Ctrl-C to stop)
  uv run python server.py --bg     # start as background process
  uv run python server.py --stop   # stop a background process
  uv run python server.py --status # check if running
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
PID_FILE = Path(".server.pid")

CHECKPOINTS = Path("checkpoints/s2-pro")
DECODER_WEIGHTS = CHECKPOINTS / "codec.pth"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_running(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/v1/health", timeout=timeout)
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

def check_models() -> None:
    required = [
        CHECKPOINTS / "model.safetensors.index.json",
        CHECKPOINTS / "model-00001-of-00002.safetensors",
        CHECKPOINTS / "model-00002-of-00002.safetensors",
        DECODER_WEIGHTS,
        CHECKPOINTS / "config.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print("Model weights not found. Run first:")
        print("  uv run python download_models.py")
        print(f"\nMissing: {missing}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def _build_command() -> list[str]:
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    return [
        sys.executable, "local_api_server.py",
        "--llama-checkpoint-path", str(CHECKPOINTS.resolve()),
        "--decoder-checkpoint-path", str(DECODER_WEIGHTS.resolve()),
        "--decoder-config-name", "modded_dac_vq",
        "--device", device,
        "--listen", f"{SERVER_HOST}:{SERVER_PORT}",
        "--workers", "1",
    ]


def start_foreground() -> None:
    check_models()
    cmd = _build_command()
    print(f"Starting fish-speech server on {SERVER_URL} …")
    print("Models loading — this takes ~30-60 s on first run.")
    print("Press Ctrl-C to stop.\n")
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


def start_background() -> subprocess.Popen:
    check_models()

    if is_running():
        print(f"Server already running at {SERVER_URL}")
        return None

    cmd = _build_command()
    log_file = open("server.log", "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"Server started (PID {proc.pid}). Logs → server.log")
    print("Waiting for models to load", end="", flush=True)

    for _ in range(180):
        time.sleep(1)
        print(".", end="", flush=True)
        if is_running():
            print(f"\nReady at {SERVER_URL}")
            return proc

    print("\nTimed out waiting for server — check server.log for errors.")
    return proc


def stop_background() -> None:
    if not PID_FILE.exists():
        if is_running():
            print("Server is running but no PID file found. Kill it manually.")
        else:
            print("Server is not running.")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        import os
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink()
        print(f"Sent SIGTERM to PID {pid}.")
    except ProcessLookupError:
        print(f"PID {pid} not found (already stopped?).")
        PID_FILE.unlink()


def status() -> None:
    running = is_running()
    pid_info = ""
    if PID_FILE.exists():
        pid_info = f" (PID {PID_FILE.read_text().strip()})"
    print(f"Server: {'RUNNING' if running else 'STOPPED'}{pid_info} — {SERVER_URL}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--stop" in args:
        stop_background()
    elif "--status" in args:
        status()
    elif "--bg" in args:
        start_background()
    else:
        start_foreground()
