"""
jarvis_desktop.py — Run JARVIS in a native macOS window
=======================================================
Instead of opening the Three.js UI in a browser tab, this renders it inside
a real native window (Cocoa WKWebView via pywebview). JARVIS itself runs as
normal underneath — same audio pipeline, same Ollama model, same skills.

It launches main.py with --no-browser (so no browser tab opens), waits for
the local UI server, then displays http://localhost:8766 in a native window.

Requires:
  pip install pywebview

Run directly:
  python3 jarvis_desktop.py

Or just double-click JARVIS.app (built by build_app.sh), which calls this.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
UI_HOST = "127.0.0.1"
UI_PORT = 8766
UI_URL  = f"http://{UI_HOST}:{UI_PORT}"

_backend: subprocess.Popen | None = None


def _wait_for_server(url: str, timeout: int = 40) -> bool:
    """Poll the UI server until it responds or timeout elapses."""
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _start_backend() -> subprocess.Popen:
    """Launch JARVIS (audio loop + UI server) without opening a browser."""
    # Pass through any CLI args (e.g. --text) the user added after the script.
    extra = sys.argv[1:]
    cmd = [PYTHON, os.path.join(HERE, "main.py"), "--no-browser", *extra]
    # Inherit stdout/stderr so the conversation log still appears if launched
    # from a terminal; when launched from the .app it goes to the app's log.
    return subprocess.Popen(cmd, cwd=HERE)


def _shutdown(*_):
    global _backend
    if _backend and _backend.poll() is None:
        try:
            _backend.terminate()
            _backend.wait(timeout=5)
        except Exception:
            _backend.kill()
    _backend = None


def main() -> None:
    global _backend

    try:
        import webview
    except ImportError:
        print("✗ pywebview is not installed.")
        print("  Install it with:  pip install pywebview")
        print("  (Falling back to launching the browser UI instead.)")
        # Graceful fallback — run main.py normally (it opens the browser)
        os.execv(PYTHON, [PYTHON, os.path.join(HERE, "main.py"), *sys.argv[1:]])
        return

    print("Starting JARVIS backend…")
    _backend = _start_backend()

    # Clean shutdown if the process is killed
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    print(f"Waiting for UI server at {UI_URL} …")
    if not _wait_for_server(UI_URL):
        print("✗ UI server did not start in time.")
        _shutdown()
        sys.exit(1)
    print("UI ready — opening native window.")

    # Create the native window. Dark background prevents a white flash on load.
    window = webview.create_window(
        title="JARVIS",
        url=UI_URL,
        width=1440,
        height=900,
        min_size=(1024, 700),
        background_color="#020a16",
        text_select=False,
    )

    # When the window closes, shut the backend down too.
    window.events.closed += _shutdown

    # webview.start() blocks on the main thread (required by Cocoa) until the
    # window is closed.
    try:
        webview.start()
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
