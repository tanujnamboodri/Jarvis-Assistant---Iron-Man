#!/bin/bash
# JARVIS launcher — HEADLESS.
# Place this file at Contents/MacOS/JARVIS inside JARVIS.app (overwrite the
# default one Platypus/py2app generates), then: chmod +x on it.
#
# What changed vs a naive launcher:
#   * No osascript / Terminal window. Telling Terminal.app to open a window
#     and run commands IS the "terminal appears on launch" problem.
#   * No `open -a Ollama` (the GUI menu-bar app). We run the plain
#     `ollama serve` daemon instead — same server, zero UI.
#   * All output goes to ~/Library/Logs/JARVIS/ so you can still debug:
#         tail -f ~/Library/Logs/JARVIS/jarvis.log
#
# PROJECT_DIR is resolved from the .app's own location, NOT hardcoded, so
# this works for anyone regardless of where they put JARVIS.app or what
# their username is. If you placed jarvis_desktop.py somewhere other than
# next to the .app itself, override PROJECT_DIR below.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="${JARVIS_PROJECT_DIR:-$APP_DIR}"
PYTHON="${JARVIS_PYTHON:-$(command -v python3)}"
OLLAMA="${JARVIS_OLLAMA_BIN:-$(command -v ollama)}"

LOGDIR="$HOME/Library/Logs/JARVIS"
mkdir -p "$LOGDIR"

# --- Ollama: start headless server only if not already running -------------
if ! curl -s --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1; then
    nohup "$OLLAMA" serve >> "$LOGDIR/ollama.log" 2>&1 &
    # Wait up to 20 s for the server to come up
    for i in $(seq 1 40); do
        curl -s --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

# --- JARVIS: run the pywebview desktop app, no terminal --------------------
cd "$PROJECT_DIR" || exit 1
# `exec` keeps python as the .app's own process, so macOS attributes the
# microphone permission prompt to JARVIS.app (NSMicrophoneUsageDescription
# in Info.plist). [VERIFY] on first launch: if macOS instead prompts for
# "python3", grant it — Homebrew python sometimes carries its own TCC
# identity regardless.
exec "$PYTHON" jarvis_desktop.py >> "$LOGDIR/jarvis.log" 2>&1
