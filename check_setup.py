#!/usr/bin/env python3
"""
JARVIS — check_setup.py
========================
Run this before your first launch. It checks the things that fail silently
or with a confusing error otherwise, and tells you exactly what to run to
fix each one. Nothing here modifies your system.

    python check_setup.py
"""
from __future__ import annotations
import importlib.util
import shutil
import sys
import urllib.request

PASS, WARN, FAIL = "✓", "!", "✗"
_had_failure = False


def check(label: str, ok: bool, fix: str = "", warn_only: bool = False) -> None:
    global _had_failure
    mark = PASS if ok else (WARN if warn_only else FAIL)
    print(f"  [{mark}] {label}")
    if not ok:
        if fix:
            print(f"        → {fix}")
        if not warn_only:
            _had_failure = True


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


print("\nJARVIS setup check\n" + "=" * 40)

# --- Python version ----------------------------------------------------
check("Python 3.10+", sys.version_info >= (3, 10),
      "Install Python 3.10 or newer (python.org, or `brew install python@3.11`)")

# --- Ollama binary + server ------------------------------------------------
ollama_bin = shutil.which("ollama")
check("Ollama installed", ollama_bin is not None,
      "Install from https://ollama.com/download")

ollama_running = False
if ollama_bin:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        ollama_running = True
    except Exception:
        pass
check("Ollama server reachable", ollama_running,
      "Run `ollama serve` in a terminal, then re-run this check", warn_only=True)

if ollama_running:
    try:
        import json
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        import config
        model_pulled = any(config.OLLAMA_MODEL in n for n in names)
        check(f"Model '{config.OLLAMA_MODEL}' pulled", model_pulled,
              f"Run: ollama pull {config.OLLAMA_MODEL}")
    except Exception:
        check("Could not verify pulled models", False,
              "Check `ollama list` manually", warn_only=True)

# --- Core Python deps --------------------------------------------------
for mod, pip_name in [("ollama", "ollama"), ("numpy", "numpy"),
                       ("requests", "requests")]:
    check(f"Python package '{pip_name}'", has_module(mod),
          f"pip install {pip_name}")

# --- Voice deps (optional but expected for full experience) ------------
check("faster-whisper (speech-to-text)", has_module("faster_whisper"),
      "pip install faster-whisper", warn_only=True)
check("sounddevice (microphone access)", has_module("sounddevice"),
      "pip install sounddevice  (needs PortAudio — see README Troubleshooting)",
      warn_only=True)
check("openwakeword (wake word)", has_module("openwakeword"),
      "pip install openwakeword onnxruntime", warn_only=True)

if sys.platform != "darwin":
    check("pyttsx3 (text-to-speech, non-macOS)", has_module("pyttsx3"),
          "pip install pyttsx3", warn_only=True)
else:
    print(f"  [{PASS}] macOS detected — using built-in `say` for speech, no extra install needed")

# --- Papers folder -------------------------------------------------------
try:
    import config
    pdf_count = len(list(config.PAPERS_DIR.glob("*.pdf")))
    check(f"Papers folder has PDFs ({config.PAPERS_DIR})", pdf_count > 0,
          "Drop research PDFs into that folder for the research assistant / RAG features",
          warn_only=True)
except Exception:
    pass

print("=" * 40)
if _had_failure:
    print("Some REQUIRED items are missing (✗ above). Fix those, then re-run this script.")
    sys.exit(1)
else:
    print("All required checks passed. Items marked (!) are optional features you can add later.")
    print("Run:  python main.py --text     to start in keyboard mode.")
