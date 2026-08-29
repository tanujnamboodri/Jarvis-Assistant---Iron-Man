"""
JARVIS — config.py  Central configuration
==========================================
Every setting a new user might want to change lives here, read from
environment variables (or a .env file — see .env.example). Nothing in the
rest of the codebase should hardcode a path, model name, or API key; if you
find one, it's a bug — open an issue.

Loading order (lowest to highest priority):
    1. Defaults below
    2. .env file in the project root (if python-dotenv is installed)
    3. Real environment variables (always win — good for Docker/CI)
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()   # loads .env from the current working directory, if present
except ImportError:
    pass  # .env support is optional — plain env vars always work


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Where your PDFs live for the research assistant / RAG index.
# Default: a "papers" folder next to this file — create it and drop PDFs in.
PAPERS_DIR = Path(_env("JARVIS_PAPERS_DIR", str(PROJECT_ROOT / "papers")))
PAPERS_DIR.mkdir(exist_ok=True)

# SQLite memory database (schedule, preferences, frequently-used apps).
MEMORY_DB_PATH = _env("JARVIS_MEMORY_DB", str(PROJECT_ROOT / "jarvis_memory.db"))

# Human-readable schedule file (markdown checklist format).
SCHEDULE_PATH = _env("JARVIS_SCHEDULE_PATH", str(PROJECT_ROOT / "schedule.md"))


# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------
# The base conversational model. 3B-class models (llama3.2:3b, phi3:mini,
# qwen2.5:3b) run acceptably on 8GB RAM CPU-only machines — that's the whole
# point of this project. Larger machines can point this at anything Ollama
# serves.
OLLAMA_MODEL = _env("JARVIS_MODEL", "llama3.2:3b")

# If you've fine-tuned a domain model (see the paper / training scripts),
# point this at it. Falls back to OLLAMA_MODEL if unset — nothing breaks if
# you skip fine-tuning entirely and just want the assistant features.
JARVIS_FINETUNED_MODEL = _env("JARVIS_FINETUNED_MODEL", OLLAMA_MODEL)

OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(_env("JARVIS_NUM_CTX", "8192"))


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------
# macOS `say` voice name. Run `say -v '?'` in Terminal to list what's
# installed on your Mac — voices vary by macOS version and language packs.
SAY_VOICE = _env("JARVIS_SAY_VOICE", "Daniel")

# Wake word engine model name (bundled with openWakeWord). "hey_jarvis" ships
# pretrained; train your own with openWakeWord's training notebook if you
# want a different phrase.
WAKE_WORD_MODEL = _env("JARVIS_WAKE_WORD", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(_env("JARVIS_WAKE_THRESHOLD", "0.5"))

# Whisper model size for speech-to-text. tiny/base/small — see stt.py's
# docstring for the speed/accuracy tradeoff on CPU-only machines.
STT_MODEL_SIZE = _env("JARVIS_STT_MODEL", "base.en")


# ---------------------------------------------------------------------------
# Optional third-party API keys (everything here has a working fallback if
# left unset — the assistant is fully functional with zero keys configured)
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # else: Open-Meteo (free, no key)
NEWS_API_KEY         = os.environ.get("NEWS_API_KEY")          # else: BBC/Google News RSS

# Startup dashboard weather location. Leave both unset to auto-detect via
# free IP geolocation (no key needed) — set explicitly to skip that lookup
# or to always show a fixed location regardless of where you're running JARVIS.
_lat = os.environ.get("JARVIS_STARTUP_LAT")
_lon = os.environ.get("JARVIS_STARTUP_LON")
STARTUP_LAT = float(_lat) if _lat else None
STARTUP_LON = float(_lon) if _lon else None

# "Domestic" news search term for the startup news dashboard.
DOMESTIC_NEWS_QUERY = _env("JARVIS_DOMESTIC_NEWS_QUERY", "world news")


# ---------------------------------------------------------------------------
# Optional smart-home integration (EMOS GoSmart / Tuya bulb)
# Entirely optional — if unset, bulb_control's functions just report failure
# and JARVIS keeps working normally. See README "Smart Home (optional)".
# ---------------------------------------------------------------------------
BULB_ID      = os.environ.get("BULB_ID")
BULB_IP      = os.environ.get("BULB_IP")
BULB_KEY     = os.environ.get("BULB_KEY")
BULB_VERSION = float(_env("BULB_VERSION", "3.3"))


# ---------------------------------------------------------------------------
# Behavior flags
# ---------------------------------------------------------------------------
# Sleep-by-double-clap. OFF by default: on a laptop the mic sits right above
# the keyboard, so typing reads as claps. Turn on only with an external mic.
SLEEP_CLAP_ENABLED = _env_bool("JARVIS_SLEEP_CLAP", False)

DRY_RUN_SYSTEM_CTRL = _env_bool("JARVIS_DRY_RUN", False)


if __name__ == "__main__":
    # `python config.py` — quick sanity check of what a fresh clone resolves to.
    print("JARVIS configuration:")
    for name, val in sorted(vars().items()):
        if name.isupper():
            shown = "•••" if ("KEY" in name or "SECRET" in name) and val else val
            print(f"  {name:24} = {shown}")
