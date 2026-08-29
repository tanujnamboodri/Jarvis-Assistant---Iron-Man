"""
JARVIS — main.py  Orchestrator
================================
The single entry point. Wires all packets into a running assistant.

Usage
-----
  # Full production mode (Windows — needs Piper + sounddevice + Ollama)
  python main.py

  # Text / keyboard mode — no audio hardware, good for dev and testing
  python main.py --text

  # Keyword-only routing — no Ollama needed (fast, limited to registered skills)
  python main.py --text --mode keyword

  # Dry-run system control — actions logged, not executed
  python main.py --text --dry-run

  # Verbose logging
  python main.py --text --loglevel DEBUG

Packet integration map
----------------------
  voice.py          ← Packet 1 (BUILT): Piper TTS + clap wake
  stt.py            ← Packet A (STUB):  faster-whisper (keyboard fallback)
  brain.py          ← Packet B (BUILT): intent router + Ollama
  skills/basic_skills.py ← BUILT:       time, date, hello, help
  skills/system_ctrl← Packet C (BUILT): apps, volume, brightness, wi-fi, power
  skills/web_agent  ← Packet D (STUB):  search, weather, news
  skills/email_notify← Packet E (STUB): email, calendar
  memory.py         ← Packet F (BUILT): SQLite persistence
  vision.py         ← Packet G (STUB):  screenshot, OCR

To swap in a real packet: implement its register() or listen() function
with the same signature, drop the file in place, and restart. Nothing
else in this file needs to change.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the jarvis/ project root is on sys.path so all modules can use
# flat imports (from contracts import ..., from voice import ..., etc.)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Logging — set up before any other imports so module-level loggers work
# ---------------------------------------------------------------------------
try:
    import ui_server as _ui
except ImportError:
    class _ui:
        @staticmethod
        def start(**kw): pass
        @staticmethod
        def broadcast(state, text=""): pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)-20s %(levelname)s: %(message)s",
)
log = logging.getLogger("jarvis.main")


# ===========================================================================
# Skill registration — add new packets here, nothing else needs to change
# ===========================================================================

def _load_skill(path: str):
    """Load a skill module from an absolute file path."""
    import importlib.util
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    # CRITICAL: register BEFORE exec. Without this, any skill that defines a
    # @dataclass under `from __future__ import annotations` (schedule.py's
    # Task) crashes with "'NoneType' object has no attribute '__dict__'" —
    # dataclass processing resolves annotations via sys.modules[module name].
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_skill_file(filename: str) -> str | None:
    """Search for a skill file in several candidate locations."""
    candidates = [
        os.path.join(_HERE, "skills", filename),   # Jarvis/skills/
        os.path.join(_HERE, filename),             # Jarvis/ root
        os.path.join(os.getcwd(), "skills", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _register_all_skills(registry, dry_run: bool = False) -> None:
    """Register every skill by loading .py files directly from disk.
    Always prints what it finds so you can see skill loading status."""

    # Find papers folder — ONLY inside Jarvis 2026/, never outside
    _papers = None
    for _n in ["Paper", "papers", "Papers", "Research", "research"]:
        _c = os.path.join(_HERE, _n)
        if os.path.isdir(_c):
            _papers = _c
            break
    if _papers is None:
        _papers = os.path.join(_HERE, "papers")
        os.makedirs(_papers, exist_ok=True)
    print(f"  [Research] Papers folder → {_papers}")

    skills = [
        ("basic_skills",       "basic_skills.py",       {}),
        ("system_ctrl",        "system_ctrl.py",        {"dry_run": dry_run}),
        ("web_agent",          "web_agent.py",          {}),
        ("research_assistant", "research_assistant.py", {"papers_dir": _papers}),
        ("schedule",           "schedule.py",           {}),
    ]

    skills_dir = os.path.join(_HERE, "skills")
    print(f"  Loading skills from: {skills_dir}")

    for label, filename, kwargs in skills:
        path = _find_skill_file(filename)
        if path is None:
            print(f"  ⚠  {label}: file not found ({filename})")
            continue
        try:
            _load_skill(path).register(registry, **kwargs)
            print(f"  ✓  {label}")
        except Exception as exc:
            print(f"  ✗  {label}: {exc}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------ args
    parser = argparse.ArgumentParser(
        description="JARVIS — local voice assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --text                 keyboard mode, auto routing\n"
            "  python main.py --text --mode keyword  no LLM needed\n"
            "  python main.py --dry-run              safe mode, no system changes\n"
        ),
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Keyboard input / print output — no audio hardware required",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the browser UI (used by the desktop app window)",
    )
    parser.add_argument(
        "--model", default="llama3.2:3b",
        help="Ollama model for conversational replies (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--mode", default="auto", choices=["auto", "keyword", "llm"],
        help=(
            "auto     = keyword fast-path, then LLM for ambiguous queries  [default]\n"
            "keyword  = keyword matching only, no LLM (works without Ollama)\n"
            "llm      = always ask the LLM to classify intent\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="System-control actions are logged but not executed",
    )
    parser.add_argument(
        "--loglevel", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: WARNING)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.loglevel))

    # --------------------------------------------------------- import modules
    # Import AFTER args are parsed so voice.configure() is called first.
    import voice
    import stt
    from stt import SLEEP_SIGNAL as _SLEEP
    from contracts import SkillRegistry
    from brain import Brain

    voice.configure(text_mode=args.text)
    _ui.start(open_browser=(not args.text and not args.no_browser))

    # ------------------------------------------------------- register skills
    registry = SkillRegistry()
    _register_all_skills(registry, dry_run=args.dry_run)

    # ----------------------------------------------------------- wire brain
    brain = Brain(
        registry=registry,
        model=args.model,
        mode=args.mode,
        speak_fn=voice.speak,
    )

    # ---- Wire the research-panel text input to the brain ----------------
    def _handle_typed_question(q: str):
        """Called when the user types a question in the research panel."""
        try:
            voice.stop()
            _ui.broadcast("processing", q)
            result = brain.route(q)
            # brain.route already speaks via speak_fn; UI gets speaking broadcasts
            # Surface which papers RAG retrieved (written citation in the panel)
            try:
                import rag as _rag
                if _rag.last_sources:
                    _ui.broadcast("sources", " · ".join(_rag.last_sources[:3]))
            except Exception:
                pass
        except Exception as e:
            log.error("Typed question failed: %s", e)
            _ui.broadcast("idle")
    try:
        _ui.set_ask_callback(_handle_typed_question)
    except Exception:
        pass

    # -------------------------------------------------- warm up Ollama model
    # Send a silent 1-token request so the model is loaded into GPU/RAM
    # before the user's first real query. Runs in background — won't delay startup.
    brain.warm_up()

    # -------------------------------------------------------- startup banner
    sep = "=" * 56
    mode_tag  = "TEXT MODE" if args.text else "AUDIO MODE"
    wake_hint = "Press Enter" if args.text else "Clap twice"
    sleep_hint = "Type 'bye'" if args.text else "Say 'bye' or 'offline'"

    print(f"\n{sep}")
    print(f"  JARVIS  [{mode_tag} | routing={args.mode.upper()}]")
    print(f"  Skills : {', '.join(registry.names)}")
    print(f"  Brain  : {args.model}")
    print(f"  Wake   : {wake_hint}")
    print(f"  Sleep  : {sleep_hint}")
    print(f"{sep}\n")

    # =====================================================================
    # Main loop:  wait for wake → greet → command loop → sleep → repeat
    # =====================================================================
    try:
        while True:
            # ---- Wake -------------------------------------------------------
            try:
                if args.text:
                    voice.wait_for_wake(text_mode=True)      # Enter key
                else:
                    # Wake word "hey jarvis" (openWakeWord). The mic stream
                    # for wake detection is cheap and local; full STT only
                    # engages after detection. Falls back to the clap
                    # detector if openwakeword isn't installed.
                    try:
                        import wake
                        wake.wait_for_wake()
                    except ImportError:
                        log.warning("openwakeword not installed — "
                                    "falling back to clap wake. "
                                    "Fix: pip install openwakeword onnxruntime")
                        voice.wait_for_wake(text_mode=False)
            except EOFError:
                # Piped stdin exhausted (e.g. end of test script) — exit cleanly
                break

            # ---- Greet ------------------------------------------------------
            _ui.broadcast("listening")
            voice.greet(text_mode=args.text)
            voice.wait_until_done()  # ensure all greeting phrases finished
            time.sleep(0.2)          # was 0.7 — fixed dead air on every wake

            # ---- Command loop -----------------------------------------------
            # Runs until the user says one of the sleep trigger words.
            while True:
                _ui.broadcast("listening")
                query = stt.listen(text_mode=args.text)

                if query is None:
                    continue

                # Double clap during listening → go back to idle/sleep
                if query is _SLEEP or query == _SLEEP:
                    _ui.broadcast("idle")
                    voice.speak_wait("Going to sleep, sir.")
                    break

                q_lo = query.lower().strip()
                if not q_lo:
                    continue

                if any(w in q_lo for w in ("bye", "goodbye", "offline",
                                            "sleep", "exit", "stop listening")):
                    _ui.broadcast("idle")
                    voice.speak_wait("Going to sleep, sir.")
                    break

                # Interrupt any ongoing speech before handling the new query
                voice.stop()
                _ui.broadcast("processing", query)
                _result = brain.route(query)
                # Broadcast category tag so UI can style Intel Feed cards correctly
                if hasattr(_result, "data") and _result.data.get("feed"):
                    _ui.broadcast("category", _result.data["feed"])
                # If a paper was just analyzed, add its briefing to the
                # brain's conversation history so follow-up questions
                # ("how do I replicate?", "what else?") get paper context.
                # If a paper was analyzed, inject its briefing into the
                # brain's conversation history so follow-up questions have context.
                if (hasattr(_result, "data")
                        and _result.data.get("path")
                        and _result.data.get("title")
                        and _result.speak):
                    title = _result.data["title"]
                    with brain._lock:
                        brain._history.append({
                            "role": "user",
                            "content": f"Tell me about the paper: {title}",
                        })
                        brain._history.append({
                            "role": "assistant",
                            "content": _result.speak,
                        })
                voice.wait_until_done()
                time.sleep(0.2)   # was 0.7 — fixed dead air after every reply

    except KeyboardInterrupt:
        print()
        _ui.broadcast("offline")
        # Use speak() not speak_wait() — speak_wait() can hang on Ctrl+C
        # because _tts_queue.join() blocks inside a threading lock.
        voice.speak("Shutting down. Goodbye, sir.")
        time.sleep(2.5)   # give the TTS thread time to finish
        sys.exit(0)


if __name__ == "__main__":
    main()
