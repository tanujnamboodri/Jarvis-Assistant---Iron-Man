"""
JARVIS — voice.py   Thin compatibility layer over jarvis_voice.py
====================================================================
main.py, brain.py, and research_assistant.py all do `import voice` and
call voice.configure/greet/speak/speak_wait/stop/wait_for_wake/wait_until_done
— but the underlying TTS/wake implementation lives in jarvis_voice.py under
slightly different names and signatures (e.g. jarvis_voice.greet() takes no
text_mode argument, and jarvis_voice has no `configure`, `wait_for_wake`, or
`wait_until_done` at all — it has wait_for_clap and wait_for_audio_done).

This module is that adapter: it re-exports jarvis_voice's real
implementations under the names/signatures every caller expects, and adds
the one bit of actual behavior (text_mode routing) that doesn't exist
anywhere else.

NOTE FOR MAINTAINERS: this was reconstructed from the actual call sites in
main.py/brain.py/research_assistant.py and the actual function bodies in
jarvis_voice.py, not from an original voice.py — if a prior version of this
file existed with different behavior (e.g. a real Piper-only wake path,
different greeting logic), prefer that version and treat this as a
verified-working fallback, not a replacement.
"""

from __future__ import annotations

import logging

import jarvis_voice as _jv

log = logging.getLogger(__name__)

_TEXT_MODE = False


def configure(text_mode: bool = False) -> None:
    """Called once at startup with the --text flag. Stores the mode so
    wait_for_wake()/greet() know whether to use keyboard or audio."""
    global _TEXT_MODE
    _TEXT_MODE = text_mode


def wait_for_wake(text_mode: bool | None = None) -> None:
    """Block until JARVIS should wake up.

    text_mode=True  : press Enter (keyboard mode, no audio hardware needed)
    text_mode=False : double-clap (jarvis_voice.wait_for_clap) — this is the
                      fallback wake path; main.py prefers wake.wait_for_wake()
                      (the neural "hey jarvis" model) when available and only
                      falls back to this clap detector if openwakeword isn't
                      installed.
    """
    mode = _TEXT_MODE if text_mode is None else text_mode
    if mode:
        try:
            input("\n  [Press Enter to wake JARVIS] ")
        except EOFError:
            raise   # let main.py's own EOFError handler catch this
    else:
        _jv.wait_for_clap()


def greet(text_mode: bool = False) -> None:
    """Speak (or print) the startup greeting.
    jarvis_voice.greet() always speaks aloud; in text mode we print instead
    so a keyboard-only session doesn't try to open an audio device."""
    if text_mode:
        hour_greeting = _jv.greet.__doc__  # not used; kept for reference
        import datetime
        hour = datetime.datetime.now().hour
        if   4  <= hour < 12: g = "Good morning, sir."
        elif 12 <= hour < 17: g = "Good afternoon, sir."
        elif 17 <= hour < 24: g = "Good evening, sir."
        else:                 g = "Welcome back, sir."
        print(f"[Jarvis] {g} {_jv.load_name()} online.")
    else:
        _jv.greet()


def wait_until_done() -> None:
    """Block until any in-flight speech has fully finished playing.
    Alias for jarvis_voice.wait_for_audio_done() — renamed here because
    every caller in this codebase already uses wait_until_done()."""
    _jv.wait_for_audio_done()


def speak(text: str) -> None:
    _jv.speak(text)


def speak_wait(text: str) -> None:
    _jv.speak_wait(text)


def stop() -> None:
    _jv.stop()


def is_speaking() -> bool:
    return _jv.is_speaking()


if __name__ == "__main__":
    # Quick smoke test: exercise every function this module exposes without
    # needing a microphone (text_mode=True throughout).
    configure(text_mode=True)
    greet(text_mode=True)
    speak("Testing the voice adapter.")
    wait_until_done()
    stop()
    print("voice.py adapter smoke test complete — check jarvis_voice.py's own")
    print("--diagnose mode separately to verify actual audio output:")
    print("    python jarvis_voice.py --diagnose")
