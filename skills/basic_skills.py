"""
skills/basic_skills.py — Time, date, greeting, and help
=========================================================
The smallest possible skill module, registered first so a fresh clone has
at least one working, demonstrable skill with zero configuration.

Deliberately named basic_skills.py, NOT builtins.py: a file loaded under the
name "builtins" via this project's importlib pattern (see main.py's
_load_skill) gets registered as sys.modules['builtins'] — which silently
replaces Python's REAL builtins module (containing print, len, isinstance,
etc.) for the rest of the process. Verified this actually happens; don't
rename this file back.
"""

from __future__ import annotations

import datetime

from contracts import IntentResult, SkillRegistry


def _handle_time(query: str) -> IntentResult:
    t = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
    return IntentResult(speak=f"It's {t}, sir.")


def _handle_date(query: str) -> IntentResult:
    d = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return IntentResult(speak=f"Today is {d}, sir.")


def _handle_hello(query: str) -> IntentResult:
    return IntentResult(speak="At your service, sir.")


def _handle_help(query: str) -> IntentResult:
    return IntentResult(speak=(
        "I can tell the time and date, control your schedule, search the "
        "web, check the weather, read the news, control basic system "
        "settings, and answer questions about your research papers, sir."
    ))


def register(registry: SkillRegistry, **kwargs) -> None:
    registry.register("time",  ["time", "clock"], _handle_time,
                      "Tells the current time of day")
    registry.register("date",  ["date", "today's date"], _handle_date,
                      "Tells today's date")
    registry.register("hello", ["hello", "hi jarvis", "hey jarvis"],
                      _handle_hello, "Basic greeting")
    registry.register("help",  ["help", "what can you do"], _handle_help,
                      "Lists what JARVIS can do")
