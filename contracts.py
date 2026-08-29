"""
contracts.py  — Shared Types & Skill Registry
==============================================
This IS Packet 0's contract definition, produced alongside Packet B
so Brain has something real to build against.

Every skill in the JARVIS system implements ONE interface:

    Handler = Callable[[str], IntentResult]

A skill receives the raw query and returns an IntentResult.
It never calls speak() itself — it just fills IntentResult.speak and
the Brain speaks it. This keeps skills testable in pure Python with no
audio device.

Usage:
    from contracts import IntentResult, SkillRegistry

    registry = SkillRegistry()

    def handle_time(query: str) -> IntentResult:
        import datetime
        t = datetime.datetime.now().strftime("%I:%M %p")
        return IntentResult(speak=f"The current time is {t}.")

    registry.register(
        name="time",
        keywords=["time", "clock", "hour", "what time"],
        handler=handle_time,
        description="Tells the current time of day",
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Core result type every handler returns
# ---------------------------------------------------------------------------
@dataclass
class IntentResult:
    """Returned by every skill handler and by Brain.route().

    speak   : text Jarvis should say out loud (None = stay silent)
    handled : False means "I don't own this query — try something else"
    data    : optional structured payload (e.g. for vision or follow-up tasks)
    """
    speak: Optional[str] = None
    handled: bool = True
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The one callable signature every skill must satisfy
# ---------------------------------------------------------------------------
Handler = Callable[[str], IntentResult]


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------
@dataclass
class SkillInfo:
    name: str
    keywords: list[str]
    handler: Handler
    description: str   # shown to the LLM for intent classification


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class SkillRegistry:
    """Holds all registered skills and offers two look-up strategies:
      match_keywords : O(n) substring scan — used as the v1 fast path
      list_skills    : returns metadata for LLM-based intent classification
    """

    def __init__(self) -> None:
        self._skills: list[SkillInfo] = []

    # -- registration --------------------------------------------------------
    def register(
        self,
        name: str,
        keywords: list[str],
        handler: Handler,
        description: str,
    ) -> None:
        """Register a skill. Later registrations with the same name replace
        earlier ones so skills can be swapped out without restarting."""
        for i, s in enumerate(self._skills):
            if s.name == name:
                self._skills[i] = SkillInfo(name, keywords, handler, description)
                return
        self._skills.append(SkillInfo(name, keywords, handler, description))

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it existed."""
        before = len(self._skills)
        self._skills = [s for s in self._skills if s.name != name]
        return len(self._skills) < before

    # -- look-up -------------------------------------------------------------
    def match_keywords(self, query: str) -> Optional[tuple[str, Handler]]:
        """Return (skill_name, handler) for the first skill whose any keyword
        appears in the lowercased query, or None if nothing matches.

        Matching rule: keyword must appear as a *word boundary* match so that
        "time" doesn't fire inside "sometime". Each keyword can be a phrase
        (e.g. "what time").
        """
        q = query.lower()
        for skill in self._skills:
            for kw in skill.keywords:
                # Use word-boundary regex for single words; plain substring for phrases.
                kw_lo = kw.lower()
                if " " in kw_lo:
                    if kw_lo in q:
                        return (skill.name, skill.handler)
                else:
                    if re.search(rf"\b{re.escape(kw_lo)}\b", q):
                        return (skill.name, skill.handler)
        return None

    def get(self, name: str) -> Optional[SkillInfo]:
        """Look up a skill by exact name."""
        for s in self._skills:
            if s.name == name:
                return s
        return None

    def list_skills(self) -> list[SkillInfo]:
        """Return all registered skills (for LLM intent prompt)."""
        return list(self._skills)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._skills]

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry({self.names})"
