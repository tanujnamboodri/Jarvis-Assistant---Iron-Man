"""
JARVIS — Schedule Skill
=======================
Reads a plain-markdown schedule file and reports the user's tasks for
TODAY / THIS MONTH / THIS YEAR, and (on request) arranges them by time or
priority. It also pushes the parsed schedule to the front-end so the Three.js
UI can open a SCHEDULE view with three columns.

Voice / text examples (routed here by the keyword router in main.py):
    "what's my schedule for today"
    "what's on this month"
    "show me my tasks this year"
    "arrange my schedule by priority"
    "sort today's tasks by time"

--------------------------------------------------------------------------
THE SCHEDULE FILE
--------------------------------------------------------------------------
Default location:  <project_root>/schedule.md
  (override with the JARVIS_SCHEDULE_FILE environment variable)

One task per line. The parser is deliberately tolerant; the canonical form is:

    - 2026-06-25 09:00 | Finish ROUGE-L significance test | priority: high
    - 2026-06-25       | Email supervisor about held-out set
    - 2026-07-10       | arXiv posting | priority: medium
    - [x] 2026-06-20   | Switch TTS voice to Daniel          (a completed task)
    - Re-run clean training pass                              (undated / "anytime")

Rules:
  * An ISO date YYYY-MM-DD makes a line "scheduled". Lines with no date go into
    an "anytime" bucket — shown under TODAY, never date-filtered.
  * An optional 24h time HH:MM may follow the date.
  * Text after the date/time (and an optional '|') is the description.
  * Optional trailing  priority: low|medium|high  sets the priority.
  * A leading '- [x]' (or the word "done" in the line) marks it complete.
  * Blank lines and lines starting with '#' (markdown headings) are ignored.

--------------------------------------------------------------------------
INTEGRATION  (the parts that touch files NOT in this module — verify them)
--------------------------------------------------------------------------
In main.py, after brain.init(...) and after the skill registry exists:

    import skills.schedule as schedule
    brain.register(
        "schedule",
        ["schedule", "agenda", "my tasks", "task list", "to-do", "to do",
         "todo", "what's on", "whats on", "what do i have", "my day"],
        schedule.handle_schedule,
        "Reads the user's schedule file and reports tasks for today/month/year",
    )
    # [VERIFY] ui_server.broadcast — replace with whatever function your
    # ui_server uses to push a JSON dict to all connected UI clients
    # (the same mechanism that already sends {"state":"view", ...}).
    schedule.set_broadcaster(ui_server.broadcast)

If you do NOT wire the broadcaster, the skill still works by voice — it just
won't open the on-screen board.

This module never rewrites schedule.md on its own. "Arrange" only changes the
ORDER shown/spoken. A safe, backup-first rewrite_sorted() is provided at the
bottom but is intentionally NOT wired to any voice command.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

# The real project provides this. The fallback shim only activates when the
# module is imported standalone (e.g. for a quick self-test) and is otherwise
# never used.
try:
    from contracts import IntentResult  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class IntentResult:  # minimal mirror of the project's contract
        speak: str = ""
        handled: bool = True

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------
_PRIORITY_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}


@dataclass
class Task:
    d: Optional[date]      # None == "anytime"
    time: str              # "HH:MM" or ""
    text: str
    priority: str          # "high" | "medium" | "low" | ""
    done: bool

    @property
    def anytime(self) -> bool:
        return self.d is None

    def _minutes(self) -> int:
        if not self.time:
            return 9999  # untimed tasks sort after timed ones within a day
        try:
            h, m = self.time.split(":")
            return int(h) * 60 + int(m)
        except ValueError:
            return 9999

    def to_dict(self) -> dict:
        return {
            "date": self.d.isoformat() if self.d else "",
            "time": self.time,
            "text": self.text,
            "priority": self.priority,
            "done": self.done,
            "anytime": self.anytime,
        }


# ---------------------------------------------------------------------------
# File location + parsing
# ---------------------------------------------------------------------------
def _schedule_path() -> Path:
    """Resolve the schedule file. Env var wins; else <project_root>/schedule.md;
    else schedule.md in the current working directory."""
    env = os.environ.get("JARVIS_SCHEDULE_FILE")
    if env:
        return Path(env).expanduser()
    # skills/ lives directly under the project root
    root = Path(__file__).resolve().parent.parent
    candidate = root / "schedule.md"
    if candidate.exists():
        return candidate
    return Path.cwd() / "schedule.md"


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PRIO_RE = re.compile(
    r"(?:priorit(?:y|ies)\s*[:=]?\s*|!)(high|medium|med|low)", re.I)
_DONE_RE = re.compile(r"\[[xX]\]|\bdone\b|\(done\)", re.I)


def _parse_line(line: str) -> Optional[Task]:
    raw = line.rstrip("\n")
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        return None

    done = bool(_DONE_RE.search(stripped))

    # Strip a leading bullet / checkbox so it isn't part of the description.
    body = re.sub(r"^\s*[-*]\s*", "", stripped)
    body = re.sub(r"^\[[ xX]\]\s*", "", body)

    # Date (optional). Remove it from the description text once captured.
    d: Optional[date] = None
    m = _DATE_RE.search(body)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            d = None
        if d:
            body = body[:m.start()] + body[m.end():]

    # Time (optional) — only meaningful when there is a date.
    t = ""
    if d:
        mt = _TIME_RE.search(body)
        if mt:
            t = f"{int(mt.group(1)):02d}:{mt.group(2)}"
            body = body[:mt.start()] + body[mt.end():]

    # Priority (optional).
    prio = ""
    mp = _PRIO_RE.search(body)
    if mp:
        p = mp.group(1).lower()
        prio = "medium" if p == "med" else p
        body = body[:mp.start()] + body[mp.end():]

    # Clean up the description: drop separators / 'done' marker / extra spaces.
    body = _DONE_RE.sub("", body)
    body = body.strip(" |\t-").strip()
    body = re.sub(r"\s{2,}", " ", body)
    if not body:
        return None

    return Task(d=d, time=t, text=body, priority=prio, done=done)


def _load_tasks() -> tuple[list[Task], Optional[str]]:
    """Return (tasks, error_message). error_message is a spoken-ready string
    when something went wrong, else None."""
    path = _schedule_path()
    if not path.exists():
        return [], (
            f"I couldn't find your schedule file, sir. "
            f"Create a file named schedule.md in the project folder and I'll read it."
        )
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.warning("Schedule read failed: %s", exc)
        return [], "I couldn't open your schedule file, sir."

    tasks = [tk for tk in (_parse_line(ln) for ln in text.splitlines()) if tk]
    return tasks, None


# ---------------------------------------------------------------------------
# Filtering / sorting / intent detection
# ---------------------------------------------------------------------------
def _bucket(tasks: list[Task], today: date) -> dict:
    out = {"today": [], "tomorrow": [], "week": [], "month": [], "year": [],
           "anytime": []}
    week_end = today + timedelta(days=6)
    for tk in tasks:
        if tk.d is None:
            out["anytime"].append(tk)
            continue
        if tk.d == today:
            out["today"].append(tk)
        if tk.d == today + timedelta(days=1):
            out["tomorrow"].append(tk)
        if today <= tk.d <= week_end:
            out["week"].append(tk)
        if tk.d.year == today.year and tk.d.month == today.month:
            out["month"].append(tk)
        if tk.d.year == today.year:
            out["year"].append(tk)
    return out


def _sort_tasks(tasks: list[Task], key: str) -> list[Task]:
    if key == "priority":
        return sorted(
            tasks,
            key=lambda t: (-_PRIORITY_RANK.get(t.priority, 0),
                           t.d or date.max, t._minutes()),
        )
    # default: chronological
    return sorted(tasks, key=lambda t: (t.d or date.max, t._minutes()))


def _detect_scope(q: str) -> str:
    if "year" in q:
        return "year"
    if "month" in q:
        return "month"
    if "week" in q or "next 7 days" in q or "seven days" in q:
        return "week"
    if "tomorrow" in q:
        return "tomorrow"
    # "today", "tonight", "my day" all mean today; this is also the default
    return "today"


def _detect_sort(q: str) -> str:
    if "priorit" in q or "importan" in q:
        return "priority"
    return "time"


def _arrange_requested(q: str) -> bool:
    return any(w in q for w in (
        "arrange", "sort", "organis", "organiz", "order by",
        "by priority", "by time", "by date", "by importance",
    ))


# ---------------------------------------------------------------------------
# Spoken summary
# ---------------------------------------------------------------------------
def _describe(t: Task, with_date: bool = False) -> str:
    bits = []
    if with_date and t.d:
        bits.append(f"{t.d.day} {t.d.strftime('%b')}")
    if t.time:
        bits.append(f"at {t.time}")
    prefix = " ".join(bits)
    text = t.text.rstrip(". ")
    return f"{prefix} {text}".strip() if prefix else text


def _next_upcoming(tasks: list[Task], today: date) -> Optional[Task]:
    future = [t for t in tasks if t.d and t.d >= today]
    pool = future or [t for t in tasks if t.d]
    if not pool:
        return None
    return min(pool, key=lambda t: (t.d, t._minutes()))


def _summarise(scope: str, buckets: dict, today: date, arranged: bool,
               sort_key: str) -> str:
    if scope == "today":
        items = buckets["today"] + buckets["anytime"]
        if not items:
            speak = "Your schedule is clear today, sir."
        else:
            n = len(items)
            shown = "; ".join(_describe(t) for t in items[:3])
            tail = f", and {n - 3} more" if n > 3 else ""
            plural = "s" if n != 1 else ""
            speak = (f"You have {n} item{plural} today, sir: {shown}{tail}. "
                     f"I've put the full board on screen for you.")
    elif scope == "tomorrow":
        items = buckets["tomorrow"]
        if not items:
            speak = "Nothing scheduled for tomorrow, sir."
        else:
            n = len(items)
            shown = "; ".join(_describe(t) for t in items[:3])
            tail = f", and {n - 3} more" if n > 3 else ""
            plural = "s" if n != 1 else ""
            speak = f"You have {n} item{plural} tomorrow, sir: {shown}{tail}."
    elif scope == "week":
        items = buckets["week"]
        if not items:
            speak = "Your week ahead is clear, sir."
        else:
            n = len(items)
            up = _next_upcoming(items, today)
            nxt = f" Next up is {_describe(up, with_date=True)}." if up else ""
            plural = "s" if n != 1 else ""
            speak = (f"You have {n} item{plural} in the next seven days, "
                     f"sir.{nxt} The full board is on screen.")
    elif scope == "month":
        items = buckets["month"]
        if not items:
            speak = f"Nothing on the calendar for {today.strftime('%B')}, sir."
        else:
            n = len(items)
            up = _next_upcoming(items, today)
            nxt = f" Next up is {_describe(up, with_date=True)}." if up else ""
            plural = "s" if n != 1 else ""
            speak = (f"You have {n} item{plural} this month, sir.{nxt} "
                     f"The full month is on screen.")
    else:  # year
        items = buckets["year"]
        if not items:
            speak = f"Nothing scheduled for {today.year}, sir."
        else:
            n = len(items)
            up = _next_upcoming(items, today)
            nxt = f" Next up is {_describe(up, with_date=True)}." if up else ""
            plural = "s" if n != 1 else ""
            speak = (f"You have {n} item{plural} this year, sir.{nxt} "
                     f"The full year is on screen.")

    if arranged:
        speak += (" I've ordered them by priority."
                  if sort_key == "priority" else " I've ordered them by time.")
    return speak


# ---------------------------------------------------------------------------
# UI bridge
# ---------------------------------------------------------------------------
_broadcaster: Optional[Callable] = None


def set_broadcaster(fn: Callable) -> None:
    """Wire the UI broadcast function. Signature must match ui_server.broadcast:
    fn(state: str, text: str = '') — the same channel already used for
    {"state":"view","text":"..."} messages. Called automatically by register()."""
    global _broadcaster
    _broadcaster = fn


def _push_to_ui(payload: dict) -> None:
    if _broadcaster is None:
        return
    try:
        import json as _json
        # ui_server.broadcast(state, text) — text is a JSON string here so the
        # JS side can JSON.parse(d.text) to get the structured schedule object.
        _broadcaster("schedule", _json.dumps(payload))
        _broadcaster("view", "schedule")
    except Exception as exc:  # never let a UI hiccup break the spoken reply
        log.debug("Schedule UI push failed: %s", exc)


# ---------------------------------------------------------------------------
# Natural-language ADD support
#   "add finish the benchmark to my schedule for Friday at 3pm"
#   "remind me to call the supervisor tomorrow"
#   "schedule a dentist appointment on the 15th, high priority"
#
# This is intentionally a best-effort parser, NOT a full NLP date engine. It
# handles the common phrasings below; anything it can't date becomes an
# "anytime" task. JARVIS always reads the result back so a mis-hear or a
# mis-parse is caught immediately, and the file is backed up (.bak) on write.
# ---------------------------------------------------------------------------
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6, "mon": 0, "tue": 1, "tues": 1, "wed": 2,
    "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}
MONTHS: dict = {}
for _i in range(1, 13):
    MONTHS[calendar.month_name[_i].lower()] = _i
    MONTHS[calendar.month_abbr[_i].lower()] = _i

# Longest-first so "june" is tried before "jun", "thursday" before "thu".
_MONTHS_RE = "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True))
_WEEKDAYS_RE = "|".join(sorted(map(re.escape, WEEKDAYS), key=len, reverse=True))


def _is_add_intent(q: str) -> bool:
    """True when the utterance is asking to ADD a task (vs read the schedule).
    The keyword router already gates this skill to schedule context, so this
    just distinguishes write-requests from read-requests."""
    s = re.sub(r"^\s*(hey\s+)?jarvis[\s,]*", "", q.strip(), flags=re.I)
    s = re.sub(r"^\s*(could|can|would|will)\s+you\s+(please\s+)?", "", s, flags=re.I)
    s = re.sub(r"^\s*please\s+", "", s, flags=re.I)
    if re.match(r"^(add|put|jot|log|note\s+down|new\s+task|create\s+a\s+task|"
                r"make\s+a\s+note|remind\s+me|schedule\s+(a|an|the|me)\b)",
                s, re.I):
        return True
    if re.search(r"\bremind\s+me\s+to\b", q, re.I):
        return True
    if re.search(r"\badd\b[^.]*\bto\s+(my|the)\s+"
                 r"(schedule|calendar|agenda|task|to-?do)", q, re.I):
        return True
    return False


# -- date / time resolution -------------------------------------------------
def _add_month(d: date) -> date:
    y = d.year + (1 if d.month == 12 else 0)
    m = 1 if d.month == 12 else d.month + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def _weekday_date(target: int, nxt: bool) -> date:
    today = date.today()
    days = (target - today.weekday()) % 7
    if days == 0 and nxt:        # "next monday" said on a Monday -> following week
        days = 7
    return today + timedelta(days=days)


def _md_date(month: int, day: int) -> Optional[date]:
    today = date.today()
    last = calendar.monthrange(today.year, month)[1]
    d = date(today.year, month, min(day, last))
    if d < today:                # date already passed -> next year
        last = calendar.monthrange(today.year + 1, month)[1]
        d = date(today.year + 1, month, min(day, last))
    return d


def _dom_date(day: int) -> date:
    today = date.today()
    last = calendar.monthrange(today.year, today.month)[1]
    d = date(today.year, today.month, min(day, last))
    if d < today:                # day-of-month already passed -> next month
        nm = _add_month(today)
        last = calendar.monthrange(nm.year, nm.month)[1]
        d = date(nm.year, nm.month, min(day, last))
    return d


def _extract_date(t: str) -> tuple[Optional[date], str]:
    """Find a date phrase in `t`; return (date_or_None, text_with_phrase_removed)."""
    def cut(m):
        return t[:m.start()] + t[m.end():]

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])), cut(m)
        except ValueError:
            pass
    m = re.search(r"\bin\s+(\d+)\s+(day|days|week|weeks)\b", t, re.I)
    if m:
        mult = 7 if "week" in m[2].lower() else 1
        return date.today() + timedelta(days=int(m[1]) * mult), cut(m)
    m = re.search(r"\bday\s+after\s+tomorrow\b", t, re.I)
    if m:
        return date.today() + timedelta(days=2), cut(m)
    m = re.search(r"\btomorrow\b", t, re.I)
    if m:
        return date.today() + timedelta(days=1), cut(m)
    m = re.search(r"\b(today|tonight|this\s+evening)\b", t, re.I)
    if m:
        return date.today(), cut(m)
    m = re.search(r"\bnext\s+week\b", t, re.I)
    if m:
        return date.today() + timedelta(days=(7 - date.today().weekday())), cut(m)
    m = re.search(r"\bnext\s+month\b", t, re.I)
    if m:
        return _add_month(date.today()), cut(m)
    m = re.search(r"\bnext\s+(" + _WEEKDAYS_RE + r")\b", t, re.I)
    if m:
        return _weekday_date(WEEKDAYS[m[1].lower()], nxt=True), cut(m)
    m = re.search(r"\b(?:this\s+|on\s+|coming\s+)?(" + _WEEKDAYS_RE + r")\b", t, re.I)
    if m:
        return _weekday_date(WEEKDAYS[m[1].lower()], nxt=False), cut(m)
    m = re.search(r"\b(" + _MONTHS_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", t, re.I)
    if m:
        return _md_date(MONTHS[m[1].lower()], int(m[2])), cut(m)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTHS_RE + r")\b",
                  t, re.I)
    if m:
        return _md_date(MONTHS[m[2].lower()], int(m[1])), cut(m)
    m = re.search(r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)\b", t, re.I)
    if m:
        return _dom_date(int(m[1])), cut(m)
    return None, t


def _extract_time(t: str) -> tuple[Optional[str], str]:
    def cut(m):
        return t[:m.start()] + t[m.end():]

    m = re.search(r"\bnoon\b", t, re.I)
    if m:
        return "12:00", cut(m)
    m = re.search(r"\bmidnight\b", t, re.I)
    if m:
        return "00:00", cut(m)
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*([ap])\.?m\.?\b", t, re.I)
    if m:
        h = int(m[1]) % 12 + (12 if m[3].lower() == "p" else 0)
        return f"{h:02d}:{m[2]}", cut(m)
    m = re.search(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", t, re.I)
    if m:
        h = int(m[1]) % 12 + (12 if m[2].lower() == "p" else 0)
        return f"{h:02d}:00", cut(m)
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m:
        return f"{int(m[1]):02d}:{m[2]}", cut(m)
    m = re.search(r"\bat\s+(\d{1,2})\b", t, re.I)
    if m and 0 <= int(m[1]) <= 23:
        return f"{int(m[1]):02d}:00", cut(m)
    return None, t


def _extract_priority(t: str) -> tuple[str, str]:
    for pat, val in [
        (r"\b(?:high\s+priority|urgent|asap|important)\b", "high"),
        (r"\b(?:low\s+priority)\b", "low"),
        (r"\b(?:medium|normal)\s+priority\b", "medium"),
    ]:
        m = re.search(pat, t, re.I)
        if m:
            return val, t[:m.start()] + t[m.end():]
    return "", t


def _clean_task_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(hey\s+)?jarvis[\s,]*", "", s, flags=re.I)
    s = re.sub(r"^\s*(could|can|would|will)\s+you\s+(please\s+)?", "", s, flags=re.I)
    s = re.sub(r"^\s*please\s+", "", s, flags=re.I)
    s = re.sub(r"^\s*(add|put|jot\s+down|log(?:\s+a\s+task)?|create\s+a\s+task(?:\s+to)?|"
               r"make\s+a\s+note(?:\s+to)?|note\s+down|new\s+task(?:\s+to)?|"
               r"remind\s+me\s+to|remind\s+me|schedule\s+(?:a|an|the|me)?)\s+",
               "", s, flags=re.I)
    # "add task X" / "add a task to X" → the verb regex above removed "add",
    # so peel the leftover "task"/"a task to" prefix here.
    s = re.sub(r"^\s*(a\s+|the\s+)?tasks?\s+(to\s+)?", "", s, flags=re.I)
    s = re.sub(r"\b(to|on|in|into)\s+(my|the)\s+"
               r"(schedule|calendar|task\s*list|to-?do\s*list|agenda)\b", "", s, flags=re.I)
    s = re.sub(r"\bfor\s+me\b", "", s, flags=re.I)
    # peel orphan prepositions/articles left at the edges by phrase removal
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"^\s*(on|at|by|for|to|the)\b\s*", "", s, flags=re.I)
        s = re.sub(r"\s*\b(on|at|by|for|to)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.-|")
    return s.strip()


def _parse_add_request(query: str) -> tuple[str, Optional[date], Optional[str], str]:
    prio, work = _extract_priority(query.strip())
    d, work = _extract_date(work)
    tm, work = _extract_time(work)
    text = _clean_task_text(work)
    return text, d, tm, prio


# -- writing ----------------------------------------------------------------
def _format_line(text: str, d: Optional[date], tm: Optional[str], prio: str) -> str:
    seg = "- "
    if d:
        when = d.isoformat()
        if tm:
            when += " " + tm
        seg += when + " | "
    seg += text
    if prio:
        seg += " | priority: " + prio
    return seg


def _append_task(text: str, d: Optional[date], tm: Optional[str],
                 prio: str) -> Optional[str]:
    path = _schedule_path()
    line = _format_line(text, d, tm, prio)
    try:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # back up before every write so a bad add is always recoverable
            path.with_suffix(path.suffix + ".bak").write_text(content, encoding="utf-8")
            if not content.endswith("\n"):
                content += "\n"
        else:
            content = "# Schedule\n\n"
        path.write_text(content + line + "\n", encoding="utf-8")
        log.info("Added task: %s", line)
        return line
    except OSError as exc:
        log.warning("Append failed: %s", exc)
        return None


# -- spoken confirmation ----------------------------------------------------
def _spoken_time(tm: str) -> str:
    if tm == "12:00":
        return "noon"
    if tm == "00:00":
        return "midnight"
    h, mn = int(tm[:2]), tm[3:]
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {ap}" if mn == "00" else f"{h12}:{mn} {ap}"


def _confirm_add(text: str, d: Optional[date], tm: Optional[str], prio: str) -> str:
    if d is None:
        when = " as an anytime task"
    elif d == date.today():
        when = " for today"
    elif d == date.today() + timedelta(days=1):
        when = " for tomorrow"
    else:
        when = f" for {d.strftime('%A')}, {d.day} {d.strftime('%B')}"
    if d is not None and tm:
        when += " at " + _spoken_time(tm)
    pr = f", marked {prio} priority" if prio else ""
    return f"Added to your schedule, sir: {text.rstrip('. ')}{when}{pr}."


_INVALID_TASK = {"this", "that", "it", "this one", "that one", ""}


def _handle_add(query: str) -> IntentResult:
    text, d, tm, prio = _parse_add_request(query)

    # Guard against writing junk: empty, anaphoric ("add this"), or a question
    # ("what should I add...") rather than an actual task.
    if (text.lower() in _INVALID_TASK or len(text) < 3
            or re.match(r"^(what|which|when|where|why|who|how|should|shall|"
                        r"do|does|did|can|could|is|are)\b", text, re.I)):
        return IntentResult(
            speak="What would you like me to add to your schedule, sir?",
            handled=True)

    line = _append_task(text, d, tm, prio)
    if line is None:
        return IntentResult(
            speak="I couldn't write to your schedule file, sir.", handled=True)

    # Refresh the on-screen board so the new task appears immediately.
    buckets, today, err = _load_sorted_buckets("time")
    if not err and buckets is not None:
        _push_to_ui(_board_payload(buckets, today, "time"))

    return IntentResult(speak=_confirm_add(text, d, tm, prio), handled=True)


# ---------------------------------------------------------------------------
# Board payload helpers (shared by read + add)
# ---------------------------------------------------------------------------
def _board_payload(buckets: dict, today: date, sort_key: str) -> dict:
    ui_today = [t.to_dict() for t in (buckets["today"] + buckets["anytime"])]
    return {
        "today": ui_today,
        "month": [t.to_dict() for t in buckets["month"]],
        "year": [t.to_dict() for t in buckets["year"]],
        "generated": today.strftime("%a %d %b %Y"),
        "sort": sort_key,
    }


def _load_sorted_buckets(sort_key: str):
    """Return (buckets, today, error). buckets is None on error."""
    tasks, err = _load_tasks()
    if err:
        return None, date.today(), err
    today = date.today()
    buckets = _bucket(tasks, today)
    for k in buckets:
        buckets[k] = _sort_tasks(buckets[k], sort_key)
    return buckets, today, None


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------
def handle_schedule(query: str) -> IntentResult:
    q = (query or "").lower()

    # WRITE path: "add X to my schedule", "remind me to X", ...
    if _is_add_intent(q):
        return _handle_add(query)

    # READ path
    sort_key = _detect_sort(q)
    buckets, today, err = _load_sorted_buckets(sort_key)
    if err:
        return IntentResult(speak=err, handled=True)

    _push_to_ui(_board_payload(buckets, today, sort_key))

    scope = _detect_scope(q)
    arranged = _arrange_requested(q)
    speak = _summarise(scope, buckets, today, arranged, sort_key)
    return IntentResult(speak=speak, handled=True)


# ---------------------------------------------------------------------------
# OPTIONAL, NOT WIRED: safe sorted rewrite of the file (creates a .bak first)
# ---------------------------------------------------------------------------
def rewrite_sorted(key: str = "time") -> Optional[Path]:
    """Rewrite schedule.md with tasks sorted by `key` ('time' or 'priority').
    Writes a <file>.bak backup first. Returns the path written, or None.
    NOT called by any voice command — invoke explicitly if you want it."""
    path = _schedule_path()
    tasks, err = _load_tasks()
    if err or not tasks:
        return None
    ordered = _sort_tasks(tasks, key)
    lines = ["# Schedule", ""]
    for t in ordered:
        chk = "[x] " if t.done else ""
        when = t.d.isoformat() if t.d else ""
        if t.time:
            when = f"{when} {t.time}".strip()
        prio = f" | priority: {t.priority}" if t.priority else ""
        lines.append(f"- {chk}{when} | {t.text}{prio}".replace("  ", " "))
    try:
        if path.exists():
            path.with_suffix(path.suffix + ".bak").write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except OSError as exc:
        log.warning("rewrite_sorted failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Registration — plugs into the Brain's SkillRegistry (same pattern as
# system_ctrl, web_agent, etc.). Called by main.py via _load_skill(path).register()
# ---------------------------------------------------------------------------
def register(registry, **kwargs) -> None:
    """Register the schedule skill.

    Automatically wires the ui_server broadcaster if ui_server is importable,
    so the on-screen board opens whenever JARVIS answers a schedule query.
    No extra kwargs required from main.py.
    """
    # Wire broadcaster — import ui_server the same way main.py does it so the
    # schedule board uses the same WebSocket channel as every other broadcast.
    try:
        import ui_server as _ui_server  # type: ignore
        set_broadcaster(_ui_server.broadcast)
        log.debug("schedule: ui_server broadcaster wired")
    except ImportError:
        log.debug("schedule: ui_server not available — board will be voice-only")

    registry.register(
        name="schedule",
        keywords=[
            # Read queries
            "schedule", "agenda", "my tasks", "task list",
            "to-do", "to do", "todo",
            "what's on", "whats on", "what do i have", "my day",
            # Add queries — capture "remind me" etc. before they reach _converse
            "remind me", "add a task", "new task",
            "note down", "jot down", "make a note",
        ],
        handler=handle_schedule,
        description=(
            "Reads and writes the user's schedule file. Reports tasks for today, "
            "this month, or this year. Adds new tasks by voice."
        ),
    )


# ---------------------------------------------------------------------------
# Self-test  (python3 skills/schedule.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    captured: list[dict] = []
    set_broadcaster(lambda msg: captured.append(msg))

    for probe in [
        "what's my schedule for today",
        "what's on this month",
        "show me my tasks this year",
        "arrange my schedule by priority",
    ]:
        print(f"\n>>> {probe}")
        r = handle_schedule(probe)
        print("SPEAK:", r.speak)

    print("\n--- last UI payload pushed ---")
    if captured:
        import json
        sched = [m for m in captured if m.get("state") == "schedule"][-1]["schedule"]
        print(json.dumps(sched, indent=2)[:1200])
