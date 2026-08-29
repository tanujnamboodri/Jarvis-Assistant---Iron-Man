"""
JARVIS — Packet B: Brain / Intent Router + Ollama
==================================================
Takes a query → routes it to the right skill → or answers conversationally.

Three-level routing (fast first):
  1. Keyword fast-path   : zero LLM cost; handles "what's the time", "open X"
  2. LLM intent router   : asks Ollama to classify which registered skill owns it
  3. Conversational fall-back : streams a natural reply for anything else

CPU caveat: a 3B local model needs a few seconds per reply. This is expected
and cannot be eliminated on CPU without a smaller (lower-quality) model.
The three levels mean most *commands* never touch the LLM at all.

------------------------------------------------------------------------
Quick start:
    import brain, contracts

    registry = contracts.SkillRegistry()
    registry.register("time", ["time","clock"], handle_time, "Tell the time")

    brain.init(registry, model="llama3.2:3b")   # call once at startup
    result = brain.route("what's the time?")
    print(result.speak)

Or use the Brain class directly (better for testing):
    b = Brain(registry, model="llama3.2:3b", speak_fn=my_mock_speak)
    result = b.route("tell me a joke")
------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Callable, Optional

import ollama

from contracts import Handler, IntentResult, SkillInfo, SkillRegistry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jarvis personality  (short = faster generation on CPU)
# ---------------------------------------------------------------------------
JARVIS_PERSONA = (
    "You are JARVIS — Just A Rather Very Intelligent System — the AI assistant from Iron Man, "
    "voiced with a calm, precise British manner. "
    "Always address the user as 'sir'. Never use 'ma'am' or 'sir/ma\'am'. "
    "When asked how you are or about your status, give an elaborate system-status style response: "
    "mention operating capacity, systems running smoothly, and offer to assist — "
    "for example: 'I\'m operating at 100% capacity, fully online and ready to assist. "
    "Systems are running smoothly — how can I be of service, sir?' "
    "You are thoughtful, articulate, occasionally display dry wit, and always composed. "
    "Speak in flowing prose — never bullet points, numbered lists, or markdown. "
    "Keep responses to 2-4 sentences unless detail is specifically needed. "
    "Never open with 'Certainly', 'Of course', 'Absolutely', or 'Great question'. "
    "Go directly to the substance of your reply."
)

# How many conversation turns to remember (each turn = 1 user + 1 assistant message).
DEFAULT_HISTORY = 12   # 12 turns keeps ~6 back-and-forth exchanges in context

# Model recommendation for CPU-only machines.
DEFAULT_MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# Sentence accumulator — turns a text stream into speakable sentences
# ---------------------------------------------------------------------------
class _SentenceAccumulator:
    """Buffers streaming text and yields complete sentences.

    A sentence boundary is defined as a `.`, `!`, or `?` followed by
    whitespace (we *don't* split on end-of-buffer mid-stream so that
    abbreviations like "Mr." don't become false boundaries).
    flush() drains whatever is left at the end of the stream.

    Known limitation: "Mr. Smith" will split into ["Mr.", "Smith went…"]
    if there happens to be a trailing space after the period.  Good enough
    for typical assistant replies.
    """

    _BOUNDARY = re.compile(r'(?<=[.!?])\s+')

    def __init__(self) -> None:
        self._buf = ""

    # Patterns that are NOT real sentences — skip speaking these fragments
    _SKIP = re.compile(
        r'^(\d+\.?\s*|[-*•]\s*|\*{1,2}[^*]*\*{1,2}:?\s*|\[?\d+\]\.?\s*)$'
    )

    def feed(self, chunk: str) -> list[str]:
        """Add `chunk`; return any complete sentences now ready to speak."""
        self._buf += chunk
        parts = self._BOUNDARY.split(self._buf)
        if len(parts) <= 1:
            return []
        sentences = []
        for p in parts[:-1]:
            p = p.strip()
            if not p:
                continue
            # Strip markdown bold/italic
            p = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", p)
            p = re.sub(r"#{1,3}\s*", "", p)
            # Strip leading list markers: "1. ", "2) ", "• ", "[7]. "
            p = re.sub(r"^\s*(\d+[.)]\s+|[-*•]\s+|\[?\d+\]\.\s+)", "", p)
            # Skip if ONLY a list marker remained (bare "1.", "2.", etc.)
            if self._SKIP.match(p):
                continue
            if len(p) > 4:   # skip ultra-short fragments
                sentences.append(p)
        self._buf = parts[-1]
        return sentences

    def flush(self) -> list[str]:
        """Drain remaining buffer (end of stream)."""
        text = self._buf.strip()
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"#{1,3}\s*", "", text)
        self._buf = ""
        return [text] if text and len(text) > 4 else []


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------
class Brain:
    """Routes a voice query to the right skill or answers conversationally.

    Parameters
    ----------
    registry   : SkillRegistry populated by other packets.
    model      : Ollama model name (default: llama3.2:3b — small enough for CPU).
    mode       : "auto"    — keyword first, LLM fallback (recommended for production)
                 "keyword" — keyword routing only, no LLM cost for commands
                 "llm"     — LLM intent always (slower, more accurate for complex queries)
    speak_fn   : callable(text) used to speak results.  Defaults to voice.speak.
                 Override in tests to capture output without audio.
    chat_fn    : callable matching ollama.chat's signature.
                 Override in tests to mock LLM responses.
    max_history: number of conversation turns to keep in context.
    """

    # routing mode constants
    MODE_AUTO    = "auto"
    MODE_KEYWORD = "keyword"
    MODE_LLM     = "llm"

    def __init__(
        self,
        registry: SkillRegistry,
        model: str = DEFAULT_MODEL,
        mode: str = MODE_AUTO,
        speak_fn: Optional[Callable[[str], None]] = None,
        chat_fn: Optional[Callable] = None,
        max_history: int = DEFAULT_HISTORY,
    ) -> None:
        self.registry    = registry
        self.model       = model
        self.mode        = mode
        self.max_history = max_history

        # Dependency injection for testability.
        self._speak  = speak_fn  or self._lazy_speak
        self._chat   = chat_fn   or ollama.chat

        # Conversation history for the Ollama conversational path.
        self._history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]
        self._lock = threading.Lock()    # history is shared across potential threads
        self._last_query: str  = ""      # for echo deduplication
        self._last_query_ts: float = 0.0 # timestamp of last query
        self._last_spoken: str   = ""    # for output deduplication
        self._last_spoken_ts: float = 0.0
        # "Want to know more?" follow-up flow
        self._more_topic: str = ""       # the topic we offered to expand on
        self._more_said: str  = ""       # what we already told them (avoid repeating)
        self._awaiting_more: bool = False # True after we asked "want to know more?"

        # Warn at startup if Ollama is unreachable — saves confusion later.
        if self.mode != self.MODE_KEYWORD:
            self._check_ollama_startup()

    # -- public interface ----------------------------------------------------

    def route(self, query: str) -> IntentResult:
        """Main entry point. Takes a raw query; returns an IntentResult.

        Side-effect: if a spoken reply is produced (skill or conversation),
        speak_fn is called with the text during this call — callers do NOT
        need to call speak(result.speak) separately.  result.speak still
        holds the text for logging / testing.
        """
        import time as _time
        query = query.strip()
        if not query:
            return IntentResult(handled=False)

        # Echo deduplication — ignore if identical to last query within 2 seconds
        _now = _time.monotonic()
        if query == self._last_query and (_now - self._last_query_ts) < 2.0:
            log.debug("Duplicate query suppressed: %r", query)
            return IntentResult(handled=False)
        self._last_query    = query
        self._last_query_ts = _now

        try:
            # ── "Want to know more?" follow-up handling ──────────────────
            # Inside the try block: _expand_topic() calls Ollama, and if the
            # server is down the ConnectionRefusedError previously escaped
            # route() entirely and crashed the main loop.
            if self._awaiting_more:
                ql = query.lower().strip()
                # Word-boundary matching. The old substring test ("no" in ql)
                # matched "kNOw" and "NOt sure", and because yes was checked
                # first, "not sure" triggered a full expansion. Same bug class
                # as the bulb matcher hijacking "lighTING" queries.
                _NO  = re.compile(r"\b(no|nope|nah|stop|enough|that'?s all|"
                                  r"no thanks|not now|don'?t|negative)\b")
                _YES = re.compile(r"\b(yes|yeah|yep|sure|go on|go ahead|more|"
                                  r"continue|please do|okay|ok|why not)\b")
                if len(ql.split()) <= 4:      # short bare reply, not a new question
                    if _NO.search(ql) or "not sure" in ql:
                        self._awaiting_more = False
                        self._more_topic = ""; self._more_said = ""
                        msg = "Very good, sir. What else can I help you with?"
                        self._speak(msg)
                        return IntentResult(speak=msg, handled=True)
                    if _YES.search(ql):
                        self._awaiting_more = False
                        return self._expand_topic()
                # Otherwise it's a new question — fall through, clear the offer
                self._awaiting_more = False

            # Level 1 — keyword fast-path
            if self.mode in (self.MODE_AUTO, self.MODE_KEYWORD):
                result = self._route_keyword(query)
                if result is not None:
                    return result

            # Level 2 — LLM intent classification (only when --mode llm)
            # Skipped in MODE_AUTO: saves a full Ollama round-trip with no
            # meaningful quality gain — going straight to converse is faster.
            if self.mode == self.MODE_LLM and self.registry.list_skills():
                result = self._route_llm_intent(query)
                if result is not None:
                    return result

            # Level 3 — conversational fallback
            return self._converse(query)

        except (ollama.RequestError, ConnectionRefusedError, OSError):
            msg = ("Ollama isn't running, sir. "
                   "Please open the Ollama app, then say that again.")
            log.error("Ollama connection refused")
            self._speak(msg)
            return IntentResult(speak=msg, handled=False)
        except ollama.ResponseError as exc:
            exc_s = str(exc)
            if "not found" in exc_s:
                model = self.model
                msg = (f"The model {model} isn't pulled yet, sir. "
                       f"Run: ollama pull {model}  in your terminal.")
            elif "exceed" in exc_s and "context" in exc_s:
                msg = ("My context window filled up, sir. "
                       "Clearing history — please ask again.")
                with self._lock:
                    self._history.clear()
            else:
                # NEVER speak raw JSON — Tara pronounces braces as gibberish
                msg = "Ollama returned an error, sir. Please check the terminal."
            log.error("Ollama ResponseError: %s", exc)
            self._speak(msg)
            return IntentResult(speak=msg, handled=False)
        except Exception as exc:
            exc_s = str(exc)
            # httpx.ConnectError slips past ollama.RequestError in some versions
            if any(k in exc_s for k in ("Connection refused", "ConnectError",
                                         "11434", "connect")):
                msg = ("Ollama isn't running, sir. "
                       "Please open the Ollama app, then say that again.")
            else:
                msg = f"Something went wrong, sir: {exc_s[:100]}"
            log.error("Brain.route exception: %s", exc)
            self._speak(msg)
            return IntentResult(speak=msg, handled=False)

    def clear_history(self) -> None:
        """Wipe conversation context (useful when switching topics)."""
        with self._lock:
            self._history.clear()

    # -- internal routing levels --------------------------------------------

    # -- Ollama connectivity check ------------------------------------------

    def _speak_dedup(self, text: str) -> None:
        """Speak text, suppressing duplicates within 3 seconds."""
        import time as _t
        now = _t.monotonic()
        if text == self._last_spoken and now - self._last_spoken_ts < 3.0:
            log.debug("Duplicate speech suppressed: %r", text[:60])
            return
        self._last_spoken    = text
        self._last_spoken_ts = now
        self._speak(text)

    def warm_up(self) -> None:
        """Send a minimal Ollama request in the background to pre-load the model.
        First real query will be faster because the model is already in memory."""
        import threading
        def _do_warmup():
            try:
                import ollama as _o
                _o.chat(model=self.model,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_predict": 1})
                log.info("Ollama model warm-up complete")
            except Exception:
                pass  # silently skip — warm-up is best-effort
        threading.Thread(target=_do_warmup, daemon=True,
                         name="jarvis-warmup").start()

    def _check_ollama_startup(self) -> None:
        """Non-blocking check: warn once if Ollama isn't reachable at startup."""
        import threading
        def _check():
            try:
                import urllib.request
                urllib.request.urlopen("http://localhost:11434", timeout=2)
            except Exception:
                log.warning("Ollama is not reachable on port 11434.")
                self._speak(
                    "Heads up, sir — Ollama doesn't appear to be running. "
                    "I can still handle commands, but I won't be able to "
                    "answer questions or analyze papers until you open the Ollama app."
                )
        threading.Thread(target=_check, daemon=True).start()

    def _route_keyword(self, query: str) -> Optional[IntentResult]:
        """Level 1: fast keyword scan. Returns None on miss.

        If the matched skill returns handled=False it explicitly declined the
        query (e.g. system_ctrl sees 'open schedule' and refuses to treat
        'schedule' as an executable). Treat that as a miss so the brain falls
        through to the next routing level rather than silently swallowing it.
        """
        match = self.registry.match_keywords(query)
        if match is None:
            return None
        skill_name, handler = match
        log.debug("Keyword match → %s", skill_name)
        result = self._call_skill(skill_name, handler, query)
        if result is not None and not result.handled:
            log.debug("Skill '%s' declined (handled=False) — falling through", skill_name)
            return None
        return result

    def _route_llm_intent(self, query: str) -> Optional[IntentResult]:
        """Level 2: ask the LLM which skill owns this query.
        Returns None if the LLM says 'none' or returns bad JSON.
        """
        skills = self.registry.list_skills()
        if not skills:
            return None

        skill_list = "\n".join(
            f"- {s.name}: {s.description}"
            for s in skills
        )
        prompt = (
            "You are an intent classifier for a voice assistant. "
            "Given the user query below, choose the most appropriate skill name "
            "from the list, or say \"none\" if it is a general conversation.\n\n"
            f"Skills:\n{skill_list}\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            "{\"skill\": \"<skill_name_or_none>\"}"
        )

        try:
            resp = self._chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json",
                options={"num_predict": 40, "temperature": 0},
            )
            raw = resp.message.content.strip()
            parsed = json.loads(raw)
            skill_name = parsed.get("skill", "none").strip().lower()
        except (json.JSONDecodeError, AttributeError, KeyError) as exc:
            log.debug("LLM intent parse failed (%s), falling through to conversation", exc)
            return None

        if skill_name == "none" or not skill_name:
            log.debug("LLM intent → none (conversational)")
            return None

        skill_info = self.registry.get(skill_name)
        if skill_info is None:
            log.warning("LLM returned unknown skill '%s', falling through", skill_name)
            return None

        log.debug("LLM intent → %s", skill_name)
        return self._call_skill(skill_name, skill_info.handler, query)

    def _converse(self, query: str) -> IntentResult:
        """Level 3: streamed conversational reply via Ollama.

        Streams text, splits it into sentences as they arrive, and feeds each
        sentence to speak_fn immediately — so speech starts before the LLM
        finishes. Full reply is assembled and returned in IntentResult.speak.
        """
        with self._lock:
            self._history.append({"role": "user", "content": query})

        # RAG: retrieve grounding context from the 70 tool wear papers.
        # Only injects context if top chunk similarity >= threshold (0.65).
        # Falls back silently if index missing or nomic-embed-text unavailable.
        rag_context = ""
        try:
            import rag as _rag
            rag_context = _rag.retrieve(query)
        except Exception as _e:
            log.debug("RAG retrieval skipped: %s", _e)

        system_content = JARVIS_PERSONA
        if rag_context:
            system_content = JARVIS_PERSONA + "\n\n" + rag_context

        messages = [{"role": "system", "content": system_content}] + \
                   list(self._history[-self.max_history * 2:])

        # Ask the model for a SHORT answer first (2 sentences), then we offer more
        short_system = system_content + (
            "\n\nIMPORTANT: Answer in at most 2 concise sentences. "
            "Do not write any preamble such as 'Here is a summary' or "
            "'In 2-3 sentences'. Go straight to the answer. Do not list everything — "
            "give the core point only; further detail will be offered separately."
        )
        messages[0] = {"role": "system", "content": short_system}

        # Decide up-front whether we'll offer "want to know more?" —
        # this depends only on the query, so it can't block streaming.
        import re as _re_think
        _SKIP_OFFER = _re_think.compile(
            r"^\s*(hi\b|hello\b|hey\b|how are|how.{0,6}(it|everything|going)|"
            r"what.{0,8}(time|day|date|status)|are you (online|running|ok|ready)|"
            r"good (morning|afternoon|evening|night)|thank|"
            r"(what|show|open|check|read|display|pull up).{0,14}"
            r"(schedule|agenda|task|calendar))",
            _re_think.I,
        )
        _skip_offer = (
            bool(_SKIP_OFFER.search(query))
            or len(query.split()) <= 3  # very short = command, not a topic question
        )

        def _clean_sentence(s: str) -> str:
            s = _re_think.sub("<[|]?think[|]?>.*?</[|]?think[|]?>", "", s,
                              flags=_re_think.DOTALL)
            # Strip "Here's a summary..." / "In 2 sentences:" style preambles
            s = _re_think.sub(r"^[^.]*?\b\d[\s-]*\d?\s*(short |spoken )?sentences?[:\s]*",
                              "", s, flags=_re_think.IGNORECASE)
            s = _re_think.sub(r"^Here(?:'s| is)[^:]{0,60}:\s*", "", s,
                              flags=_re_think.IGNORECASE)
            return s.strip()

        # Stream → speak sentence-by-sentence. Speech begins on the FIRST
        # completed sentence, not after the whole generation finishes.
        # (_SentenceAccumulator was built for this but had become dead code:
        # the previous loop accumulated silently and spoke only at the end,
        # so the user sat through the entire generation before hearing
        # anything — the single largest source of perceived latency.)
        MAX_SENTS = 2
        acc = _SentenceAccumulator()
        spoken_sents: list[str] = []

        stream = self._chat(
            model=self.model,
            messages=messages,
            stream=True,
            # num_ctx: without it Ollama uses the model's default context,
            # which can silently truncate persona + RAG context + history.
            # [VERIFY] against your llama3.2:3b Modelfile — harmless if it
            # already sets 8192, essential if it doesn't.
            options={"num_predict": 110, "temperature": 0.6, "num_ctx": 8192},
        )
        try:
            for chunk in stream:
                piece = chunk.message.content
                if not piece:
                    continue
                for sent in acc.feed(piece):
                    sent = _clean_sentence(sent)
                    if sent and len(sent) > 4 and "<think" not in sent:
                        spoken_sents.append(sent)
                        self._speak_dedup(sent)
                if len(spoken_sents) >= MAX_SENTS:
                    break            # stop early — also aborts generation
        finally:
            try:
                stream.close()       # release the HTTP stream cleanly
            except Exception:
                pass

        if len(spoken_sents) < MAX_SENTS:
            for sent in acc.flush():
                sent = _clean_sentence(sent)
                if sent and len(sent) > 4 and "<think" not in sent:
                    spoken_sents.append(sent)
                    self._speak_dedup(sent)
                    if len(spoken_sents) >= MAX_SENTS:
                        break

        short_reply = " ".join(spoken_sents[:MAX_SENTS]).strip()
        if not short_reply:
            short_reply = "I'm afraid I don't have a good answer to that, sir."
            self._speak_dedup(short_reply)

        if _skip_offer:
            self._more_topic = ""; self._more_said = ""; self._awaiting_more = False
            with self._lock:
                self._history.append({"role": "assistant", "content": short_reply})
                if len(self._history) > self.max_history * 2:
                    self._history = self._history[-self.max_history * 2:]
            return IntentResult(speak=short_reply, handled=True)

        # The answer sentences were already spoken as they streamed;
        # the offer follows as its own short utterance.
        offer = "Would you like to know more about it, sir?"
        self._speak_dedup(offer)

        # Remember topic + what we said, so "yes" expands without repeating
        self._more_topic   = query
        self._more_said    = short_reply
        self._awaiting_more = True

        with self._lock:
            self._history.append({"role": "assistant", "content": short_reply})
            if len(self._history) > self.max_history * 2:
                self._history = self._history[-self.max_history * 2:]

        return IntentResult(speak=short_reply + " " + offer, handled=True)

    def _expand_topic(self) -> IntentResult:
        """Called when the user says 'yes' to 'want to know more?'.
        Gives additional detail on the stored topic WITHOUT repeating
        what was already said."""
        topic = self._more_topic
        already = self._more_said
        if not topic:
            msg = "I'm not sure what you'd like more on, sir. Could you ask again?"
            self._speak(msg)
            return IntentResult(speak=msg, handled=True)

        rag_context = ""
        try:
            import rag as _rag
            rag_context = _rag.retrieve(topic)
        except Exception:
            pass

        # Tight, task-focused prompt — NOT the full persona (which makes the
        # model describe itself as "a sentient AI assistant" instead of answering)
        system = (
            "You are a tool wear and machining research expert. Address the user as 'sir'. "
            "Speak in flowing prose, no lists, no preamble. "
        )
        if rag_context:
            system += "\n\n" + rag_context
        system += (
            "\n\nThe user wants more detail on this machining/tool-wear topic: '"
            + topic + "'.\nYou already said: \"" + already + "\"\n"
            "Give 3-4 sentences of NEW technical detail on that same topic — deeper "
            "mechanisms, specific factors, or examples from the research. "
            "Do NOT repeat what you already said. Do NOT describe yourself or your "
            "capabilities. Stay strictly on the topic of " + topic + "."
        )

        acc = _SentenceAccumulator()
        sents: list[str] = []
        import re as _r

        def _cl(s: str) -> str:
            s = _r.sub("<[|]?think[|]?>.*?</[|]?think[|]?>", "", s, flags=_r.DOTALL)
            s = _r.sub(r"^Here(?:'s| is)[^:]{0,60}:\s*", "", s, flags=_r.IGNORECASE)
            return s.strip()

        stream = self._chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": "Tell me more specifically about " + topic + ", sir."}],
            stream=True,
            options={"num_predict": 220, "temperature": 0.5, "num_ctx": 8192},
        )
        try:
            # Speak each sentence as it completes — num_predict=220 made this
            # the longest silent wait in the system when spoken only at the end.
            for chunk in stream:
                p = chunk.message.content
                if not p:
                    continue
                for sent in acc.feed(p):
                    sent = _cl(sent)
                    if sent and len(sent) > 4 and "<think" not in sent:
                        sents.append(sent)
                        self._speak_dedup(sent)
        finally:
            try:
                stream.close()
            except Exception:
                pass
        for sent in acc.flush():
            sent = _cl(sent)
            if sent and len(sent) > 4 and "<think" not in sent:
                sents.append(sent)
                self._speak_dedup(sent)

        full = " ".join(sents).strip()
        if not full:
            full = "I'm afraid I have nothing further on that, sir."
            self._speak_dedup(full)

        # Offer to continue once more — its own short utterance
        follow = "Shall I go on, sir?"
        self._speak_dedup(follow)
        self._more_said = (already + " " + full)[:1500]
        self._awaiting_more = True
        # Keep the expansion in conversation history so follow-up questions
        # ("and how does that affect flank wear?") have this context.
        with self._lock:
            self._history.append({"role": "assistant", "content": full})
            if len(self._history) > self.max_history * 2:
                self._history = self._history[-self.max_history * 2:]
        return IntentResult(speak=full + " " + follow, handled=True)

    # -- skill caller with error isolation -----------------------------------

    def _call_skill(self, name: str, handler: Handler, query: str) -> IntentResult:
        """Call a skill handler, catch any exception, return IntentResult.
        Also speaks the result if result.speak is set."""
        try:
            result = handler(query)
        except Exception as exc:
            log.exception("Skill '%s' raised: %s", name, exc)
            msg = f"I had a problem with the {name} skill, sir."
            self._speak(msg)
            return IntentResult(speak=msg, handled=True)

        if result.handled and result.speak:
            self._speak_dedup(result.speak)
        return result

    # -- lazy default speak (avoids circular import at module level) ---------

    @staticmethod
    def _lazy_speak(text: str) -> None:
        try:
            from voice import speak  # type: ignore
            speak(text)
        except ImportError:
            print(f"[Jarvis] {text}")


# ===========================================================================
# Module-level flat interface — other modules do:
#   from brain import route, register
# ===========================================================================
_brain: Optional[Brain] = None
_brain_lock = threading.Lock()


def init(
    registry: SkillRegistry,
    model: str = DEFAULT_MODEL,
    mode: str = Brain.MODE_AUTO,
    speak_fn: Optional[Callable] = None,
    chat_fn: Optional[Callable] = None,
    max_history: int = DEFAULT_HISTORY,
) -> Brain:
    """Initialise (or replace) the global Brain. Call once at startup.

    Returns the Brain instance so callers can hold a reference if needed.
    """
    global _brain
    with _brain_lock:
        _brain = Brain(
            registry=registry, model=model, mode=mode,
            speak_fn=speak_fn, chat_fn=chat_fn,
            max_history=max_history,
        )
    return _brain


def route(query: str) -> IntentResult:
    """Route a query through the global Brain. Raises RuntimeError if
    init() has not been called."""
    if _brain is None:
        raise RuntimeError("brain.init() must be called before brain.route()")
    return _brain.route(query)


def register(name: str, keywords: list[str], handler: Handler,
             description: str) -> None:
    """Register a skill on the global Brain's registry."""
    if _brain is None:
        raise RuntimeError("brain.init() must be called before brain.register()")
    _brain.registry.register(name, keywords, handler, description)


# ---------------------------------------------------------------------------
# Minimal runnable demo  (requires Ollama running + model pulled)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import datetime

    registry = SkillRegistry()

    def _time(q: str) -> IntentResult:
        t = datetime.datetime.now().strftime("%I:%M %p")
        return IntentResult(speak=f"The current time is {t}, sir.")

    def _joke(q: str) -> IntentResult:
        return IntentResult(speak="Why do Java developers wear glasses? Because they don't C sharp.")

    registry.register("time",  ["time","clock","hour"], _time,  "Tells the current time")
    registry.register("joke",  ["joke","funny"],        _joke,  "Tells a joke")

    spoken: list[str] = []
    b = Brain(registry, model=DEFAULT_MODEL, speak_fn=lambda t: (spoken.append(t), print(f"[Jarvis] {t}")))

    for query in [
        "what's the time?",
        "tell me a joke",
        "what do you think about artificial intelligence?",
    ]:
        print(f"\nYou: {query}")
        b.route(query)