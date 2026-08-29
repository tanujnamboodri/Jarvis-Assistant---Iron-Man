"""
JARVIS — skills/research_assistant.py   Research Assistant
===========================================================
Analyzes academic papers (PDFs) and answers questions about them.

Workflow:
  1. Drop one or more PDFs into the  jarvis/papers/  folder.
  2. Say one of:
       "Jarvis, analyze the paper"
       "Jarvis, summarize the latest paper"
       "Jarvis, what methodology did they use?"
       "Jarvis, what were the main findings?"
       "Jarvis, load the Chen 2024 paper"
       "Jarvis, compare the two papers"
       "Jarvis, what papers do I have?"
  3. JARVIS speaks a structured briefing — problem, approach,
     findings, significance — using Ollama locally (no cloud, no key).

CPU strategy:
  - Only the most informative sections are sent to Ollama, not the
    full paper. This keeps inference time to ~5–15 s on CPU.
  - Abstract + Introduction + Conclusion are extracted first because
    they contain 90% of the analytical content.
  - num_ctx=4096 lets Ollama see ~3000 tokens of paper text — enough
    for a complete understanding of most papers.
  - num_predict is capped to keep spoken responses concise.

Install:
  pip install pdfplumber pypdf
  (pypdf is the fallback if pdfplumber fails on a specific PDF)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)


def _strip_llm_preamble(text: str) -> str:
    """Deterministically remove meta-preambles a small model adds despite
    prompt instructions — e.g. 'Here's a summary of the text in 2-3 short
    spoken sentences: "..."'. The prompts already forbid this; a 3B model
    ignores that regularly, so enforcement has to happen here."""
    t = text.strip()
    t = re.sub(r"^(sure|certainly|of course)[,.!]?\s*", "", t, flags=re.I)
    t = re.sub(r"^here(?:'s| is)\s+(?:a|the)?\s*"
               r"(?:summary|answer|response|breakdown|analysis|overview)"
               r"[^:.\"]{0,80}[:.]\s*", "", t, flags=re.I)
    t = re.sub(r"^based on (?:this|the) paper[,:]?\s*", "", t, flags=re.I)
    t = re.sub(r"^in\s+\d+(?:\s*-\s*\d+)?\s+(?:short\s+)?(?:spoken\s+)?"
               r"sentences?[:,]?\s*", "", t, flags=re.I)
    t = t.strip()
    if len(t) > 2 and t[0] in "\"“" and t[-1] in "\"”":
        t = t[1:-1].strip()
    elif len(t) > 1 and t[0] in "\"“" and t.count('"') == 1:
        t = t[1:].strip()
    return t

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# IntentResult — import from contracts or inline fallback
# ---------------------------------------------------------------------------
try:
    from contracts import IntentResult          # type: ignore
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class IntentResult:                         # type: ignore
        speak:   Optional[str] = None
        handled: bool = True
        data:    dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# PDF reading libraries — graceful fallback chain
# ---------------------------------------------------------------------------
try:
    import fitz as _fitz          # PyMuPDF — best extractor, try first
    HAS_PYMUPDF = True
except ImportError:
    _fitz = None
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    pdfplumber = None
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    PdfReader = None
    HAS_PYPDF = False

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    _ollama = None
    HAS_OLLAMA = False

# ---------------------------------------------------------------------------
# Section-header patterns for academic papers
# Covers most common journal/conference paper structures.
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = {
    "abstract":     re.compile(r"\bAbstract\b", re.IGNORECASE),
    "introduction": re.compile(r"\b(?:1\.?\s+)?Introduction\b", re.IGNORECASE),
    "methodology":  re.compile(
        r"\b(?:\d+\.?\s+)?(?:Method(?:ology|s)|Experimental|Materials?\s+and\s+Methods?)\b",
        re.IGNORECASE,
    ),
    "results":      re.compile(
        r"\b(?:\d+\.?\s+)?(?:Results?(?:\s+and\s+Discussion)?|Findings?)\b",
        re.IGNORECASE,
    ),
    "discussion":   re.compile(r"\b(?:\d+\.?\s+)?Discussion\b", re.IGNORECASE),
    "conclusion":   re.compile(r"\b(?:\d+\.?\s+)?Conclusion(?:s)?\b", re.IGNORECASE),
}

# How many tokens (rough: chars/4) to send to Ollama per section
_SEC_CHAR_LIMITS = {
    "abstract":     2_000,   # usually short, give it full space
    "introduction": 2_000,   # contains the core problem statement
    "methodology":  1_500,   # key for replication questions
    "results":      1_500,   # key for findings questions
    "conclusion":   1_500,   # key for gaps and significance
}
_TOTAL_CONTEXT_CHARS = 8_000   # ~2000 tokens — 2x more content, better answers
                                # Safe: llama3.2:3b num_ctx=4096, prompt ≈ 200t


class ResearchAssistant:
    """Holds state for the currently loaded paper and handles all research intents.

    Parameters
    ----------
    papers_dir  : folder to watch for PDFs (default: jarvis/papers/)
    ollama_model: Ollama model for analysis and Q&A
    """

    def __init__(
        self,
        papers_dir: Optional[str] = None,
        ollama_model: str = "llama3.2:3b",
    ) -> None:
        # Find the papers folder — searches common names so
        # "Paper", "Papers", "papers", "Research" all work.
        if papers_dir:
            self.papers_dir = papers_dir
        else:
            # Search in sensible locations — never in home dir directly
            # HERE is the directory containing research_assistant.py
            # (either Jarvis2026/skills/ or Jarvis2026/ depending on install)
            parent = os.path.dirname(HERE)   # skills/ → Jarvis2026/
            # Also check cwd (where main.py was run from)
            cwd    = os.getcwd()

            found = None
            for base in [HERE, parent, cwd]:
                for name in ["papers", "Paper", "Papers",
                             "research", "Research"]:    # no Documents!
                    candidate = os.path.join(base, name)
                    try:
                        if os.path.isdir(candidate):
                            found = candidate
                            break
                    except PermissionError:
                        continue
                if found:
                    break

            if found:
                self.papers_dir = found
                log.info("Papers folder: %s", found)
            else:
                # Default — tell the user exactly where to put the folder
                self.papers_dir = os.path.join(cwd, "papers")
                log.info("Papers folder (default): %s", self.papers_dir)
        self.model       = ollama_model
        self._current:   Optional[dict] = None
        self._history:   list[dict]     = []
        self._text_cache: dict[str, str] = {}

        try:
            os.makedirs(self.papers_dir, exist_ok=True)
        except PermissionError:
            pass
        print(f"  [Research] Papers folder: {self.papers_dir}")
        self._summary_cache = self._load_summary_cache()  # persists between restarts

    def _pre_speak(self, message: str) -> None:
        """Speak a short contextual message and wait for it to finish
        before starting a slow Ollama call — gives immediate audio feedback."""
        try:
            import voice as _v
            _v.speak(message)
            _v.wait_until_done()
        except Exception:
            print(f"[Jarvis] {message}")

    # =========================================================================
    # Persistent summary cache  (survives restarts)
    # =========================================================================

    _CACHE_FILE = ".jarvis_summaries.json"

    def _load_summary_cache(self) -> dict:
        """Load summary cache from JSON file in the papers folder."""
        import json
        cache_path = os.path.join(self.papers_dir, self._CACHE_FILE)
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_summary(self, path: str, summary: str) -> None:
        """Persist a paper summary keyed by path + modification time."""
        import json
        try:
            mtime = str(os.path.getmtime(path))
            self._summary_cache[path] = {"summary": summary, "mtime": mtime}
            cache_path = os.path.join(self.papers_dir, self._CACHE_FILE)
            with open(cache_path, "w") as f:
                json.dump(self._summary_cache, f, indent=2)
        except Exception as exc:
            log.debug("Summary cache write failed: %s", exc)

    def _get_cached_summary(self, path: str) -> Optional[str]:
        """Return cached summary if file hasn't changed since it was cached.
        The strip is applied on READ as well: summaries cached before the
        preamble fix have 'Here's a summary...' baked into the JSON file and
        would otherwise keep replaying forever."""
        entry = self._summary_cache.get(path)
        if not entry:
            return None
        try:
            if str(os.path.getmtime(path)) == entry.get("mtime"):
                cached = entry.get("summary")
                return _strip_llm_preamble(cached) if cached else cached
        except Exception:
            pass
        return None

    # =========================================================================
    # Public handlers — each registered with SkillRegistry
    # =========================================================================

    def handle_analyze(self, query: str) -> IntentResult:
        """Analyze a paper. If multiple papers exist and none is specified, ask which one."""
        q_lo = query.lower()

        # Detect "write / create / draft a paper" — user wants to author, not analyze
        if any(w in q_lo for w in ("write", "create", "compose", "draft",
                                    "generate a", "make a", "help me write",
                                    "can you write", "could you write")):
            return IntentResult(
                speak=(
                    "I can analyse and break down existing papers for you, sir, "
                    "but writing a full research paper from scratch is beyond my current scope. "
                    "I can help you understand the methodology and findings of a reference paper "
                    "if that would assist your own research."
                )
            )

        hint = self._parse_paper_hint(query)

        # If multiple papers and no specific number/name given → ask which one
        pdfs = self._sorted_papers()
        if len(pdfs) > 1 and self._parse_paper_number(query) is None and not hint:
            names  = [os.path.splitext(os.path.basename(p))[0] for p in pdfs]
            listed = ", ".join(f"Paper {i+1}: {n}" for i, n in enumerate(names))
            return IntentResult(
                speak=f"I have {len(pdfs)} papers, sir. {listed}. "
                      f"Which one would you like me to analyse?"
            )

        path = self._find_paper(hint)

        if path is None:
            if pdfs and hint and self._parse_paper_number(hint) is not None:
                return IntentResult(
                    speak=f"I only have {len(pdfs)} paper{'s' if len(pdfs)>1 else ''}, sir. "
                          f"Say 'list papers' to hear what is available."
                )
            return IntentResult(
                speak=(
                    f"I couldn't find any PDF in the papers folder, sir. "
                    f"Please drop a paper into {self.papers_dir} and ask again."
                )
            )

        # Extract & cache text
        text, meta = self._load_paper(path)
        if not text:
            return IntentResult(
                speak="I found the file but couldn't extract any text from it, sir. "
                       "It may be a scanned image PDF — try a text-based PDF."
            )

        title   = meta.get("title") or os.path.splitext(os.path.basename(path))[0]
        authors = meta.get("author", "")
        name    = f"{title} by {authors}" if authors else title

        # Check persistent cache first — avoids re-running Ollama on known papers
        cached = self._get_cached_summary(path)
        if cached:
            log.debug("Using cached summary for %s", title)
            return IntentResult(speak=cached, data={"path": path, "title": title})

        # Speak immediately so user knows we are working
        self._pre_speak("One moment, sir.")

        # Build focused context for Ollama
        sections = self._extract_sections(text)
        context  = self._build_context(sections, text)

        # Ask Ollama for a spoken briefing
        analysis = self._ask_ollama(context, _ANALYSIS_PROMPT, max_tokens=80, query=query)
        if not analysis:
            analysis = self._extractive_briefing(sections, text)

        # Cache for next time
        self._save_summary(path, analysis)

        return IntentResult(speak=analysis, data={"path": path, "title": title})

    def handle_question(self, query: str) -> IntentResult:
        """Answer a specific question about the currently loaded paper."""
        if self._current is None:
            # Try to auto-load the latest paper first
            path = self._find_paper()
            if path:
                self.handle_analyze("analyze")
            else:
                return IntentResult(
                    speak="No paper is loaded, sir. "
                          "Please say 'analyze paper' first."
                )

        sections = self._current.get("sections", {})
        text     = self._current.get("text", "")
        context  = self._build_context(sections, text)

        # Pre-speak once, then route to the right section
        self._pre_speak("One moment, sir.")
        q_lo = query.lower()
        if any(w in q_lo for w in ("method", "approach", "how did they", "technique", "experiment")):
            focus = sections.get("methodology") or sections.get("introduction") or text[:1500]
            prompt = _METHODOLOGY_PROMPT
        elif any(w in q_lo for w in ("finding", "result", "outcome", "performance", "accuracy", "what did they find")):
            focus = sections.get("results") or sections.get("conclusion") or text[:1500]
            prompt = _FINDINGS_PROMPT
        elif any(w in q_lo for w in ("conclusion", "significance", "implication", "future", "recommend")):
            focus = sections.get("conclusion") or text[-1500:]
            prompt = _CONCLUSION_PROMPT
        elif any(w in q_lo for w in ("abstract", "overview", "brief", "summary", "gist")):
            focus = sections.get("abstract") or text[:1000]
            prompt = _SUMMARY_PROMPT
        else:
            focus  = context
            prompt = _QA_PROMPT.format(question=query)

        answer = self._ask_ollama(focus[:_TOTAL_CONTEXT_CHARS], prompt, max_tokens=180, query=query)
        if not answer:
            answer = "I couldn't generate an answer from the paper's content, sir."

        return IntentResult(speak=answer, data={"query": query})

    def handle_replicate(self, query: str) -> IntentResult:
        """Step-by-step guide to replicate the experiment in the loaded paper."""
        if self._current is None:
            path = self._find_paper()
            if path:
                self.handle_analyze("analyze")
            else:
                return IntentResult(speak="No paper loaded, sir. Say paper first.")
        self._pre_speak("One moment, sir.")
        sections = self._current.get("sections", {})
        text     = self._current.get("text", "")
        focus = (sections.get("methodology", "") + "\n" +
                 sections.get("results", "")[:500] + "\n" +
                 sections.get("introduction", "")[:400])
        if len(focus.strip()) < 100:
            focus = text[:_TOTAL_CONTEXT_CHARS]
        answer = self._ask_ollama(focus[:_TOTAL_CONTEXT_CHARS],
                                  _REPLICATION_PROMPT, max_tokens=240)
        if not answer:
            answer = "I couldn't extract replication steps from this paper, sir."
        return IntentResult(speak=answer, data={"query": query})

    def handle_gaps(self, query: str) -> IntentResult:
        """Identify research gaps and limitations in the loaded paper."""
        if self._current is None:
            path = self._find_paper()
            if path:
                self.handle_analyze("analyze")
            else:
                return IntentResult(speak="No paper loaded, sir. Say paper first.")
        self._pre_speak("One moment, sir.")
        sections = self._current.get("sections", {})
        text     = self._current.get("text", "")
        focus = (sections.get("conclusion", "") + "\n" +
                 sections.get("discussion", "") + "\n" +
                 sections.get("introduction", "")[:400])
        if len(focus.strip()) < 100:
            focus = text[:_TOTAL_CONTEXT_CHARS]
        answer = self._ask_ollama(focus[:_TOTAL_CONTEXT_CHARS],
                                  _GAPS_PROMPT, max_tokens=200)
        if not answer:
            answer = "I couldn't identify research gaps from this paper, sir."
        return IntentResult(speak=answer, data={"query": query})

    def handle_compare(self, query: str) -> IntentResult:
        """Compare the two most recently analyzed papers."""
        if len(self._history) < 2:
            return IntentResult(
                speak="I only have one paper loaded, sir. "
                      "Load a second paper and ask me to compare them."
            )
        self._pre_speak("One moment, sir.")

        p1, p2 = self._history[-1], self._history[-2]
        ctx = (
            f"PAPER 1: {p1.get('title','?')}\n{p1.get('context','')[:1500]}\n\n"
            f"PAPER 2: {p2.get('title','?')}\n{p2.get('context','')[:1500]}"
        )
        answer = self._ask_ollama(ctx, _COMPARE_PROMPT, max_tokens=220, query=query)
        if not answer:
            answer = (
                f"Paper one is '{p1.get('title','unknown')}' and "
                f"paper two is '{p2.get('title','unknown')}'. "
                "I couldn't generate a detailed comparison, sir."
            )
        return IntentResult(speak=answer)

    def handle_list_papers(self, query: str) -> IntentResult:
        """List all PDFs numbered so the user can say 'analyze paper 2'."""
        pdfs = self._sorted_papers()
        if not pdfs:
            return IntentResult(
                speak=f"The papers folder is empty, sir. "
                      f"Drop some PDFs into {self.papers_dir}."
            )
        names = [os.path.splitext(os.path.basename(p))[0] for p in pdfs]
        count = len(pdfs)

        if count == 1:
            spoken = f"You have one paper, sir. Paper 1: {names[0]}. Just say paper 1 to analyze it."
        else:
            # Speak each paper with its number
            parts  = [f"Paper {i+1}: {n}" for i, n in enumerate(names)]
            listed = ". ".join(parts)
            spoken = (
                f"You have {count} papers, sir. {listed}. "
                f"Say the number to analyze any one of them."
            )
        return IntentResult(speak=spoken, data={"count": count, "names": names})

    def handle_load_paper(self, query: str) -> IntentResult:
        """Load a specific paper by name without running full analysis."""
        hint = self._parse_paper_hint(query)
        path = self._find_paper(hint)
        if path is None:
            return IntentResult(
                speak=f"I couldn't find a paper matching '{hint}', sir."
            )
        return self.handle_analyze(query)   # load + analyze

    # =========================================================================
    # PDF reading
    # =========================================================================

    def _all_pdfs(self) -> list[str]:
        """Return all PDF files in papers_dir.
        Accepts .pdf extension OR files without extension that start
        with the PDF magic bytes %PDF — so 'Paper1' works as well as
        'Paper1.pdf'."""
        if not os.path.isdir(self.papers_dir):
            return []
        try:
            entries = os.listdir(self.papers_dir)
        except PermissionError:
            log.error("Permission denied reading: %s", self.papers_dir)
            return []
        results = []
        for f in entries:
            path = os.path.join(self.papers_dir, f)
            if not os.path.isfile(path):
                continue
            if f.lower().endswith(".pdf"):
                results.append(path)
            elif "." not in f:          # no extension — check magic bytes
                try:
                    with open(path, "rb") as fh:
                        if fh.read(4) == b"%PDF":
                            results.append(path)
                except OSError:
                    pass
        return results

    def _sorted_papers(self) -> list[str]:
        """Return all PDFs sorted alphabetically — consistent numbering."""
        return sorted(self._all_pdfs(),
                      key=lambda p: os.path.basename(p).lower())

    def _parse_paper_number(self, query):
        """Return 0-based index from queries like 'paper 2', '3', 'second'."""
        m = re.search("[0-9]+", query)
        if m:
            return int(m.group()) - 1   # "paper 2" -> index 1
        ordinals = {
            "first": 0, "second": 1, "third": 2,
            "fourth": 3, "fifth": 4, "sixth": 5,
            "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
        }
        q_lo = query.lower()
        for word, idx in ordinals.items():
            if word in q_lo:
                return idx
        return None


    def _find_paper(self, hint: Optional[str] = None) -> Optional[str]:
        """Return the best matching PDF path, or None."""
        pdfs = self._all_pdfs()
        if not pdfs:
            return None
        if hint:
            # Check for a number first ("paper 2", "the third one")
            idx = self._parse_paper_number(hint)
            if idx is not None:
                ordered = self._sorted_papers()
                if 0 <= idx < len(ordered):
                    return ordered[idx]
                return None   # number out of range
            # Fuzzy name match
            hint_lo = hint.lower()
            scored  = sorted(
                pdfs,
                key=lambda p: sum(
                    w in os.path.basename(p).lower() for w in hint_lo.split()
                ),
                reverse=True,
            )
            if scored:
                return scored[0]
        # Default: first alphabetically (same order as list)
        return self._sorted_papers()[0] if pdfs else None

    def _load_paper(self, path: str) -> tuple[str, dict]:
        """Extract text and metadata from a PDF. Returns (text, meta)."""
        # Full cache hit: text AND current state both match this paper
        if path in self._text_cache and self._current and self._current.get("path") == path:
            return self._text_cache[path], self._current.get("meta", {})
        # Partial hit: text cached but _current is a different paper — rebuild state
        if path in self._text_cache:
            text     = self._text_cache[path]
            sections = self._extract_sections(text)
            context  = self._build_context(sections, text)
            self._current = {
                "path": path, "text": text, "meta": {},
                "sections": sections, "context": context,
                "title": _guess_title(text),
            }
            self._history = (self._history[-1:] + [self._current]) if self._history else [self._current]
            return text, {}

        text, meta = "", {}

        # Tier 1: PyMuPDF — handles the widest range of PDF types
        if HAS_PYMUPDF and _fitz is not None:
            try:
                doc   = _fitz.open(path)
                pages = [page.get_text() for page in doc]
                text  = "\n".join(pages)
                # Try to get metadata
                m    = doc.metadata or {}
                meta = {
                    "title":  m.get("title", "") or "",
                    "author": m.get("author", "") or "",
                }
                doc.close()
                log.debug("PyMuPDF extracted %d chars", len(text))
            except Exception as exc:
                log.warning("PyMuPDF failed (%s), trying pdfplumber", exc)

        # Tier 2: pdfplumber
        if (not text or len(text) < 100) and HAS_PDFPLUMBER and pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                log.debug("pdfplumber extracted %d chars", len(text))
            except Exception as exc:
                log.warning("pdfplumber failed (%s), trying pypdf", exc)

        # Tier 3: pypdf
        if (not text or len(text) < 100) and HAS_PYPDF and PdfReader is not None:
            try:
                reader = PdfReader(path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
                m      = reader.metadata or {}
                meta   = {
                    "title":  str(m.get("/Title", "") or ""),
                    "author": str(m.get("/Author", "") or ""),
                }
                log.debug("pypdf extracted %d chars", len(text))
            except Exception as exc:
                log.error("pypdf also failed: %s", exc)

        # Clean up text
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Build current paper state
        sections = self._extract_sections(text)
        context  = self._build_context(sections, text)
        self._current = {
            "path":     path,
            "text":     text,
            "meta":     meta,
            "sections": sections,
            "context":  context,
            "title":    meta.get("title") or _guess_title(text),
        }
        # Keep last 2 papers for comparison
        self._history = (self._history[-1:] + [self._current]) if self._history else [self._current]
        self._text_cache[path] = text

        return text, meta

    # =========================================================================
    # Section extraction
    # =========================================================================

    def _extract_sections(self, text: str) -> dict[str, str]:
        """Identify and extract key sections from the paper text."""
        lines   = text.splitlines()
        found   = {}          # section_name → start line index
        order   = []          # (line_idx, section_name)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < 3 or len(stripped) > 120:
                continue
            for name, pat in _SECTION_PATTERNS.items():
                if pat.match(stripped) or (len(stripped) < 60 and pat.search(stripped)):
                    if name not in found:
                        found[name] = i
                        order.append((i, name))
                    break

        order.sort()
        sections: dict[str, str] = {}

        for idx, (start_line, sec_name) in enumerate(order):
            end_line = order[idx + 1][0] if idx + 1 < len(order) else len(lines)
            section_text = "\n".join(lines[start_line:end_line]).strip()
            # Trim to limit
            limit    = _SEC_CHAR_LIMITS.get(sec_name, 1000)
            sections[sec_name] = section_text[:limit]

        return sections

    def _build_context(self, sections: dict, full_text: str) -> str:
        """Assemble the most informative sections for Ollama, within token budget.
        Priority order: abstract first (densest summary), then conclusion,
        then intro, then results/discussion/methodology for detail.
        """
        parts    = []
        used     = 0
        priority = ["abstract", "conclusion", "introduction",
                    "results", "discussion", "methodology"]

        for sec in priority:
            if sec in sections and used < _TOTAL_CONTEXT_CHARS:
                limit   = _SEC_CHAR_LIMITS.get(sec, 1_000)
                snippet = sections[sec][:limit]
                parts.append(f"[{sec.upper()}]\n{snippet}")
                used += len(snippet)

        if used < 500 and full_text:
            # No sections found — just use the beginning of the paper
            parts.append(f"[PAPER TEXT]\n{full_text[:_TOTAL_CONTEXT_CHARS]}")

        return "\n\n".join(parts)

    # =========================================================================
    # Ollama analysis
    # =========================================================================

    def _ask_ollama(self, context: str, prompt_template: str,
                    max_tokens: int = 180, query: str = "") -> Optional[str]:
        """Send context + prompt to Ollama with optional RAG grounding."""
        if not HAS_OLLAMA or _ollama is None:
            return None
        full_prompt = prompt_template.replace("{context}", context)

        # Inject RAG context from the paper corpus as system message
        system_msg = (
            "You are JARVIS, a research assistant fine-tuned on tool wear and machining. "
            "Respond with precise, factual answers grounded in published research. "
            "Spoken prose only — no bullet points. Address the user as 'sir'."
        )
        if query:
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                import rag as _rag
                rag_ctx = _rag.retrieve(query)
                if rag_ctx:
                    system_msg += "\n\n" + rag_ctx
            except Exception as _e:
                log.debug("RAG skipped in research_assistant: %s", _e)

        try:
            resp = _ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": full_prompt},
                ],
                stream=False,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                    "num_ctx":     8192,
                },
            )
            return _strip_llm_preamble(resp.message.content.strip())
        except Exception as exc:
            log.warning("Ollama request failed: %s", exc)
            return None
    def _extractive_briefing(self, sections: dict, full_text: str) -> str:
        """Build a basic spoken briefing from raw extracted text if Ollama fails."""
        abstract = sections.get("abstract", "")
        concl    = sections.get("conclusion", "")
        chosen   = abstract or concl or full_text[:600]
        # Normalize all whitespace so embedded newlines don't reach the speaker
        chosen = re.sub(r"\s+", " ", chosen.strip())
        sents  = re.split(r"(?<=[.!?])\s+", chosen)
        return " ".join(sents[:4]) if sents else "No readable content found."

    # =========================================================================
    # Query parsing helpers
    # =========================================================================

    _LOAD_STRIP = re.compile(
        r"^(?:load|open|analyze|analyse|read|summarize|summarise|"
        r"tell me about|what(?:'s|\s+is) in)\s+(?:the\s+)?(?:paper|pdf|file|document)?\s*",
        re.IGNORECASE,
    )

    def _parse_paper_hint(self, query: str) -> Optional[str]:
        """Extract a paper name/keyword from a query string."""
        cleaned = self._LOAD_STRIP.sub("", query).strip().rstrip("?.")
        # Remove very generic words
        cleaned = re.sub(r"\b(?:latest|recent|last|new|this|that)\b", "", cleaned, flags=re.I).strip()
        return cleaned if len(cleaned) > 2 else None


# =========================================================================
# Ollama prompt templates
# =========================================================================
# Every prompt follows the same 3 rules:
#   1. Start the answer immediately — no meta-commentary
#   2. Hard word limit — enforced by max_tokens in the caller
#   3. No bullet points, no headings, plain spoken prose

_ANALYSIS_PROMPT = (
    "Write 2 short sentences about this paper. Under 50 words total.\n"
    "Start with the topic immediately — e.g. 'Researchers propose...' or 'This paper introduces...'\n"
    "Do NOT write 'Here is a summary' or 'This paper is about' or 'Based on the paper'.\n"
    "No greetings. No bullet points. Spoken prose only.\n"
    "Cover: the core problem and the main result.\n\n"
    "Paper:\n{context}"
)

_REPLICATION_PROMPT = (
    "Explain how to replicate this experiment. Under 80 words.\n"
    "Start with the first action immediately — e.g. 'First, obtain...' or 'To replicate this...'\n"
    "Do NOT start with 'Based on this paper', 'Here are the steps', or any meta-text.\n"
    "Cover: data or materials needed, tools or software, the key procedure, how to verify results.\n"
    "Plain spoken sentences. No bullet points.\n\n"
    "Paper:\n{context}"
)

_GAPS_PROMPT = (
    "State 2 specific research gaps or limitations from this paper. Under 60 words.\n"
    "Start directly — e.g. 'The study does not address...' or 'One limitation is...'\n"
    "Do NOT start with 'Based on this paper', 'Here are the gaps', or any meta-text.\n"
    "Be specific. Plain spoken sentences. No bullet points.\n\n"
    "Paper:\n{context}"
)

_METHODOLOGY_PROMPT = (
    "Describe the methodology of this paper in 2 sentences. Under 50 words.\n"
    "Start directly — e.g. 'The researchers used...' or 'The approach involves...'\n"
    "Do NOT start with 'Based on this paper' or 'Here is the methodology'.\n"
    "Focus on what they did and how. Plain spoken sentences, no bullet points.\n\n"
    "Paper:\n{context}"
)

_FINDINGS_PROMPT = (
    "State the 2 most important findings from this paper. Under 60 words.\n"
    "Start directly — e.g. 'The model achieved...' or 'Results show...' or 'The study found...'\n"
    "Do NOT start with 'Here is a summary', 'The findings are', or 'Based on this paper'.\n"
    "Include key numbers or metrics if relevant. Plain spoken sentences, no bullet points.\n\n"
    "Paper:\n{context}"
)

_CONCLUSION_PROMPT = (
    "State the conclusion of this paper in 2 sentences. Under 50 words.\n"
    "Start directly — e.g. 'The authors conclude...' or 'This work shows...'\n"
    "Do NOT start with 'Based on this paper' or 'Here are the conclusions'.\n"
    "Plain sentences, no bullet points.\n\n"
    "Paper:\n{context}"
)

_SUMMARY_PROMPT = (
    "One sentence. What is this paper about? Under 30 words.\n"
    "Start with the topic directly. No meta-text.\n\n"
    "Paper:\n{context}"
)

_QA_PROMPT = (
    "Answer this question about the paper in 2 sentences. Under 50 words. Be specific.\n"
    "Start directly with the answer. Do NOT begin with 'Based on the paper' or 'According to'.\n"
    "No bullet points.\n\n"
    "Question: {question}\n\n"
    "Paper:\n{{context}}"
)

_COMPARE_PROMPT = (
    "Compare these two papers in 3 sentences. Under 70 words.\n"
    "Start directly — e.g. 'The first paper examines...' or 'While paper one...\n"
    "Do NOT start with 'Based on these papers' or 'Here is a comparison'.\n"
    "Cover: what each studied, how approaches differ, whose findings matter more.\n"
    "No bullet points.\n\n"
    "{context}"
)
# =========================================================================
# Utility
# =========================================================================

def _guess_title(text: str) -> str:
    """Try to extract the paper title from the first few lines of text."""
    lines = [l.strip() for l in text.splitlines()[:12] if l.strip()]
    # Title is usually the longest short line near the top
    candidates = [l for l in lines if 10 < len(l) < 180 and not l.startswith("http")]
    if candidates:
        return candidates[0]
    return "Unknown paper"


# =========================================================================
# register()  —  integrates with Brain's SkillRegistry
# =========================================================================

def register(registry, papers_dir: Optional[str] = None,
             ollama_model: str = "llama3.2:3b", **kwargs) -> None:
    """Register all research-assistant skills with contracts.SkillRegistry."""
    ra = ResearchAssistant(papers_dir=papers_dir, ollama_model=ollama_model)

    # IMPORTANT: specific skills are registered FIRST.
    # This prevents "paper" in a question like "what are the gaps in this paper?"
    # from matching analyze_paper before research_gaps gets a chance.

    registry.register(
        name="list_papers",
        keywords=[
            "list papers", "what papers", "how many papers", "papers do i have",
            "show papers", "available papers",
        ],
        handler=ra.handle_list_papers,
        description="Lists all PDF papers in the papers folder.",
    )

    registry.register(
        name="compare_papers",
        keywords=[
            "compare", "compare the papers", "compare to", "difference between",
            "how do they differ", "which is better", "versus",
        ],
        handler=ra.handle_compare,
        description="Compares the two most recently analyzed papers.",
    )

    registry.register(
        name="research_gaps",
        keywords=[
            "research gap", "gaps", "limitations", "limitation",
            "what is missing", "what they missed", "open question",
            "future work", "not addressed", "drawback", "weakness",
            "what could be improved", "what remains", "unanswered",
            "critique", "shortcoming",
        ],
        handler=ra.handle_gaps,
        description="Identifies research gaps and limitations in the loaded paper.",
    )

    registry.register(
        name="replicate_experiment",
        keywords=[
            "replicate", "reproduce", "implement", "recreate",
            "how can i replicate", "how to replicate", "how to reproduce",
            "step by step", "what equipment", "what materials",
            "what tools", "what software", "procedure", "protocol",
            "how would i", "how do i implement", "workflow",
        ],
        handler=ra.handle_replicate,
        description="Explains how to replicate the experiment from the loaded paper.",
    )

    registry.register(
        name="paper_question",
        keywords=[
            "methodology", "what method", "how did they", "what approach",
            "findings", "what did they find", "results", "what were the results",
            "conclusion", "what do they conclude", "abstract", "brief summary",
            "research question", "hypothesis", "dataset",
            "tell me more", "what else", "more about",
            "explain", "elaborate", "significance", "contribution",
        ],
        handler=ra.handle_question,
        description="Answers specific questions about the loaded paper.",
    )

    # analyze_paper registered LAST — catches "paper", "paper 1", "paper 2"
    # only when no specific skill matched first.
    registry.register(
        name="analyze_paper",
        keywords=[
            "paper", "analyze", "analyse", "summarize", "summarise",
            "analyze paper", "analyse paper",
            "analyze the paper", "analyse the paper",
            "read the paper", "summarize the paper",
            "tell me about the paper", "what is this paper",
            "what does it say", "load paper", "latest paper",
        ],
        handler=ra.handle_analyze,
        description="Analyzes a research paper PDF and gives a short spoken summary.",
    )
