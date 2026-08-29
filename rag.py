"""
rag.py — Retrieval-Augmented Generation for JARVIS
====================================================
Retrieves relevant paper chunks at query time and formats them as context
for the fine-tuned `jarvis` Ollama model. This is the inference-time component.
It does NOT touch finetune.py or the LoRA weights.

Pipeline:
  build  : papers → chunks → embed (nomic-embed-text) → save index to disk
  query  : embed query → cosine top-k → formatted context string

Setup (one-time):
  ollama pull nomic-embed-text      # ~274 MB, needs internet once
  python3 rag.py --build            # indexes all PDFs (~5-10 min)

Integration:
  import rag
  context = rag.retrieve("What causes flank wear?")   # returns str or ""
  # inject context into Ollama system prompt

Notes:
  - Query expansion uses llama3.2:3b (NOT jarvis — fine-tuned model cannot
    follow general instructions and returns empty strings for expansion prompts)
  - nomic-embed-text does not bridge manufacturing domain synonymy well
    (e.g. "cutting force" ≠ "vibration monitoring" in embedding space).
    Query expansion via LLM mitigates this significantly.
  - Returns "" (not None) when unavailable — safe to concatenate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request

import numpy as np

log = logging.getLogger("jarvis.rag")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
TEXT_CACHE = os.path.join(HERE, ".paper_texts.json")
INDEX_VEC  = os.path.join(HERE, ".rag_index.npy")
INDEX_META = os.path.join(HERE, ".rag_index.json")

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import OLLAMA_MODEL as _CFG_BASE, JARVIS_FINETUNED_MODEL as _CFG_TUNED
except ImportError:
    _CFG_BASE, _CFG_TUNED = "llama3.2:3b", "llama3.2:3b"

OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
EXPAND_MODEL  = _CFG_BASE     # base model handles instruction-following; a
                              # fine-tuned domain model often can't (see notes)
GEN_MODEL     = _CFG_TUNED   # only used for --ask CLI mode. Defaults to the
                              # base model if you haven't fine-tuned your own —
                              # RAG works fully without ever fine-tuning anything.

CHUNK_WORDS   = 220
CHUNK_OVERLAP = 40
TOP_K         = 4
MAX_PER_SRC   = 1               # max chunks per paper (prevents one paper dominating)
SIM_THRESHOLD = 0.65            # skip RAG if best match is below this

# ── Cached index (loaded once, reused across calls) ───────────────────────────
_vecs:  np.ndarray | None = None
_meta:  list[dict]        = []
_ready: bool              = False
last_sources: list[str]   = []   # paper names from the most recent retrieve()


# ── Ollama helpers ────────────────────────────────────────────────────────────
def _post(path: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        OLLAMA_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _embed(text: str) -> list[float] | None:
    """Embed text using nomic-embed-text. Tries modern API then legacy."""
    try:
        r = _post("/api/embed", {"model": EMBED_MODEL, "input": text})
        return r.get("embeddings", [r.get("embedding")])[0]
    except Exception:
        pass
    try:
        r = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
        return r.get("embedding")
    except Exception as e:
        log.debug("embed failed: %s", e)
        return None


def _expand_query(query: str) -> str:
    """Use llama3.2:3b to rephrase the query into domain-rich keywords.
    Falls back to original query on any error.
    NOTE: uses base model, NOT jarvis — fine-tuned model can't follow
    general prompts and returns empty strings for this task."""
    try:
        r = _post("/api/chat", {
            "model":   EXPAND_MODEL,
            "stream":  False,
            "options": {"num_predict": 60, "temperature": 0.0},
            "messages": [{
                "role": "user",
                "content": (
                    "Expand this manufacturing/machining query into 10-15 "
                    "technical keywords and synonyms that would appear in academic "
                    "papers. Return only the keywords, comma-separated, nothing else.\n\n"
                    f"Query: {query}"
                ),
            }],
        })
        expanded = r["message"]["content"].strip()
        if expanded and len(expanded) > 10:
            return f"{query} {expanded}"
    except Exception as e:
        log.debug("query expansion failed: %s", e)
    return query


# ── PDF text loading ──────────────────────────────────────────────────────────
def _load_paper_texts() -> dict[str, str]:
    """Load cached PDF texts. Falls back to re-extracting if cache missing."""
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE) as f:
            raw = json.load(f)
        # Handle both flat {name: text} and nested {name: {text: ...}} schemas
        texts = {}
        for k, v in raw.items():
            if isinstance(v, str):
                texts[k] = v
            elif isinstance(v, dict):
                texts[k] = v.get("text", v.get("content", ""))
        if texts:
            return texts

    log.info("No paper text cache found — extracting from PDFs")
    return _extract_all_pdfs()


def _extract_all_pdfs() -> dict[str, str]:
    texts = {}
    for folder in ["papers", "Paper", "Papers", "Research", "research"]:
        d = os.path.join(HERE, folder)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith(".pdf"):
                    path  = os.path.join(d, fname)
                    text  = _read_pdf(path)
                    if text:
                        texts[os.path.splitext(fname)[0]] = text
            break
    if texts:
        with open(TEXT_CACHE, "w") as f:
            json.dump(texts, f, ensure_ascii=False)
    return texts


def _read_pdf(path: str) -> str:
    for fn in [_pdfminer, _pdfplumber, _pypdf]:
        try:
            t = fn(path)
            if len(t) > 200:
                return t
        except Exception:
            pass
    return ""


def _pdfminer(p):
    from pdfminer.high_level import extract_text
    return extract_text(p)

def _pdfplumber(p):
    import pdfplumber
    with pdfplumber.open(p) as f:
        return "\n".join(pg.extract_text() or "" for pg in f.pages)

def _pypdf(p):
    from pypdf import PdfReader
    return "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)


# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk(text: str, paper_name: str) -> list[dict]:
    words  = text.split()
    step   = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i:i + CHUNK_WORDS])
        if len(chunk_text) < 80:
            continue
        chunks.append({"paper": paper_name, "text": chunk_text, "offset": i})
    return chunks


# ── Build index ───────────────────────────────────────────────────────────────
def build(force: bool = False) -> None:
    """Embed all paper chunks and save index to disk."""
    if not force and os.path.exists(INDEX_VEC) and os.path.exists(INDEX_META):
        print("Index already exists. Use --force to rebuild.")
        return

    texts  = _load_paper_texts()
    if not texts:
        print("✗ No paper texts found. Add PDFs to the papers/ folder.")
        return

    chunks: list[dict] = []
    for name, text in texts.items():
        chunks.extend(_chunk(text, name))

    print(f"Embedding {len(chunks)} chunks from {len(texts)} papers ...")
    vecs = []
    failed = 0
    for i, c in enumerate(chunks):
        v = _embed(c["text"])
        if v is None:
            failed += 1
            vecs.append(np.zeros(768))
        else:
            vecs.append(np.array(v, dtype=np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(chunks)} embedded")

    mat = np.array(vecs, dtype=np.float32)
    # Normalise rows so cosine similarity = dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat /= norms

    np.save(INDEX_VEC, mat)
    with open(INDEX_META, "w") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"✓ Index built: {len(chunks)} chunks  ({failed} embedding failures)")
    if failed:
        print("  Pull nomic-embed-text: ollama pull nomic-embed-text")


# ── Load index ────────────────────────────────────────────────────────────────
def _load_index() -> bool:
    global _vecs, _meta, _ready
    if _ready:
        return True
    if not os.path.exists(INDEX_VEC) or not os.path.exists(INDEX_META):
        return False
    try:
        _vecs  = np.load(INDEX_VEC)
        with open(INDEX_META) as f:
            _meta = json.load(f)
        _ready = True
        log.info("RAG index loaded: %d chunks", len(_meta))
        return True
    except Exception as e:
        log.warning("Failed to load RAG index: %s", e)
        return False


# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(
    query:        str,
    k:            int  = TOP_K,
    expand:       bool = True,
    sim_threshold: float = SIM_THRESHOLD,
) -> str:
    """
    Return a formatted context string of top-k relevant paper chunks.
    Returns "" if index unavailable, nomic-embed-text unreachable, or
    top similarity below sim_threshold (query is not paper-relevant).
    Safe to concatenate directly into a system prompt.
    """
    if not _load_index():
        return ""

    # Query expansion dramatically improves recall for domain synonymy gaps
    expanded = _expand_query(query) if expand else query

    vec = _embed(expanded)
    if vec is None:
        return ""

    q = np.array(vec, dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-9)

    scores = _vecs @ q                    # cosine similarities (normalised)
    top_idx = np.argsort(scores)[::-1]

    if scores[top_idx[0]] < sim_threshold:
        log.debug("RAG: top similarity %.3f < threshold %.2f — skipping",
                  scores[top_idx[0]], sim_threshold)
        return ""

    # Select top-k with max_per_source deduplication
    seen_papers: dict[str, int] = {}
    selected: list[tuple[int, float]] = []

    for idx in top_idx:
        if len(selected) >= k:
            break
        paper = _meta[idx]["paper"]
        if seen_papers.get(paper, 0) >= MAX_PER_SRC:
            continue
        seen_papers[paper] = seen_papers.get(paper, 0) + 1
        selected.append((idx, float(scores[idx])))

    if not selected:
        last_sources.clear()
        return ""

    # Record source paper names for the UI to display
    last_sources.clear()
    for idx, _ in selected:
        nm = _meta[idx]["paper"]
        if nm not in last_sources:
            last_sources.append(nm)

    parts = ["The following excerpts from published research papers may be relevant:"]
    for rank, (idx, sim) in enumerate(selected, 1):
        chunk = _meta[idx]
        parts.append(
            f"\n[{rank}] From: {chunk['paper']} (relevance: {sim:.2f})\n"
            f"{chunk['text'].strip()}"
        )
    parts.append(
        "\nUse the above excerpts to ground your answer where relevant, but respond "
        "in your own words as flowing speech. Never say 'excerpt 1' or 'excerpt [1]' "
        "or reference the excerpts by number — instead weave the facts in naturally, "
        "e.g. 'research on carbide tools shows...' or name the paper if useful."
    )

    return "\n".join(parts)


# ── CLI for standalone testing ────────────────────────────────────────────────
def _cli_ask(query: str, model: str, k: int, verbose: bool) -> None:
    context = retrieve(query, k=k)
    if not context:
        print("(No RAG context — index missing or query below threshold)")
    elif verbose:
        print("─── RAG Context ───")
        print(context[:800], "..." if len(context) > 800 else "")
        print("─── End Context ───\n")

    import urllib.request
    payload = {
        "model":   model,
        "stream":  False,
        "options": {"num_predict": 300, "temperature": 0.2},
        "messages": [
            {"role": "system",
             "content": "You are JARVIS, a research assistant. " + (context if context else "")},
            {"role": "user", "content": query},
        ],
    }
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            OLLAMA_URL + "/api/chat", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        print(resp["message"]["content"])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="JARVIS RAG module")
    p.add_argument("--build",   action="store_true", help="Build/rebuild the index")
    p.add_argument("--force",   action="store_true", help="Force rebuild even if index exists")
    p.add_argument("--ask",     type=str,  default=None)
    p.add_argument("--model",   type=str,  default=GEN_MODEL)
    p.add_argument("--k",       type=int,  default=TOP_K)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.build or args.force:
        build(force=args.force)
    elif args.ask:
        _cli_ask(args.ask, args.model, args.k, args.verbose)
    else:
        p.print_help()
