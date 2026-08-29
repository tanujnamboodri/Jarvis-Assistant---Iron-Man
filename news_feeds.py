"""
news_feeds.py — Unified real-time news fetcher for JARVIS
=========================================================
Sources (config in news_config.json):
  global_geopolitical : NewsAPI/GNews/NewsData (key required)
  domestic_india      : NewsAPI/GNews/NewsData (key required)
  research            : arXiv (no key) + Crossref (key optional)
  manufacturing       : RSS — The Manufacturer, SME
  machining           : RSS — Modern Machine Shop (mmsonline)

Public API:
  fetch(category)  -> list[str]   3-5 headlines
  fetch_all()      -> dict[str, list[str]]
All functions degrade gracefully — failures return [] and never raise.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

log = logging.getLogger("jarvis.news")

_HERE   = os.path.dirname(os.path.abspath(__file__))
_CONFIG = os.path.join(_HERE, "news_config.json")
_UA     = {"User-Agent": "Mozilla/5.0 (JARVIS research assistant)"}
_N      = 4   # headlines per category


def _cfg() -> dict:
    try:
        with open(_CONFIG) as f:
            return json.load(f)
    except Exception as e:
        log.warning("news_config.json not loaded: %s", e)
        return {}


def _get(url: str, timeout: int = 8) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        log.debug("fetch failed %s: %s", url[:60], e)
        return None


def _rss_titles(url: str, n: int = _N) -> list[str]:
    raw = _get(url)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
        titles = [i.find("title").text for i in root.iter("item")
                  if i.find("title") is not None and i.find("title").text]
        # Atom fallback (arXiv uses Atom)
        if not titles:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            titles = [e.text.strip() for e in root.findall(".//a:entry/a:title", ns)
                      if e.text]
        return titles[:n]
    except Exception as e:
        log.debug("rss parse failed: %s", e)
        return []


# ── Keyed news APIs ───────────────────────────────────────────────────────────
def _api_news(section: dict) -> list[str]:
    provider = section.get("provider", "newsapi")
    key      = section.get("api_key", "")
    if not key or key.startswith("PASTE"):
        return []
    q = urllib.parse.quote(section.get("query", "news"))

    if provider == "newsapi":
        url = (f"https://newsapi.org/v2/top-headlines?q={q}"
               + (f"&country={section['country']}" if section.get("country") else "")
               + f"&pageSize={_N}&apiKey={key}")
        # top-headlines with q+country can be empty — fall back to everything
        raw = _get(url)
        arts = []
        if raw:
            arts = json.loads(raw).get("articles", [])
        if not arts:
            url = f"https://newsapi.org/v2/everything?q={q}&sortBy=publishedAt&pageSize={_N}&apiKey={key}"
            raw = _get(url)
            if raw:
                arts = json.loads(raw).get("articles", [])
        return [a["title"] for a in arts[:_N] if a.get("title")]

    if provider == "gnews":
        url = f"https://gnews.io/api/v4/search?q={q}&max={_N}&apikey={key}"
        raw = _get(url)
        if raw:
            return [a["title"] for a in json.loads(raw).get("articles", [])[:_N]]
        return []

    if provider == "newsdata":
        url = f"https://newsdata.io/api/1/latest?q={q}&apikey={key}" \
              + (f"&country={section['country']}" if section.get("country") else "")
        raw = _get(url)
        if raw:
            return [a["title"] for a in json.loads(raw).get("results", [])[:_N]]
        return []

    return []


# ── Research: arXiv + Crossref ───────────────────────────────────────────────
def _arxiv(query: str) -> list[str]:
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(query)
           + f"&sortBy=submittedDate&sortOrder=descending&max_results={_N}")
    return _rss_titles(url)


def _crossref(query: str, api_key: str = "") -> list[str]:
    url = ("https://api.crossref.org/works?query="
           + urllib.parse.quote(query)
           + f"&sort=created&order=desc&rows={_N}")
    try:
        req = urllib.request.Request(url, headers=dict(
            _UA, **({"crossref-api-key": f"Bearer {api_key}"} if api_key else {})))
        with urllib.request.urlopen(req, timeout=8) as r:
            items = json.loads(r.read())["message"]["items"]
        return [i["title"][0] for i in items if i.get("title")][:_N]
    except Exception as e:
        log.debug("crossref failed: %s", e)
        return []


# ── YouTube channel → latest video ID (works even when not live) ─────────────
YT_CHANNELS = {
    "geopolitics": "UC-rEK7l-X_iD8aXkX8azYZA",   # Times Now
    "india":       "UCz8QaiQxApLq8sLNcszYyJw",   # Firstpost
    "science":     "UCBX5er6E37_yWB3gCM32p3g",   # Science News
    "manufacturing": "",   # playlist-based, no single channel
}

def latest_video_id(channel_id: str) -> str | None:
    """Fetch the most recent video ID from a YouTube channel's RSS feed.
    Works regardless of live status. Returns None on failure."""
    if not channel_id:
        return None
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    raw = _get(url)
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
        ns = {"yt": "http://www.youtube.com/xml/schemas/2015",
              "a":  "http://www.w3.org/2005/Atom"}
        vid = root.find(".//yt:videoId", ns)
        if vid is not None and vid.text:
            return vid.text
        # Fallback: parse from <link> href
        link = root.find(".//a:entry/a:link", ns)
        if link is not None:
            href = link.get("href", "")
            if "watch?v=" in href:
                return href.split("watch?v=")[1].split("&")[0]
    except Exception as e:
        log.debug("latest_video_id failed: %s", e)
    return None


def channel_videos() -> dict:
    """Return {category: latest_video_id} for all configured channels."""
    out = {}
    for cat, cid in YT_CHANNELS.items():
        vid = latest_video_id(cid)
        if vid:
            out[cat] = vid
    return out


# ── Public API ────────────────────────────────────────────────────────────────
def fetch(category: str) -> list[str]:
    """Fetch headlines for one category. Returns [] on any failure."""
    c = _cfg()
    try:
        if category == "global_geopolitical":
            return _api_news(c.get("global_geopolitical", {}))
        if category == "domestic_india":
            return _api_news(c.get("domestic_india", {}))
        if category == "research":
            r   = c.get("research", {})
            out = _arxiv(r.get("arxiv_query", "all:manufacturing"))
            out += _crossref(r.get("crossref_query", "tool wear"),
                             r.get("crossref_api_key", ""))
            return out[:_N + 2]
        if category == "manufacturing":
            out = []
            for u in c.get("manufacturing_rss", []):
                out += _rss_titles(u, 2)
            if not out:   # publisher RSS moved/blocked — Google News fallback
                out = _rss_titles(
                    "https://news.google.com/rss/search?q=manufacturing+industry"
                    "&hl=en-US&gl=US&ceid=US:en", _N)
            return out[:_N]
        if category == "machining":
            out = []
            for u in c.get("machining_rss", []):
                out += _rss_titles(u, _N)
            if not out:
                out = _rss_titles(
                    "https://news.google.com/rss/search?q=CNC+machining+cutting+tools"
                    "&hl=en-US&gl=US&ceid=US:en", _N)
            return out[:_N]
    except Exception as e:
        log.warning("fetch(%s) failed: %s", category, e)
    return []


def fetch_all() -> dict:
    """All categories. Keyed APIs silently skipped if no key configured."""
    return {cat: fetch(cat) for cat in (
        "global_geopolitical", "domestic_india", "research",
        "manufacturing", "machining")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for cat, titles in fetch_all().items():
        print(f"\n[{cat}]")
        for t in titles:
            print("  •", t[:90])
