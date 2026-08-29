"""
JARVIS — skills/web_agent.py  Web Agent  (Packet D)
====================================================
Search, read + summarise, weather, news, Wikipedia.
Returns natural spoken responses via IntentResult.speak.

All speak strings are TTS-safe:
  - No newlines, no bullet symbols, no raw URLs.
  - Formatted as natural spoken sentences.

Weather tiers (no key → key):
  1. Open-Meteo + Nominatim  — completely free, no API key ever required.
  2. OpenWeatherMap           — free tier (1 000 calls/day); set
                                OPENWEATHER_API_KEY if you want this.

News tiers:
  1. BBC / Reuters RSS        — completely free, no key, always works.
  2. NewsAPI                  — richer topic search; set NEWS_API_KEY.
                                ⚠️  Free tier is developer-only and
                                restricted — verify current terms at
                                newsapi.org before relying on it.

Search:
  1. DDG Instant Answer API   — official, free, factual queries.
  2. DDG Lite scrape          — fallback; fragile (HTML structure changes).
  Summarisation uses Ollama locally (no cloud, no key).

Install:
  pip install requests beautifulsoup4

Optional (already present from Packet B):
  pip install ollama
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import unquote

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError as _e:
    HAS_WEB_DEPS = False
    _WEB_DEP_ERROR = str(_e)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IntentResult — import from contracts; inline fallback for isolation testing
# ---------------------------------------------------------------------------
try:
    from contracts import IntentResult       # type: ignore
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class IntentResult:                      # type: ignore
        speak:   Optional[str] = None
        handled: bool = True
        data:    dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Optional Ollama (for summarisation)
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    _ollama    = None
    HAS_OLLAMA = False

# ---------------------------------------------------------------------------
# WMO weather code → human description
# ---------------------------------------------------------------------------
_WMO: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}

_BASE_RSS   = "https://news.google.com/rss"
# `when:2d` restricts results to the last 48h — without it Google News search
# feeds are relevance-ordered and happily return week-old stories first.
_SEARCH_RSS = "https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
_HU_RSS     = "https://news.google.com/rss/search?q={q}+when:2d&hl=en-HU&gl=HU&ceid=HU:en"

_RSS_FEEDS = {
    "top":         _BASE_RSS + "?hl=en-US&gl=US&ceid=US:en",
    "world":       _SEARCH_RSS.format(q="world+news"),
    "geopolitics": _SEARCH_RSS.format(q="geopolitics+international+relations+diplomacy"),
    "politics":    _SEARCH_RSS.format(q="politics+government+policy"),
    "domestic":    _HU_RSS.format(q="Hungary+domestic+news"),
    "hungary":     _HU_RSS.format(q="Hungary"),
    "economy":     _SEARCH_RSS.format(q="economy+economics+finance+markets"),
    "business":    _SEARCH_RSS.format(q="business"),
    "technology":  _SEARCH_RSS.format(q="technology"),
    "ai":          _SEARCH_RSS.format(q="artificial+intelligence+machine+learning"),
    "science":     _SEARCH_RSS.format(q="science+research"),
    "space":       _SEARCH_RSS.format(q="space+NASA+astronomy+SpaceX"),
    "health":      _SEARCH_RSS.format(q="health+medicine+medical"),
    "climate":     _SEARCH_RSS.format(q="climate+change+environment"),
    "energy":      _SEARCH_RSS.format(q="energy+oil+gas+renewable+nuclear"),
    "defense":     _SEARCH_RSS.format(q="military+defense+war+security"),
    "sports":      _SEARCH_RSS.format(q="sports+football+soccer+tennis"),
}

_TOPIC_MAP = {
    "geopolit":"geopolitics", "international relat":"geopolitics",
    "foreign policy":"geopolitics", "diplomacy":"geopolitics",
    "domestic":"domestic", "local news":"domestic",
    "hungary":"hungary", "hungarian":"hungary",
    "politic":"politics", "government":"politics",
    "econom":"economy", "financ":"economy", "market":"economy",
    "technolog":"technology", "artific":"ai", "machine learn":"ai",
    "scienc":"science", "space":"space", "nasa":"space", "astronom":"space",
    "health":"health", "medic":"health",
    "climat":"climate", "environment":"climate",
    "energy":"energy", "nuclear":"energy",
    "military":"defense", "defense":"defense", "defence":"defense", "conflict":"defense",
    "sport":"sports", "football":"sports", "soccer":"sports",
    "business":"business", "world":"world",
}


# ===========================================================================
# WebAgent
# ===========================================================================
class WebAgent:
    """Encapsulates all web-retrieval skills.

    Parameters
    ----------
    session         : injectable requests.Session (use for tests).
    ollama_model    : Ollama model for summarisation.
    weather_api_key : OpenWeatherMap key (optional; Open-Meteo used if absent).
    news_api_key    : NewsAPI key (optional; BBC RSS used if absent).
    """

    def __init__(
        self,
        session:         Optional[requests.Session] = None,
        ollama_model:    str  = "llama3.2:3b",
        weather_api_key: Optional[str] = None,
        news_api_key:    Optional[str] = None,
    ) -> None:
        self._session      = session or self._make_session()
        self._ollama_model = ollama_model
        self._weather_key  = weather_api_key or os.getenv("OPENWEATHER_API_KEY")
        self._news_key     = news_api_key    or os.getenv("NEWS_API_KEY")
        self._last_location: Optional[str] = None  # remember last weather city

    # =========================================================================
    # Public handlers — each registered as a skill with SkillRegistry
    # =========================================================================

    def handle_search(self, query: str) -> IntentResult:
        """Search online and return a spoken summary."""
        terms = self._parse_search_terms(query)
        try:
            # Pass 1: DDG Instant Answer (official API, factual queries)
            instant = self._ddg_instant(terms)
            if instant:
                return IntentResult(
                    speak=f"Here is what I found. {instant}",
                    data={"source": "ddg_instant", "terms": terms},
                )

            # Pass 2: scrape top URLs, read and summarise the first good page
            urls = self._ddg_search_urls(terms, n=3)
            if not urls:
                return IntentResult(speak="I couldn't find any results for that, sir.")

            for url in urls:
                text = self._read_webpage(url)
                if text and len(text) > 100:
                    summary = self._summarize(text, context=terms)
                    return IntentResult(
                        speak=f"Here is what I found. {summary}",
                        data={"source": url, "terms": terms},
                    )

            return IntentResult(speak="I found some links but couldn't read them, sir.")

        except Exception as exc:
            log.error("Search error: %s", exc)
            return IntentResult(speak="The search failed, sir. Please check your connection.")

    def handle_weather(self, query: str) -> IntentResult:
        """Current or tomorrow weather. Remembers last city for follow-ups."""
        location = self._parse_location(query)
        if location is None:
            if self._last_location:
                location = self._last_location
            else:
                return IntentResult(speak="Which city would you like the weather for, sir?")
        else:
            self._last_location = location

        q = query.lower()
        want_tomorrow = any(w in q for w in
                            ("tomorrow", "will it", "will be", "next day",
                             "how will", "how it will", "forecast"))
        try:
            lat, lon, city = self._geocode(location)
            if want_tomorrow:
                t_str, desc, city = self._weather_tomorrow(lat, lon, city)
                return IntentResult(
                    speak=f"Tomorrow in {city} expect {desc} with {t_str}, sir.",
                    data={"city": city},
                )
            temp, desc, city = self._weather_open_meteo(location)
            return IntentResult(
                speak=f"The weather in {city} is {desc} with a temperature of {temp:.0f} degrees Celsius, sir.",
                data={"city": city, "temp": temp},
            )
        except Exception as e1:
            log.warning("Open-Meteo failed (%s); trying OpenWeatherMap.", e1)

        if not self._weather_key:
            return IntentResult(speak="I couldn't fetch the weather, sir. Set OPENWEATHER_API_KEY or check your internet connection.")
        try:
            temp, desc, city = self._weather_openweathermap(location)
            return IntentResult(
                speak=f"The weather in {city} is {desc} with a temperature of {temp:.0f} degrees Celsius, sir.",
                data={"city": city, "temp": temp},
            )
        except Exception as exc:
            log.error("Weather lookup failed: %s", exc)
            return IntentResult(speak="I was unable to retrieve the weather right now, sir.")


    def handle_news(self, query: str) -> IntentResult:
        """Return top news headlines. Say 'more' or 'other' for next batch."""
        topic  = self._parse_news_topic(query)
        q_lo   = query.lower()
        offset = 5 if any(w in q_lo for w in
                          ("other", "more", "different", "else", "another")) else 0

        if topic.startswith("custom:"):
            feed_url = _SEARCH_RSS.format(q=topic[7:].replace(" ","+"))
            feed_key = "custom"
        else:
            feed_key = topic if topic in _RSS_FEEDS else "top"
            feed_url = _RSS_FEEDS.get(feed_key, _RSS_FEEDS["top"])

        try:
            all_titles = self._rss_headlines(feed_url, n=10)
            titles = all_titles[offset:offset + 5]
            if not titles:
                titles = all_titles[:5]
            if titles:
                joined = ". ".join(titles) + "."
                label = topic[7:] if topic.startswith("custom:") else ("today's top" if feed_key == "top" else feed_key)
                speak = f"Here are the latest {label} headlines. {joined}"
                return IntentResult(
                    speak=speak,
                    data={"source": "google_news", "feed": feed_key, "titles": titles},
                )
        except Exception as e1:
            log.warning("RSS failed (%s); trying NewsAPI.", e1)

        # Tier 2: NewsAPI — needs key
        if not self._news_key:
            return IntentResult(
                speak=(
                    "I couldn't fetch the news right now, sir. "
                    "Set NEWS_API_KEY or check your internet connection."
                )
            )
        try:
            titles = self._newsapi_headlines(topic, n=3)
            if not titles:
                return IntentResult(speak=f"No news found for {topic}, sir.")
            joined = ". ".join(titles) + "."
            speak  = f"Here are the latest headlines on {topic}. {joined}"
            return IntentResult(
                speak=speak,
                data={"source": "newsapi", "topic": topic, "titles": titles},
            )
        except Exception as exc:
            log.error("NewsAPI error: %s", exc)
            return IntentResult(speak="I was unable to fetch the news right now, sir.")

    def handle_wikipedia(self, query: str) -> IntentResult:
        """Look up a topic on Wikipedia and return a spoken summary."""
        terms = self._parse_search_terms(query)
        try:
            summary = self._wikipedia_summary(terms)
            if summary:
                return IntentResult(
                    speak=summary,
                    data={"source": "wikipedia", "topic": terms},
                )
            return IntentResult(
                speak=f"I couldn't find a Wikipedia entry for {terms}, sir.",
            )
        except Exception as exc:
            log.error("Wikipedia error: %s", exc)
            return IntentResult(
                speak="I had trouble reaching Wikipedia, sir."
            )

    # =========================================================================
    # Weather backends
    # =========================================================================

    def _weather_tomorrow(self, lat: float, lon: float, city: str) -> tuple[str, str, str]:
        """Fetch tomorrow's high/low temp and weather description."""
        resp = self._session.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                    "timezone": "auto", "forecast_days": 2},
            timeout=10,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        t_max = daily.get("temperature_2m_max", [None, None])[1]
        t_min = daily.get("temperature_2m_min", [None, None])[1]
        code  = daily.get("weathercode", [0, 0])[1]
        desc  = _WMO_CODES.get(int(code) if code else 0, "mixed conditions")
        t_str = (f"a high of {t_max:.0f} and a low of {t_min:.0f} degrees Celsius"
                 if t_max is not None and t_min is not None else "unknown temperature")
        return t_str, desc, city

    def _geocode(self, location: str) -> tuple[float, float, str]:
        """Nominatim geocoding (OpenStreetMap) — free, no key."""
        resp = self._session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"Location not found: {location!r}")
        lat  = float(data[0]["lat"])
        lon  = float(data[0]["lon"])
        # "Budapest, Pest megye, ..." → keep just "Budapest"
        city = data[0].get("display_name", location).split(",")[0].strip()
        return lat, lon, city

    def _weather_open_meteo(self, location: str) -> tuple[float, str, str]:
        """Open-Meteo weather — completely free, no API key."""
        lat, lon, city = self._geocode(location)
        resp = self._session.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current_weather": "true",
                "temperature_unit": "celsius",
            },
            timeout=10,
        )
        resp.raise_for_status()
        current     = resp.json()["current_weather"]
        temp        = float(current["temperature"])
        code        = int(current["weathercode"])
        description = _WMO.get(code, "variable conditions")
        return temp, description, city

    def _weather_openweathermap(self, location: str) -> tuple[float, str, str]:
        """OpenWeatherMap — needs OPENWEATHER_API_KEY (free tier: 1 000/day)."""
        resp = self._session.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": self._weather_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data        = resp.json()
        temp        = float(data["main"]["temp"])
        description = data["weather"][0]["description"]
        city        = data["name"]
        return temp, description, city

    # =========================================================================
    # Search backends
    # =========================================================================

    def _ddg_instant(self, terms: str) -> Optional[str]:
        """DDG Instant Answer API — official, free, factual queries only."""
        try:
            resp = self._session.get(
                "https://api.duckduckgo.com/",
                params={"q": terms, "format": "json", "no_html": "1",
                        "skip_disambig": "1"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("AbstractText"):
                return data["AbstractText"]
            # Related topic first entry as fallback
            topics = data.get("RelatedTopics", [])
            if topics and isinstance(topics[0], dict):
                return topics[0].get("Text") or None
        except Exception as exc:
            log.debug("DDG Instant Answer failed: %s", exc)
        return None

    def _ddg_search_urls(self, terms: str, n: int = 3) -> list[str]:
        """Scrape DDG Lite for result URLs — fragile fallback.

        ⚠️  DDG may change their HTML at any time, breaking this.
        """
        try:
            resp = self._session.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": terms},
                timeout=10,
            )
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                m = re.search(r"uddg=(https?[^&]+)", a["href"])
                if m:
                    links.append(unquote(m.group(1)))   # FIX: URL-decode
            return list(dict.fromkeys(links))[:n]       # dedup + limit
        except Exception as exc:
            log.debug("DDG Lite scrape failed: %s", exc)
            return []

    def _read_webpage(self, url: str) -> Optional[str]:
        """Fetch a URL and return clean body text (≤ 8 000 chars)."""
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form"]):
                tag.decompose()
            tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
            text = "\n".join(t.get_text(strip=True) for t in tags) \
                   if tags else soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text[:8000]
        except Exception as exc:
            log.debug("Read error for %s: %s", url, exc)
            return None

    # =========================================================================
    # News backends
    # =========================================================================

    def _rss_headlines(self, feed_url: str, n: int = 10) -> list[str]:
        """Parse an RSS feed and return up to n cleaned headline strings,
        newest first (Google News search feeds are relevance-ordered, so we
        sort on pubDate ourselves)."""
        resp = self._session.get(feed_url, timeout=10)
        resp.raise_for_status()
        root  = ET.fromstring(resp.content)

        from email.utils import parsedate_to_datetime
        dated = []
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            if not title:
                continue
            try:
                ts = parsedate_to_datetime(item.findtext("pubDate", ""))
            except Exception:
                ts = None
            dated.append((ts, title))

        # Newest first; undated items go last.
        import datetime as _dt
        _epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        dated.sort(key=lambda p: p[0] or _epoch, reverse=True)

        headlines, seen = [], set()
        for _, title in dated:
            # Strip " - Source Name" suffix from Google News titles
            title = re.sub(" - [^-]{2,50}$", "", title).strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                headlines.append(title)
            if len(headlines) >= n:
                break
        return headlines

    def _newsapi_headlines(self, topic: str, n: int = 5) -> list[str]:
        """NewsAPI headlines — requires NEWS_API_KEY."""
        is_top = topic.lower() in ("top", "headlines", "latest", "")
        url    = ("https://newsapi.org/v2/top-headlines"
                  if is_top else "https://newsapi.org/v2/everything")
        params = ({"country": "us", "apiKey": self._news_key}
                  if is_top else {"q": topic, "apiKey": self._news_key,
                                  "pageSize": n, "sortBy": "publishedAt"})
        resp     = self._session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles[:n] if a.get("title")]

    # =========================================================================
    # Wikipedia
    # =========================================================================

    def _wikipedia_summary(self, topic: str) -> Optional[str]:
        """Wikipedia REST API summary — free, no key required."""
        slug = topic.strip().replace(" ", "_")
        resp = self._session.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            timeout=10,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data    = resp.json()
        extract = data.get("extract", "").strip()
        # Keep to 2-3 sentences so it's speakable
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:3]) if sentences else None

    # =========================================================================
    # Summarisation
    # =========================================================================

    @staticmethod
    def _strip_llm_preamble(text: str) -> str:
        """Deterministically remove meta-preambles a small model adds despite
        instructions — e.g. 'Here's a summary of the text in 2-3 short spoken
        sentences: "..."'. Prompt-level prohibitions alone are NOT reliable
        on a 3B model; this enforcement layer is."""
        t = text.strip()
        # "Here's a summary of the text in 2-3 short spoken sentences:" etc.
        t = re.sub(r"^(sure|certainly|of course)[,.!]?\s*", "", t, flags=re.I)
        t = re.sub(r"^here(?:'s| is)\s+(?:a|the)?\s*"
                   r"(?:summary|answer|response|breakdown|analysis|overview)"
                   r"[^:.\"]{0,80}[:.]\s*", "", t, flags=re.I)
        t = re.sub(r"^in\s+\d+(?:\s*-\s*\d+)?\s+(?:short\s+)?(?:spoken\s+)?"
                   r"sentences?[:,]?\s*", "", t, flags=re.I)
        # Unwrap if the whole remaining reply is quoted
        t = t.strip()
        if len(t) > 2 and t[0] in "\"“" and t[-1] in "\"”":
            t = t[1:-1].strip()
        elif len(t) > 1 and t[0] in "\"“" and t.count('"') == 1:
            t = t[1:].strip()          # dangling opening quote only
        return t

    def _summarize(self, text: str, context: str = "") -> str:
        """Try Ollama for summarisation; fall back to extractive."""
        if HAS_OLLAMA and _ollama is not None:
            try:
                prompt = (
                    f"Summarise the key points of the text below so they can "
                    f"be spoken aloud. Two to three sentences. Begin directly "
                    f"with the substance — no preamble such as 'Here is a "
                    f"summary', and do not wrap the answer in quotation marks. "
                    f"Context: {context}\n\nText:\n{text[:3000]}"
                )
                resp = _ollama.chat(
                    model=self._ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={"num_predict": 120, "temperature": 0.3},
                )
                summary = self._strip_llm_preamble(
                    resp.message.content.strip())   # FIX: .message.content
                if summary:
                    return summary
            except Exception as exc:
                log.warning("Ollama summarisation failed: %s", exc)
        return self._extractive_summary(text, n=3)

    def _extractive_summary(self, text: str, n: int = 3) -> str:
        """Return the first n sentences of text — no external dependencies."""
        if not text:
            return "No content available."
        # Collapse all whitespace (including newlines from HTML joins) to a
        # single space before splitting on sentence boundaries.
        text      = re.sub(r"\s+", " ", text.strip())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chosen    = [s.strip() for s in sentences[:n] if s.strip()]
        suffix    = "..." if len(sentences) > n else ""
        return " ".join(chosen) + suffix

    # =========================================================================
    # Query parsing
    # =========================================================================

    _SEARCH_STRIP = re.compile(
        r"^(search\s+(for|about|on)?|look\s+up|find(\s+out)?|"
        r"google|tell\s+me\s+about|what\s+is|who\s+is)\s+",
        re.IGNORECASE,
    )
    _WEATHER_LOC = re.compile(
        r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z\s,]+?)(?:\s+(?:today|now|tonight|forecast))?$",
        re.IGNORECASE,
    )
    _WEATHER_STRIP = re.compile(
        r"\b(weather|forecast|temperature|how(?:\'s| is) the|what\'s the|"
        r"today|currently|right now)\b",
        re.IGNORECASE,
    )
    _NEWS_STRIP = re.compile(
        r"^(news|headlines|what\'s happening|latest news|read me the news|"
        r"any news)(\s+on|\s+about|\s+in)?\s*",
        re.IGNORECASE,
    )

    def _parse_search_terms(self, query: str) -> str:
        terms = self._SEARCH_STRIP.sub("", query).strip().rstrip("?.")
        return terms or query

    def _parse_location(self, query: str) -> Optional[str]:
        """Extract city from a weather query.
        Returns None if no city is found (caller uses last known location)."""
        q = query.strip().rstrip("?.,!")

        # Try "in/at/for/check CITY [today|tomorrow|now|tonight|this week]"
        m = re.search(
            r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z\s]+?)"
            r"(?:\s+(?:today|tomorrow|now|tonight|this week|next week|forecast))?\s*$",
            q, re.IGNORECASE,
        )
        if m:
            city = m.group(1).strip()
            if city and city.lower() not in self._LOCATION_BLOCKLIST:
                return city

        # Fallback: strip ALL question/filler words, use what remains as city
        stop = {
            "weather", "forecast", "temperature", "check",
            "how", "is", "are", "was", "will", "be", "it", "going",
            "what", "the", "any", "there", "rain", "raining", "snow",
            "today", "tomorrow", "now", "currently", "right", "please",
            "outside", "like", "degrees", "celsius", "fahrenheit",
            "cold", "hot", "warm", "sunny", "cloudy", "rainy",
            "tell", "me", "give", "show",
        }
        words = [w for w in q.split() if w.lower() not in stop]
        city  = " ".join(words).strip()
        return city if city else None   # None = no city found, use last known

    def _parse_news_topic(self, query: str) -> str:
        import re as _re; q = query.lower().strip()
        m = _re.search(r"news (?:about|on|regarding) (.+)", q)
        if m: return "custom:" + m.group(1).strip().rstrip("?.,") 
        best_key, best_len = "top", 0
        for kw, fk in _TOPIC_MAP.items():
            if kw in q and len(kw) > best_len: best_key, best_len = fk, len(kw)
        if best_len > 0: return best_key
        cleaned = self._NEWS_STRIP.sub("", query).strip().rstrip("?.")
        return cleaned if cleaned else "top"

    # =========================================================================
    # Session factory
    # =========================================================================

    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        return s


# =============================================================================
# register() — plugs WebAgent into the Brain's SkillRegistry
# =============================================================================

def register(registry, **kwargs) -> None:
    """Register all web-agent skills with a contracts.SkillRegistry."""
    if not HAS_WEB_DEPS:
        print(f"  [Web Agent] Missing dependencies: {_WEB_DEP_ERROR}")
        print("  [Web Agent] Fix: pip install requests beautifulsoup4")
        return
    agent = WebAgent(**{k: v for k, v in kwargs.items()
                        if k in ("session", "ollama_model",
                                 "weather_api_key", "news_api_key")})

    registry.register(
        name="web_search",
        keywords=["search", "look up", "find", "google", "who is", "what is",
                  "tell me about", "find out"],
        handler=agent.handle_search,
        description=(
            "Searches the web and summarises results. "
            "Use for factual questions, current events, or research."
        ),
    )
    registry.register(
        name="weather",
        keywords=[
            "weather", "forecast", "temperature", "rain", "sunny",
            "cold", "hot", "how's the weather", "what's the weather",
            # Tomorrow / forecast queries
            "tomorrow", "will it rain", "will it snow", "will it be",
            "what will be", "how will", "how it will",
            "temperature tomorrow", "weather tomorrow", "degrees",
        ],
        handler=agent.handle_weather,
        description="Current or tomorrow's weather for any city.",
    )
    registry.register(
        name="news",
        keywords=["news","headlines","latest","any news","current events",
                  "geopolitical","geopolitics","domestic news","hungarian news",
                  "politics news","economic news","technology news","science news",
                  "health news","climate news","sports news","defense news",
                  "space news","energy news","ai news","news about","news on"],
        handler=agent.handle_news,
        description="News by topic: geopolitics, domestic, tech, health, sports, etc.",
    )
    registry.register(
        name="wikipedia",
        keywords=["wikipedia", "wiki", "what is", "who is", "explain",
                  "tell me about", "definition of"],
        handler=agent.handle_wikipedia,
        description="Looks up a topic on Wikipedia.",
    )
