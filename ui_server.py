"""
JARVIS — ui_server.py   Browser UI Bridge
==========================================
Two local servers:
  HTTP  port 8766 — serves jarvis_ui.html  (fixes CORS/file:// issues)
  WS    port 8765 — streams state updates to the browser in real time

The file:// protocol blocks CDN scripts (Three.js) in most browsers.
Serving over HTTP fixes that and makes the WebSocket origin match.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

log = logging.getLogger(__name__)

HERE     = os.path.dirname(os.path.abspath(__file__))
_loop:    Optional[asyncio.AbstractEventLoop] = None
_clients: set  = set()
_ask_callback = None   # set by brain/main: called with (question_text) when UI sends a question

def set_ask_callback(fn):
    """Register a function to call when the browser submits a typed question."""
    global _ask_callback
    _ask_callback = fn
_started: bool = False


# ── HTTP server — serves jarvis_ui.html ──────────────────────────────────────
class _HTMLHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = os.path.join(HERE, "ui", "jarvis_ui.html")
        try:
            data = open(html, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, "jarvis_ui.html not found")

    def log_message(self, *args):
        pass  # suppress access log noise


def _start_http() -> None:
    try:
        HTTPServer(("localhost", 8766), _HTMLHandler).serve_forever()
    except OSError as exc:
        log.error("UI HTTP server failed (port 8766): %s", exc)


# ── WebSocket server — broadcasts state ──────────────────────────────────────
def _fetch_startup_data() -> dict:
    """Fetch news + weather + service health for the startup dashboard."""
    import xml.etree.ElementTree as ET
    result = {"weather": None, "news": [], "services": {
        "audio":   True,   # alive if server is running
        "mic":     True,
        "stt":     True,
        "ai_core": True,
        "internet": False,
        "weather_svc": False,
        "news_svc": False,
    }}
    try:
        import urllib.request as _ur, json as _js
        try:
            from config import STARTUP_LAT, STARTUP_LON
        except ImportError:
            # No config.py present, or these keys not set — falls back to
            # IP-based geolocation via a free, keyless lookup rather than
            # baking one person's city into a public template.
            STARTUP_LAT = STARTUP_LON = None
        if STARTUP_LAT is None or STARTUP_LON is None:
            try:
                with _ur.urlopen("http://ip-api.com/json/?fields=lat,lon", timeout=3) as _geo:
                    _loc = _js.loads(_geo.read())
                STARTUP_LAT, STARTUP_LON = _loc["lat"], _loc["lon"]
            except Exception:
                STARTUP_LAT, STARTUP_LON = 51.5074, -0.1278  # London — neutral default
        _url = ("https://api.open-meteo.com/v1/forecast"
                f"?latitude={STARTUP_LAT}&longitude={STARTUP_LON}"
                "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
                "&wind_speed_unit=kmh")
        with _ur.urlopen(_url, timeout=6) as _r:
            c = _js.loads(_r.read())["current"]
        codes = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                 45:"Foggy",48:"Foggy",51:"Light drizzle",61:"Light rain",
                 63:"Rain",71:"Light snow",80:"Showers",95:"Thunderstorm"}
        cond = codes.get(c["weather_code"], "Variable")
        result["weather"] = {
            "temp": round(c["temperature_2m"]),
            "humidity": c["relative_humidity_2m"],
            "wind": round(c["wind_speed_10m"]),
            "condition": cond,
        }
        result["services"]["internet"]    = True
        result["services"]["weather_svc"] = True
    except Exception as e:
        log.debug("Startup weather fetch failed: %s", e)

    # News feeds — fetch 3 headlines per category
    try:
        from config import DOMESTIC_NEWS_QUERY
    except ImportError:
        DOMESTIC_NEWS_QUERY = "world news"   # generic default for a fresh clone
    feeds = [
        ("domestic",   f"https://news.google.com/rss/search?q={DOMESTIC_NEWS_QUERY.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"),
        ("global",     "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en"),
        ("technology", "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"),
        ("science",    "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en"),
    ]
    try:
        import urllib.request as _ur2
        for feed_key, url in feeds:
            try:
                req = _ur2.Request(url, headers={"User-Agent":"Mozilla/5.0"})
                with _ur2.urlopen(req, timeout=6) as _r:
                    xml = ET.fromstring(_r.read())
                titles = [item.find("title").text for item in
                          xml.iter("item") if item.find("title") is not None][:3]
                if titles:
                    result["news"].append({"category": feed_key, "titles": titles})
                    result["services"]["news_svc"] = True
                    result["services"]["internet"] = True
            except Exception as e:
                log.debug("Feed %s failed: %s", feed_key, e)
    except Exception as e:
        log.debug("News fetch failed: %s", e)

    # Extended sources: API news (if keys configured), arXiv/Crossref, sector RSS
    try:
        import news_feeds
        extra_map = {
            "global_geopolitical": "geopolitics",
            "domestic_india":      "india",
            "research":            "research",
            "manufacturing":       "manufacturing",
            "machining":           "machining",
        }
        for src, key in extra_map.items():
            titles = news_feeds.fetch(src)
            if titles:
                result["news"].append({"category": key, "titles": titles})
                result["services"]["news_svc"] = True
        # Resolve latest YouTube video IDs (works even when channels aren't live)
        try:
            result["videos"] = news_feeds.channel_videos()
        except Exception as _ve:
            log.debug("channel_videos failed: %s", _ve)
            result["videos"] = {}
    except Exception as e:
        log.debug("news_feeds failed: %s", e)
    return result


async def _ws_handler(websocket) -> None:
    _clients.add(websocket)
    log.debug("Browser connected (%d total)", len(_clients))
    # Push startup dashboard data to this client
    import threading as _threading
    def _push_startup():
        import time as _time
        _time.sleep(0.8)   # let browser render first
        data = _fetch_startup_data()
        msg  = json.dumps({"state": "startup", "startup": data})
        import asyncio as _asyncio
        async def _s():
            try:
                await websocket.send(msg)
            except Exception:
                pass
        _asyncio.run_coroutine_threadsafe(_s(), _loop)
    _threading.Thread(target=_push_startup, daemon=True).start()
    try:
        # Send current idle state immediately on connect
        await websocket.send(json.dumps({"state": "idle", "text": ""}))
        # Listen for inbound messages (typed questions from the research panel)
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "ask" and msg.get("text"):
                q = msg["text"].strip()
                if q and _ask_callback:
                    # Run the (blocking) brain call in a thread so we don't
                    # block the event loop.
                    import threading as _th
                    _th.Thread(target=_ask_callback, args=(q,), daemon=True).start()
    finally:
        _clients.discard(websocket)


async def _ws_serve() -> None:
    import websockets  # noqa: PLC0415
    async with websockets.serve(_ws_handler, "localhost", 8765):
        await asyncio.Future()  # run until cancelled


# ── Public API ────────────────────────────────────────────────────────────────

def start(open_browser: bool = True) -> None:
    """Start both servers. Call once at JARVIS startup."""
    global _loop, _started
    if _started:
        return
    _started = True

    # 1. HTTP server (serves the HTML file)
    threading.Thread(target=_start_http, daemon=True, name="jarvis-http").start()

    # 2. WebSocket server
    try:
        import websockets  # noqa: F401 — check it's installed
        _loop = asyncio.new_event_loop()

        def _run_ws():
            asyncio.set_event_loop(_loop)
            _loop.run_until_complete(_ws_serve())

        threading.Thread(target=_run_ws, daemon=True, name="jarvis-ws").start()
        log.info("UI ready — http://localhost:8766")

    except ImportError:
        log.warning("websockets not installed. Run: pip install websockets")

    # 3. Open browser after a short delay (server needs a moment to start)
    if open_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open("http://localhost:8766/jarvis_ui.html")
        threading.Thread(target=_open, daemon=True).start()


def broadcast(state: str, text: str = "") -> None:
    """Send a state update to all connected browser tabs.
    Safe to call from any thread. No-op if no browser is connected."""
    if _loop is None or not _clients:
        return
    msg = json.dumps({"state": state, "text": str(text)})

    async def _send():
        for ws in list(_clients):
            try:
                await ws.send(msg)
            except Exception:
                _clients.discard(ws)

    asyncio.run_coroutine_threadsafe(_send(), _loop)
