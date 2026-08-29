# JARVIS — a local, offline voice assistant for research & daily tasks

A fully local voice assistant inspired by Iron Man's JARVIS. No cloud APIs,
no accounts, no data leaving your machine. Built around a small (3B-parameter)
language model that runs on consumer laptop hardware (developed and tested
on an 8GB M1 MacBook Air), with a fine-tunable research-assistant mode for
domain-specific Q&A over your own PDF corpus.

This project underlies a research paper comparing LoRA fine-tuning vs. RAG
for small-model domain adaptation — see `research/` for the training and
evaluation scripts, if you want to reproduce or extend that work. You don't
need any of that to just use JARVIS as an assistant.

**Status:** actively developed hobby/research project, not a polished
product. Expect rough edges — see [Known Limitations](#known-limitations)
and please open issues.

---

## What it can do

- **Conversational assistant** — ask questions, get spoken (or typed) replies,
  streamed sentence-by-sentence so it starts talking before it's done thinking.
- **Wake word** — say "hey Jarvis" instead of pressing a button (via
  [openWakeWord](https://github.com/dscripka/openWakeWord), fully local).
- **System control** — volume, brightness, opening/closing apps — sandboxed to
  a fixed allowlist of actions, never passes raw text to a shell.
- **Schedule / reminders** — natural-language "remind me to X tomorrow at 3pm."
- **Web agent** — weather, news headlines, general web search — works with
  zero API keys (falls back to Open-Meteo / public RSS), or add keys for
  richer results.
- **Research assistant** — point it at a folder of PDFs; ask about them by
  voice, get answers grounded in retrieval (RAG) over your own corpus.
- **Smart home (optional)** — local Tuya/EMOS GoSmart bulb control, no cloud.

## What it deliberately does NOT do

- No file deletion or modification capability for the LLM — it can *open*
  applications from an allowlist, nothing more.
- No shell access from spoken/typed text under any code path.
- No telemetry, no analytics, no phone-home of any kind.

---

## Quick start

**Requirements:** Python 3.10+, [Ollama](https://ollama.com/download),
~4GB free RAM for the default model, a microphone (optional — text mode
needs nothing but a keyboard).

```bash
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # edit if you want non-default settings

ollama pull llama3.2:3b            # ~2GB download, one time

python check_setup.py              # tells you exactly what's missing, if anything
```

**Text mode (no microphone needed, good first run):**
```bash
python main.py --text
```

**Full voice mode:**
```bash
python main.py
```
Say "hey Jarvis," wait for the greeting, then talk normally. Say "bye" or
"go to sleep" to end a session (JARVIS keeps running, just stops listening
until the wake word again).

---

## Configuration

Everything configurable lives in `config.py`, populated from environment
variables — set them directly or put them in a `.env` file (see
`.env.example` for every option with comments). **A fresh clone with no
`.env` at all still runs** — every setting has a working default or falls
back to a free/keyless alternative.

The two you'll most likely want to change:

```bash
# .env
JARVIS_MODEL=llama3.2:3b       # any Ollama-pullable model
JARVIS_PAPERS_DIR=./papers     # your PDF corpus for the research assistant
```

---

## Research Assistant / RAG setup

1. Put PDFs in `papers/` (or wherever `JARVIS_PAPERS_DIR` points).
2. First launch builds an embedding index automatically (uses
   `nomic-embed-text` via Ollama — pull it once: `ollama pull nomic-embed-text`).
3. Ask questions naturally: *"Jarvis, what does the tool wear literature say
   about flank wear at high cutting speeds?"*

If you want to fine-tune your own domain model instead of (or in addition
to) RAG, see `research/` for the LoRA training pipeline used in the
accompanying paper. That's a separate, optional workflow — RAG alone works
out of the box with zero training.

---

## Wake word tuning

The bundled `hey_jarvis` model from openWakeWord works out of the box, but
detection sensitivity depends heavily on your microphone, room, and accent —
there is no universally correct threshold. If it's missing your wake word or
triggering on the TV, adjust:

```bash
# .env
JARVIS_WAKE_THRESHOLD=0.5   # lower = more sensitive (more false positives)
```

Test changes live with `python wake.py` — it prints a detection event each
time it hears the wake word, without launching the full assistant.

---

## Platform notes

| Feature | macOS | Windows | Linux |
|---|---|---|---|
| Text-to-speech | built-in `say` (no install) | `pyttsx3` (SAPI) | `pyttsx3` (may need `espeak`) |
| Volume control | `osascript` | `pycaw` | not yet implemented — PR welcome |
| Brightness control | `osascript` | `screen-brightness-control` | not yet implemented — PR welcome |
| App launch | `open -a` | `subprocess`/`startfile` | `subprocess` |

Developed and daily-driven on macOS (Apple Silicon). Windows paths are
implemented but less tested — please file issues with your exact error if
something breaks. Linux system-control is genuinely incomplete; conversation,
schedule, research assistant, and web agent all work fine there today.

## Desktop app packaging (macOS)

`JARVIS_app_launcher.sh` is a headless launcher script for wrapping this as
a `.app` bundle (no visible Terminal or Ollama GUI window). See comments in
that file for how to wire it into an Automator/Platypus-built `.app`.

---

## Known limitations

- **`builtins.py` isn't included yet** — a small skill module for simple
  time/date/greeting commands. `main.py` loads it if present and prints a
  one-line warning and continues normally if it's missing (nothing crashes).
  It's a good first-contribution PR if you want one: follow the `register(registry)`
  pattern in `schedule.py`.
- **`vision.py` and `email_notify.py` are documented but not built** (see
  `main.py`'s own module docstring, which calls them "STUB" packets). They
  aren't wired into skill loading at all — screenshot/OCR and email/calendar
  notification are not currently assistant capabilities.
- **CPU inference is slow, and that's expected.** A 3B model on a laptop CPU
  takes a few seconds per reply. This isn't a bug to be optimized away — it's
  the tradeoff for zero cloud dependency and zero API cost. If you want
  faster replies, use a smaller model (`phi3:mini`) or point Ollama at a GPU.
- **Linux system control is incomplete** (see table above) — contributions
  welcome.
- **Wake word false-positive rate varies by environment** and needs local
  tuning; there's no universal threshold that works for everyone.
- This is a personal research project's codebase made public, not a
  maintained product — response time on issues will vary.

## Troubleshooting

**`sounddevice` fails to install:** it needs the PortAudio system library.
- macOS: `brew install portaudio`
- Ubuntu/Debian: `sudo apt install portaudio19-dev`
- Windows: usually installs fine via pip alone; if not, use the prebuilt
  wheel from [PyPI](https://pypi.org/project/sounddevice/).

**"Ollama isn't running, sir":** start it with `ollama serve` in a terminal
(or launch the Ollama desktop app), then try again.

**Wake word never triggers:** confirm your mic is being read at all with
`python wake.py` (should print scores near 0 during silence — if it prints
nothing, sounddevice isn't seeing your microphone). Check OS mic permissions
for your terminal/Python.

---

## Contributing

Issues and PRs welcome. If you add a new skill module, follow the pattern in
`schedule.py` or `web_agent.py` — a `register(registry, **kwargs)` function
and an `IntentResult` return contract (see `contracts.py`).

## License

MIT — see `LICENSE`. Use it, fork it, build your own JARVIS.
