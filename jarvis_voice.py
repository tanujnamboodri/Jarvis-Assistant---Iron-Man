"""
JARVIS — Voice Foundation  (Mac + Windows)
==========================================
TTS priority chain:
  1. Piper          — best quality, British male, fully offline
                      Mac: needs  brew install espeak-ng  first
                      Windows: works after pip install piper-tts
  2. macOS say      — built-in on EVERY Mac, zero install needed
                      Best British male voice: Daniel
                      Activate: just run on Mac — auto-detected
  3. pyttsx3        — Windows SAPI fallback, pip install pyttsx3
  4. Print-only     — last resort; at least you see what it says

Run  python jarvis_voice.py --diagnose  to find out which tier is active
and play a spoken test so you know audio routing is working.
"""

import datetime
import glob
import os
import platform
import queue
import subprocess
import sys
import threading

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except OSError:
    # PortAudio isn't installed on this machine. Only Piper playback and
    # --diagnose need sd; text-mode speech (macOS `say` / pyttsx3) never
    # touches it. Without this guard, --text mode couldn't even start on a
    # machine without PortAudio — defeating the whole point of text mode.
    sd = None
    HAS_SOUNDDEVICE = False

IS_MAC     = platform.system() == "Darwin"


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio from sr_in to sr_out.
    Priority: samplerate (best) → scipy (good) → numpy interp (fallback).
    All three produce clean output; samplerate avoids numpy/scipy conflicts."""
    if sr_in == sr_out:
        return audio
    ratio = sr_out / sr_in
    # Tier 1: samplerate (libsamplerate — pip install samplerate)
    try:
        import samplerate as _sr
        return _sr.resample(audio, ratio, "sinc_best").astype(np.float32)
    except Exception:
        pass
    # Tier 2: scipy polyphase (best quality when available)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_out, sr_in)
        return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        pass
    # Tier 3: numpy linear interpolation (always works)
    n_out = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio), n_out, endpoint=False),
        np.arange(len(audio)), audio,
    ).astype(np.float32)
IS_WINDOWS = platform.system() == "Windows"

# ===========================================================================
# Piper  (tier 1)
# ===========================================================================
try:
    from piper import PiperVoice, SynthesisConfig
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False

# ===========================================================================
# pyttsx3  (tier 3 — Windows SAPI / Mac NSS via pyttsx3)
# ===========================================================================
try:
    import pyttsx3 as _pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# ===========================================================================
# CONFIG
# ===========================================================================
HERE            = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR       = os.path.join(HERE, "voices")
PREFERRED_VOICE          = "jarvis-high"         # MCU JARVIS voice
PREFERRED_VOICE_FALLBACK = "en_GB-alan-medium"   # fallback if JARVIS not downloaded

SAMPLE_RATE    = 16_000
CHUNK_FRAMES   = 1_600

WAKE_MODE      = "double"
PEAK_THRESHOLD = 0.45
REFRACTORY_S   = 0.20
MIN_GAP_S      = 0.12
MAX_GAP_S      = 0.80

SPEECH_LENGTH_SCALE = 0.93  # tuned via voice_tuner.py

# Mac "say" voice — Daniel is a genuine British male voice on every Mac.
# Change to "Oliver" or "Rishi" if you prefer.
# Run  say -v '?'  in Terminal to see all available voices.
SAY_VOICE = "Daniel"


# ===========================================================================
# TTS — queue-fed background thread
# ===========================================================================
_tts_queue: queue.Queue = queue.Queue()
_say_proc = None            # currently-running `say` subprocess (for interruption)
_stop_flag = False          # set True to abort current + queued speech
_is_speaking = False        # True while audio is actively playing (half-duplex guard)

def is_speaking() -> bool:
    """True while JARVIS is actively producing speech audio."""
    return _is_speaking
_active_tier: str = "unknown"


def _find_voice_model() -> str:
    """Find the voice model. Priority: jarvis-high → jarvis-medium → alan → any .onnx"""
    for name in ["jarvis-high", "jarvis-medium", PREFERRED_VOICE, PREFERRED_VOICE_FALLBACK]:
        candidate = os.path.join(VOICE_DIR, name + ".onnx")
        if os.path.exists(candidate):
            return candidate
    found = sorted(glob.glob(os.path.join(VOICE_DIR, "*.onnx")))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"No .onnx voice found in {VOICE_DIR}\n"
        f"  Download JARVIS voice (MIT, sounds like the movie):\n"
        f"    curl -L https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/high/jarvis-high.onnx -o voices/jarvis-high.onnx\n"
        f"    curl -L https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/high/jarvis-high.onnx.json -o voices/jarvis-high.onnx.json\n"
        f"  Or use generic British voice: python -m piper.download_voices en_GB-alan-medium --download-dir voices"
    )


def _banner(kind: str, msg: str, extra: str = "") -> None:
    mark = "=" if kind == "ok" else "!"
    print(f"\n{mark*56}")
    print(f"  [Jarvis TTS] {msg}")
    if extra:
        print(f"  {extra}")
    print(f"{mark*56}\n")


def _tts_worker() -> None:
    global _active_tier

    # ── Mac fast path: skip Piper, use say -v Veena directly ─────────────
    if IS_MAC:
        _active_tier = "say"
        _banner("ok", f"macOS say -v {SAY_VOICE} ✓")
        import time as _time, subprocess as _sp
        while True:
            text = _tts_queue.get()
            if text is None:
                _tts_queue.task_done(); break
            _parts, _got = [text], 1
            _dl = _time.monotonic() + 0.005
            while _time.monotonic() < _dl:
                try:
                    nx = _tts_queue.get_nowait(); _got += 1
                    if nx is None: _tts_queue.task_done(); _got -= 1; break
                    _parts.append(nx)
                except Exception: break
            combined = "  ".join(_parts)
            # Sanitize before speaking — removes preambles, markdown, gibberish sources
            import re as _re
            clean = combined
            # 1. LLaMA special tokens: <|eot_id|> etc. -> gibberish buzz
            clean = _re.sub(r'<\|[^|>]{1,40}\|>', ' ', clean)
            clean = _re.sub(r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', ' ', clean)
            # 2. Web-agent preambles
            clean = _re.sub(r'Here\s+is\s+what\s+I\s+found[.,]*\s*', '', clean, flags=_re.IGNORECASE)
            clean = _re.sub(r"Here['\u2019]?s?\s+a\s+\d[^:]{0,60}:\s*", '', clean, flags=_re.IGNORECASE)
            # 3. Markdown / math
            clean = _re.sub(r'[*_`#{}|<>\\^~]', ' ', clean)
            clean = clean.replace('\u00b5','micro').replace('\u00b0',' degrees ')
            clean = clean.replace('\u03b1','alpha').replace('\u03b2','beta')
            clean = clean.replace('\u03b3','gamma').replace('\u03c3','sigma')
            clean = clean.replace('\u03bc','mu').replace('\u2026','.')
            # 4. Smart punctuation
            clean = clean.replace('\u201c','"').replace('\u201d','"')
            clean = clean.replace('\u2018',"'").replace('\u2019',"'")
            clean = clean.replace('\u2014',', ').replace('\u2013',' to ')
            # 5. ASCII only -- strips Devanagari, Hindi, Arabic, Cyrillic
            # Cut at first Devanagari/CJK/Arabic char — model has drifted to
            # another language; everything after is hallucinated drift.
            _drift = _re.search(r'[\u0900-\u097F\u4e00-\u9fff\u0600-\u06FF]', clean)
            if _drift:
                clean = clean[:_drift.start()].strip()
            clean = clean.encode('ascii','ignore').decode('ascii')
            # 6. Repetition spam
            clean = _re.sub(r'([.!?,\-]){2,}', r'\1', clean)
            clean = _re.sub(r'(\w)\1{4,}', r'\1\1', clean)
            # 7. Whitespace
            clean = _re.sub(r'\s{2,}', ' ', clean).strip()
            # 8. Skip if no real words (prevents noisy/silent bursts)
            if len([w for w in clean.split() if any(c.isalpha() for c in w)]) < 2:
                for _ in range(_got): _tts_queue.task_done()
                continue
            global _say_proc, _stop_flag, _is_speaking
            if _stop_flag:
                for _ in range(_got): _tts_queue.task_done()
                continue
            try:
                # Broadcast text to UI BEFORE speaking so text+audio start together
                try:
                    import ui_server as _ui; _ui.broadcast("speaking", clean)
                except Exception: pass
                _is_speaking = True
                _say_proc = _sp.Popen(["say", "-v", SAY_VOICE, clean],
                                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                _say_proc.wait()
                _say_proc = None
            except Exception as e:
                print(f"[Jarvis TTS] say error: {e}")
            finally:
                _is_speaking = False
                for _ in range(_got): _tts_queue.task_done()
        return

    # ── Tier 1: Piper (non-Mac) ──────────────────────────────────────────
    if HAS_PIPER:
        try:
            model_path = _find_voice_model()
            voice      = PiperVoice.load(model_path, use_cuda=False)
            # Try to build SynthesisConfig with noise_scale; fall back if not supported
            try:
                syn = SynthesisConfig(
                    length_scale=SPEECH_LENGTH_SCALE,
                    noise_scale=0.84,   # tuned via voice_tuner.py
                    volume=1.0,
                )
            except TypeError:
                syn = SynthesisConfig(
                    length_scale=SPEECH_LENGTH_SCALE,
                    volume=1.0,
                )
            sr         = voice.config.sample_rate
            _active_tier = "piper"

            try:
                out_name = sd.query_devices(sd.default.device[1])["name"]
            except Exception:
                out_name = "default"

            _banner("ok",
                    "Piper voice loaded ✓",
                    f"Voice={os.path.basename(model_path)}  Rate={sr}Hz  Out={out_name}")

            while True:
                text = _tts_queue.get()
                if text is None:
                    _tts_queue.task_done()
                    break
                # Drain window: collect phrases queued within 60ms and batch them.
                # IMPORTANT: track count — must call task_done() once per get() call.
                import time as _time
                _deadline = _time.monotonic() + 0.005  # 5ms — minimal drain, fast sentence transitions
                _parts = [text]
                _got   = 1        # already got one item above
                while _time.monotonic() < _deadline:
                    try:
                        _next = _tts_queue.get_nowait()
                        _got += 1
                        if _next is None:
                            _tts_queue.task_done()
                            _got -= 1   # None sentinel handled separately
                            break
                        _parts.append(_next)
                    except Exception:
                        break
                text = "  ".join(_parts)   # combine: double-space = natural Piper pause
                try:
                    chunks = list(voice.synthesize(text, syn_config=syn))
                    if chunks:
                        raw = b"".join(c.audio_int16_bytes for c in chunks)
                        if IS_MAC:
                            # afplay uses CoreAudio directly — handles sample rate
                            # conversion natively with zero crackling on Apple Silicon
                            import tempfile, wave as _wave
                            with tempfile.NamedTemporaryFile(
                                    suffix=".wav", delete=False) as _f:
                                _tmp = _f.name
                            # Normalise + fade-in/out to eliminate inter-phrase clicks
                            _arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                            # Remove DC offset
                            _arr -= np.mean(_arr)
                            # Normalise to 80% headroom
                            _peak = np.max(np.abs(_arr))
                            if _peak > 0:
                                                    _arr = _arr / _peak * 13050  # volume=0.45 (tuned)
                            # 8ms fade-in and fade-out — eliminates click at phrase boundaries
                            _fade = int(sr * 0.041)
                            if len(_arr) > _fade * 2:
                                _arr[:_fade]  *= np.linspace(0.0, 1.0, _fade)
                                _arr[-_fade:] *= np.linspace(1.0, 0.0, _fade)
                            _arr = _arr.astype(np.int16)
                            # Scale trailing silence: 235ms for batched (greeting), 60ms for single sentences
                            _trail_s = 0.235 if len(_parts) > 1 else 0.04
                            _silence = np.zeros(int(sr * _trail_s), dtype=np.int16)
                            with _wave.open(_tmp, "wb") as _wf:
                                _wf.setnchannels(1)
                                _wf.setsampwidth(2)   # int16
                                _wf.setframerate(sr)
                                _wf.writeframes(np.concatenate([_arr, _silence]).tobytes())
                            subprocess.run(["afplay", _tmp],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                            os.unlink(_tmp)
                        else:
                            # Windows / Linux — use sounddevice with resampling
                            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            try:
                                dev_sr = int(sd.query_devices(
                                    sd.default.device[1])["default_samplerate"])
                            except Exception:
                                dev_sr = 48000
                            if sr != dev_sr:
                                audio = _resample(audio, sr, dev_sr)
                            sd.play(audio, dev_sr)
                            sd.wait()
                except Exception as exc:
                    print(f"[Jarvis TTS] Piper playback error: {exc}")
                finally:
                    for _ in range(_got):   # one task_done per item consumed from queue
                        _tts_queue.task_done()
            return

        except Exception as exc:
            hint = ""
            if "espeak" in str(exc).lower():
                hint = "Fix: brew install espeak-ng   (then restart)"
            elif "No .onnx" in str(exc):
                hint = "Fix: python -m piper.download_voices en_GB-alan-medium --download-dir voices"
            _banner("err", f"Piper FAILED — {exc}", hint)

    # ── Tier 2: macOS say command ─────────────────────────────────────────
    # Every Mac ships with `say`. No installation required.
    # `say -v Daniel` gives a genuine British male voice.
    if IS_MAC:
        # Verify the voice exists
        try:
            check = subprocess.run(
                ["say", "-v", SAY_VOICE, ""],
                capture_output=True, timeout=3
            )
            voice_ok = check.returncode == 0
        except Exception:
            voice_ok = False

        if not voice_ok:
            # Fall back to system default voice if Daniel not found
            SAY_VOICE_USED = None
            _banner("ok", "Using macOS say (system default voice)")
        else:
            SAY_VOICE_USED = SAY_VOICE
            _banner("ok",
                    f"Using macOS say -v {SAY_VOICE} ✓",
                    "Tip: brew install espeak-ng → unlocks better Piper voice later")

        _active_tier = "mac_say"

        while True:
            text = _tts_queue.get()
            if text is None:
                _tts_queue.task_done()
                break
            try:
                cmd = ["say"]
                if SAY_VOICE_USED:
                    cmd += ["-v", SAY_VOICE_USED]
                cmd.append(text)
                subprocess.run(cmd, check=True)
            except Exception as exc:
                print(f"[Jarvis TTS] say error: {exc}")
            finally:
                _tts_queue.task_done()
        return

    # ── Tier 3: pyttsx3 (Windows SAPI) ───────────────────────────────────
    if HAS_PYTTSX3:
        try:
            engine = _pyttsx3.init()
            voices = engine.getProperty("voices")
            for v in voices:
                if any(kw in (v.id + v.name) for kw in ("GB", "United Kingdom", "English")):
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 155)
            engine.setProperty("volume", 1.0)
            _active_tier = "pyttsx3"
            _banner("ok", "Using pyttsx3 (Windows SAPI) ✓")

            while True:
                text = _tts_queue.get()
                if text is None:
                    _tts_queue.task_done()
                    break
                # Drain window: collect phrases queued within 60ms and batch them.
                # IMPORTANT: track count — must call task_done() once per get() call.
                import time as _time
                _deadline = _time.monotonic() + 0.005  # 5ms — minimal drain, fast sentence transitions
                _parts = [text]
                _got   = 1        # already got one item above
                while _time.monotonic() < _deadline:
                    try:
                        _next = _tts_queue.get_nowait()
                        _got += 1
                        if _next is None:
                            _tts_queue.task_done()
                            _got -= 1   # None sentinel handled separately
                            break
                        _parts.append(_next)
                    except Exception:
                        break
                text = "  ".join(_parts)   # combine: double-space = natural Piper pause
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"[Jarvis TTS] pyttsx3 error: {exc}")
                finally:
                    _tts_queue.task_done()
            return
        except Exception as exc:
            _banner("err", f"pyttsx3 failed: {exc}")

    # ── Tier 4: print-only ────────────────────────────────────────────────
    _active_tier = "silent"
    print("\n" + "!"*56)
    print("  [Jarvis TTS] No TTS engine could load.")
    if IS_MAC:
        print("  Mac fix options:")
        print("    A) Piper:  brew install espeak-ng  (then restart)")
        print("    B) pyttsx3: pip install pyttsx3")
    else:
        print("  Fix: pip install pyttsx3")
    print("!"*56 + "\n")

    while True:
        item = _tts_queue.get()
        _tts_queue.task_done()
        if item is None:
            break


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def stop() -> None:
    """Immediately stop current speech and clear the queue.
    Called when the user asks a new question while JARVIS is still speaking."""
    global _stop_flag, _say_proc, _is_speaking
    _stop_flag = True
    # Drain pending queue items
    try:
        while True:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
    except queue.Empty:
        pass
    # Kill the currently-speaking process
    if _say_proc is not None:
        try:
            _say_proc.terminate()
        except Exception:
            pass
    # Also kill any stray say processes (belt and suspenders)
    try:
        subprocess.run(["killall", "say"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass
    _is_speaking = False
    _stop_flag = False


def wait_for_audio_done() -> None:
    """Block until the queue drains AND the say process has fully exited.
    Prevents the mic from opening while the speaker is still playing JARVIS's
    own tail audio (which the mic would otherwise capture as input)."""
    import time as _t
    _tts_queue.join()
    # Wait for the running say process to fully exit
    for _ in range(60):  # up to ~6s
        if _say_proc is None:
            break
        _t.sleep(0.1)
    _t.sleep(0.35)  # let the audio device flush its tail


def speak(text: str) -> None:
    print(f"[Jarvis] {text}")
    _tts_queue.put(str(text))


def speak_wait(text: str) -> None:
    speak(text)
    _tts_queue.join()


# ===========================================================================
# DIAGNOSTICS  —  python jarvis_voice.py --diagnose
# ===========================================================================

def diagnose() -> None:
    import time
    time.sleep(0.3)   # let the TTS thread print its banner first
    sep = "=" * 56
    print(f"\n{sep}")
    print("  JARVIS — Diagnostics")
    print(f"  Platform : {platform.system()} {platform.mac_ver()[0] if IS_MAC else ''}")
    print(f"  Active TTS tier: {_active_tier}")
    print(sep)

    # Audio output devices
    print("\n--- Audio output devices ---")
    if not HAS_SOUNDDEVICE:
        print("  sounddevice/PortAudio not installed — device listing "
              "unavailable, but macOS `say` / pyttsx3 speech still works.")
    else:
        try:
            devices    = sd.query_devices()
            _, out_idx = sd.default.device
            for i, d in enumerate(devices):
                if d["max_output_channels"] > 0:
                    marker = " ← DEFAULT" if i == out_idx else ""
                    print(f"  {i:2d}: {d['name']}{marker}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Sound test (pure tone, no Piper)
    print("\n--- Sound test (440 Hz tone) ---")
    try:
        t    = np.linspace(0, 0.8, int(44100 * 0.8), dtype=np.float32)
        tone = 0.35 * np.sin(2 * np.pi * 440 * t)
        sd.play(tone, 44100)
        sd.wait()
        print("  Did you hear a beep?  (yes = sounddevice works)")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        if IS_MAC:
            print("  Try:  brew install portaudio  then  pip install sounddevice --force-reinstall")

    # Mac say test
    if IS_MAC:
        print("\n--- macOS say test ---")
        try:
            subprocess.run(["say", "-v", SAY_VOICE, "Hello sir, this is Jarvis."], check=True)
            print(f"  say -v {SAY_VOICE}: OK — did you hear it?")
        except Exception as exc:
            print(f"  say failed: {exc}")

    # Piper voice files
    print("\n--- Piper voice files ---")
    print(f"  Looking in: {VOICE_DIR}")
    onnx_files = sorted(glob.glob(os.path.join(VOICE_DIR, "*.onnx")))
    if onnx_files:
        for f in onnx_files:
            print(f"  ✓ {os.path.basename(f)}  ({os.path.getsize(f)//1_048_576} MB)")
    else:
        print("  ✗ No .onnx files — run:")
        print(f"      python -m piper.download_voices {PREFERRED_VOICE} --download-dir voices")
        if IS_MAC:
            print("      brew install espeak-ng")

    # espeak-ng check (Mac)
    if IS_MAC:
        print("\n--- espeak-ng (required by Piper on Mac) ---")
        try:
            r = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True)
            print(f"  ✓ {r.stdout.strip() or 'found'}")
        except FileNotFoundError:
            print("  ✗ Not installed — Piper will fail until you run:")
            print("      brew install espeak-ng")

    # Spoken test via active TTS
    print("\n--- Spoken greeting test (via active TTS) ---")
    speak_wait("Hello sir. Diagnostics complete. Jarvis is ready.")

    print(f"\n{sep}")
    print("  If you heard the greeting above, Jarvis is working.")
    print("  If not, check the device list — run with the right")
    print("  output device selected in System Settings → Sound.")
    print(f"{sep}\n")


# ===========================================================================
# CLAP WAKE DETECTION
# ===========================================================================
class ClapDetector:
    def __init__(self, sample_rate, chunk_frames, mode="double",
                 peak_threshold=0.45, refractory_s=0.20,
                 min_gap_s=0.12, max_gap_s=0.80):
        self.mode           = mode
        self.peak_threshold = peak_threshold
        chunk_s             = chunk_frames / sample_rate
        self.refractory     = int(round(refractory_s / chunk_s))
        self.min_gap        = int(round(min_gap_s    / chunk_s))
        self.max_gap        = int(round(max_gap_s    / chunk_s))
        self._prev_loud     = False
        self._cooldown      = 0
        self._first_onset   = None

    @staticmethod
    def _peak(chunk):
        if chunk.size == 0: return 0.0
        return float(np.max(np.abs(chunk.astype(np.float32)))) / 32768.0

    def feed(self, chunk):
        loud = self._peak(chunk) >= self.peak_threshold
        if self._cooldown > 0: self._cooldown -= 1
        if self._first_onset is not None:
            self._first_onset += 1
            if self._first_onset > self.max_gap:
                self._first_onset = None

        onset = loud and not self._prev_loud and self._cooldown == 0
        self._prev_loud = loud
        fired = False
        if onset:
            self._cooldown = self.refractory
            if self.mode == "single":
                fired = True
            elif self._first_onset is not None and self._first_onset >= self.min_gap:
                fired = True
                self._first_onset = None
            else:
                self._first_onset = 0
        return fired


def wait_for_clap() -> None:
    det = ClapDetector(SAMPLE_RATE, CHUNK_FRAMES, mode=WAKE_MODE,
                       peak_threshold=PEAK_THRESHOLD, refractory_s=REFRACTORY_S,
                       min_gap_s=MIN_GAP_S, max_gap_s=MAX_GAP_S)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype=np.int16, blocksize=CHUNK_FRAMES) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_FRAMES)
            if det.feed(chunk[:, 0]):
                return


# ===========================================================================
# GREETING
# ===========================================================================
def load_name() -> str:
    try:
        with open(os.path.join(HERE, "assistant_name.txt")) as fh:
            return fh.read().strip() or "Jarvis"
    except FileNotFoundError:
        return "Jarvis"


def greet() -> None:
    """Single greeting phrase reduces echo surface area — fewer words
    echoing in the room before the mic opens for the first command."""
    hour = datetime.datetime.now().hour
    if   4  <= hour < 12:  greeting = "Good morning, sir."
    elif 12 <= hour < 17:  greeting = "Good afternoon, sir."
    elif 17 <= hour < 24:  greeting = "Good evening, sir."
    else:                  greeting = "Welcome back, sir."
    speak_wait(f"{greeting} {load_name()} online.")


def on_wake() -> None:
    greet()


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    if "--diagnose" in sys.argv:
        diagnose()
        return

    clap_word = "twice" if WAKE_MODE == "double" else "once"
    speak_wait(f"Systems online. Clap {clap_word} to wake me.")
    try:
        while True:
            print(f"\n(idle — clap {clap_word} to wake)")
            wait_for_clap()
            on_wake()
    except KeyboardInterrupt:
        speak_wait("Going offline. Goodbye.")
    finally:
        _tts_queue.put(None)
        _tts_queue.join()


if __name__ == "__main__":
    main()
