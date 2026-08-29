"""
JARVIS — stt.py  Speech-to-Text  (Packet A — faster-whisper)
=============================================================
Replaces the Google STT stub with fully offline, CPU-local transcription.

Architecture
------------
listen()
  └─ text mode  : input()  (keyboard, dev / tests)
  └─ audio mode : _capture_utterance()   ← VAD-based mic capture
                  └─ _transcribe()       ← faster-whisper inference

VAD (Voice Activity Detection) is done in TWO passes:
  Pass 1 — amplitude-based (_capture_utterance):
      Fast, zero-cost, done on the raw PCM stream.  Ends recording after
      SILENCE_AFTER seconds of quiet.  Same constants as jarvis_voice.py
      so the two modules are interchangeable without retuning.
  Pass 2 — Silero VAD (inside faster-whisper, vad_filter=True):
      A neural VAD model that strips silence fragments that slipped through
      pass 1 and suppresses hallucinations on near-silence input.

Model recommendation (CPU-only machine)
----------------------------------------
  tiny    — fastest  (~0.5 s/utterance), lower accuracy
  base    — good balance (~1–2 s),   recommended default  ← default here
  small   — better accuracy (~2–4 s), still feasible on CPU
  medium  — too slow on CPU for real-time use

First run downloads the model to ~/.cache/huggingface/hub/ (needs internet
once; fully offline after that).

Install
-------
  pip install faster-whisper

Public interface
----------------
  listen(text_mode=False) -> str | None
  transcribe_file(path)   -> str | None  (for WAV-based testing)
  configure(model_size, language, beam_size, compute_type)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Returned by listen() when the user double-claps to sleep
SLEEP_SIGNAL = "__sleep__"

# ---------------------------------------------------------------------------
# Optional runtime dependencies
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    sd = None
    HAS_SD = False
    log.warning("sounddevice not available — audio capture disabled.")

try:
    from faster_whisper import WhisperModel as _WhisperModel
    HAS_FW = True
except ImportError:
    _WhisperModel = None
    HAS_FW = False
    log.warning(
        "faster-whisper not installed. "
        "Run: pip install faster-whisper\n"
        "       --text mode will still work without it."
    )

# ---------------------------------------------------------------------------
# Tunable configuration  (override via configure() before first listen())
# ---------------------------------------------------------------------------
try:
    from config import STT_MODEL_SIZE as _DEFAULT_MODEL_SIZE
except ImportError:
    _DEFAULT_MODEL_SIZE = "base.en"   # config.py not present — safe fallback

_cfg: dict = {
    "model_size":   _DEFAULT_MODEL_SIZE,  # base.en = English-only, better accuracy on technical terms
    "language":     "en",     # fixed language = faster (skips 30-s detection)
    "beam_size":    5,         # 5 = noticeably better accuracy; still <1s on M1
    "compute_type": "int8",    # int8 = fastest on CPU; float32 if int8 fails
}

# ---------------------------------------------------------------------------
# VAD / capture constants — identical to jarvis_voice.py for interoperability
# ---------------------------------------------------------------------------
SAMPLE_RATE   = 16_000    # Hz — Whisper's native rate, no resampling needed
CHUNK_FRAMES  = 1_600     # 100 ms per chunk
SILENCE_DB    = -38.0     # dBFS threshold; below = silence
MAX_SECONDS   = 8.0       # hard cap on one utterance

# Sleep-by-clap is OFF by default: on a MacBook the microphone sits directly
# above the keyboard, so typing produces sharp transients that the detector
# reads as claps — two keystrokes inside the gap window = "double clap" =
# JARVIS goes to sleep mid-sentence. Saying "bye" still sleeps. Re-enable
# only if you use an external mic away from the keyboard.
SLEEP_CLAP_ENABLED = False
SILENCE_AFTER = 1.2       # seconds of trailing silence that ends utterance
                          # (was 2.0 — Silero VAD pass-2 + speech_pad_ms already
                          #  protect against mid-sentence cutoff, so the long
                          #  amplitude-level tail was pure added latency)
VOICE_BEFORE  = 0.3       # minimum seconds of speech before recording starts

# ---------------------------------------------------------------------------
# Model — lazy-loaded once on first audio call
# ---------------------------------------------------------------------------
_model: Optional[object] = None
_model_lock = threading.Lock()


def configure(
    model_size:   Optional[str] = None,
    language:     Optional[str] = None,
    beam_size:    Optional[int] = None,
    compute_type: Optional[str] = None,
) -> None:
    """Override model config. Must be called before the first listen() call."""
    global _model
    if _model is not None:
        log.warning("configure() called after model was already loaded — ignored.")
        return
    if model_size   is not None: _cfg["model_size"]   = model_size
    if language     is not None: _cfg["language"]     = language
    if beam_size    is not None: _cfg["beam_size"]    = beam_size
    if compute_type is not None: _cfg["compute_type"] = compute_type


def _get_model() -> Optional[object]:
    """Return the cached WhisperModel, loading it on first call."""
    global _model
    if not HAS_FW:
        return None
    if _model is None:
        with _model_lock:
            if _model is None:
                logging.getLogger("faster_whisper").setLevel(logging.WARNING)
                size = _cfg["model_size"]
                size_hint = {"tiny": "~75 MB", "base": "~145 MB",
                             "small": "~465 MB"}.get(size, "")
                log.info("Loading faster-whisper '%s' %s…", size, size_hint)
                _model = _WhisperModel(
                    size,
                    device="cpu",
                    compute_type=_cfg["compute_type"],
                    cpu_threads=0,   # 0 = all available cores via OpenMP
                )
                log.info("Whisper model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _rms_dbfs(chunk: np.ndarray) -> float:
    """RMS level of an int16 chunk in dBFS.  -96 dBFS = silence."""
    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
    return -96.0 if rms == 0 else 20.0 * np.log10(rms / 32768.0)


def _capture_utterance() -> Optional[np.ndarray]:
    """Capture one spoken utterance from the microphone.

    Returns an int16 numpy array at SAMPLE_RATE Hz, or None on
    silence / hardware error.
    """
    if not HAS_SD:
        log.error("sounddevice not available — cannot capture audio.")
        return None

    frames_buffer: list[np.ndarray] = []
    voiced_frames: list[np.ndarray] = []
    silence_chunks = 0
    speech_chunks  = 0
    recording_live = False

    silence_limit = int(SILENCE_AFTER / (CHUNK_FRAMES / SAMPLE_RATE))
    voice_min     = int(VOICE_BEFORE  / (CHUNK_FRAMES / SAMPLE_RATE))
    max_chunks    = int(MAX_SECONDS   / (CHUNK_FRAMES / SAMPLE_RATE))

    log.debug("Listening…")

    # Run a second ClapDetector on the same stream for sleep-clap detection.
    # Disabled by default (SLEEP_CLAP_ENABLED) — keyboard typing near the
    # built-in mic false-triggers it. Say "bye" to sleep instead.
    sleep_det = None
    if SLEEP_CLAP_ENABLED:
        try:
            from jarvis_voice import ClapDetector as _CD  # type: ignore
            sleep_det = _CD(SAMPLE_RATE, CHUNK_FRAMES, mode="double",
                            peak_threshold=0.65)
        except Exception:
            sleep_det = None

    # Half-duplex guard: do not open the mic while JARVIS is still speaking,
    # otherwise the mic captures JARVIS's own voice from the speakers.
    try:
        import jarvis_voice as _jv
        import time as _t
        _waited = 0.0
        while _jv.is_speaking() and _waited < 15.0:
            _t.sleep(0.1); _waited += 0.1
        _t.sleep(0.35)  # let speaker tail flush before opening mic
    except Exception:
        pass

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.int16, blocksize=CHUNK_FRAMES,
        ) as stream:
            # Discard the first 3 chunks (0.3 s) as mic warmup.
            # (Was 20 chunks = 2.0 s — a fixed 2-second penalty added to
            #  EVERY turn of conversation. CoreAudio streams settle well
            #  within 300 ms; raise this only if you hear onset clicks.)
            for _ in range(3):
                stream.read(CHUNK_FRAMES)

            for _ in range(max_chunks):
                chunk, _ = stream.read(CHUNK_FRAMES)
                chunk = chunk[:, 0]                      # stereo → mono

                # If JARVIS started speaking, abandon this capture (half-duplex)
                try:
                    import jarvis_voice as _jv2
                    if _jv2.is_speaking():
                        return None
                except Exception:
                    pass

                # Check for sleep clap before running VAD
                if sleep_det and sleep_det.feed(chunk):
                    log.debug("Sleep clap detected.")
                    return SLEEP_SIGNAL

                is_speech = _rms_dbfs(chunk) > SILENCE_DB

                if is_speech:
                    speech_chunks  += 1
                    silence_chunks  = 0
                    frames_buffer.append(chunk)
                    if speech_chunks >= voice_min and not recording_live:
                        recording_live = True
                        log.debug("Speech onset — recording.")
                else:
                    silence_chunks += 1
                    frames_buffer.append(chunk)           # keep trailing silence

                if recording_live:
                    voiced_frames = list(frames_buffer)  # full snapshot
                    if silence_chunks >= silence_limit:
                        log.debug("End of utterance.")
                        break

    except Exception as exc:
        log.error("Microphone capture error: %s", exc)
        return None

    if not voiced_frames or speech_chunks < voice_min:
        return None      # only silence or too short

    return np.concatenate(voiced_frames)   # int16, shape (N,)


def _transcribe(audio_int16: np.ndarray,
                model: Optional[object] = None) -> Optional[str]:
    """Transcribe int16 audio to text using faster-whisper.

    The `model` parameter is exposed for testing (inject a mock).
    In production, pass None and the cached model is used.
    """
    m = model if model is not None else _get_model()
    if m is None:
        log.error("faster-whisper model not available.")
        return None

    # Convert int16 PCM → float32 in [-1, 1] — the format Whisper expects.
    # This matches what faster_whisper.audio.decode_audio() produces.
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    try:
        segments, _info = m.transcribe(
            audio_f32,
            language         = _cfg["language"],
            beam_size        = _cfg["beam_size"],
            temperature      = 0.0,        # greedy = deterministic + fastest on CPU
            initial_prompt   = ("JARVIS voice assistant. Tool wear, machining, "
                                "cutting speed, carbide, weather, news, papers, "
                                "research, Budapest, Hungary."),
            vad_filter       = True,       # Silero VAD pass-2: remove silence gaps
            vad_parameters   = {
                "min_silence_duration_ms": 800,  # wait longer before cutting off
                "speech_pad_ms":           200,
            },
            without_timestamps      = True,   # skip timestamp computation = faster
            condition_on_previous_text = False,  # avoids repetition on short audio
        )
    except Exception as exc:
        log.error("Whisper model error: %s", exc)
        return None

    # segments is a lazy generator — exhaust it to get all text
    parts = [s.text.strip() for s in segments if s.text.strip()]
    if not parts:
        return None

    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def listen(text_mode: bool = False) -> Optional[str]:
    """Return the user's next spoken (or typed) command as lowercased text.

    Parameters
    ----------
    text_mode : keyboard input instead of microphone (dev / CI / --text flag)

    Returns None on silence / empty input / unrecoverable error.
    """
    if text_mode:
        try:
            raw = input("\n  You: ").strip()
            return raw.lower() if raw else None
        except EOFError:
            return "bye"
        except KeyboardInterrupt:
            return "bye"

    audio = _capture_utterance()
    if audio is None:
        return None
    if isinstance(audio, str):   # SLEEP_SIGNAL is a str; numpy arrays are not
        return SLEEP_SIGNAL

    log.debug("Captured %d samples (%.1f s) — transcribing…",
              len(audio), len(audio) / SAMPLE_RATE)
    try:
        return _transcribe(audio)
    except Exception as exc:
        log.error("Transcription error: %s", exc)
        return None


def transcribe_file(path: str) -> Optional[str]:
    """Transcribe an audio file (WAV, MP3, FLAC, …) from disk.

    Convenience function for WAV-based testing and offline benchmarking.
    Uses the same _transcribe() path as live audio.
    """
    if not HAS_FW:
        log.error("faster-whisper not installed.")
        return None
    model = _get_model()
    if model is None:
        return None

    from faster_whisper.audio import decode_audio
    audio_f32 = decode_audio(path, sampling_rate=SAMPLE_RATE)

    segments, _info = model.transcribe(
        audio_f32,
        language                    = _cfg["language"],
        beam_size                   = _cfg["beam_size"],
        temperature                 = 0.0,
        vad_filter                  = True,
        without_timestamps          = True,
        condition_on_previous_text  = False,
    )
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).lower() if parts else None
