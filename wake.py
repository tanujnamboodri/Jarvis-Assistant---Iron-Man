"""
JARVIS — wake.py  Wake-word engine ("Hey Jarvis")
==================================================
Replaces the clap detector with a real neural wake word, Siri-style:
the mic stream runs continuously at near-zero CPU cost, and the full
STT pipeline only engages after "hey jarvis" is detected.

Engine: openWakeWord (Apache-2.0, fully local, no account, no API key).
The pretrained "hey_jarvis" model ships with the package (0.4.x) or is
downloaded once (0.6.x) and runs offline forever after — consistent with
the paper's zero-cloud-dependency claim. Inference is a tiny ONNX model:
one 80 ms frame per prediction, comfortably real-time on the M1's CPU.

Install (Mac):
    pip install openwakeword onnxruntime

API compatibility note — VERIFIED against the real package:
  * openwakeword 0.4.x : Model(wakeword_model_paths=[...])   (tested end-to-end)
  * openwakeword 0.6.x : Model(wakeword_models=[...], inference_framework="onnx")
    0.6.x also needs a one-time openwakeword.utils.download_models()
    (internet once; offline after). [VERIFY on the Mac: which version pip
    resolves — 0.6.x should install there since the tflite pin is Linux-only.]

Integration (main.py) — replace the clap wake with:

    import wake
    ...
    # instead of: voice.wait_for_wake(text_mode=args.text)
    if args.text:
        voice.wait_for_wake(text_mode=True)      # Enter key, unchanged
    else:
        wake.wait_for_wake()                     # blocks until "hey jarvis"

Tuning:
  * THRESHOLD 0.5 is the openWakeWord default. Raise toward 0.7 if you get
    false wakes from the TV / your own speech; lower toward 0.35 if it
    misses you across the room. Tune empirically in your office —
    detection rates vary by accent, mic, and room, so don't trust any
    number you haven't measured yourself.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE  = 16_000
FRAME        = 1_280          # 80 ms — openWakeWord's expected hop size
THRESHOLD    = 0.5            # detection score threshold (0..1)
COOLDOWN_S   = 2.0            # ignore re-triggers right after a detection

try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    sd = None
    HAS_SD = False
    log.warning("sounddevice not available — wake word disabled.")

_model = None
_model_key: Optional[str] = None     # key inside predict() score dict


def _load_model():
    """Load the hey_jarvis model, handling both openwakeword API generations."""
    global _model, _model_key
    if _model is not None:
        return _model

    import openwakeword
    from openwakeword.model import Model

    # 0.6.x: models are downloaded, new kwarg names.
    try:
        try:
            from openwakeword.utils import download_models
            download_models(model_names=["hey_jarvis"])
        except Exception:
            pass  # 0.4.x bundles models in the wheel; nothing to download
        _model = Model(wakeword_models=["hey_jarvis"],
                       inference_framework="onnx")
        log.info("openWakeWord loaded (0.6.x API).")
    except TypeError:
        # 0.4.x: explicit model path, old kwarg name.  (Verified end-to-end.)
        path = openwakeword.models["hey_jarvis"]["model_path"]
        _model = Model(wakeword_model_paths=[path])
        log.info("openWakeWord loaded (0.4.x API).")

    # Discover the score-dict key ("hey_jarvis" or "hey_jarvis_v0.1")
    probe = _model.predict(np.zeros(FRAME, dtype=np.int16))
    _model_key = next((k for k in probe if "jarvis" in k.lower()),
                      next(iter(probe)))
    _model.reset()
    return _model


def wait_for_wake(threshold: float = THRESHOLD) -> bool:
    """Block until 'hey jarvis' is heard. Returns True on detection.

    Runs a persistent low-cost mic stream; suppresses detection while
    JARVIS itself is speaking (half-duplex guard) so it can't wake on
    its own voice saying 'Jarvis'.
    """
    if not HAS_SD:
        log.error("No audio input available.")
        return False

    model = _load_model()
    model.reset()

    # Optional half-duplex guard against self-triggering
    try:
        import jarvis_voice as _jv
        _speaking = _jv.is_speaking
    except Exception:
        _speaking = lambda: False

    log.info("Listening for wake word 'hey jarvis'…")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype=np.int16, blocksize=FRAME) as stream:
        while True:
            frame, _ = stream.read(FRAME)
            if _speaking():
                model.reset()          # don't accumulate JARVIS's own audio
                continue
            score = model.predict(frame[:, 0])[_model_key]
            if score >= threshold:
                log.info("Wake word detected (score %.2f).", score)
                model.reset()
                time.sleep(0.05)       # let the tail of the phrase pass
                return True


if __name__ == "__main__":
    # Live test:  python wake.py   → say "hey jarvis", watch it trigger.
    logging.basicConfig(level=logging.INFO)
    while True:
        wait_for_wake()
        print(">>> WAKE <<<")
        time.sleep(COOLDOWN_S)
