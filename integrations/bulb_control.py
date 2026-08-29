"""
bulb_control.py - Local on/off control of an EMOS GoSmart (Tuya) Wi-Fi bulb.

EMOS GoSmart runs on the Tuya platform, so a Wi-Fi EMOS bulb speaks the standard
Tuya LAN protocol and can be driven with tinytuya - no app, no cloud, fully
offline once the local key has been extracted one time.

Install:
    pip install tinytuya

One-time values you must fill in (see setup notes in chat):
    DEVICE_ID  - Tuya device id     (from `python -m tinytuya wizard`)
    DEVICE_IP  - bulb's LAN IP       (from `python -m tinytuya scan`)
    LOCAL_KEY  - device local key    (from the wizard's devices.json)
    VERSION    - protocol version    (scan reports 3.3 / 3.4 / 3.5)

Keep LOCAL_KEY out of git. Load from env vars or an untracked file.
"""

import os

try:
    import tinytuya
    HAS_TINYTUYA = True
except ImportError:
    tinytuya = None
    HAS_TINYTUYA = False
    # Not fatal: main.py registers this as an optional skill and should
    # skip it entirely if HAS_TINYTUYA is False. `pip install tinytuya`
    # only if you actually have a Tuya/EMOS GoSmart bulb.

# --- config -- [VERIFY] fill in from the tinytuya scan/wizard for YOUR bulb ---
try:
    from config import BULB_ID as _CFG_ID, BULB_IP as _CFG_IP, \
                       BULB_KEY as _CFG_KEY, BULB_VERSION as _CFG_VER
except ImportError:
    _CFG_ID = _CFG_IP = _CFG_KEY = None
    _CFG_VER = 3.3

# Fully optional. Leave unset and JARVIS works normally — light_on/light_off
# just return False. Fill in via config.py or these env vars, whichever you
# already use elsewhere in your setup.
DEVICE_ID = os.environ.get("BULB_ID",      _CFG_ID or "")
DEVICE_IP = os.environ.get("BULB_IP",      _CFG_IP or "")
LOCAL_KEY = os.environ.get("BULB_KEY",     _CFG_KEY or "")
VERSION   = float(os.environ.get("BULB_VERSION", str(_CFG_VER)))
_CONFIGURED = HAS_TINYTUYA and bool(DEVICE_ID and DEVICE_IP and LOCAL_KEY)
# -----------------------------------------------------------------------------


def _bulb():
    """Build a device handle with a short timeout so an unreachable bulb fails
    fast (~4s) instead of blocking the caller for 40s+ on tinytuya defaults.
    No caching/persistent socket: a fresh connect per toggle is negligible on a
    LAN and avoids stale-socket and cache-invalidation pitfalls."""
    d = tinytuya.BulbDevice(DEVICE_ID, DEVICE_IP, LOCAL_KEY, version=VERSION)
    d.set_socketTimeout(2)
    d.set_socketRetryLimit(1)
    return d


def _ok(resp):
    """tinytuya does NOT raise on failure - it returns a dict like
    {'Error': 'Network Error: Unable to Connect', 'Err': '901'}.
    Treat the presence of an Error/Err key as failure."""
    return not (isinstance(resp, dict) and ("Error" in resp or "Err" in resp))


def light_on():
    if not _CONFIGURED:
        print("[bulb] not configured — set BULB_ID/BULB_IP/BULB_KEY to use this feature.")
        return False
    try:
        return _ok(_bulb().turn_on())
    except Exception as e:
        print(f"[bulb] on failed: {e}")
        return False


def light_off():
    if not _CONFIGURED:
        print("[bulb] not configured — set BULB_ID/BULB_IP/BULB_KEY to use this feature.")
        return False
    try:
        return _ok(_bulb().turn_off())
    except Exception as e:
        print(f"[bulb] off failed: {e}")
        return False


# Bonus, beyond the on/off ask: brightness/colour are one call away if you want them.
#   _bulb().set_brightness_percentage(70)   # 1-100
#   _bulb().set_colour(255, 120, 0)         # R,G,B
# (Each opens its own connection; fine for occasional use.)


if __name__ == "__main__":
    # manual test:  python bulb_control.py on   |   off   |   status
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "status"
    if arg == "on":
        print("on:", light_on())
    elif arg == "off":
        print("off:", light_off())
    else:
        try:
            print(_bulb().status())
        except Exception as e:
            print("status failed:", e)
