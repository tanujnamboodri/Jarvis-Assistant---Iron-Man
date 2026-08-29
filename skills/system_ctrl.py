"""
JARVIS — Packet C: System Control
==================================
Controls apps, folders, volume, brightness, Wi-Fi, shutdown, and screenshot.

Exposes a SINGLE handler (SystemControlSkill.handle) to the Brain's registry.
Internally dispatches to sub-handlers via ordered regex matching.

------------------------------------------------------------------------
Setup (Windows):
    pip install psutil screen-brightness-control pycaw comtypes

Brightness note:
    Works automatically on laptops. Desktop external monitors require
    DDC/CI to be enabled in the monitor's OSD menu. If get_brightness()
    fails, screen_brightness_control will raise — we catch and explain.

Wi-Fi note:
    netsh adapter control requires Administrator privileges.
    The adapter name defaults to "Wi-Fi". If it differs on your machine,
    check with:  netsh interface show interface
    Then set:    SystemControlSkill.WIFI_ADAPTER = "Your Adapter Name"
------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import sys
import logging
import subprocess
import threading
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import IntentResult from contracts.py (Packet 0).
# Falls back to a minimal inline definition if contracts.py isn't on the path
# yet — so this file is testable in isolation.
# ---------------------------------------------------------------------------
try:
    from contracts import IntentResult  # type: ignore
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class IntentResult:  # type: ignore
        speak: Optional[str] = None
        handled: bool = True
        data: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Optional dependencies — each feature degrades gracefully if missing
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    log.warning("psutil not installed — close-app will be unavailable.")

try:
    import screen_brightness_control as sbc
    HAS_SBC = True
except ImportError:
    sbc = None
    HAS_SBC = False
    log.warning("screen_brightness_control not installed — brightness control unavailable.")

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAS_PYCAW = True
except ImportError:
    HAS_PYCAW = False
    log.warning("pycaw/comtypes not installed — volume control unavailable.")

# ---------------------------------------------------------------------------
# Voice-name → Windows executable map
# Covers the most common apps a user might say aloud.
# Values starting with "ms-" are protocol handlers (use os.startfile).
# ---------------------------------------------------------------------------
_APP_MAP: dict[str, str] = {
    # Built-in Windows
    "notepad":               "notepad",
    "calculator":            "calc",
    "paint":                 "mspaint",
    "task manager":          "taskmgr",
    "file explorer":         "explorer",
    "explorer":              "explorer",
    "control panel":         "control",
    "cmd":                   "cmd",
    "command prompt":        "cmd",
    "terminal":              "wt",
    "powershell":            "powershell",
    "snipping tool":         "SnippingTool",
    "settings":              "ms-settings:",
    # Browsers
    "chrome":                "chrome",
    "google chrome":         "chrome",
    "firefox":               "firefox",
    "edge":                  "msedge",
    # Microsoft Office / 365
    "word":                  "winword",
    "excel":                 "excel",
    "powerpoint":            "powerpnt",
    "outlook":               "outlook",
    "email":                 "outlook",
    "mail":                  "outlook",
    "teams":                 "teams",
    "onenote":               "onenote",
    # Common third-party
    "spotify":               "spotify",
    "discord":               "discord",
    "slack":                 "slack",
    "zoom":                  "zoom",
    "vscode":                "code",
    "vs code":               "code",
    "visual studio code":    "code",
    "steam":                 "steam",
    "vlc":                   "vlc",
}

# JARVIS internal view / feature names — never run these as executables.
# When the user says "open schedule" or "open dashboard", system_ctrl catches
# "open" first. We decline (handled=False) so brain.py falls through to the
# schedule skill or the builtins view-switcher instead.
_JARVIS_VIEWS: frozenset = frozenset({
    "schedule", "schedule planner", "my schedule",
    "dashboard", "news", "news dashboard",
    "research", "research assistant", "research view",
    "globe", "home", "intel", "intel feed",
})


class SystemControlSkill:
    """All Windows system-control actions in one place.

    Parameters
    ----------
    dry_run : If True, actions are logged but never executed — safe for tests.
    wifi_adapter : Name of the Wi-Fi adapter as shown by `netsh interface show
                   interface`. Defaults to "Wi-Fi" (correct for most machines).
    """

    # Try these adapter names in order if the first fails.
    _WIFI_FALLBACKS = ["Wi-Fi", "Wireless Network Connection", "WiFi", "WLAN"]

    def __init__(self, dry_run: bool = False,
                 wifi_adapter: Optional[str] = None) -> None:
        self.dry_run      = dry_run
        self.wifi_adapter = wifi_adapter or self._WIFI_FALLBACKS[0]

    # =========================================================================
    # Public entry point — the ONE handler exposed to the Brain's registry
    # =========================================================================
    def handle(self, query: str) -> IntentResult:
        """Dispatch a system-control query to the right sub-handler."""
        return self._dispatch(query.lower().strip(), query)

    # =========================================================================
    # Internal dispatcher  (ordered: most-specific patterns checked first)
    # =========================================================================
    def _dispatch(self, q: str, raw: str) -> IntentResult:
        # Cancel shutdown must come before the shutdown check.
        if re.search(r'\bcancel\s*(shutdown|restart)\b', q):
            return self._handle_cancel_shutdown(raw)
        if re.search(r'\b(shutdown|restart)\b', q):
            return self._handle_power(raw)
        if re.search(r'\bscreenshot\b', q):
            return self._handle_screenshot(raw)
        if re.search(r'\b(close|kill|stop)\b', q):
            return self._handle_close_app(raw)
        if re.search(r'\b(open folder|browse folder|browse)\b', q):
            return self._handle_open_folder(raw)
        if re.search(r'\b(open|launch|start)\b', q):
            return self._handle_open_app(raw)
        if re.search(r'\b(volume|sound|speaker|mute|unmute)\b', q):
            return self._handle_volume(raw)
        if re.search(r'\b(brightness|brighter|dimmer|dim)\b', q):
            return self._handle_brightness(raw)
        if re.search(r'\b(wifi|wi-fi|wireless|internet)\b', q):
            return self._handle_wifi(raw)

        return IntentResult(
            speak="I'm not sure which system action you meant, sir.",
            handled=False,
        )

    # =========================================================================
    # Sub-handlers
    # =========================================================================

    # --- Open application ----------------------------------------------------
    def _handle_open_app(self, text: str) -> IntentResult:
        m = re.search(r'\b(?:open|launch|start)\s+(.+)', text, re.IGNORECASE)
        if not m:
            return IntentResult(speak="I couldn't work out which app to open, sir.")

        spoken_name = m.group(1).strip().lower().rstrip(".")

        # Decline silently — let brain.py route to the schedule skill or
        # the builtins view-switcher. handled=False signals a miss so the
        # brain falls through without stopping here.
        if spoken_name in _JARVIS_VIEWS:
            log.debug("'%s' is a JARVIS view — declining system_ctrl", spoken_name)
            return IntentResult(speak="", handled=False)

        executable  = _APP_MAP.get(spoken_name, spoken_name)
        display     = spoken_name.title()

        def action() -> None:
            if executable.startswith("ms-"):
                # Protocol handler (e.g. ms-settings:) — must use startfile.
                os.startfile(executable)  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                # macOS: `open -a` resolves app names via LaunchServices.
                # List form = no shell, so transcribed speech can never be
                # interpreted as shell syntax ("open x; rm -rf ~" is inert).
                r = subprocess.run(["open", "-a", executable],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise FileNotFoundError(
                        f"No application named '{display}' found.")
                return
            try:
                # shell=False: explicit PATH search, no injection risk.
                subprocess.Popen([executable], shell=False)
            except FileNotFoundError:
                # Try with .exe extension as a fallback.
                try:
                    subprocess.Popen([executable + ".exe"], shell=False)
                except FileNotFoundError:
                    if sys.platform == "win32":
                        # Windows last resort: ShellExecute via startfile
                        # resolves registered apps WITHOUT invoking cmd.exe,
                        # so no metacharacter injection is possible.
                        os.startfile(executable)  # type: ignore[attr-defined]
                    else:
                        raise
                # SECURITY: the previous version fell back to
                # subprocess.Popen(executable, shell=True), which passed
                # raw transcribed/typed user text to /bin/sh on every
                # platform. That path is deliberately gone.

        return self._execute(
            action_name=f"Open {display}",
            action_fn=action,
            speak_ok=f"Opening {display}, sir.",
        )

    # --- Close application ---------------------------------------------------
    def _handle_close_app(self, text: str) -> IntentResult:
        if not HAS_PSUTIL:
            return IntentResult(
                speak="I can't close apps right now — psutil isn't installed, sir."
            )
        m = re.search(r'\b(?:close|kill|stop)\s+(.+)', text, re.IGNORECASE)
        if not m:
            return IntentResult(speak="I couldn't work out which app to close, sir.")

        target = m.group(1).strip().lower().rstrip(".")
        display = target.title()

        def action() -> None:
            hit = [
                p for p in psutil.process_iter(["pid", "name"])
                if p.info["name"] and target in p.info["name"].lower()
            ]
            if not hit:
                raise RuntimeError(f"No running process matched '{target}'.")
            for proc in hit:
                proc.terminate()
            # Give processes a moment; kill any that are still alive.
            _, alive = psutil.wait_procs(hit, timeout=3)
            for proc in alive:
                proc.kill()

        return self._execute(
            action_name=f"Close {display}",
            action_fn=action,
            speak_ok=f"Closed {display}, sir.",
        )

    # --- Open folder ---------------------------------------------------------
    def _handle_open_folder(self, text: str) -> IntentResult:
        m = re.search(r'\b(?:open folder|browse folder|browse)\s+(.+)',
                      text, re.IGNORECASE)
        if not m:
            return IntentResult(speak="Which folder would you like me to open, sir?",
                                handled=False)

        path = os.path.expandvars(os.path.expanduser(m.group(1).strip()))

        if not self.dry_run and not os.path.exists(path):
            return IntentResult(
                speak=f"I can't find the path {path}, sir. Please check it exists."
            )

        def action() -> None:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

        return self._execute(
            action_name=f"Open folder {path}",
            action_fn=action,
            speak_ok=f"Opened the folder, sir.",
        )

    # --- Volume --------------------------------------------------------------
    def _handle_volume(self, text: str) -> IntentResult:
        if not HAS_PYCAW:
            return IntentResult(
                speak="Volume control requires pycaw on Windows, sir. "
                      "Install it with: pip install pycaw comtypes"
            )
        q = text.lower()

        def action() -> None:
            devices   = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_,
                                         CLSCTX_ALL, None)
            vol = cast(interface, POINTER(IAudioEndpointVolume))

            if "mute" in q and "unmute" not in q:
                vol.SetMute(1, None)
                return
            if "unmute" in q:
                vol.SetMute(0, None)
                return

            current = vol.GetMasterVolumeLevelScalar()

            if "up" in q or "increase" in q or "louder" in q:
                target = min(current + 0.10, 1.0)
            elif "down" in q or "decrease" in q or "quieter" in q:
                target = max(current - 0.10, 0.0)
            else:
                num = re.search(r'(\d{1,3})', q)
                if num:
                    target = max(0.0, min(int(num.group(1)) / 100.0, 1.0))
                else:
                    raise ValueError("Couldn't determine the volume direction or level.")

            vol.SetMute(0, None)           # auto-unmute when adjusting
            vol.SetMasterVolumeLevelScalar(target, None)

        q_lower = text.lower()
        if "mute" in q_lower and "unmute" not in q_lower:
            verb = "Muted"
        elif "unmute" in q_lower:
            verb = "Unmuted"
        elif "up" in q_lower or "increase" in q_lower or "louder" in q_lower:
            verb = "Volume up"
        elif "down" in q_lower or "decrease" in q_lower or "quieter" in q_lower:
            verb = "Volume down"
        else:
            verb = "Volume set"

        return self._execute(
            action_name="Adjust volume",
            action_fn=action,
            speak_ok=f"{verb}, sir.",
        )

    # --- Brightness ----------------------------------------------------------
    def _handle_brightness(self, text: str) -> IntentResult:
        if not HAS_SBC:
            return IntentResult(
                speak="Brightness control isn't available, sir. "
                      "Install screen_brightness_control with pip."
            )
        q = text.lower()

        def action() -> None:
            try:
                levels = sbc.get_brightness()
                current = levels[0] if levels else 50
            except Exception:
                raise RuntimeError(
                    "Couldn't read the display brightness. "
                    "If this is a desktop, enable DDC/CI in your monitor's menu, sir."
                )

            if "increase" in q or "up" in q or "brighter" in q:
                target = min(current + 10, 100)
            elif "decrease" in q or "down" in q or "dimmer" in q or "dim" in q:
                target = max(current - 10, 0)
            else:
                num = re.search(r'(\d{1,3})', q)
                if num:
                    target = max(0, min(int(num.group(1)), 100))
                else:
                    raise ValueError("Couldn't determine a brightness level.")

            sbc.set_brightness(target)

        return self._execute(
            action_name="Adjust brightness",
            action_fn=action,
            speak_ok="Brightness adjusted, sir.",
        )

    # --- Wi-Fi ---------------------------------------------------------------
    def _handle_wifi(self, text: str) -> IntentResult:
        if sys.platform != "win32":
            return IntentResult(
                speak="Wi-Fi control via netsh is only available on Windows, sir."
            )
        q = text.lower()
        state   = "enabled" if any(w in q for w in ("on", "enable")) else "disabled"
        display = "on" if state == "enabled" else "off"

        def action() -> None:
            # FIX: shell=False + list — the correct form on all platforms.
            # (The original used shell=True + list, which silently drops args on Windows.)
            adapter = self.wifi_adapter
            for candidate in [adapter] + [a for a in self._WIFI_FALLBACKS
                                           if a != adapter]:
                result = subprocess.run(
                    ["netsh", "interface", "set", "interface",
                     f"name={candidate}", f"admin={state}"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return   # success
            # All candidates failed — surface the last error.
            raise PermissionError(
                f"netsh failed. This usually means the process needs "
                f"Administrator rights, or the adapter name isn't one of "
                f"{self._WIFI_FALLBACKS}. Check with: netsh interface show interface"
            )

        return self._execute(
            action_name=f"Wi-Fi {display}",
            action_fn=action,
            speak_ok=f"Wi-Fi turned {display}, sir.",
        )

    # --- Shutdown / Restart --------------------------------------------------
    def _handle_power(self, text: str) -> IntentResult:
        q = text.lower()
        action_type = "restart" if "restart" in q else "shutdown"

        # Safety gate — require spoken confirmation.
        if "confirm" not in q:
            return IntentResult(
                speak=f"Just to confirm, sir — say '{action_type} confirm' "
                      f"and I'll proceed. You have 60 seconds to cancel "
                      f"by saying 'cancel {action_type}'.",
                handled=True,
            )

        def action() -> None:
            if sys.platform == "win32":
                flag = "/r" if action_type == "restart" else "/s"
                subprocess.Popen(
                    ["shutdown", flag, "/t", "60", "/c",
                     "JARVIS initiated — say cancel shutdown to abort."]
                )
            else:
                flag = "-r" if action_type == "restart" else "-h"
                subprocess.Popen(["sudo", "shutdown", flag, "+1"])

        verb = "Restarting" if action_type == "restart" else "Shutting down"
        return self._execute(
            action_name=action_type.title(),
            action_fn=action,
            speak_ok=f"{verb} in 60 seconds, sir. Say 'cancel {action_type}' to abort.",
        )

    # --- Cancel shutdown / restart -------------------------------------------
    def _handle_cancel_shutdown(self, text: str) -> IntentResult:
        def action() -> None:
            if sys.platform == "win32":
                subprocess.run(["shutdown", "/a"], check=True)
            else:
                subprocess.run(["sudo", "shutdown", "-c"], check=True)

        return self._execute(
            action_name="Cancel shutdown",
            action_fn=action,
            speak_ok="Shutdown cancelled, sir.",
        )

    # --- Screenshot ----------------------------------------------------------
    def _handle_screenshot(self, text: str) -> IntentResult:
        def action() -> None:
            try:
                import pyautogui  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "pyautogui isn't installed. Run: pip install pyautogui"
                )
            path = os.path.join(os.path.expanduser("~"), "Pictures",
                                "jarvis_screenshot.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pyautogui.screenshot().save(path)

        return self._execute(
            action_name="Screenshot",
            action_fn=action,
            speak_ok="Screenshot saved to your Pictures folder, sir.",
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================
    def _execute(self, action_name: str, action_fn, speak_ok: str) -> IntentResult:
        """Run action_fn; return a spoken IntentResult.  Honours dry_run."""
        if self.dry_run:
            log.info("[DRY RUN] Would execute: %s", action_name)
            return IntentResult(speak=f"[Dry run] {speak_ok}", handled=True)
        try:
            action_fn()
            return IntentResult(speak=speak_ok, handled=True)
        except Exception as exc:
            log.error("System control error (%s): %s", action_name, exc)
            # Speak the exception message — it's intentionally user-friendly.
            return IntentResult(speak=str(exc), handled=True)


# =============================================================================
# Public registration function — plugs into the Brain's SkillRegistry
# =============================================================================
def register(registry, dry_run: bool = False,
             wifi_adapter: Optional[str] = None) -> None:
    """Register the system-control skill with a contracts.SkillRegistry.

    Parameters
    ----------
    registry     : contracts.SkillRegistry instance.
    dry_run      : If True, no real system changes are made (safe for tests).
    wifi_adapter : Override the Wi-Fi adapter name (default: "Wi-Fi").
    """
    ctrl = SystemControlSkill(dry_run=dry_run, wifi_adapter=wifi_adapter)

    registry.register(
        name="system_control",
        keywords=[
            "open", "launch", "start",
            "close", "kill", "stop",
            "volume", "mute", "unmute", "louder", "quieter",
            "brightness", "brighter", "dimmer",
            "wifi", "wi-fi", "wireless",
            "shutdown", "restart",
            "screenshot",
            "open folder", "browse",
        ],
        handler=ctrl.handle,
        description=(
            "Controls Windows system settings: open/close apps, adjust volume "
            "and brightness, toggle Wi-Fi, take screenshots, shutdown or restart."
        ),
    )


# =============================================================================
# Isolation test suite  (python system_ctrl.py)
# =============================================================================
if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from contracts import SkillRegistry, IntentResult as _IR
    except ImportError:
        print("contracts.py not found — using inline IntentResult stub")
        SkillRegistry = None

    ok = True

    def check(label, cond, note=""):
        global ok
        tag = "PASS" if cond else "FAIL"
        print(f"[{tag}] {label}" + (f"  ({note})" if note else ""))
        ok = ok and bool(cond)

    ctrl = SystemControlSkill(dry_run=True)

    print("=== Dispatch routing ===")
    cases = [
        ("open notepad",          "open"),
        ("launch chrome",         "open"),
        ("close notepad",         "close"),
        ("volume up",             "volume"),
        ("increase brightness",   "brightness"),
        ("wi-fi off",             "wifi"),
        ("shutdown",              "shutdown"),
        ("shutdown confirm",      "shutdown"),
        ("cancel shutdown",       "cancel"),
        ("screenshot please",     "screenshot"),
        ("open folder ~/Desktop", "folder"),
    ]
    for cmd, tag in cases:
        r = ctrl.handle(cmd)
        check(f"'{cmd}' → handled", r.handled, r.speak)

    print("\n=== IntentResult contract ===")
    r = ctrl.handle("open notepad")
    check("result has .speak attribute",  hasattr(r, "speak"))
    check("result has .handled attribute", hasattr(r, "handled"))
    check("result does NOT have .success", not hasattr(r, "success"))
    check("result does NOT have .message", not hasattr(r, "message"))

    print("\n=== Safety gate: shutdown without confirm ===")
    r = ctrl.handle("shutdown")
    check("blocked without 'confirm'", "confirm" in r.speak.lower())

    print("\n=== Safety gate: shutdown with confirm ===")
    r = ctrl.handle("shutdown confirm")
    check("proceeds with 'confirm'", r.handled)
    check("mentions 60 seconds or cancel", any(w in r.speak.lower() for w in ("60", "cancel")))

    print("\n=== App map coverage ===")
    for voice_name, exe in [("chrome","chrome"), ("email","outlook"),
                              ("settings","ms-settings:"), ("vscode","code")]:
        check(f"APP_MAP['{voice_name}'] == '{exe}'", _APP_MAP.get(voice_name) == exe)

    print("\n=== Graceful degradation (missing libs) ===")
    r = SystemControlSkill(dry_run=False)._dispatch(
        "volume up", "volume up"
    )
    if not HAS_PYCAW:
        check("volume without pycaw → speaks explanation",
              r.speak is not None and "pycaw" in r.speak.lower())
    else:
        check("volume with pycaw → handled", r.handled)

    if not HAS_PSUTIL:
        r2 = ctrl._handle_close_app("close notepad")
        check("close without psutil → speaks explanation",
              r2.speak is not None and "psutil" in r2.speak.lower())

    print("\n=== SkillRegistry integration ===")
    if SkillRegistry:
        reg = SkillRegistry()
        register(reg, dry_run=True)
        check("registered name is 'system_control'",
              "system_control" in reg.names)
        skill = reg.get("system_control")
        r3 = skill.handler("open notepad")
        check("handler callable via registry returns IntentResult",
              hasattr(r3, "speak") and r3.handled)
        # Keyword matching via Brain's registry
        match = reg.match_keywords("open chrome")
        check("keyword 'open' routes to system_control",
              match is not None and match[0] == "system_control")
    else:
        print("[SKIP] SkillRegistry tests — contracts.py not found")

    print()
    print("ALL TESTS PASSED ✓" if ok else "SOME TESTS FAILED ✗")
    _sys.exit(0 if ok else 1)
