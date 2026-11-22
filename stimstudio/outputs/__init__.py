"""Live output: audio (continuous / wavelet), Howl remote, Intiface, a dry run.

Ported from HWL Viewer's ``outputs`` package (itself ported from Howl). The
Bluetooth Coyote sinks are intentionally not included: StimStudio reaches the
Coyote 3 through the Howl app.

Nothing here imports Qt, and ``sounddevice`` is imported lazily, so the package
stays importable (and testable) on a machine without it. A sink whose
dependency is missing is listed but disabled, with the reason as its tooltip.
"""

from __future__ import annotations

import importlib.util
import queue
from dataclasses import dataclass

from .base import (EV_BATTERY, EV_DEVICES, EV_ERROR, EV_FRAME, EV_POWER, EV_STATUS,
                   EV_STOPPED, MAX_PLAY_SECONDS, POWER_MAX, PULSE_INTERVAL,
                   PULSE_RATE, STATUS_CONNECTED, STATUS_CONNECTING,
                   STATUS_DISCONNECTED, ClockedSink, Controls, FrequencyMap, LiveInput,
                   OutputPulse, OutputSink, PulseClock, PulseSource, Window,
                   clamp, encode_hwl, format_frame)

__all__ = [
    "OUTPUTS", "OutputInfo", "available", "make", "info",
    "PulseSource", "OutputSink", "ClockedSink", "OutputPulse", "Controls", "LiveInput",
    "Window", "FrequencyMap", "PulseClock", "clamp", "format_frame", "encode_hwl",
    "PULSE_RATE", "PULSE_INTERVAL", "POWER_MAX", "MAX_PLAY_SECONDS",
    "EV_STATUS", "EV_BATTERY", "EV_POWER", "EV_ERROR", "EV_STOPPED", "EV_FRAME",
    "EV_DEVICES",
    "STATUS_DISCONNECTED", "STATUS_CONNECTING", "STATUS_CONNECTED",
]


@dataclass(frozen=True, slots=True)
class OutputInfo:
    """One entry in the device combo box."""

    key: str
    label: str
    module: str          # submodule of this package holding the sink
    attr: str            # class name inside that module
    requires: tuple[str, ...] = ()
    #: must be connected explicitly before Play (network); local sinks connect on Play
    explicit_connect: bool = False
    verified: bool = True       # False adds a "not tested on hardware" warning

    @property
    def missing(self) -> tuple[str, ...]:
        """Required modules that are not installed."""
        return tuple(m for m in self.requires if importlib.util.find_spec(m) is None)

    @property
    def unavailable_reason(self) -> str:
        missing = self.missing
        if not missing:
            return ""
        return f"needs {', '.join(missing)} — install it to enable this output"


# Order is the order shown in the panel; the dry run sits last as a fallback.
OUTPUTS: tuple[OutputInfo, ...] = (
    OutputInfo("howl", "Coyote 3 via Howl (remote)", "howl_remote", "HowlRemoteSink",
               explicit_connect=True),
    OutputInfo("continuous", "Audio (continuous)", "audio", "ContinuousSink",
               requires=("sounddevice",)),
    OutputInfo("wavelet", "Audio (wavelet)", "audio", "WaveletSink",
               requires=("sounddevice",)),
    OutputInfo("buttplug", "Intiface / Buttplug (vibrators)", "buttplug", "ButtplugSink",
               explicit_connect=True, verified=False),
    OutputInfo("dryrun", "Dry run (no device)", "dryrun", "DryRunSink"),
)

_BY_KEY = {i.key: i for i in OUTPUTS}


def info(key: str) -> OutputInfo:
    return _BY_KEY[key]


def available(key: str) -> bool:
    return not _BY_KEY[key].missing


def make(key: str, source: PulseSource,
         events: "queue.Queue[tuple[str, object]]") -> OutputSink:
    """Build the sink for *key*. Imports its backend only at this point."""
    i = _BY_KEY[key]
    if i.missing:
        raise RuntimeError(i.unavailable_reason)
    module = importlib.import_module(f"{__name__}.{i.module}")
    return getattr(module, i.attr)(source, events)
