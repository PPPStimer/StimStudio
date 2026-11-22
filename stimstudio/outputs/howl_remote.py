"""Remote control of the Howl Android app over its HTTP API.

Unlike the Coyotes, this does not stream pulses on a clock: an HTTP round trip
to a phone over Wi-Fi is far too slow and jittery for 40 Hz packets, and it is
not how Howl's API is shaped anyway. Instead the whole window is handed over
once as a `.hwl` blob (`/load_hwl`) and Howl plays it locally with its own
copy of the same conversion math `outputs/base.py` is ported from. From here
we only push power/mute/frequency-range/channel-swap live and poll `/status`
for playback position and any change made on the phone itself.

Endpoints and payloads are ported from the reference client in Howl's own
repo (`remote_control/howl_remote.html`, github.com/Amethyst-Sysadmin/Howl),
and the sink has been tested with the Howl app on a phone.

No soft-start ramp: Howl gets the file in one shot rather than frame by frame,
so there is nothing to ease in against, and the base class's `gain()` ramp is
deliberately left unused here.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import urllib.error
import urllib.request

from .base import (EV_ERROR, EV_POWER, EV_STOPPED, PULSE_RATE,
                   STATUS_CONNECTED, STATUS_CONNECTING, STATUS_DISCONNECTED,
                   FrequencyMap, OutputSink, encode_hwl)

DEFAULT_PORT = 4695
POLL_INTERVAL = 0.4      # power/mute push + player-status poll cadence
REQUEST_TIMEOUT = 3.0


def _http_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """The one seam a test can replace instead of touching a real socket."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    return json.loads(raw) if raw else {}


class HowlLink:
    """One background worker thread doing blocking HTTP calls, queued in order.

    Mirrors `ble.py`'s rule: an unimportant call is dropped rather than queued
    when one is already in flight — a backlog of stale status polls is worse
    than a gap. Power/mute pushes are already change-gated and infrequent, so
    they go through as "important" and are never dropped.
    """

    def __init__(self, on_error):
        self._on_error = on_error
        self.host = ""
        self.port = DEFAULT_PORT
        self.key = ""
        self._queue: "queue.Queue[tuple[str, dict, object] | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._busy = False
        self._lock = threading.Lock()

    def configure(self, host: str, port: int, key: str):
        self.host, self.port, self.key = host, port, key

    def call(self, path: str, body: dict | None = None, *, callback=None,
             important: bool = True) -> bool:
        """Queue one request. Returns False when an unimportant call was dropped."""
        if not self.host:
            return False
        with self._lock:
            if not important and self._busy:
                return False
            self._busy = True
        self._ensure_thread()
        self._queue.put((path, body or {}, callback))
        return True

    def close(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(1.0)
            self._thread = None

    # ── the worker ──────────────────────────────────────────────────────
    def _ensure_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="howl-link", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, body, callback = item
            try:
                result = self._request(path, body)
            except Exception as exc:                      # noqa: BLE001
                result = None
                self._on_error(f"{path}: {exc}")
            with self._lock:
                self._busy = False
            if callback is not None:
                callback(result)

    def _request(self, path: str, body: dict) -> dict:
        url = f"http://{self.host}:{self.port}{path}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        return _http_post(url, body, headers, REQUEST_TIMEOUT)


def local_subnets() -> list[str]:
    """Every /24 prefix ("192.168.1") this PC looks like it has an interface on.

    Two heuristics, unioned, since neither alone is reliable on a machine with
    more than one active adapter (Wi-Fi + Ethernet, a VPN, a virtual switch
    from Docker/VirtualBox/Hyper-V — all common):

    - The outbound-route guess: opening a UDP socket to a public address never
      actually sends a packet, it just asks the OS which local interface/IP
      routing would use for that destination — usually the "main" adapter,
      but a full-tunnel VPN can make that the *wrong* one.
    - Every address `gethostbyname_ex` reports for this host, which tends to
      also catch whichever other adapter is sitting on the LAN the phone is
      actually on.

    No extra dependency does real interface enumeration cross-platform, so
    this stays a best effort rather than a proper adapter list.
    """
    import socket
    prefixes: set[str] = set()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        prefixes.add(sock.getsockname()[0].rsplit(".", 1)[0])
    except OSError:
        pass
    finally:
        sock.close()

    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        addrs = []
    for ip in addrs:
        if not ip.startswith("127."):
            prefixes.add(ip.rsplit(".", 1)[0])

    return sorted(prefixes)


def local_subnet_hosts() -> list[str]:
    """Every candidate address across every /24 this PC appears to be on."""
    return [f"{prefix}.{i}" for prefix in local_subnets() for i in range(1, 255)]


def scan_for_howl(port: int = DEFAULT_PORT, timeout: float = 0.3,
                  max_workers: int = 64, hosts: list[str] | None = None) -> list[str]:
    """Probe candidate hosts' Howl port; return the ones that answer.

    A bare TCP connect is all this checks — good enough to shortlist the Host
    field without needing the right access key up front, and much cheaper
    than a full HTTP round trip against a few hundred addresses.
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor

    candidates = local_subnet_hosts() if hosts is None else hosts
    if not candidates:
        return []

    def probe(host: str) -> str | None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return host
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        found = [host for host in pool.map(probe, candidates) if host is not None]
    return sorted(found, key=lambda ip: tuple(int(part) for part in ip.split(".")))


class HowlRemoteSink(OutputSink):
    """Push the current window to Howl and drive its player remotely."""

    label      = "Howl (remote)"
    requires   = ()
    needs_link = True
    verified   = True        # tested with the Howl app on a phone
    supports_live = False    # Howl plays a whole file: no live monitoring

    # Not used for local byte conversion the way the Coyotes use it — Howl maps
    # frequency on its own end. Only its Hz bounds matter here, forwarded as-is
    # to `/set_freq_range` so the on-phone mapping matches what the panel shows.
    default_freq_map = FrequencyMap()

    def __init__(self, source, events):
        super().__init__(source, events)
        self._link = HowlLink(lambda msg: self._post(EV_ERROR, msg))
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()
        self._last_pushed: tuple[int, int, bool] | None = None
        self._last_freq_range: tuple[int, int] | None = None
        self._last_swap: bool | None = None
        self._device_position = 0.0
        self._device_duration = 0.0
        self._uploaded_window = None    # identity of the Window last sent to the phone

    # ── settings ────────────────────────────────────────────────────────
    def configure(self, settings: dict):
        host = str(settings.get("remote_host", "")).strip()
        port = int(settings.get("remote_port", DEFAULT_PORT) or DEFAULT_PORT)
        key = str(settings.get("remote_key", "")).strip()
        self._link.configure(host, port, key)

        lo = int(settings.get("freq_min", self.default_freq_map.min_hz))
        hi = int(settings.get("freq_max", self.default_freq_map.max_hz))
        freq_range = (lo, hi) if lo <= hi else (hi, lo)
        if freq_range != self._last_freq_range:
            sent = self._link.call("/set_freq_range",
                                   {"min": freq_range[0], "max": freq_range[1]}, important=True)
            if sent:
                self._last_freq_range = freq_range

        swap = bool(settings.get("swap_channels", False))
        if swap != self._last_swap:
            sent = self._link.call("/set_swap_channels", {"value": swap}, important=True)
            if sent:
                self._last_swap = swap

    # ── lifecycle ───────────────────────────────────────────────────────
    def connect(self):
        if not self._link.host:
            self._post(EV_ERROR, "set the Howl device's IP address first")
            return
        self._set_status(STATUS_CONNECTING)
        self._link.call("/status", {}, callback=self._on_connected, important=True)

    def _on_connected(self, result):
        if result is None:
            self._set_status(STATUS_DISCONNECTED)
            return
        self._set_status(STATUS_CONNECTED)
        self._start_poll()

    def disconnect(self):
        self.stop(panic=True)
        self._stop_poll()
        self._link.close()
        self._last_pushed = None
        self._last_freq_range = None
        self._last_swap = None
        self._set_status(STATUS_DISCONNECTED)

    def start(self, at_pulse: float = 0.0):
        if self._playing:
            return
        self._device_position = at_pulse / PULSE_RATE
        super().start(at_pulse)
        self._upload(at_pulse)

    def _upload(self, at_pulse: float):
        """Push the current window and resume at `at_pulse`.

        Called both from `start()` and, mid-playback, from the poll loop when
        an edit (a new render) swaps out the window for a new one — Howl has no
        incremental update, so a changed window means a fresh `/load_hwl`.
        """
        window = self.source.window
        self._uploaded_window = window
        payload = {
            "title": "StimStudio",
            "hwl": base64.b64encode(encode_hwl(window.pulses)).decode("ascii"),
            "play": False,
            "loop": window.loop,
        }
        self._link.call("/load_hwl", payload,
                        callback=lambda r: self._on_loaded(r, at_pulse), important=True)

    def _on_loaded(self, result, at_pulse: float):
        if result is None:
            self._playing = False
            self._post(EV_STOPPED, "load failed")
            return
        self._link.call("/start_player", {"from": at_pulse / PULSE_RATE}, important=True)

    def stop(self, *, panic: bool = False):
        if not self._playing:
            return
        self._playing = False
        self._link.call("/stop_player", {}, important=True)
        self._post(EV_STOPPED, "panic" if panic else "stopped")

    def seek(self, at_pulse: float):
        """Jump within the file already loaded on the phone — cheap, and
        unlike `_upload` it does not interrupt playback to get there."""
        if not self._playing:
            return
        seconds = at_pulse / PULSE_RATE
        self._device_position = seconds
        self._link.call("/seek", {"position": seconds}, important=True)

    def position(self) -> float:
        return self._device_position

    # ── live power / mute / status ─────────────────────────────────────
    def _start_poll(self):
        self._poll_stop.clear()
        thread = threading.Thread(target=self._poll_loop, name="howl-poll", daemon=True)
        self._poll_thread = thread
        thread.start()

    def _stop_poll(self):
        self._poll_stop.set()
        thread, self._poll_thread = self._poll_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(1.0)

    def _poll_loop(self):
        while not self._poll_stop.wait(POLL_INTERVAL):
            self._push_controls_if_changed()
            if not self._playing:
                continue
            if self.source.window is not self._uploaded_window:
                # An effects edit swapped the window under us — re-upload and
                # resume where the phone last reported it had gotten to.
                self._upload(self._device_position * PULSE_RATE)
            else:
                self._link.call("/status", {}, callback=self._on_status, important=False)

    def _push_controls_if_changed(self):
        controls = self.source.controls
        power = (controls.effective_a, controls.effective_b, controls.muted)
        if power == self._last_pushed:
            return
        self._last_pushed = power
        self._link.call("/set_power", {"power_a": power[0], "power_b": power[1]}, important=True)
        self._link.call("/set_mute", {"value": power[2]}, important=True)

    def _on_status(self, result):
        if not isinstance(result, dict):
            return
        options = result.get("options") or {}
        player = result.get("player") or {}
        power_a, power_b = options.get("power_a"), options.get("power_b")
        if power_a is not None and power_b is not None:
            self._post(EV_POWER, (power_a, power_b))
        if player.get("position") is not None:
            self._device_position = float(player["position"])
        if player.get("duration") is not None:
            self._device_duration = float(player["duration"])
        if self._playing and player.get("playing") is False:
            self._playing = False
            self._post(EV_STOPPED, "finished")
