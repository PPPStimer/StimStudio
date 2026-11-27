"""Output panel: device choice and connection, power A/B with limits, device
settings (frequency range, sound card, wave shape, Howl address) and safety.

Adapted from HWL Viewer's OutputPanel; the transport moved to the toolbar.
Power always starts at 0 — it is deliberately not restored between sessions.
"""

from __future__ import annotations

import json
import queue
import threading

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox,
                             QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
                             QTableWidget, QVBoxLayout, QWidget)

from ... import outputs
from ...outputs import POWER_MAX, STATUS_CONNECTED, STATUS_CONNECTING, STATUS_DISCONNECTED
from ...outputs.buttplug import ASSIGN_LABELS, ASSIGNMENTS, DEFAULT_URL as BUTTPLUG_URL
from ...outputs.howl_remote import DEFAULT_PORT as HOWL_PORT
from .. import metrics, theme
from ..settings import app_settings

#: Which settings rows each output shows.
SETTINGS_ROWS = {
    "howl": ("remotehost", "remotekey", "remotescan", "freq", "swap"),
    "continuous": ("sounddev", "freq", "shape", "monitor"),
    "wavelet": ("sounddev", "freq", "shape", "carrier", "width", "fade", "monitor"),
    "dryrun": ("freq", "log"),
    "buttplug": ("bpurl", "bpdevices", "bphint"),
}


def _spin(lo, hi, value, slot, width=56) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setMinimumWidth(width)   # not fixed: grows to hold its longest value
    s.setKeyboardTracking(False)
    s.valueChanged.connect(slot)
    return s


def _dspin(lo, hi, value, step, slot) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(2)
    s.setValue(value)
    s.setMinimumWidth(64)
    s.setKeyboardTracking(False)
    s.valueChanged.connect(slot)
    return s


def _row(title: str, *widgets, stretch: bool = False) -> QWidget:
    holder = QWidget()
    box = QHBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(4)
    caption = QLabel(title)
    caption.setMinimumWidth(84)   # lines the rows up; longer text grows
    box.addWidget(caption)
    for w in widgets:
        box.addWidget(w, 1 if stretch else 0)
    if not stretch:
        box.addStretch()
    return holder


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {theme.BORDER}; max-height: 1px; border: none;")
    return line


class DevicePanel(QScrollArea):
    def __init__(self, playback, parent=None):
        super().__init__(parent)
        self.pb = playback
        self.settings = app_settings()
        self._suspend = 0
        self.setWidgetResizable(True)
        # a narrow dock (small screen, large display scale) scrolls rather than cuts
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QWidget()
        self.setWidget(body)
        self.box = QVBoxLayout(body)
        self.box.setContentsMargins(10, 8, 10, 10)
        self.box.setSpacing(6)
        self.rows: dict[str, QWidget] = {}

        self._build_device()
        self.box.addWidget(_rule())
        self._build_power()
        self.box.addWidget(_rule())
        self._build_settings()
        self.box.addWidget(_rule())
        self._build_safety()
        self._build_monitor()
        self.box.addStretch(1)

        self._scan_results: "queue.Queue[list[str]]" = queue.Queue()
        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(100)
        self._scan_timer.timeout.connect(self._poll_scan)

        playback.statusChanged.connect(self._set_status)
        playback.powerReported.connect(self._apply_reported_power)
        playback.logLine.connect(self.log.appendPlainText)
        playback.devicesReported.connect(self.set_bp_devices)
        playback.playingChanged.connect(lambda _p: self._sync_enabled())

        key = str(self.settings.value("output/device", "continuous"))
        idx = self.device.findData(key)
        if idx < 0 or not outputs.available(key):
            idx = next((i for i in range(self.device.count())
                        if outputs.available(self.device.itemData(i))), 0)
        self.device.setCurrentIndex(idx)
        self._on_device_changed()

    # ── construction ────────────────────────────────────────────────────────
    def _build_device(self):
        self.device = QComboBox()
        for i in outputs.OUTPUTS:
            self.device.addItem(i.label, i.key)
            index = self.device.count() - 1
            if i.missing:
                self.device.model().item(index).setEnabled(False)
                self.device.setItemData(index, i.unavailable_reason, Qt.ItemDataRole.ToolTipRole)
        self.device.currentIndexChanged.connect(self._on_device_changed)
        self.box.addWidget(QLabel("Device"))
        self.box.addWidget(self.device)

        s = self.settings
        self.remote_host = QLineEdit(str(s.value("output/howl_host", "")))
        self.remote_host.setPlaceholderText("192.168.1.42")
        self.remote_host.editingFinished.connect(self._on_settings_changed)
        self.remote_port = _spin(1, 65535, int(s.value("output/howl_port", HOWL_PORT)),
                                 self._on_settings_changed, width=64)
        self.rows["remotehost"] = _row("Host", self.remote_host, self.remote_port, stretch=True)
        self.remote_key = QLineEdit(str(s.value("output/howl_key", "")))
        self.remote_key.setPlaceholderText("access key from Howl's settings")
        self.remote_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.remote_key.editingFinished.connect(self._on_settings_changed)
        self.rows["remotekey"] = _row("Access key", self.remote_key, stretch=True)
        self.scan_btn = QPushButton("Scan this PC's subnet for Howl")
        self.scan_btn.setObjectName("secondary")
        self.scan_btn.clicked.connect(self._on_scan)
        self.rows["remotescan"] = self.scan_btn
        for key in ("remotehost", "remotekey", "remotescan"):
            self.box.addWidget(self.rows[key])

        # Intiface / Buttplug
        self.bp_url = QLineEdit(str(s.value("output/buttplug_url", BUTTPLUG_URL)))
        self.bp_url.setPlaceholderText(BUTTPLUG_URL)
        self.bp_url.editingFinished.connect(self._on_settings_changed)
        self.bp_scan = QPushButton("Scan")
        self.bp_scan.setObjectName("secondary")
        self.bp_scan.setToolTip("Ask Intiface to look for devices (connect first)")
        self.bp_scan.clicked.connect(self._on_bp_scan)
        self.rows["bpurl"] = _row("Server", self.bp_url, self.bp_scan, stretch=True)
        self.bp_table = QTableWidget(0, 2)
        self.bp_table.setHorizontalHeaderLabels(["Device", "Follows"])
        self.bp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.bp_table.verticalHeader().setVisible(False)
        self.bp_table.setMaximumHeight(140)
        self.rows["bpdevices"] = self.bp_table
        self.bp_hint = QLabel("Start Intiface Central, press its server button, then Connect. "
                              "Only the level is sent — vibrators have no pulse frequency.")
        self.bp_hint.setWordWrap(True)
        self.bp_hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.rows["bphint"] = self.bp_hint
        try:
            self.bp_assign = dict(json.loads(str(s.value("output/buttplug_assign", "{}"))))
        except (ValueError, TypeError):
            self.bp_assign = {}
        for key in ("bpurl", "bpdevices", "bphint"):
            self.box.addWidget(self.rows[key])

        row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.status_lbl = QLabel()
        row.addWidget(self.connect_btn)
        row.addWidget(self.status_lbl, 1)
        self.box.addLayout(row)
        self.warn_lbl = QLabel()
        self.warn_lbl.setWordWrap(True)
        self.warn_lbl.setStyleSheet(f"color: {theme.DANGER};")
        self.box.addWidget(self.warn_lbl)

    def _build_power(self):
        head = QHBoxLayout()
        head.addWidget(QLabel("Power"))
        head.addStretch()
        self.link = QCheckBox("link A/B")
        self.link.toggled.connect(lambda _on: self._on_power_changed("a"))
        head.addWidget(self.link)
        self.box.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.addWidget(QLabel("value"), 0, 2, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(QLabel("max"), 0, 3, Qt.AlignmentFlag.AlignHCenter)
        self.sliders: dict[str, QSlider] = {}
        self.spins: dict[str, QSpinBox] = {}
        self.limits: dict[str, QSpinBox] = {}
        for row, side in enumerate(("a", "b"), start=1):
            grid.addWidget(QLabel(side.upper()), row, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, POWER_MAX)
            slider.valueChanged.connect(lambda _v, s=side: self._on_slider(s))
            spin = QSpinBox()
            spin.setRange(0, POWER_MAX)
            spin.setMinimumWidth(54)
            spin.valueChanged.connect(lambda _v, s=side: self._on_spin(s))
            limit = QSpinBox()
            limit.setRange(0, POWER_MAX)
            limit.setMinimumWidth(54)
            limit.setValue(int(self.settings.value(f"output/limit_{side}", POWER_MAX)))
            limit.setToolTip("Hard ceiling for this channel — the slider cannot pass it")
            limit.valueChanged.connect(lambda _v, s=side: self._on_limit(s))
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(limit, row, 3)
            self.sliders[side], self.spins[side], self.limits[side] = slider, spin, limit
            slider.setMaximum(limit.value())
            spin.setMaximum(limit.value())
        grid.setColumnStretch(1, 1)
        self.box.addLayout(grid)

    def _build_settings(self):
        self.freq_min = _spin(1, 20000, 10, self._on_settings_changed)   # range: the output's
        self.freq_max = _spin(1, 20000, 100, self._on_settings_changed)
        self.rows["freq"] = _row("Frequency", self.freq_min, QLabel(".."), self.freq_max,
                                 QLabel("Hz"))
        self.swap = QCheckBox("swap A ↔ B")
        self.swap.toggled.connect(self._on_settings_changed)
        self.rows["swap"] = _row("Channels", self.swap)
        self.sound_dev = metrics.shrinkable_combo(QComboBox())
        self.sound_dev.currentIndexChanged.connect(self._on_settings_changed)
        self.rows["sounddev"] = _row("Sound device", self.sound_dev, stretch=True)
        self.shape = QComboBox()
        self.shape.addItems(["sine", "square", "triangle", "trapezoid"])
        self.shape.currentIndexChanged.connect(self._on_settings_changed)
        self.rows["shape"] = _row("Wave shape", self.shape, stretch=True)
        self.carrier = _spin(100, 20000, 1000, self._on_settings_changed, width=64)
        self.rows["carrier"] = _row("Carrier", self.carrier, QLabel("Hz"))
        self.wl_width = _spin(1, 64, 5, self._on_settings_changed)
        self.rows["width"] = _row("Wavelet", self.wl_width, QLabel("cycles"))
        self.wl_fade = _dspin(0.0, 1.0, 0.5, 0.05, self._on_settings_changed)
        self.rows["fade"] = _row("Fade", self.wl_fade)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        self.log.setMaximumBlockCount(300)
        self.rows["log"] = self.log
        for key in ("freq", "swap", "sounddev", "shape", "carrier", "width", "fade", "log"):
            self.box.addWidget(self.rows[key])

    def _build_safety(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Soft start"))
        self.soft_start = _dspin(0.0, 10.0, float(self.settings.value("output/soft_start", 2.0)),
                                 0.5, lambda _v: self._push_controls())
        row.addWidget(self.soft_start)
        row.addWidget(QLabel("s"))
        row.addStretch()
        self.mute = QCheckBox("mute")
        self.mute.toggled.connect(lambda _on: self._push_controls())
        row.addWidget(self.mute)
        self.box.addLayout(row)
        hint = QLabel("Space — play / stop · Esc — instant stop (panic)")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        hint.setWordWrap(True)
        self.box.addWidget(hint)

    def _build_monitor(self):
        """Scope / spectrum / meters of the sound card signal (audio outputs only)."""
        from .output_monitor import OutputMonitor
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(_rule())
        box.addWidget(QLabel("Output signal"))
        self.monitor = OutputMonitor(self.pb)
        box.addWidget(self.monitor)
        self.rows["monitor"] = holder
        self.box.addWidget(holder)

    # ── device ──────────────────────────────────────────────────────────────
    # ── Intiface devices ────────────────────────────────────────────────────
    def set_bp_devices(self, devices: list) -> None:
        self.bp_table.setRowCount(0)
        for d in devices:
            row = self.bp_table.rowCount()
            self.bp_table.insertRow(row)
            kinds = ", ".join(sorted({k for k, _steps in d.get("actuators", [])})) or "—"
            item = QLabel(f"  {d.get('name', '?')}  <span style='color:{theme.TEXT_DIM}'>"
                          f"({kinds})</span>")
            self.bp_table.setCellWidget(row, 0, item)
            combo = QComboBox()
            for key in ASSIGNMENTS:
                combo.addItem(ASSIGN_LABELS[key], key)
            name = str(d.get("name", ""))
            combo.setCurrentIndex(max(0, combo.findData(self.bp_assign.get(name, "max"))))
            combo.currentIndexChanged.connect(
                lambda _i, n=name, c=combo: self._on_bp_assign(n, c.currentData()))
            self.bp_table.setCellWidget(row, 1, combo)
        if not devices:
            self.bp_table.insertRow(0)
            self.bp_table.setCellWidget(0, 0, QLabel("  no devices — press Scan"))

    def _on_bp_assign(self, name: str, mode: str) -> None:
        self.bp_assign[name] = mode
        self._on_settings_changed()

    def _on_bp_scan(self) -> None:
        sink = self.pb.sink
        if sink is None or not hasattr(sink, "scan") or self.pb.status != STATUS_CONNECTED:
            self.pb.message.emit("Connect to Intiface first.")
            return
        sink.scan(True)
        self.pb.message.emit("Scanning for devices…")

    def current_key(self) -> str:
        return self.device.currentData()

    def _on_device_changed(self):
        key = self.current_key()
        self.settings.setValue("output/device", key)
        self.pb.select_output(key)
        info = outputs.info(key)
        visible = SETTINGS_ROWS.get(key, ("freq",))
        for name, w in self.rows.items():
            w.setVisible(name in visible)
        if "sounddev" in visible:
            self._fill_sound_devices()
        self._suspend += 1
        try:
            if self.pb.sink is not None:
                fmap = self.pb.sink.default_freq_map
                for spin, value in ((self.freq_min, fmap.min_hz), (self.freq_max, fmap.max_hz)):
                    spin.setRange(fmap.min_limit, fmap.max_limit)
                    spin.setValue(value)
        finally:
            self._suspend -= 1
        self.warn_lbl.setVisible(not info.verified)
        self.warn_lbl.setText("⚠ Not tested with real hardware yet — raise the power slowly.")
        self.connect_btn.setVisible(info.explicit_connect or key in ("continuous", "wavelet"))
        self._on_settings_changed()
        self._set_status(self.pb.status)

    def _fill_sound_devices(self):
        self._suspend += 1
        try:
            self.sound_dev.clear()
            self.sound_dev.addItem("system default", None)
            try:
                import sounddevice as sd
                for index, dev in enumerate(sd.query_devices()):
                    if dev["max_output_channels"] >= 2:
                        self.sound_dev.addItem(f"{index}: {dev['name']}", index)
            except Exception as exc:  # noqa: BLE001
                self.pb.message.emit(f"Could not list sound devices: {exc}")
            metrics.widen_popup(self.sound_dev)
            self.sound_dev.setToolTip(self.sound_dev.currentText())
        finally:
            self._suspend -= 1

    def _on_connect_clicked(self):
        if self.pb.status == STATUS_DISCONNECTED:
            self._on_settings_changed()
            self.pb.connect_output()
        else:
            self.pb.disconnect_output()
        self._sync_enabled()

    def _set_status(self, status: str):
        color = {STATUS_CONNECTED: "#4ec9b0", STATUS_CONNECTING: theme.ACCENT}.get(
            status, theme.TEXT_DIM)
        self.status_lbl.setText(f"● {status}")
        self.status_lbl.setStyleSheet(f"color: {color};")
        self.connect_btn.setText("Disconnect" if status == STATUS_CONNECTED else "Connect")
        self._sync_enabled()

    def _sync_enabled(self):
        self.device.setEnabled(not self.pb.playing)

    # ── settings ────────────────────────────────────────────────────────────
    def settings_dict(self) -> dict:
        return {
            "sound_device": self.sound_dev.currentData(),
            "shape": self.shape.currentText(),
            "carrier": self.carrier.value(),
            "wavelet_width": self.wl_width.value(),
            "wavelet_fade": self.wl_fade.value(),
            "remote_host": self.remote_host.text(),
            "remote_port": self.remote_port.value(),
            "remote_key": self.remote_key.text(),
            "freq_min": self.freq_min.value(),
            "freq_max": self.freq_max.value(),
            "swap_channels": self.swap.isChecked(),
            "buttplug_url": self.bp_url.text().strip() or BUTTPLUG_URL,
            "buttplug_assign": dict(self.bp_assign),
        }

    def _on_settings_changed(self, *_):
        if self._suspend:
            return
        self.pb.configure(self.settings_dict(), (self.freq_min.value(), self.freq_max.value()))
        if self.current_key() == "buttplug":
            self.settings.setValue("output/buttplug_url", self.bp_url.text().strip())
            self.settings.setValue("output/buttplug_assign", json.dumps(self.bp_assign))
        if self.current_key() == "howl":
            self.settings.setValue("output/howl_host", self.remote_host.text())
            self.settings.setValue("output/howl_port", self.remote_port.value())
            self.settings.setValue("output/howl_key", self.remote_key.text())
        self._push_controls()

    # ── power ───────────────────────────────────────────────────────────────
    def _on_slider(self, side):
        if self._suspend:
            return
        self._suspend += 1
        try:
            self.spins[side].setValue(self.sliders[side].value())
        finally:
            self._suspend -= 1
        self._on_power_changed(side)

    def _on_spin(self, side):
        if self._suspend:
            return
        self._suspend += 1
        try:
            self.sliders[side].setValue(self.spins[side].value())
        finally:
            self._suspend -= 1
        self._on_power_changed(side)

    def _on_limit(self, side):
        limit = self.limits[side].value()
        self.settings.setValue(f"output/limit_{side}", limit)
        self._suspend += 1
        try:
            self.sliders[side].setMaximum(limit)
            self.spins[side].setMaximum(limit)
        finally:
            self._suspend -= 1
        self._on_power_changed(side)

    def _on_power_changed(self, side):
        if self.link.isChecked() and not self._suspend:
            other = "b" if side == "a" else "a"
            value = min(self.spins[side].value(), self.limits[other].value())
            self._suspend += 1
            try:
                self.spins[other].setValue(value)
                self.sliders[other].setValue(value)
            finally:
                self._suspend -= 1
        self._push_controls()

    def _push_controls(self):
        self.settings.setValue("output/soft_start", self.soft_start.value())
        self.pb.set_controls(
            power_a=self.spins["a"].value(), power_b=self.spins["b"].value(),
            limit_a=self.limits["a"].value(), limit_b=self.limits["b"].value(),
            muted=self.mute.isChecked(), soft_start=self.soft_start.value())

    def set_power(self, a: int, b: int) -> None:
        for side, value in (("a", a), ("b", b)):
            self.spins[side].setValue(min(int(value), self.limits[side].value()))

    def _apply_reported_power(self, a: int, b: int):
        """Howl reports its real strength (changed on the phone or the box)."""
        self._suspend += 1
        try:
            for side, value in (("a", a), ("b", b)):
                value = min(int(value), self.limits[side].value())
                self.spins[side].setValue(value)
                self.sliders[side].setValue(value)
        finally:
            self._suspend -= 1
        self._push_controls()

    # ── Howl subnet scan ────────────────────────────────────────────────────
    def _on_scan(self):
        from ...outputs import howl_remote
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning…")
        port = self.remote_port.value()
        self._scan_prefixes = howl_remote.local_subnets()
        hosts = [f"{p}.{i}" for p in self._scan_prefixes for i in range(1, 255)]

        def worker():
            try:
                found = howl_remote.scan_for_howl(port=port, hosts=hosts)
            except Exception:  # noqa: BLE001 — never leave the button stuck
                found = []
            self._scan_results.put(found)

        threading.Thread(target=worker, name="howl-scan", daemon=True).start()
        self._scan_timer.start()

    def _poll_scan(self):
        try:
            found = self._scan_results.get_nowait()
        except queue.Empty:
            return
        self._scan_timer.stop()
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan this PC's subnet for Howl")
        if not found:
            scanned = ", ".join(f"{p}.0/24" for p in self._scan_prefixes) or "no subnet"
            QMessageBox.information(
                self, "Scan finished",
                f"Nothing answered on port {self.remote_port.value()}.\n\nScanned: {scanned}\n\n"
                "Make sure the phone is on the same Wi-Fi as this PC and that Howl's "
                "remote-access API is enabled in its settings.")
            return
        if len(found) == 1:
            self._pick_host(found[0])
            return
        menu = QMenu(self)
        for ip in found:
            menu.addAction(ip, lambda _c=False, h=ip: self._pick_host(h))
        menu.exec(self.scan_btn.mapToGlobal(self.scan_btn.rect().bottomLeft()))

    def _pick_host(self, host: str):
        self.remote_host.setText(host)
        self._on_settings_changed()
        self.pb.message.emit(f"Found Howl at {host}.")
