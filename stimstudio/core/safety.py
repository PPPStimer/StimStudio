"""Safety check: find the parts of a render that are likely to be unpleasant.

Works on a :class:`~.render.ProjectRender` (the same arrays the outputs play)
and reports :class:`Issue` spans. Thresholds live in :class:`SafetyRules`.
None of this is medical advice — it flags patterns that are known to be
uncomfortable with e-stim (sudden onsets, long stretches at the top, very low
pulse rates at high level, abrupt triphase position changes).

Rules:

* ``jump`` — the level rises fast (within ``jump_window_s``) by more than
  ``jump_rise`` above the peak of the preceding ``context_s``. Repeated beats
  at the same level therefore do not count; a hit after silence does.
* ``cold_start`` — after ``silence_s`` of near silence (or at the start), the
  level passes ``cold_level`` within ``cold_window_s`` (a missing warm-up).
* ``sustained`` — the 1-second mean stays above ``sustained_level`` for longer
  than ``sustained_s``.
* ``low_freq`` — a channel above ``low_freq_amp`` while its pulse rate is below
  ``low_freq_hz`` for at least ``low_freq_min_s``.
* ``position_jump`` — a triphase position moves more than
  ``position_jump`` (disk units) within ``jump_window_s`` while audible.
* ``dose`` (info) — the share of the project above ``dose_level``.

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .pulse import FS, L_AMP, L_FREQ, R_AMP, R_FREQ

HIGH, WARN, INFO = "high", "warn", "info"
SEVERITY_ORDER = {HIGH: 0, WARN: 1, INFO: 2}
KIND_LABELS = {"jump": "Sudden jump", "cold_start": "No warm-up", "sustained": "Long at the top",
               "low_freq": "Low rate at high level", "position_jump": "Position jump",
               "dose": "Total intensity"}


@dataclass(frozen=True)
class SafetyRules:
    jump_rise: float = 0.4
    jump_high: float = 0.7
    jump_window_s: float = 0.1
    context_s: float = 2.0
    silence_level: float = 0.03
    silence_s: float = 3.0
    cold_level: float = 0.5
    cold_window_s: float = 2.0
    sustained_level: float = 0.85
    sustained_s: float = 90.0
    low_freq_hz: float = 15.0
    low_freq_amp: float = 0.8
    low_freq_min_s: float = 2.0
    position_jump: float = 0.5
    dose_level: float = 0.7
    dose_share: float = 0.4


@dataclass
class Issue:
    t0: float
    t1: float
    kind: str
    severity: str
    message: str
    track_id: str | None = None
    value: float = 0.0               # the measured quantity (rise, seconds, …)

    @property
    def key(self) -> str:
        """Stable id for "ignore this one" (kind + start, to 0.1 s)."""
        return f"{self.kind}@{self.t0:.1f}"

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


@dataclass
class SafetyReport:
    issues: list[Issue] = field(default_factory=list)
    profile_bin_s: float = 0.25
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # per-bin max

    @property
    def worst(self) -> str | None:
        if not self.issues:
            return None
        return min((i.severity for i in self.issues), key=SEVERITY_ORDER.__getitem__)

    def count(self, severity: str) -> int:
        return sum(i.severity == severity for i in self.issues)


# ── helpers ─────────────────────────────────────────────────────────────────
def level(master: np.ndarray) -> np.ndarray:
    """Per-pulse level: the louder channel."""
    if not len(master):
        return np.zeros(0, np.float32)
    return np.maximum(master[:, L_AMP], master[:, R_AMP]).astype(np.float32)


def level_profile(lv: np.ndarray, bin_s: float = 0.25) -> np.ndarray:
    """Per-bin maximum of a level signal (for the heat strip and overview)."""
    per = max(1, int(round(bin_s * FS)))
    if not len(lv):
        return np.zeros(0, np.float32)
    return np.maximum.reduceat(lv, np.arange(0, len(lv), per)).astype(np.float32)


def _runs(mask: np.ndarray, merge_gap: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """``[start, end)`` index runs where *mask* is true (gaps ≤ merge_gap joined)."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs: list[list[int]] = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= merge_gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return [(s, e) for s, e in runs if e - s >= min_len]


def _past_max(x: np.ndarray, lag: int, length: int) -> np.ndarray:
    """``out[i] = max(x[i - lag - length : i - lag])`` (0 before the start)."""
    padded = np.concatenate([np.zeros(lag + length, x.dtype), x])
    win = sliding_window_view(padded[:len(padded) - lag], length)
    return win.max(axis=1)[:len(x)]


def _culprit(render, i0: int, i1: int, project) -> str | None:
    """The track contributing the most level over ``[i0, i1)``."""
    best, best_v = None, 0.0
    for track in getattr(project, "tracks", []):
        if track.type not in ("standard", "triphase") or track.muted:
            continue
        tr = render.tracks.get(track.id)
        if tr is None:
            continue
        if track.type == "triphase" and tr.triphase is not None:
            v = float(tr.triphase[i0:i1, 0].max(initial=0.0)) * track.gain
        else:
            v = float(level(tr.pulses[i0:i1]).max(initial=0.0)) * track.gain
        if v > best_v:
            best, best_v = track.id, v
    return best


def _t(i: int) -> float:
    return i / FS


# ── the rules ───────────────────────────────────────────────────────────────
def _jumps(lv, rules) -> list[tuple[int, int, float]]:
    w = max(1, int(round(rules.jump_window_s * FS)))
    ctx = max(1, int(round(rules.context_s * FS)))
    before = _past_max(lv, w, ctx)
    rise = lv - before
    out = []
    for s, e in _runs(rise > rules.jump_rise, merge_gap=int(0.5 * FS)):
        k = s + int(np.argmax(rise[s:e]))
        out.append((s, e, float(rise[k])))
    return out


def _cold_starts(lv, rules) -> list[tuple[int, int, float]]:
    quiet = lv < rules.silence_level
    sil = max(1, int(round(rules.silence_s * FS)))
    win = max(1, int(round(rules.cold_window_s * FS)))
    out = []
    starts = [0] if len(lv) and not quiet[0] else []
    for s, e in _runs(quiet):
        if e - s >= sil and e < len(lv):
            starts.append(e)
    for s in starts:
        seg = lv[s:s + win]
        peak = float(seg.max(initial=0.0))
        if peak > rules.cold_level:
            out.append((s, s + len(seg), peak))
    return out


def _sustained(lv, rules) -> list[tuple[int, int, float]]:
    if len(lv) < FS:
        return []
    mean1 = np.convolve(lv, np.ones(FS) / FS, mode="same")
    need = int(round(rules.sustained_s * FS))
    return [(s, e, (e - s) / FS) for s, e in _runs(mean1 > rules.sustained_level, min_len=need)]


def _low_freq(master, freq_range, rules) -> list[tuple[int, int, float]]:
    lo, hi = freq_range
    need = max(1, int(round(rules.low_freq_min_s * FS)))
    mask = np.zeros(len(master), bool)
    for a, f in ((L_AMP, L_FREQ), (R_AMP, R_FREQ)):
        hz = lo + master[:, f] * (hi - lo)
        mask |= (master[:, a] > rules.low_freq_amp) & (hz < rules.low_freq_hz)
    out = []
    for s, e in _runs(mask, merge_gap=int(0.25 * FS), min_len=need):
        hz = lo + np.minimum(master[s:e, L_FREQ], master[s:e, R_FREQ]) * (hi - lo)
        out.append((s, e, float(hz.min())))
    return out


def _position_jumps(tri, rules) -> list[tuple[int, int, float]]:
    w = max(1, int(round(rules.jump_window_s * FS)))
    if tri is None or len(tri) <= w:
        return []
    xy = tri[:, 1:3].astype(np.float64) * 2.0 - 1.0          # 0..1 storage → disk units
    d = np.zeros(len(tri))
    d[w:] = np.hypot(*(xy[w:] - xy[:-w]).T)
    audible = np.minimum(tri[:, 0], np.concatenate([np.zeros(w), tri[:-w, 0]])) > 0.3
    out = []
    # hopping patterns (saltation, electrode hops) are one region, not one issue per hop
    for s, e in _runs((d > rules.position_jump) & audible, merge_gap=int(3.0 * FS)):
        out.append((s, e, float(d[s:e].max())))
    return out


def check(render, project=None, rules: SafetyRules = SafetyRules(),
          ignored=()) -> SafetyReport:
    """Run every rule over *render*; *ignored* holds :attr:`Issue.key` values."""
    master = np.asarray(render.master, np.float32)
    lv = level(master)
    tri = getattr(render, "triphase", None)
    if tri is not None and len(tri) == len(lv):
        lv = np.maximum(lv, tri[:, 0])              # triphase volume counts as level
    freq_range = tuple(getattr(project, "freq_range_hz", (10.0, 100.0)))
    issues: list[Issue] = []

    def add(kind, sev, s, e, msg, value):
        issues.append(Issue(_t(s), _t(e), kind, sev, msg,
                            _culprit(render, s, max(e, s + 1), project) if project else None,
                            value))

    for s, e, rise in _jumps(lv, rules):
        before = float(lv[max(0, s - 1)])
        add("jump", HIGH if rise >= rules.jump_high else WARN, s, e,
            f"Level jumps by {rise:.2f} within {rules.jump_window_s * 1000:.0f} ms "
            f"(from about {before:.2f}).", rise)
    jump_starts = {round(i.t0, 1) for i in issues}
    for s, e, peak in _cold_starts(lv, rules):
        if round(_t(s), 1) in jump_starts:
            continue                                 # already reported as a jump
        add("cold_start", WARN, s, e,
            f"Reaches {peak:.2f} within {rules.cold_window_s:.0f} s of starting — "
            "add a fade-in or a softer opening.", peak)
    for s, e, secs in _sustained(lv, rules):
        add("sustained", WARN, s, e,
            f"Stays above {rules.sustained_level:.2f} for {secs:.0f} s without a break.", secs)
    for s, e, hz in _low_freq(master, freq_range, rules):
        add("low_freq", WARN, s, e,
            f"Pulse rate down to {hz:.0f} Hz at a high level — feels harsh.", hz)
    for s, e, d in _position_jumps(tri if tri is not None and len(tri) == len(lv) else None,
                                   rules):
        add("position_jump", WARN, s, e,
            f"Triphase position moves {d:.2f} within {rules.jump_window_s * 1000:.0f} ms.", d)
    if len(lv) > 60 * FS:
        share = float((lv > rules.dose_level).mean())
        if share >= rules.dose_share:
            issues.append(Issue(0.0, _t(len(lv)), "dose", INFO,
                                f"{share * 100:.0f} % of the project is above "
                                f"{rules.dose_level:.2f} — plan some rests.", value=share))
    ignored = set(ignored)
    issues = [i for i in issues if i.key not in ignored]
    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.t0))
    return SafetyReport(issues, 0.25, level_profile(lv, 0.25))


# ── fixes ───────────────────────────────────────────────────────────────────
def limiter_for(issue: Issue) -> dict:
    """A ``slew_limit`` effect dict that would tame *issue* (jump / cold start)."""
    from ..effects import get_effect
    ramp_s = 2.0 if issue.kind == "cold_start" else 1.0
    return get_effect("slew_limit").make(rise=round(1.0 / ramp_s, 2))


def apply_limiter(effects: list[dict], issue: Issue) -> list[dict]:
    """*effects* with a slew limiter added (or an existing one tightened)."""
    want = limiter_for(issue)
    out = [dict(e) for e in effects]
    for e in out:
        if e.get("type") == "slew_limit" and e.get("enabled", True):
            e["rise"] = min(float(e.get("rise", 2.0)), want["rise"])
            return out
    return out + [want]


def rest_points(issue: Issue, every_s: float = 60.0, depth: float = 0.45,
                length_s: float = 2.0, ramp_s: float = 0.5) -> list[tuple[float, float]]:
    """Envelope points (multiply mode) that put short rests into a sustained span."""
    pts: list[tuple[float, float]] = []
    span = issue.t1 - issue.t0
    count = max(1, int(span // every_s))
    for k in range(count):
        mid = issue.t0 + span * (k + 0.5) / count
        a = mid - length_s / 2
        pts += [(a - ramp_s, 1.0), (a, depth), (a + length_s, depth),
                (a + length_s + ramp_s, 1.0)]
    return pts


def merge_points(existing, new) -> list[tuple[float, float]]:
    """*existing* envelope points with *new* ones; old points inside the new
    span are dropped (the new shape wins there)."""
    if not new:
        return list(existing)
    lo, hi = min(t for t, _ in new), max(t for t, _ in new)
    kept = [(t, v) for t, v in existing if t < lo or t > hi]
    return sorted(kept + list(new))
