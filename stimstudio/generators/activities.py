"""Faithful ports of Howl's activities (from Activity.kt).

Each activity is a simulation stepped at 1/40 s per pulse. ``ActivityGenerator``
drives the simulation exactly like Howl's ActivityHost: ``run_simulation(dt)``
then ``get_pulse()``, with the first step using dt=0. All randomness flows
through the engine's seeded RNG, so a seed reproduces a render.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.pulse import FS, PulseTrack
from ..engine import mathutils as mu
from ..engine.components import (
    ActivityManager,
    NiceSmoother,
    Pulse,
    Timer,
    WaveManager,
)
from ..engine.noise import NoiseGenerator
from ..engine.waves import (
    HERMITE,
    LINEAR,
    SMALL_AMOUNT,
    TMAX,
    CyclicalWave,
    WavePoint,
    WaveShape,
    create_repeated_wave,
)
from .base import Generator, ParamSpec, register

DT = 1.0 / FS


def _wave(name, points, interp=HERMITE):
    return CyclicalWave(WaveShape(name, [WavePoint(*p) for p in points], interp))


# =====================================================================
# Base generator that runs an activity simulation
# =====================================================================
class ActivityGenerator(Generator):
    stochastic = True
    category = "activity"
    activity_cls = None

    def make_activity(self, params):
        return self.activity_cls(params)

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        mu.set_rng(seed)
        activity = self.make_activity(self.merged_params(params))
        activity.initialise()
        l_amp = np.empty(n, np.float32)
        r_amp = np.empty(n, np.float32)
        l_freq = np.empty(n, np.float32)
        r_freq = np.empty(n, np.float32)
        for i in range(n):
            activity.run_simulation(0.0 if i == 0 else DT)
            p = activity.get_pulse()
            l_amp[i] = p.amp_a
            r_amp[i] = p.amp_b
            l_freq[i] = p.freq_a
            r_freq[i] = p.freq_b
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()

    def segment_boundaries(self, n: int, params: dict, seed: int = 0) -> list[float]:
        mu.set_rng(seed)
        activity = self.make_activity(self.merged_params(params))
        activity.initialise()
        out: list[float] = []
        prev = None
        for i in range(n):
            activity.run_simulation(0.0 if i == 0 else DT)
            label = activity.segment_label()
            if label is not None and label != prev:
                if i > 0:
                    out.append(i / FS)
                prev = label
        return out

    def manual_params_at(self, n: int, params: dict, seed: int = 0) -> dict:
        mu.set_rng(seed)
        activity = self.make_activity(self.merged_params(params))
        activity.initialise()
        for i in range(n):
            activity.run_simulation(0.0 if i == 0 else DT)
        return activity.manual_snapshot()


class BaseActivity:
    """Common scaffolding mirroring Howl's abstract Activity."""

    def __init__(self, params=None):
        self.params = params or {}
        self.manager = ActivityManager()

    def initialise(self):
        pass

    def run_simulation(self, dt):
        self.manager.update(dt)

    def get_pulse(self) -> Pulse:
        raise NotImplementedError

    def segment_label(self):
        """Identifier of the current internal segment/stage, or None if the
        activity has no discrete segments. A change marks a boundary."""
        return None

    def manual_snapshot(self) -> dict:
        """Current controllable parameter values, for reuse in manual control.
        A "manual": True key means the snapshot can be applied to manual mode."""
        return {}


# =====================================================================
# Infinite licks
# =====================================================================
class LickActivity(BaseActivity):
    LICK_TYPES = ("UNIDIRECTIONAL", "BIDIRECTIONAL")
    AMP_TYPES = ("CONSISTENT", "LICK", "DIP", "RAMP", "FLICKS")

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.manual = bool(self.params.get("manual", False))
        self.speed = float(self.params.get("speed", 1.0))
        self._seg = 0
        self.lick_start = mu.random_in_range(0.0, 1.0)
        self.lick_end = mu.random_in_range(0.0, 1.0)
        self.lick_type = mu.choice(self.LICK_TYPES)
        self.amp_type = mu.choice(self.AMP_TYPES)

    def initialise(self):
        self.wave.add_wave(_wave("bidirectional", [(0.0, 0.0), (0.5, 1.0)]))
        self.wave.add_wave(_wave("unidirectional", [(0.0, 0.0), (TMAX, 1.0)]))
        self.wave.add_wave(_wave("consistent", [(0.0, 0.85), (0.5, 0.95)]))
        self.wave.add_wave(_wave("lick", [
            (0.0, 0.0), (0.05, 0.6), (0.5, 1.0), (0.95, 0.6), (TMAX, 0.0)]))
        self.wave.add_wave(_wave("dip", [
            (0.0, 0.9), (0.35, 0.8), (0.5, 0.6), (0.65, 0.8)]))
        self.wave.add_wave(_wave("ramp", [(0.0, 0.4), (0.5, 0.7), (TMAX, 1.0)]))
        self.wave.add_wave(_wave("flicks", [
            (0.0, 0.0), (0.0833, 1.0), (0.25, 1.0), (0.3333, 0.0), (0.4166, 1.0),
            (0.5833, 1.0), (0.6666, 0.0), (0.75, 1.0), (0.9166, 1.0)]))
        self.wave.set_speed_jitter(0.5)
        self.wave.set_amplitude_jitter(0.1)
        self.manager.register(self.wave)
        if self.manual:
            self.lick_type = self.params.get("lick_type", self.lick_type)
            self.amp_type = self.params.get("amp_type", self.amp_type)
            self.lick_start = float(self.params.get("lick_start", self.lick_start))
            self.lick_end = float(self.params.get("lick_end", self.lick_end))
            self.wave.set_speed(self.speed)
            self.wave.restart()
        else:
            self.new_lick()

    def new_lick(self):
        self._seg += 1
        self.lick_type = mu.choice(self.LICK_TYPES)
        self.amp_type = mu.choice(self.AMP_TYPES)
        self.lick_start = mu.random_in_range(0.0, 1.0)
        self.lick_end = mu.random_in_range(0.0, 1.0)
        distance = abs(self.lick_end - self.lick_start)
        max_speed = (-3.0 * distance) + 4.5
        speed = mu.next_double(0.3, max_speed)
        desired_time = mu.next_double(1.0, 5.0)
        reps = max(1, int(desired_time * speed))
        self.wave.restart()
        self.wave.set_speed(speed)
        self.wave.stop_after_iterations(reps, self.new_lick)

    def get_pulse(self):
        wave_name = "unidirectional" if self.lick_type == "UNIDIRECTIONAL" else "bidirectional"
        amp_name = self.amp_type.lower()
        position = self.wave.get_position(wave_name)
        power = self.wave.get_position(amp_name)
        lick_position = (self.lick_end - self.lick_start) * position + self.lick_start
        amp = power * self.wave.current_amplitude
        freq_a = lick_position * 0.5 + 0.5
        freq_b = lick_position * 0.52 + 0.48
        amp_a, amp_b = mu.calculate_positional_effect(amp, lick_position, 1.0)
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def segment_label(self):
        return self._seg

    def manual_snapshot(self):
        return {
            "manual": True,
            "speed": round(self.wave.base_speed.value, 3),
            "lick_start": round(self.lick_start, 3),
            "lick_end": round(self.lick_end, 3),
            "lick_type": self.lick_type,
            "amp_type": self.amp_type,
        }


# =====================================================================
# Penetration
# =====================================================================
class PenetrationActivity(BaseActivity):
    SPEED_RANGE = (0.3, 3.0)
    SPEED_CHANGE_RATE_RANGE = (0.05, 0.3)
    FEEL_RANGE = (0.5, 1.0)
    FEEL_CHANGE_RATE_RANGE = (0.05, 0.1)

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.manual = bool(self.params.get("manual", False))
        self.speed = float(self.params.get("speed", 1.0))
        self.manual_feel = float(self.params.get("feel", 0.75))
        self.feel = NiceSmoother(mu.random_in_range(*self.FEEL_RANGE))
        self.speed_change_timer = Timer(lambda: mu.random_in_range(1.0, 20.0), True, self._speed_change)
        self.feel_change_timer = Timer(lambda: mu.random_in_range(2.0, 10.0), True, self._feel_change)

    def initialise(self):
        self.wave.add_wave(_wave("penetration", [
            (0.0, 0.0, 0.0), (0.4, 0.95, 0.2), (0.5, 0.97, 0.0), (0.6, 0.95, -0.2)]))
        self.wave.set_speed_jitter(0.2)
        self.wave.set_amplitude_jitter(0.1)
        self.wave.set_speed(0.5)
        self.manager.register(self.speed_change_timer)
        self.manager.register(self.feel_change_timer)
        self.manager.register(self.feel)
        self.manager.register(self.wave)
        if self.manual:
            self.wave.set_speed(self.speed)
            self.feel.set_immediately(self.manual_feel)
        else:
            self._speed_change()
            self._feel_change()
            self.speed_change_timer.start()
            self.feel_change_timer.start()

    def _speed_change(self):
        self.wave.set_target_speed(mu.random_in_range(*self.SPEED_RANGE),
                                   mu.random_in_range(*self.SPEED_CHANGE_RATE_RANGE))

    def _feel_change(self):
        self.feel.set_target(mu.random_in_range(*self.FEEL_RANGE),
                             mu.random_in_range(*self.FEEL_CHANGE_RATE_RANGE))

    def get_pulse(self):
        position, velocity = self.wave.get_position_and_velocity("penetration")
        scaled_velocity = mu.scale_velocity(velocity, 0.1)
        amp_factor = (self.wave.current_speed - self.SPEED_RANGE[0]) / (self.SPEED_RANGE[1] - self.SPEED_RANGE[0])
        amp_factor = 0.8 + amp_factor * 0.2
        freq_a = mu.calculate_feel_adjustment(position * 0.7, self.feel.value)
        freq_b = mu.calculate_feel_adjustment(scaled_velocity * 0.5 + position * 0.4, self.feel.value)
        amp_a = position * amp_factor
        amp_b = (scaled_velocity * 0.6 + position * 0.4) * amp_factor
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def manual_snapshot(self):
        return {
            "manual": True,
            "speed": round(self.wave.base_speed.value, 3),
            "feel": round(self.feel.value, 3),
        }


# =====================================================================
# Sliding vibrator
# =====================================================================
class VibroActivity(BaseActivity):
    FREQ_RANGE = (0.0, 1.0)
    VIBE_POWER = 0.9
    POSITION_RANGE = (0.0, 1.0)
    MOVE_SPEED_RANGE = (0.08, 0.2)

    def __init__(self, params=None):
        super().__init__(params)
        self.manual = bool(self.params.get("manual", False))
        self.hold_probability = float(self.params.get("hold_probability", 0.25))
        self.pulse_time = float(self.params.get("pulse_time", 0.3))
        self.pulse_duty_cycle = float(self.params.get("pulse_duty_cycle", 1.0))
        self.frequency = mu.random_in_range(*self.FREQ_RANGE)
        self.position = NiceSmoother(mu.random_in_range(*self.POSITION_RANGE))
        self._pulse_tracker = 0.0
        self._vibe_active = False
        self.frequency_change_timer = Timer(lambda: mu.random_in_range(5.0, 30.0), True, self._frequency_change)
        self.hold_timer = Timer(lambda: mu.random_in_range(1.0, 3.0), False, self._new_target)

    def initialise(self):
        self.position.rate = mu.random_in_range(*self.MOVE_SPEED_RANGE)
        self.manager.register(self.frequency_change_timer)
        self.manager.register(self.hold_timer)
        self.manager.register(self.position)
        if self.manual:
            self.position.set_immediately(float(self.params.get("position", 0.5)))
            self.frequency = float(self.params.get("frequency", 0.5))
        else:
            self._new_target()
            self.frequency_change_timer.start()

    def _frequency_change(self):
        self.frequency = mu.random_in_range(*self.FREQ_RANGE)

    def _new_target(self):
        target = mu.random_in_range(*self.POSITION_RANGE)
        self.position.rate = mu.random_in_range(*self.MOVE_SPEED_RANGE)
        self.position.set_target(target, on_reached=self._target_reached)

    def _target_reached(self):
        if mu.rng().random() < self.hold_probability:
            self.hold_timer.reset()
        else:
            self._new_target()

    def run_simulation(self, dt):
        super().run_simulation(dt)
        self._pulse_tracker += dt
        pulse_phase = self._pulse_tracker % self.pulse_time
        off_duration = self.pulse_time * (1.0 - self.pulse_duty_cycle)
        self._vibe_active = pulse_phase >= off_duration

    def get_pulse(self):
        amp_a = amp_b = 0.0
        if self._vibe_active:
            amp_a, amp_b = mu.calculate_positional_effect(self.VIBE_POWER, self.position.value, 1.0)
        return Pulse(amp_a, amp_b, self.frequency, self.frequency)

    def manual_snapshot(self):
        return {
            "manual": True,
            "position": round(self.position.value, 3),
            "frequency": round(self.frequency, 3),
        }


# =====================================================================
# Milkmaster 3000
# =====================================================================
class MilkerActivity(BaseActivity):
    WOMP_START_FREQ = 0.0
    WOMP_END_FREQ = 0.7
    WOMP_START_SPEED = 0.3
    WOMP_END_SPEED = 2.5
    WOMP_RATE_RANGE = (0.1, 0.15)
    BUZZ_FREQ_A_RANGE = (0.0, 0.3)
    BUZZ_FREQ_B_RANGE = (0.7, 1.0)
    BUZZ_SPEED_RANGE = (0.4, 0.8)

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.stage = "WOMP"
        self.buzz_freq_a_start = self.buzz_freq_a_end = 0.75
        self.buzz_freq_b_start = self.buzz_freq_b_end = 0.75
        self.reverse_womp = False
        self.buzz_timer = Timer(lambda: mu.random_in_range(6.0, 12.0), False, self._womp_start)

    def initialise(self):
        self.wave.add_wave(_wave("womp", [(0.0, 1.0, 0.0), (TMAX, 0.0, 0.0)]))
        self.wave.add_wave(_wave("buzz", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)]))
        self.manager.register(self.wave)
        self.manager.register(self.buzz_timer)
        self._womp_start()

    def _womp_start(self):
        self.stage = "WOMP"
        self.wave.restart()
        rate = mu.random_in_range(*self.WOMP_RATE_RANGE)
        self.reverse_womp = mu.next_bool()
        self.wave.set_speed(self.WOMP_START_SPEED)
        self.wave.set_target_speed(
            self.WOMP_END_SPEED, rate,
            lambda: self.wave.stop_at_end_of_cycle(self._buzz_start))

    def _buzz_start(self):
        self.stage = "BUZZ"
        self.buzz_freq_a_start = mu.random_in_range(*self.BUZZ_FREQ_A_RANGE)
        self.buzz_freq_a_end = mu.random_in_range(*self.BUZZ_FREQ_A_RANGE)
        self.buzz_freq_b_start = mu.random_in_range(*self.BUZZ_FREQ_B_RANGE)
        self.buzz_freq_b_end = mu.random_in_range(*self.BUZZ_FREQ_B_RANGE)
        self.wave.restart()
        self.wave.set_speed(mu.random_in_range(*self.BUZZ_SPEED_RANGE))
        self.buzz_timer.start()

    def get_pulse(self):
        if self.stage == "WOMP":
            position = self.wave.get_position("womp")
            adj = 1 - position if self.reverse_womp else position
            amp_a, amp_b = mu.calculate_positional_effect(0.9, adj, 1.0)
            freq = adj * (self.WOMP_END_FREQ - self.WOMP_START_FREQ) + self.WOMP_START_FREQ
            return Pulse(amp_a, amp_b, freq, freq)
        position = self.wave.get_position("buzz")
        phase = self.buzz_timer.progress
        freq_a = phase * (self.buzz_freq_a_end - self.buzz_freq_a_start) + self.buzz_freq_a_start
        freq_b = phase * (self.buzz_freq_b_end - self.buzz_freq_b_start) + self.buzz_freq_b_start
        amp = 0.8 + 0.1 * position
        return Pulse(amp, amp, freq_a, freq_b)

    def segment_label(self):
        return self.stage

    def manual_snapshot(self):
        return {"stage": self.stage}


# =====================================================================
# Chaos
# =====================================================================
class ChaosActivity(BaseActivity):
    def __init__(self, params=None):
        super().__init__(params)
        self.cycle_time = float(self.params.get("cycle_time", 1.0))
        self._counter = 0.0
        self.amp_a = self.amp_b = self.freq_a = self.freq_b = 0.0

    def initialise(self):
        self._randomise()

    def _randomise(self):
        self.amp_a = mu.random_in_range(0.0, 1.0)
        self.amp_b = mu.random_in_range(0.0, 1.0)
        self.freq_a = mu.random_in_range(0.0, 1.0)
        self.freq_b = mu.random_in_range(0.0, 1.0)

    def run_simulation(self, dt):
        super().run_simulation(dt)
        self._counter += dt
        if self._counter > self.cycle_time:
            self._randomise()
            self._counter -= self.cycle_time

    def get_pulse(self):
        return Pulse(self.amp_a, self.amp_b, self.freq_a, self.freq_b)


# =====================================================================
# Luxury HJ
# =====================================================================
class LuxuryHJActivity(BaseActivity):
    HJ_SPEED_RANGE = (0.3, 3.0)
    HJ_SPEED_CHANGE_RATE_RANGE = (0.05, 0.3)
    BONUS_SPEED_RANGE = (1.5, 5.0)
    START_FREQ = 0.15
    END_FREQ = 0.65
    BONUS_START_FREQ = 0.8
    BONUS_END_FREQ = 1.0

    def __init__(self, params=None):
        super().__init__(params)
        self.manual = bool(self.params.get("manual", False))
        self.speed = float(self.params.get("speed", 1.0))
        self.bonus_probability = float(self.params.get("bonus_probability", 0.7))
        self.amplitude_jitter = float(self.params.get("amplitude_jitter", 0.15))
        self.timing_jitter = float(self.params.get("timing_jitter", 0.15))
        self.hj = WaveManager()
        self.bonus = WaveManager()
        self.bonus_channel = 0
        self.speed_change_timer = Timer(lambda: mu.random_in_range(1.0, 20.0), True, self._speed_change)
        self.bonus_timer = Timer(lambda: mu.random_in_range(10.0, 25.0), False)

    def initialise(self):
        self.hj.add_wave(_wave("hj", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)]))
        self.bonus.add_wave(_wave("bonus", [(0.0, 0.6, 0.0), (0.5, 1.0, 0.0)]))
        self.hj.set_speed_jitter(self.timing_jitter)
        self.hj.set_amplitude_jitter(self.amplitude_jitter)
        self.hj.set_amplitude_jitter_ease_in(1.0)
        self.bonus.set_speed_jitter(0.5)
        self.bonus.set_amplitude_jitter(0.075)
        self.bonus.set_speed(mu.random_in_range(*self.BONUS_SPEED_RANGE))
        self.hj.set_speed(mu.random_in_range(*self.HJ_SPEED_RANGE))
        self.hj.base_speed.rate = mu.random_in_range(*self.HJ_SPEED_CHANGE_RATE_RANGE)
        self.manager.register(self.speed_change_timer)
        self.manager.register(self.bonus_timer)
        self.manager.register(self.hj)
        self.manager.register(self.bonus)
        if self.manual:
            self.hj.set_speed(self.speed)
        else:
            self.speed_change_timer.start()

    def _speed_change(self):
        self.hj.set_target_speed(mu.random_in_range(*self.HJ_SPEED_RANGE),
                                 mu.random_in_range(*self.HJ_SPEED_CHANGE_RATE_RANGE))

    def _start_bonus(self, channel=None):
        self.bonus.set_speed(mu.random_in_range(*self.BONUS_SPEED_RANGE))
        self.bonus_channel = mu.next_int(2) if channel is None else channel
        self.bonus_timer.start()

    def run_simulation(self, dt):
        super().run_simulation(dt)
        if (not self.manual and not self.bonus_timer.is_running
                and mu.rng().random() < (self.bonus_probability * dt) / 60.0):
            self._start_bonus()

    def get_pulse(self):
        position = self.hj.get_position("hj")
        base_amp = self.hj.current_amplitude
        amp_a, amp_b = mu.calculate_positional_effect(base_amp, position, 1.0)
        freq_a = ((position * (self.END_FREQ - self.START_FREQ)) + self.START_FREQ) * 0.98
        freq_b = (position * (self.END_FREQ - self.START_FREQ)) + self.START_FREQ
        if self.bonus_timer.is_running:
            weight = 0.7
            amp_bonus = self.bonus.get_position("bonus")
            freq_bonus = (amp_bonus * (self.BONUS_END_FREQ - self.BONUS_START_FREQ)) + self.BONUS_START_FREQ
            if self.bonus_channel == 0:
                amp_a = amp_bonus * weight + amp_a * (1.0 - weight)
                freq_a = freq_bonus * weight + freq_a * (1.0 - weight)
            else:
                amp_b = amp_bonus * weight + amp_b * (1.0 - weight)
                freq_b = freq_bonus * weight + freq_b * (1.0 - weight)
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def manual_snapshot(self):
        return {"manual": True, "speed": round(self.hj.base_speed.value, 3)}


# =====================================================================
# Opposites
# =====================================================================
class OppositesActivity(BaseActivity):
    AMP_RANGE = (0.0, 1.0)
    FREQ_RANGE = (0.0, 1.0)
    SPEED_RANGE = (0.5, 3.0)
    SPEED_CHANGE_RATE_RANGE = (0.2, 0.5)
    BASE_CHANGE_RATE_RANGE = (0.15, 0.4)

    def __init__(self, params=None):
        super().__init__(params)
        self.amp_a = NiceSmoother(mu.random_in_range(*self.AMP_RANGE))
        self.freq_a = NiceSmoother(mu.random_in_range(*self.FREQ_RANGE))
        self.overall_speed = NiceSmoother(mu.random_in_range(*self.SPEED_RANGE), self.SPEED_RANGE)
        self.speed_change_timer = Timer(lambda: mu.random_in_range(10.0, 20.0), True, self._speed_change)

    def initialise(self):
        self._new_amp_target(self.amp_a)
        self._new_freq_target(self.freq_a)
        self._speed_change()
        self.speed_change_timer.start()
        self.manager.register(self.amp_a)
        self.manager.register(self.freq_a)
        self.manager.register(self.overall_speed)
        self.manager.register(self.speed_change_timer)

    def _speed_change(self):
        self.overall_speed.set_target(mu.random_in_range(*self.SPEED_RANGE),
                                      mu.random_in_range(*self.SPEED_CHANGE_RATE_RANGE))

    def _new_freq_target(self, freq):
        rate = mu.random_in_range(*self.BASE_CHANGE_RATE_RANGE) * self.overall_speed.value
        freq.set_target(mu.random_in_range(*self.FREQ_RANGE), rate,
                        on_reached=lambda: self._new_freq_target(freq))

    def _new_amp_target(self, amp):
        rate = mu.random_in_range(*self.BASE_CHANGE_RATE_RANGE) * self.overall_speed.value
        amp.set_target(mu.random_in_range(*self.AMP_RANGE), rate,
                       on_reached=lambda: self._new_amp_target(amp))

    def get_pulse(self):
        amp_b = 1.0 - self.amp_a.value
        freq_b = 1.0 - self.freq_a.value
        return Pulse(self.amp_a.value, amp_b, self.freq_a.value, freq_b)


# =====================================================================
# Calibration 1 / 2
# =====================================================================
class Calibration1Activity(BaseActivity):
    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()

    def initialise(self):
        self.wave.add_wave(_wave("calibration", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)]))
        self.wave.set_speed(0.25)
        self.manager.register(self.wave)

    def get_pulse(self):
        position = self.wave.get_position("calibration")
        amp_a, amp_b = mu.calculate_positional_effect(0.9, position, 1.0)
        return Pulse(amp_a, amp_b, 0.5, 0.5)


class Calibration2Activity(BaseActivity):
    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.phase = 0  # 0=A, 1=B, 2=BOTH
        self.phase_timer = Timer(16.0, True, self._next_phase)

    def initialise(self):
        self.wave.add_wave(_wave("calibration", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)]))
        self.wave.set_speed(0.25)
        self.manager.register(self.phase_timer)
        self.manager.register(self.wave)
        self.phase_timer.start()

    def _next_phase(self):
        self.phase = (self.phase + 1) % 3

    def get_pulse(self):
        frequency = self.wave.get_position("calibration")
        amp_a = 0.0 if self.phase == 1 else 0.9
        amp_b = 0.0 if self.phase == 0 else 0.9
        return Pulse(amp_a, amp_b, frequency, frequency)

    def segment_label(self):
        return self.phase

    def manual_snapshot(self):
        return {"phase": ("A", "B", "BOTH")[self.phase]}


# =====================================================================
# BJ Megamix
# =====================================================================
class BJActivity(BaseActivity):
    LICK_FREQ_RANGE = (0.8, 1.0)
    BJ_SPEED_RANGE = (0.2, 1.2)
    BJ_SPEED_CHANGE_RATE_RANGE = (0.03, 0.2)
    FULL_LICK_SPEED_RANGE = (0.3, 1.0)
    TIP_LICK_SPEED_RANGE = (0.5, 3.0)
    STAGES = ("FullLick", "TipLick", "Suck", "Deepthroat")

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.manual = bool(self.params.get("manual", False))
        self.speed = float(self.params.get("speed", 1.0))
        self.deepthroat_freq = _wave("deepthroatFrequencyConverter", [
            (0.0, 1.0), (0.7, 0.0), (TMAX, 0.3)])
        self.stage = self.params.get("stage", None) if self.manual else None
        if self.stage not in self.STAGES:
            self.stage = mu.choice(self.STAGES)
        self.primary_timer = Timer(lambda: mu.random_in_range(20.0, 60.0), False,
                                   lambda: self.wave.stop_at_end_of_cycle(self._next_stage))
        self.secondary_timer = Timer(lambda: mu.random_in_range(6.0, 20.0), False,
                                     lambda: self.wave.stop_at_end_of_cycle(self._next_stage))
        self.speed_change_timer = Timer(lambda: mu.random_in_range(1.0, 20.0), True, self._speed_change)

    def initialise(self):
        self.wave.add_wave(_wave("position", [(0.0, 0.0, 0.0), (0.35, 1.0, 0.0)]))
        self.wave.add_wave(_wave("bidirectional", [(0.0, 0.0, 0.0), (0.5, 1.0, 0.0)]))
        self.wave.add_wave(_wave("unidirectional", [(0.0, 0.0, 0.0), (TMAX, 1.0, 0.0)]))
        self.manager.register(self.wave)
        self.manager.register(self.speed_change_timer)
        self.manager.register(self.primary_timer)
        self.manager.register(self.secondary_timer)
        if self.manual:
            self._set_stage(self.stage, manual=True)
            self.wave.set_speed(self.speed)
        else:
            self.speed_change_timer.start()
            self._set_stage(self.stage)

    def _speed_change(self):
        if self.stage in ("FullLick", "TipLick"):
            return
        self.wave.set_target_speed(mu.random_in_range(*self.BJ_SPEED_RANGE),
                                   mu.random_in_range(*self.BJ_SPEED_CHANGE_RATE_RANGE))

    def _set_stage(self, stage, manual=False):
        self.stage = stage
        self.wave.restart()
        if stage == "FullLick":
            self.wave.set_speed_jitter(0.4)
            self.wave.set_amplitude_jitter(0.15)
            if not manual:
                self.wave.set_speed(mu.random_in_range(*self.FULL_LICK_SPEED_RANGE))
        elif stage == "TipLick":
            self.wave.set_speed_jitter(0.4)
            self.wave.set_amplitude_jitter(0.15)
            if not manual:
                self.wave.set_speed(mu.random_in_range(*self.TIP_LICK_SPEED_RANGE))
        else:
            self.wave.set_speed_jitter(0.2)
            self.wave.set_amplitude_jitter(0.1)
            if not manual:
                self.wave.set_speed(mu.random_in_range(*self.BJ_SPEED_RANGE))
        if not manual:
            self.wave.base_speed.rate = mu.random_in_range(*self.BJ_SPEED_CHANGE_RATE_RANGE)
            if stage in ("Deepthroat", "Suck"):
                self.primary_timer.reset()
            else:
                self.secondary_timer.reset()

    def _next_stage(self):
        nxt = mu.choice([s for s in self.STAGES if s != self.stage])
        self._set_stage(nxt)

    def get_pulse(self):
        if self.stage == "FullLick":
            position, velocity = self.wave.get_position_and_velocity("unidirectional")
            sv = mu.scale_velocity(velocity, 0.1)
            amp_a, amp_b = mu.calculate_positional_effect(sv, position, 1.0)
            freq_a = mu.scale_between(position, *self.LICK_FREQ_RANGE) - 0.1
            freq_b = mu.scale_between(position, *self.LICK_FREQ_RANGE)
        elif self.stage == "TipLick":
            position, velocity = self.wave.get_position_and_velocity("bidirectional")
            lick_position = mu.scale_between(position, 0.6, 1.0)
            sv = mu.scale_velocity(velocity, 0.1)
            amp_a, amp_b = mu.calculate_positional_effect(sv, lick_position, 1.0)
            freq_a = mu.scale_between(position, *self.LICK_FREQ_RANGE) - 0.1
            freq_b = mu.scale_between(position, *self.LICK_FREQ_RANGE)
        elif self.stage == "Suck":
            position = self.wave.get_position("position")
            base_amp = self.wave.current_amplitude
            amp_a, amp_b = mu.calculate_engulf_effect(base_amp, position, 0.7, 0.4)
            freq_a = mu.lerp(0.7, 0.3, mu.smoothstep(position))
            freq_b = mu.lerp(0.9, 0.3, mu.smoothstep(position))
        else:  # Deepthroat
            position = self.wave.get_position("position")
            base_amp = self.wave.current_amplitude
            amp_a, amp_b = mu.calculate_engulf_effect(base_amp, position, 0.8, 0.3)
            freq_a = position
            freq_b = self.deepthroat_freq.get_position(position)
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def segment_label(self):
        return self.stage

    def manual_snapshot(self):
        return {
            "manual": True,
            "speed": round(self.wave.base_speed.value, 3),
            "stage": self.stage,
        }


# =====================================================================
# Fast/slow
# =====================================================================
class FastSlowActivity(BaseActivity):
    MIN_SPEED = 0.15
    MAX_SPEED = 5.0
    SPEED_CHANGE_RATE_RANGE = (0.1, 0.3)
    FREQ_RANGE = (0.0, 1.0)
    SWITCH_PROBABILITY = 0.1
    WAVE_SHAPE_CHANGE_PROBABILITY = 0.2

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.wave2 = WaveManager()
        self.accelerating = False
        self.switch = mu.next_bool()
        self.freq_switch = mu.next_bool()
        self.possible = [
            _wave("sawtooth", [(0.0, 0.0, 0.0), (TMAX, 0.9, 0.0)], LINEAR),
            _wave("reverseSawtooth", [(0.0, 0.9, 0.0), (TMAX, 0.0, 0.0)], LINEAR),
            _wave("hermiteSawtooth", [(0.0, 0.0, 0.0), (TMAX, 0.9, 0.0)], HERMITE),
            _wave("hermiteReverseSawtooth", [(0.0, 0.9, 0.0), (TMAX, 0.0, 0.0)], HERMITE),
        ]
        self.wave_a = self.possible[0]
        self.wave_b = self.possible[0]
        self._seg = 0

    def initialise(self):
        for w in self.possible:
            self.wave.add_wave(w)
            self.wave2.add_wave(w)
        self.manager.register(self.wave)
        self.manager.register(self.wave2)
        self._next_iteration()

    def _next_iteration(self):
        self._seg += 1
        self.accelerating = not self.accelerating
        start = self.MIN_SPEED if self.accelerating else self.MAX_SPEED
        target = self.MAX_SPEED if self.accelerating else self.MIN_SPEED
        rate = mu.random_in_range(*self.SPEED_CHANGE_RATE_RANGE)
        self.wave.set_speed(start)
        self.wave.set_target_speed(target, rate, self._next_iteration)
        self.wave2.set_speed(target)
        self.wave2.set_target_speed(start, rate)
        if mu.rng().random() < self.WAVE_SHAPE_CHANGE_PROBABILITY:
            self.wave_a = mu.choice(self.possible)
        if mu.rng().random() < self.WAVE_SHAPE_CHANGE_PROBABILITY:
            self.wave_b = mu.choice(self.possible)
        if mu.rng().random() < self.SWITCH_PROBABILITY:
            self.switch = not self.switch
        if mu.rng().random() < self.SWITCH_PROBABILITY:
            self.freq_switch = not self.freq_switch

    def get_pulse(self):
        phase = (self.wave.current_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        inv_phase = 1.0 - phase
        amp_a = self.wave2.get_position(self.wave_a.name)
        amp_b = self.wave.get_position(self.wave_b.name)
        freq_a = mu.scale_between(inv_phase, *self.FREQ_RANGE)
        freq_b = mu.scale_between(phase, *self.FREQ_RANGE)
        if self.switch:
            amp_a, amp_b = amp_b, amp_a
        if self.freq_switch:
            freq_a, freq_b = freq_b, freq_a
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def segment_label(self):
        return self._seg


# =====================================================================
# Simplex
# =====================================================================
class SimplexActivity(BaseActivity):
    PRESETS = {
        "STANDARD": dict(amp_time=(0.2, 4.0), amp_rot=(0.0, math.pi * 0.2),
                         change=(0.1, 0.5), amp_radius=0.3, freq_radius=0.2),
        "PRO": dict(amp_time=(0.2, 0.8), amp_rot=(math.pi * 0.5, math.pi * 4.0),
                    change=(0.2, 0.5), amp_radius=0.4, freq_radius=0.3),
        "TURBO": dict(amp_time=(0.1, 0.5), amp_rot=(math.pi * 3, math.pi * 6.0),
                      change=(0.2, 0.5), amp_radius=0.6, freq_radius=0.3),
    }
    FREQ_TIME_SPEED = 0.2
    FREQ_ROTATION_SPEED = 0.1

    def __init__(self, params=None):
        super().__init__(params)
        preset = str(self.params.get("preset", "PRO")).upper()
        self.cfg = self.PRESETS.get(preset, self.PRESETS["PRO"])
        self.noise = NoiseGenerator()
        self.elapsed = 0.0
        self.phase_time = 0.0
        self.phase_rotation = 0.0
        self.amp_time_speed = NiceSmoother(mu.random_in_range(*self.cfg["amp_time"]), (0.0, math.pi * 10.0))
        self.amp_rot_speed = NiceSmoother(mu.random_in_range(*self.cfg["amp_rot"]), (0.0, math.pi * 10.0))
        self.amp_time_timer = Timer(lambda: mu.random_in_range(10.0, 50.0), True, self._amp_time_change)
        self.amp_rot_timer = Timer(lambda: mu.random_in_range(10.0, 50.0), True, self._amp_rot_change)

    def initialise(self):
        self.manager.register(self.amp_time_speed)
        self.manager.register(self.amp_rot_speed)
        self.manager.register(self.amp_time_timer)
        self.manager.register(self.amp_rot_timer)
        self.amp_time_timer.start()
        self.amp_rot_timer.start()

    def _amp_time_change(self):
        self.amp_time_speed.set_target(mu.random_in_range(*self.cfg["amp_time"]),
                                       mu.random_in_range(*self.cfg["change"]))

    def _amp_rot_change(self):
        self.amp_rot_speed.set_target(mu.random_in_range(*self.cfg["amp_rot"]),
                                      mu.random_in_range(*self.cfg["change"]))

    def run_simulation(self, dt):
        super().run_simulation(dt)
        self.elapsed += dt
        self.phase_time += self.amp_time_speed.value * dt
        self.phase_rotation += self.amp_rot_speed.value * dt

    def get_pulse(self):
        amp_a, amp_b = self.noise.get_noise(self.phase_time, self.phase_rotation,
                                            self.cfg["amp_radius"], 2, True)
        freq_a, freq_b = self.noise.get_noise(self.elapsed * self.FREQ_TIME_SPEED,
                                             self.elapsed * self.FREQ_ROTATION_SPEED,
                                             self.cfg["freq_radius"], 1, True)
        return Pulse(amp_a, amp_b, freq_a, freq_b)


# =====================================================================
# Relentless
# =====================================================================
class RelentlessActivity(BaseActivity):
    SPEED_RANGE = (0.2, 0.4)
    SPEED_BIAS = 2.5
    FREQ_RANGE = (0.0, 1.0)
    POWER_RANGE = (0.8, 0.95)
    REPEAT_OPTIONS = (1, 2, 3, 4)

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.freq_a = mu.random_in_range(*self.FREQ_RANGE)
        self.freq_b = mu.random_in_range(*self.FREQ_RANGE)
        self.swap = mu.next_bool()
        self._seg = 0
        self.iteration_timer = Timer(lambda: mu.random_in_range(1.0, 30.0), True,
                                     lambda: self.wave.stop_at_end_of_cycle(self._next_iteration))

    def initialise(self):
        self.manager.register(self.wave)
        self.manager.register(self.iteration_timer)
        self.iteration_timer.start()
        self._next_iteration()

    def _random_wave_shape(self):
        has_hold = mu.rng().random() < 0.3
        split_attack = mu.rng().random() < 0.5
        split_decay = mu.rng().random() < 0.5
        points = []
        peak = mu.random_in_range(*self.POWER_RANGE)
        attack_w = mu.random_in_range(0.5, 3.0)
        decay_w = mu.random_in_range(0.5, 3.0)
        hold_w = mu.random_in_range(0.25, 1.5) if has_hold else 0.0
        total = attack_w + decay_w + hold_w
        attack_p = attack_w / total
        decay_p = decay_w / total
        hold_p = hold_w / total
        x = 0.0
        points.append(WavePoint(x, 0.0))
        if split_attack:
            ratio = mu.random_in_range(0.3, 0.7)
            points.append(WavePoint(x + attack_p * ratio, peak * mu.random_in_range(0.2, 0.8)))
        x += attack_p
        points.append(WavePoint(x, peak))
        if has_hold:
            x += hold_p
            points.append(WavePoint(x, peak))
        if split_decay:
            ratio = mu.random_in_range(0.3, 0.7)
            points.append(WavePoint(x + decay_p * ratio, peak * mu.random_in_range(0.2, 0.8)))
        return WaveShape("randomWave", points, HERMITE)

    def _next_iteration(self):
        self._seg += 1
        speed = mu.random_in_range(*self.SPEED_RANGE, self.SPEED_BIAS)
        self.freq_a = mu.random_in_range(*self.FREQ_RANGE)
        self.freq_b = mu.random_in_range(*self.FREQ_RANGE)
        self.swap = mu.next_bool()
        long_wave = CyclicalWave(self._random_wave_shape())
        repeats = mu.choice(self.REPEAT_OPTIONS)
        short_wave = create_repeated_wave(long_wave, repeats, "shortWave")
        self.wave.add_wave(long_wave, "longWave")
        self.wave.add_wave(short_wave, "shortWave")
        self.wave.set_speed(speed)
        self.wave.restart()

    def get_pulse(self):
        long_amp = self.wave.get_position("longWave")
        short_amp = self.wave.get_position("shortWave")
        amp_a, amp_b = (long_amp, short_amp) if self.swap else (short_amp, long_amp)
        return Pulse(amp_a, amp_b, self.freq_a, self.freq_b)

    def segment_label(self):
        return self._seg


# =====================================================================
# Overflowing
# =====================================================================
class OverflowingActivity(BaseActivity):
    SPEED_RANGE = (0.05, 0.1)
    FREQ_RANGE = (0.6, 1.0)
    FREQ_CHANGE_RATE_RANGE = (0.02, 0.05)
    LONG_POWER_RANGE = (0.85, 0.99)
    SHORT_POWER_RANGE = (0.8, 0.99)

    def __init__(self, params=None):
        super().__init__(params)
        self.wave = WaveManager()
        self.freq_a = NiceSmoother(mu.random_in_range(*self.FREQ_RANGE))
        self.freq_b = NiceSmoother(mu.random_in_range(*self.FREQ_RANGE))
        self._seg = 0

    def initialise(self):
        self._update_freq_target(self.freq_a)
        self._update_freq_target(self.freq_b)
        self.manager.register(self.wave)
        self.manager.register(self.freq_a)
        self.manager.register(self.freq_b)
        self._next_iteration()

    def _long_wave_shape(self):
        power = mu.random_in_range(*self.LONG_POWER_RANGE)
        low_power = power * 0.85
        high_portion = mu.random_in_range(0.8, 0.9)
        num_laps = mu.random_int_in_range(5, 20)
        lap_inc = high_portion / (num_laps * 2.0)
        points = []
        t = 0.0
        points.append(WavePoint(t, 0.0))
        t = (1.0 - high_portion) / 2.0
        points.append(WavePoint(t, low_power))
        for _ in range(num_laps):
            t += lap_inc
            points.append(WavePoint(t, power))
            t += lap_inc
            points.append(WavePoint(t, low_power))
        return WaveShape("longWave", points, HERMITE)

    def _update_freq_target(self, freq):
        freq.set_target(mu.random_in_range(*self.FREQ_RANGE),
                        mu.random_in_range(*self.FREQ_CHANGE_RATE_RANGE),
                        on_reached=lambda: self._update_freq_target(freq))

    def _next_iteration(self):
        self._seg += 1
        speed = mu.random_in_range(*self.SPEED_RANGE)
        power_short = mu.random_in_range(*self.SHORT_POWER_RANGE)
        long_wave = CyclicalWave(self._long_wave_shape())
        short_shape = _wave("shortWaveShape", [(0.0, 0.0), (0.5, power_short)])
        repeats = mu.random_int_in_range(4, 10)
        short_wave = create_repeated_wave(short_shape, repeats, "shortWave")
        self.wave.add_wave(long_wave, "longWave")
        self.wave.add_wave(short_wave, "shortWave")
        self.wave.set_speed(speed)
        self.wave.restart()
        self.wave.stop_at_end_of_cycle(self._next_iteration)

    def get_pulse(self):
        long_amp = self.wave.get_position("longWave")
        short_amp = self.wave.get_position("shortWave")
        if short_amp > long_amp:
            short_amp = long_amp
        return Pulse(short_amp, long_amp, self.freq_a.value, self.freq_b.value)

    def segment_label(self):
        return self._seg


# =====================================================================
# Succubus
# =====================================================================
class SuccubusActivity(BaseActivity):
    SPEED_RANGE = (0.06, 2.0)
    SPEED_BIAS = 3.5
    SPEED_CHANGE_RATE_RANGE = (0.03, 0.2)
    PROPORTION_RANGE = (0.0, 1.0)
    PROPORTION_CHANGE_RATE = 0.05
    WAVE_SHAPE_CHANGE_PROBABILITY = 0.3
    PROPORTION_CHANGE_PROBABILITY = 0.3
    SPEED_CHANGE_PROBABILITY = 0.3
    WAVE_POINTS_RANGE = (2, 6)

    def __init__(self, params=None):
        super().__init__(params)
        self.wm1 = WaveManager()
        self.wm2 = WaveManager()
        self.wms = [self.wm1, self.wm2]
        self.amp_prop_a = NiceSmoother(mu.random_in_range(*self.PROPORTION_RANGE))
        self.amp_prop_b = NiceSmoother(mu.random_in_range(*self.PROPORTION_RANGE))
        self.freq_prop_a = NiceSmoother(mu.random_in_range(*self.PROPORTION_RANGE))
        self.freq_prop_b = NiceSmoother(mu.random_in_range(*self.PROPORTION_RANGE))
        self.shape_timer = Timer(lambda: mu.random_in_range(10.0, 40.0), True, self._shape_change)
        self.speed_timer = Timer(lambda: mu.random_in_range(10.0, 30.0), True, self._speed_change)
        self.proportion_timer = Timer(lambda: mu.random_in_range(10.0, 30.0), True, self._proportion_change)

    def initialise(self):
        for wm in self.wms:
            wm.add_wave(self._random_wave(mu.random_int_in_range(*self.WAVE_POINTS_RANGE)), "amp")
            wm.add_wave(self._random_wave(mu.random_int_in_range(*self.WAVE_POINTS_RANGE)), "freq")
            wm.set_speed(mu.random_in_range(*self.SPEED_RANGE, self.SPEED_BIAS))
            self.manager.register(wm)
        for c in (self.shape_timer, self.speed_timer, self.proportion_timer,
                  self.amp_prop_a, self.amp_prop_b, self.freq_prop_a, self.freq_prop_b):
            self.manager.register(c)
        self.shape_timer.start()
        self.speed_timer.start()
        self.proportion_timer.start()

    def _random_wave(self, num_points):
        points = []
        times = []
        max_power = 0.95
        lower_bound = 0.8
        min_spacing = 0.05
        for _ in range(num_points):
            while True:
                t = mu.random_in_range(0.0, TMAX)
                if all(abs(it - t) >= min_spacing for it in times):
                    break
            times.append(t)
            points.append(WavePoint(t, mu.random_in_range(0.0, max_power)))
        if all(p.position < lower_bound for p in points):
            idx = mu.next_int(num_points)
            points[idx] = WavePoint(points[idx].time, mu.random_in_range(lower_bound, max_power))
        return CyclicalWave(WaveShape("randomWave", points, HERMITE))

    def _proportion_change(self):
        for sm in (self.amp_prop_a, self.amp_prop_b, self.freq_prop_a, self.freq_prop_b):
            if mu.rng().random() < self.PROPORTION_CHANGE_PROBABILITY:
                sm.set_target(mu.random_in_range(*self.PROPORTION_RANGE), self.PROPORTION_CHANGE_RATE)

    def _shape_change(self):
        for wm in self.wms:
            if mu.rng().random() < self.WAVE_SHAPE_CHANGE_PROBABILITY:
                wm.add_wave(self._random_wave(mu.random_int_in_range(*self.WAVE_POINTS_RANGE)), "amp")
            if mu.rng().random() < self.WAVE_SHAPE_CHANGE_PROBABILITY:
                wm.add_wave(self._random_wave(mu.random_int_in_range(*self.WAVE_POINTS_RANGE)), "freq")

    def _speed_change(self):
        for wm in self.wms:
            if mu.rng().random() < self.SPEED_CHANGE_PROBABILITY:
                wm.set_target_speed(mu.random_in_range(*self.SPEED_RANGE, self.SPEED_BIAS),
                                    mu.random_in_range(*self.SPEED_CHANGE_RATE_RANGE))

    def get_pulse(self):
        ap1 = self.wm1.get_position("amp")
        ap2 = self.wm2.get_position("amp")
        fp1 = self.wm1.get_position("freq")
        fp2 = self.wm2.get_position("freq")
        amp_a = ap1 * self.amp_prop_a.value + ap2 * (1.0 - self.amp_prop_a.value)
        amp_b = ap1 * self.amp_prop_b.value + ap2 * (1.0 - self.amp_prop_b.value)
        freq_a = fp1 * self.freq_prop_a.value + fp2 * (1.0 - self.freq_prop_a.value)
        freq_b = fp1 * self.freq_prop_b.value + fp2 * (1.0 - self.freq_prop_b.value)
        return Pulse(amp_a, amp_b, freq_a, freq_b)


# =====================================================================
# Sine time
# =====================================================================
class SineTimeActivity(BaseActivity):
    SINE_MAG_RANGE = (0.1, 0.3)
    SINE_SPEED_RANGE = (0.4, 0.88)
    FREQ_RANGE = (0.5, 0.75)
    FREQ_SHIFT_RANGE = (-0.25, 0.25)
    AMP_RANGE = (0.8, 1.0)
    FADE_TIME = 0.5

    def __init__(self, params=None):
        super().__init__(params)
        self.manual = bool(self.params.get("manual", False))
        self._seg = 0
        self.sine_speed = 0.2
        self.freq_change = 0.0
        self.sine_mag = NiceSmoother(0.2, (0.0, 1.0))
        self.offset = NiceSmoother(0.0, (-math.pi, math.pi))
        self.sine_phase = 0.0
        self.freq_a = mu.random_in_range(*self.FREQ_RANGE)
        self.freq_b = mu.random_in_range(*self.FREQ_RANGE)
        self.pattern_timer = Timer(lambda: mu.random_in_range(8.0, 15.0), False,
                                   lambda: self.break_timer.reset())
        self.break_timer = Timer(lambda: mu.random_in_range(0.8, 5.0, 2.5), False, self._next_iteration)

    def initialise(self):
        self.manager.register(self.pattern_timer)
        self.manager.register(self.break_timer)
        self.manager.register(self.offset)
        self.manager.register(self.sine_mag)
        if self.manual:
            self.sine_mag.set_immediately(float(self.params.get("sine_mag", 0.2)))
            self.sine_speed = float(self.params.get("sine_speed", 0.6))
            self.freq_change = float(self.params.get("freq_change", 0.0))
            self.offset.set_immediately(float(self.params.get("offset", 0.0)))
        self._next_iteration()

    def _next_iteration(self):
        self._seg += 1
        if not self.manual:
            self.sine_mag.set_immediately(mu.random_in_range(*self.SINE_MAG_RANGE))
            self.sine_speed = mu.random_in_range(*self.SINE_SPEED_RANGE)
            self.freq_change = mu.random_in_range(*self.FREQ_SHIFT_RANGE)
            self.offset.set_immediately(mu.random_in_range(-math.pi, math.pi))
        self.freq_a = mu.random_in_range(*self.FREQ_RANGE)
        self.freq_b = mu.random_in_range(*self.FREQ_RANGE)
        self.pattern_timer.reset()

    def run_simulation(self, dt):
        super().run_simulation(dt)
        self.sine_phase += 2.0 * math.pi * self.sine_speed * dt
        self.sine_phase %= 2.0 * math.pi

    def get_pulse(self):
        if self.break_timer.is_running:
            return Pulse()
        fade_in = mu.clamp(self.pattern_timer.elapsed_time / self.FADE_TIME, 0.0, 1.0)
        fade_out = mu.clamp(self.pattern_timer.remaining_time / self.FADE_TIME, 0.0, 1.0)
        fade = min(fade_in, fade_out)
        base_amp = 1.0 - self.sine_mag.value
        amp = math.sqrt(1.0 - max(0.0, self.SINE_MAG_RANGE[1] - self.sine_mag.value))
        phase_a = math.sin(self.sine_phase)
        phase_b = math.sin(self.sine_phase + self.offset.value)
        amp_a = mu.clamp((base_amp + self.sine_mag.value * phase_a) * fade * amp, 0.0, 1.0)
        amp_b = mu.clamp((base_amp + self.sine_mag.value * phase_b) * fade * amp, 0.0, 1.0)
        freq_a = mu.clamp(self.freq_a + self.freq_change * phase_a, 0.0, 1.0)
        freq_b = mu.clamp(self.freq_b + self.freq_change * phase_b, 0.0, 1.0)
        return Pulse(amp_a, amp_b, freq_a, freq_b)

    def segment_label(self):
        return "break" if self.break_timer.is_running else self._seg

    def manual_snapshot(self):
        return {
            "manual": True,
            "sine_mag": round(self.sine_mag.value, 3),
            "sine_speed": round(self.sine_speed, 3),
            "offset": round(self.offset.value, 3),
            "freq_change": round(self.freq_change, 3),
        }


# =====================================================================
# Generator registrations
# =====================================================================
def _make(type_id, label, activity, specs=()):
    cls = type(
        f"{activity.__name__}Generator",
        (ActivityGenerator,),
        {
            "type_id": type_id,
            "label": label,
            "activity_cls": activity,
            "param_specs": classmethod(lambda cls, _s=tuple(specs): list(_s)),
        },
    )
    return register(cls)


_make("licks", "Infinite licks", LickActivity, [
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("speed", "Speed (manual)", "float", 1.0, 0.1, 5.0, 0.1),
    ParamSpec("lick_start", "Lick start (manual)", "float", 0.5, 0.0, 1.0, 0.01),
    ParamSpec("lick_end", "Lick end (manual)", "float", 0.5, 0.0, 1.0, 0.01),
    ParamSpec("lick_type", "Lick type (manual)", "choice", "BIDIRECTIONAL",
              choices=("UNIDIRECTIONAL", "BIDIRECTIONAL")),
    ParamSpec("amp_type", "Amp type (manual)", "choice", "LICK",
              choices=("CONSISTENT", "LICK", "DIP", "RAMP", "FLICKS")),
])
_make("penetration", "Penetration", PenetrationActivity, [
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("speed", "Speed (manual)", "float", 1.0, 0.3, 3.0, 0.05),
    ParamSpec("feel", "Feel exponent (manual)", "float", 0.75, 0.5, 1.0, 0.05),
])
_make("vibrator", "Sliding vibrator", VibroActivity, [
    ParamSpec("hold_probability", "Hold probability", "float", 0.25, 0.0, 1.0, 0.05),
    ParamSpec("pulse_time", "Pulse time (s)", "float", 0.3, 0.1, 1.0, 0.05),
    ParamSpec("pulse_duty_cycle", "Pulse duty cycle", "float", 1.0, 0.1, 1.0, 0.05),
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("position", "Position (manual)", "float", 0.5, 0.0, 1.0, 0.01),
    ParamSpec("frequency", "Frequency (manual)", "float", 0.5, 0.0, 1.0, 0.01),
])
_make("milkmaster", "Milkmaster 3000", MilkerActivity)
_make("chaos", "Chaos", ChaosActivity, [
    ParamSpec("cycle_time", "Cycle time (s)", "float", 1.0, 0.1, 5.0, 0.1),
])
_make("luxury_hj", "Luxury HJ", LuxuryHJActivity, [
    ParamSpec("bonus_probability", "Bonus probability", "float", 0.7, 0.0, 3.0, 0.1),
    ParamSpec("amplitude_jitter", "Amplitude jitter", "float", 0.15, 0.0, 0.5, 0.05),
    ParamSpec("timing_jitter", "Timing jitter", "float", 0.15, 0.0, 0.5, 0.05),
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("speed", "Stroke speed (manual)", "float", 1.0, 0.3, 3.0, 0.05),
])
_make("opposites", "Opposites", OppositesActivity)
_make("calibration1", "Calibration 1", Calibration1Activity)
_make("calibration2", "Calibration 2", Calibration2Activity)
_make("bj_megamix", "BJ Megamix", BJActivity, [
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("speed", "Speed (manual)", "float", 1.0, 0.1, 3.0, 0.05),
    ParamSpec("stage", "Stage (manual)", "choice", "Suck",
              choices=("FullLick", "TipLick", "Suck", "Deepthroat")),
])
_make("fast_slow", "Fast/slow", FastSlowActivity)
_make("simplex", "Simplex", SimplexActivity, [
    ParamSpec("preset", "Preset", "choice", "PRO", choices=("STANDARD", "PRO", "TURBO")),
])
_make("relentless", "Relentless", RelentlessActivity)
_make("overflowing", "Overflowing", OverflowingActivity)
_make("succubus", "Succubus", SuccubusActivity)
_make("sine_time", "Sine time", SineTimeActivity, [
    ParamSpec("manual", "Manual control", "bool", False),
    ParamSpec("sine_mag", "Sine magnitude (manual)", "float", 0.2, 0.0, 0.4, 0.01),
    ParamSpec("sine_speed", "Sine speed (manual)", "float", 0.6, 0.2, 1.0, 0.01),
    ParamSpec("offset", "Sine offset (manual)", "float", 0.0, -3.14159, 3.14159, 0.05),
    ParamSpec("freq_change", "Freq change (manual)", "float", 0.0, -0.25, 0.25, 0.01),
])
