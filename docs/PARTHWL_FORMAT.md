# `.parthwl` file format

A `.parthwl` file is JSON holding one e-stim clip: a generator definition and, optionally, the rendered pulses. The format comes from WaveHowl; StimStudio reads and writes it for clip export (**File → Export clip as .parthwl**), clip import (**File → Import .parthwl clip**) and presets.

This page has two parts: how StimStudio uses the format, then the original WaveHowl specification (version 1).

---

## How StimStudio uses the format

### Writing

* **`source`** — an extra top-level field: `"generator"` or `"pulses"`.
* **Generator clips:** `generator` holds the real generator (`type`, `seed`, `params`), and `trim_start_s` is the clip's trim. StimStudio has many more generator types than WaveHowl; their ids and parameters are defined in `stimstudio/generators`.
* **Pulse clips** (imported data, recorded takes, glued clips; patterns are written rendered as pulse clips too): `generator` is the placeholder `{"type": "waveform", "seed": 0, "params": {}}`, `trim_start_s` is `0`, and the pulses are in `cache`.
* **`cache`** is always written. It holds the clip's source render, before the clip's effects and fades.
* **`effects`** is StimStudio's chain. It runs in list order; every effect has `channels` (`["A"]`, `["B"]` or `["A", "B"]`) and `enabled`; `amplitude_feel` and `frequency_feel` use `feel` directly as the exponent (`amp = amp ^ feel`). StimStudio also has effects that WaveHowl doesn't (Tremolo, Tempo gate, Slew limiter and others).
* **Fades:** `end_at_zero` + `fade_seconds` as in WaveHowl, plus `start_at_zero` + `fade_in_seconds` for a fade-in.

### Reading

Files from WaveHowl, HWL Viewer and StimStudio all import as clips.

* With `source` present, it decides between a generator clip and a pulse clip. Without it, a placeholder generator (`waveform` with empty `params`) together with a `cache` means a pulse clip (HWL Viewer files); anything else is a generator clip (WaveHowl files).
* A generator clip is re-rendered from its definition; its `cache` is not used.
* Effects without a `channels` key are WaveHowl-style and are converted: they are sorted into WaveHowl's canonical order (which also accepts `scale_frequency`, placed just before `scale_amplitude`), effects outside that list are dropped, `feel` becomes `1 / feel`, and the noise effects get the generator's seed.

`python -m stimstudio info file.parthwl` prints a summary of a file.

---

## WaveHowl specification (version 1)

**Version 1 — WaveHowl estim segment format**

---

### Overview

A `.parthwl` file is a **UTF-8 encoded JSON** file representing one estim signal segment for the DG Lab Coyote 3. It contains the generator definition (source of truth) and an optional pre-rendered pulse cache. A sequencer can either decode the cache directly or re-render from the definition.

---

### Top-level structure

```json
{
  "format":       "parthwl",
  "version":      1,
  "fs":           40,
  "name":         "My segment",
  "duration_s":   10.0,
  "trim_start_s": 0.0,
  "end_at_zero":  false,
  "fade_seconds": 0.3,
  "generator": { ... },
  "effects":  [ ... ],
  "channel_map": { "left": "A", "right": "B" },
  "cache": { ... }
}
```

#### Required fields

| Field | Type | Description |
|---|---|---|
| `format` | string | Always `"parthwl"`. Use this to identify the file type. |
| `version` | int | Always `1` in this revision. |
| `fs` | int | Sample rate in pulses/second. Always `40` (Coyote 3). |
| `name` | string | Human-readable segment name. |
| `duration_s` | float | Length of the rendered segment in seconds. |
| `generator` | object | Generator definition — see below. |
| `effects` | array | Effect chain — see below. May be empty `[]`. |
| `channel_map` | object | Maps `"left"` and `"right"` to Coyote channels (`"A"` or `"B"`). Default: `{"left":"A","right":"B"}`. |

#### Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `trim_start_s` | float | `0.0` | Seconds to skip from the start of the generator's output. The generator still simulates from t=0 (keeps stateful activities deterministic); only the output window shifts. |
| `end_at_zero` | bool | `false` | If true, amplitude is linearly faded to 0 over the final `fade_seconds` for clean concatenation. |
| `fade_seconds` | float | `0.3` | Length of the end-fade in seconds (only used when `end_at_zero` is true). |
| `cache` | object | absent | Pre-rendered pulse data — see below. |

---

### Generator object

```json
"generator": {
  "type":   "waveform",
  "seed":   0,
  "params": { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | Generator type ID — see generator types below. |
| `seed` | int | RNG seed for stochastic generators. Deterministic: same seed always produces the same output. |
| `params` | object | Key/value parameter map specific to the generator type. |

---

### Generator types

#### `"waveform"` — shaped amplitude + frequency envelopes

Both amplitude and frequency channels are driven by an independent periodic waveform.

| Param | Type | Default | Range | Description |
|---|---|---|---|---|
| `amp_shape` | choice | `"sine"` | see shapes | Amplitude envelope waveform shape |
| `amp_speed` | float | `0.5` | 0–20 Hz | Amplitude oscillation frequency |
| `amp_min` | float | `0.0` | 0–1 | Amplitude lower bound |
| `amp_max` | float | `1.0` | 0–1 | Amplitude upper bound |
| `freq_shape` | choice | `"sine"` | see shapes | Frequency envelope waveform shape |
| `freq_speed` | float | `0.2` | 0–20 Hz | Frequency oscillation frequency |
| `freq_min` | float | `0.2` | 0–1 | Frequency lower bound |
| `freq_max` | float | `0.8` | 0–1 | Frequency upper bound |
| `duty` | float | `0.5` | 0.01–0.99 | Duty cycle (used by `"pulse"` shape) |
| `phase` | float | `0.0` | 0–1 | Initial phase offset (in cycles) |
| `channel_phase` | float | `0.0` | 0–1 | Additional phase offset for the right/B channel relative to left/A |

**Waveform shapes** (used by `amp_shape`, `freq_shape`):
`"sine"`, `"triangle"`, `"saw"` (rising), `"ramp_down"` (falling), `"square"`, `"pulse"` (duty-controlled).

All shapes produce a unipolar value in `[0, 1]`.

---

#### `"channels"` — independent A/B channels with panning

| Param | Type | Default | Range | Description |
|---|---|---|---|---|
| `amp_shape` | choice | `"sine"` | see shapes | Amplitude envelope shape |
| `amp_speed` | float | `0.8` | 0–20 Hz | Amplitude oscillation rate |
| `amp_min` | float | `0.1` | 0–1 | Amplitude lower bound |
| `amp_max` | float | `1.0` | 0–1 | Amplitude upper bound |
| `freq` | float | `0.5` | 0–1 | Static frequency value for both channels |
| `pan_speed` | float | `0.25` | 0–10 Hz | Pan oscillation rate |
| `pan_depth` | float | `1.0` | 0–1 | Pan depth (0 = no panning, 1 = full) |
| `mode` | choice | `"ping_pong"` | — | Panning mode: `"ping_pong"`, `"phase"`, `"static"` |
| `phase_offset` | float | `0.5` | 0–1 | A/B phase offset (used in `"phase"` mode) |
| `duty` | float | `0.5` | 0.01–0.99 | Duty cycle |

**Pan modes:**
- `"ping_pong"` — amplitude slides back and forth between A and B via a triangle wave
- `"phase"` — A and B share the same envelope, offset by `phase_offset` cycles
- `"static"` — both channels identical, no panning

---

#### `"wav"` — WAV audio file import

Derives amplitude from windowed RMS and frequency from zero-crossing rate.

| Param | Type | Default | Description |
|---|---|---|---|
| `path` | string | `""` | Absolute path to the WAV/audio file. Must remain valid for re-render. |
| `gain` | float | `1.0` | Amplitude gain multiplier (0–4) |
| `floor_hz` | float | `0.0` | Audio frequency that maps to 0.0 in the pulse output |
| `ceil_hz` | float | `500.0` | Audio frequency that maps to 1.0 in the pulse output |

Stereo files map L→left channel, R→right channel. Mono files duplicate to both channels.

---

#### `"activity_*"` — Howl activity ports (stochastic, seeded)

Faithful ports of Howl's built-in activities. All are stochastic — `seed` controls their output. Use `"params": {}` unless noted otherwise.

| `type` | Label |
|---|---|
| `activity_licks` | Licks |
| `activity_penetration` | Penetration |
| `activity_vibrator` | Vibrator |
| `activity_milkmaster` | Milkmaster |
| `activity_chaos` | Chaos |
| `activity_luxury_hj` | Luxury HJ |
| `activity_opposites` | Opposites |
| `activity_calibration1` | Calibration 1 |
| `activity_calibration2` | Calibration 2 |
| `activity_bj_megamix` | BJ Megamix |
| `activity_fast_slow` | Fast/Slow |
| `activity_simplex` | Simplex |
| `activity_relentless` | Relentless |
| `activity_overflowing` | Overflowing |
| `activity_succubus` | Succubus |
| `activity_sine_time` | Sine Time |

---

### Effects array

Effects are post-processing steps applied after the generator runs. They are applied in **canonical order** (listed below) regardless of their order in the JSON array. An effect object with `"enabled": false` is skipped entirely.

**Canonical pipeline order:**

1. `channel_swap`
2. `invert_frequencies`
3. `frequency_adjust`
4. `random_amplitude_noise`
5. `random_frequency_noise`
6. `amplitude_feel`
7. `frequency_feel`
8. `scale_amplitude`

After all effects, every channel is clamped to `[0, 1]`.

---

#### Effect definitions

##### `channel_swap`
Swaps left and right channels (both amplitude and frequency). No extra params.
```json
{"type": "channel_swap"}
```

##### `invert_frequencies`
Replaces each frequency value `f` with `1 - f`. No extra params.
```json
{"type": "invert_frequencies"}
```

##### `frequency_adjust`
Shifts all frequency values by a constant offset, then clamps to `[0, 1]`.

| Param | Type | Default | Description |
|---|---|---|---|
| `amount` | float | `0.0` | Additive frequency offset (can be negative) |

```json
{"type": "frequency_adjust", "amount": 0.1}
```

##### `random_amplitude_noise`
Adds a smoothly interpolated random curve to amplitude channels.

| Param | Type | Default | Description |
|---|---|---|---|
| `amount` | float | `0.0` | Max noise deviation (can be negative) |
| `speed` | float | `4.0` | Noise speed in control points per second |

##### `random_frequency_noise`
Same as `random_amplitude_noise` but applied to frequency channels. Same params.

##### `amplitude_feel`
Power-curve shaping on amplitude: `amp = amp ^ (1 / feel)`.
- `feel < 1` → emphasizes peaks
- `feel > 1` → flattens, softens response
- `feel = 1` → no change

| Param | Type | Default |
|---|---|---|
| `feel` | float | `1.0` |

##### `frequency_feel`
Same as `amplitude_feel` but applied to frequency channels.

##### `scale_amplitude`
Multiplies both amplitude channels by a scalar.

| Param | Type | Default |
|---|---|---|
| `value` | float | `1.0` |

```json
{"type": "scale_amplitude", "value": 0.8}
```

---

### Cache object (optional)

When present, the cache holds a pre-rendered binary snapshot so a sequencer can skip re-rendering.

```json
"cache": {
  "encoding": "base64-f32le",
  "n":        400,
  "fs":       40,
  "data":     "<base64 string>"
}
```

| Field | Type | Description |
|---|---|---|
| `encoding` | string | Always `"base64-f32le"` |
| `n` | int | Number of pulses encoded in `data` |
| `fs` | int | Sample rate (always 40) |
| `data` | string | Base64-encoded pulse data — see binary layout below |

#### Binary layout of `data`

Decode the base64 string to get `n * 16` bytes. Layout: N consecutive pulses, each pulse is 4 × `float32` little-endian:

```
[l_amp: f32le] [r_amp: f32le] [l_freq: f32le] [r_freq: f32le]   ← repeated N times
```

All values are normalized to `[0, 1]`. Total decoded size = `n * 16` bytes.

This is the same per-pulse layout as Howl's native `.hwl` format (which prefixes the stream with an 8-byte `YEAHBOI!` magic header; `.parthwl` cache data has no header).

---

### Example file

```json
{
  "format": "parthwl",
  "version": 1,
  "fs": 40,
  "name": "gentle wave",
  "duration_s": 5.0,
  "trim_start_s": 0.0,
  "end_at_zero": true,
  "fade_seconds": 0.3,
  "generator": {
    "type": "waveform",
    "seed": 0,
    "params": {
      "amp_shape": "sine",
      "amp_speed": 0.5,
      "amp_min": 0.2,
      "amp_max": 0.8,
      "freq_shape": "triangle",
      "freq_speed": 0.2,
      "freq_min": 0.3,
      "freq_max": 0.7,
      "duty": 0.5,
      "phase": 0.0,
      "channel_phase": 0.25
    }
  },
  "effects": [
    {"type": "scale_amplitude", "value": 0.9},
    {"type": "amplitude_feel", "feel": 1.2}
  ],
  "channel_map": {"left": "A", "right": "B"}
}
```

---

### Reading a `.parthwl` file in another program

1. Parse as UTF-8 JSON.
2. Verify `format == "parthwl"` and `version == 1`.
3. **Use cache if present:** check `cache.encoding == "base64-f32le"`, base64-decode `cache.data`, unpack `cache.n` pulses as `[l_amp, r_amp, l_freq, r_freq]` float32-LE tuples.
4. **Re-render if no cache:** use `generator.type` + `generator.params` + `generator.seed` to run the generator for `round((trim_start_s + duration_s) * fs)` pulses, discard the first `round(trim_start_s * fs)` pulses, then apply effects in canonical order. If `end_at_zero` is true, fade amplitude to 0 over the final `fade_seconds`.
5. Map `channel_map.left` and `channel_map.right` to the target Coyote channels (A or B).
