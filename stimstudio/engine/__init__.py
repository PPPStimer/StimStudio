"""Faithful Python port of Howl's pattern-generation engine.

This package mirrors the Kotlin classes used by Howl's activities so the
generated signals match the original app's behavior:
  - mathutils: interpolation + helper math + a seeded RNG context
  - waves: WaveShape/WavePoint/CyclicalWave + JitterHandler
  - components: ActivityComponent/ActivityManager/Timer/NiceSmoother/WaveManager
  - noise: NoiseGenerator (OpenSimplex-based)

Activities step a simulation at HWL_PULSE_TIME (1/40 s) per pulse and read a
Pulse each step, exactly like Howl's ActivityHost.
"""
