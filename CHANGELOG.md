# Changelog

All notable changes to StimStudio are recorded here, briefly.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-11-27

### Added
- **Output signal monitor** in the Output dock (audio outputs only): a two-channel oscilloscope (window 2 ms – 1 s, trigger on A or B, Hold), a spectrum with the strongest frequency per channel, an XY (Lissajous) view with the phase of B against A, and RMS / peak meters in dBFS. It shows what the sound card actually receives, after power, mute, soft start and fades.
- **Video fast-forward x2 / x5 / x10** in the Video dock. A playing output pauses with its usual fade and resumes from the new spot with soft start when fast-forward is turned off. PANIC, Stop or reaching the end cancels the resume.
- **Funscript export** (File → Export funscript, or Export… → Funscript): amplitude and frequency of channels A and B. Writes four files (`.amp_a/.amp_b/.freq_a/.freq_b`) or one multi-axis file, with point simplification (tolerance) and range / stems support. Exported files import back onto the same lanes.
- **Signal summary** in the export dialog for WAV / FLAC: format, sample rate, length, file size, mode, frequencies, wave shape, triphase handling, level and the settings source, shown before Export / Cancel.
- **View → Interface size** (applied at the next start) and `--scale X` on the command line: 110–200 % scales everything on top of the Windows display scale, 80–90 % makes the text smaller.
- `tools/ui_audit.py`: the interface on other screen sizes and display scales, checked on one monitor — 18 profiles (1366×768 to 3840×2160, at 80 %, 100 % and the usual Windows scales), screenshots plus a report of cut text, contents cut by scroll areas, windows larger than the screen and a hidden PANIC button; `tests/test_ui_scale.py` runs it on the three tightest profiles.
- **Marker window** with a name and a colour (swatches or any colour). It opens on Shift+M, on a double-click on the ruler and on a double-click on a flag; Cancel adds nothing. New markers start with the colour used last.

### Changed
- Audio (continuous): the tone's pitch range can be set anywhere from 20 Hz to 20 kHz (was 500–1000 Hz only; the default stays 500–1000 Hz). Audio (wavelet): carrier up to 20 kHz (was 8 kHz).
- Tabs (tabbed docks, Browser, Effects) follow the theme: the active tab blue, the others grey, white text. The native Windows style drew them light, unreadable under the theme's light text.
- The Output panel's sound device list shows whole device names (they were cut at 48 characters); the field narrows and the list and tooltip show the full name.
- File → Export audio now opens the export dialog (set to WAV) instead of writing right after the file dialog.
- The marker context menu has a single "Name and colour…" entry instead of separate Rename / Colour.

### Fixed
- Interface size 80 % / 90 % no longer leaves display faults (tabs drawn over panels, tree arrows on the wrong rows, old pixels until playback repainted the window): Qt can't draw at a device pixel ratio under 1, so below 100 % only the text gets smaller.
- The interface on small screens and at large display scales: the main window no longer grows past the screen (on first start it was 1520 px tall even on 1920×1080 at 100 %, and at 150 % it could not fit at all). Docks and big windows (Color rhythm, Triphase audio, Session builder, Video analysis, Beat marks from peaks) scroll instead, dialogs are sized to the screen, and a window saved on a bigger monitor is brought back onto the screen.
- PANIC stays in sight in a narrow window (its own toolbar; it used to fold into the » menu), and tabbed docks show whole names instead of "Insp…".
- The Output panel no longer cuts its right side in a narrow dock (a long sound card name or the key hint made it wider than the dock; it now wraps and scrolls), and the right dock column starts at least ~320 px wide on small screens.
- The frequency fields of the Output panel keep the exact Hz typed (some values came back 1 Hz lower, e.g. 50 → 49).
- Cut text: the Output panel's number fields (1000 Hz), the export dialog's signal summary (wrapped lines), clip beat-mark labels raised from 9 to 10 px.
- Howl activities Luxury HJ, Penetration, Sliding vibrator, Chaos and Succubus now show their stages in the Pattern editor (◀ Stage / Fit stage / Stage ▶), like Infinite licks. A Luxury HJ bonus burst is a stage of its own and freezes into manual mode with new Bonus (A / B) and Bonus speed parameters.

## [1.0.0] — 2026-11-05

First public release.

- Multitrack song editor with standard, triphase, modulator (audio) and video tracks; clips, fades, crossfades, envelopes, markers, loop, undo.
- Generators: Howl activities, shapes, rhythm (Euclidean, polyrhythm, groove), electrotherapy, tactile illusions, noise and chaos, breath and body, manual draw / steps, triphase; beat marks and pattern clips.
- Effects per clip, per track and on the master; automation of effect parameters; modulation routes from audio.
- Outputs: Coyote 3 via the Howl app (remote API), audio (continuous / wavelet, phase-true triphase), Intiface / Buttplug, dry run; power limits, soft start, mute, PANIC on Esc.
- Import of audio (to lanes or as a modulator, two-channel and three-phase), funscript, beat timestamps (CSV), .hwl and .parthwl; video sync with filmstrip, motion → stim, scene cuts → markers, colour rhythm.
- Export of .hwl and WAV / FLAC (mix or stems, whole project / loop / selection); presets; session builder; safety check; live recording; autosave and recovery.
