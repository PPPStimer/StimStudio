# Beat Timestamps CSV Format

**BeatGen → "Export Beat Timestamps (CSV)"**

---

## Overview

A plain CSV file listing the start time of every beat, grouped by track. Each track occupies **two adjacent columns**: beat time and mark.

```csv
Kick,Kick mark,Snare,Snare mark
0.500,,0.750,
2.000,pattern start,,
2.500,,,
3.500,pattern end,,
4.000,pattern start; pattern end,,
```

---

## Encoding & syntax

| Property | Value |
|---|---|
| Encoding | UTF-8 **with BOM** (`EF BB BF`). Strip it, e.g. Python `encoding="utf-8-sig"`. |
| Delimiter | `,` (comma) |
| Quoting | Standard RFC 4180: a field containing `,`, `"` or a newline is wrapped in `"…"`, and inner quotes are doubled. Use a real CSV parser, not `split(",")`. |
| Line ending | `\r\n` |
| Decimal separator | `.` (always, regardless of locale) |

---

## Structure

**Row 1: header.** Contains `2 × N` cells for `N` tracks:

| Column | Content |
|---|---|
| `2k` (even, 0-based) | Track name |
| `2k + 1` (odd) | Track name + `" mark"` |

**Rows 2+: data.** Row `i` holds the `i`-th beat of every track.

| Cell | Type | Description |
|---|---|---|
| time | float, 3 decimals | Beat start time in **seconds** from the start of the audio. |
| mark | string | Empty, `pattern start`, `pattern end`, or several marks joined by `"; "`. |

---

## Rules

- **Columns are independent.** Each track's column pair is its own list; cells in the same row are *not* related in time. Read every column pair top-down.
- **Padding.** Tracks have different beat counts. Once a track runs out of beats, both of its cells are empty (`""`) for the remaining rows. Stop reading that track at its first empty time cell.
- **Order.** Within a track, times are strictly ascending, with no duplicates.
- **Marks.** `pattern start` is on the first beat of a pattern and `pattern end` on its last beat. Beats between them, and standalone beats, have an empty mark. A one-beat pattern, or a beat shared by two adjacent patterns, carries both marks: `pattern start; pattern end`. Split on `"; "`.
- **Track pairing.** Pair columns **by position** (0+1, 2+3, …), not by name. Track names are not unique and may themselves end in `" mark"`.
- **Empty tracks.** A track with no beats still has its header columns. All of its data cells are empty.
- **Empty file.** If no track has a time value, the file may contain only the header row.

---

## Reference reader (Python)

```python
import csv

def read_beat_timestamps(path):
    """Return [(track_name, [(time_s, [marks...]), ...]), ...]."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    tracks = []
    for c in range(0, len(header), 2):
        beats = []
        for row in data:
            time = row[c] if c < len(row) else ""
            if time == "":
                break
            mark = row[c + 1] if c + 1 < len(row) else ""
            beats.append((float(time), mark.split("; ") if mark else []))
        tracks.append((header[c], beats))
    return tracks
```
