import pytest

from stimstudio.core.model import BeatMarksSource
from stimstudio.io.beat_csv import blocks_to_clips, read_beat_timestamps, split_blocks

SPEC_EXAMPLE = ("Kick,Kick mark,Snare,Snare mark\r\n"
                "0.500,,0.750,\r\n"
                "2.000,pattern start,,\r\n"
                "2.500,,,\r\n"
                "3.500,pattern end,,\r\n"
                "4.000,pattern start; pattern end,,\r\n")


def write(tmp_path, text, name="beats.csv"):
    path = tmp_path / name
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    return path


def test_reads_spec_example(tmp_path):
    tracks = read_beat_timestamps(write(tmp_path, SPEC_EXAMPLE))
    assert [t.name for t in tracks] == ["Kick", "Snare"]
    kick, snare = tracks
    assert kick.beats == [(0.5, []), (2.0, ["pattern start"]), (2.5, []),
                          (3.5, ["pattern end"]), (4.0, ["pattern start", "pattern end"])]
    assert snare.beats == [(0.75, [])]


def test_quoted_names_padding_and_empty_tracks(tmp_path):
    text = ('"A, b","A, b mark",Empty,Empty mark,"x mark","x mark mark"\r\n'
            "1.000,,,,0.250,\r\n"
            "2.000,,,,,\r\n")
    tracks = read_beat_timestamps(write(tmp_path, text))
    assert [t.name for t in tracks] == ["A, b", "Empty", "x mark"]
    assert [len(t.beats) for t in tracks] == [2, 0, 1]


def test_header_only_and_bad_time(tmp_path):
    tracks = read_beat_timestamps(write(tmp_path, "Kick,Kick mark\r\n"))
    assert len(tracks) == 1 and tracks[0].beats == []
    with pytest.raises(ValueError, match="row 2"):
        read_beat_timestamps(write(tmp_path, "Kick,Kick mark\r\nabc,\r\n", "bad.csv"))


S, E = "pattern start", "pattern end"


def test_blocks_from_spec_example():
    beats = [(0.5, []), (2.0, [S]), (2.5, []), (3.5, [E]), (4.0, [S, E])]
    assert split_blocks(beats) == [[0.5], [2.0, 2.5, 3.5], [4.0]]


def test_shared_beat_lies_in_both_blocks():
    beats = [(1.0, [S]), (1.5, []), (2.0, [S, E]), (2.5, []), (3.0, [E])]
    assert split_blocks(beats) == [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]]


def test_loose_runs_and_open_group():
    beats = [(0.0, []), (0.5, []), (1.0, [S]), (1.5, [E]), (2.0, []), (2.2, []),
             (3.0, [S]), (3.5, [])]
    assert split_blocks(beats) == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.2], [3.0, 3.5]]


def test_blocks_to_clips_tails_and_no_overlap():
    clips = blocks_to_clips([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [10.0]], "Kick", offset_s=0.5)
    assert [c.name for c in clips] == ["Kick 1", "Kick 2", "Kick 3"]
    a, b, c = clips
    assert isinstance(a.source, BeatMarksSource)
    assert a.start_s == pytest.approx(1.5) and a.length_s == pytest.approx(1.0)   # cut at shared
    assert [m.t_s for m in b.source.marks] == pytest.approx([0.0, 0.5, 1.0])
    assert b.length_s == pytest.approx(1.5)          # median gap tail
    assert c.length_s == pytest.approx(0.5)          # single beat: default tail
