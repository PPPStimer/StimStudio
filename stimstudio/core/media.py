"""Media files a project refers to: audio clips, the video, file parameters of
generators (WAV import).

In memory every path is absolute. A saved project stores each path twice:
``path`` relative to the project folder (POSIX slashes, so a project made on
Windows opens elsewhere) and, under ``abs_paths``, the absolute path at save
time. Loading tries the relative path first — the project folder may have been
moved or copied together with its media — and falls back to the absolute one.
A file on another drive than the project (no relative path exists) is stored
absolute only.

*Collect files* copies every file into one folder next to the project so the
whole thing can be moved or shared; *relink* points missing files at their new
location, found by name in a folder.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .model import AudioFileSource, Clip, GeneratorSource, Project, VideoFileSource

#: Name of the folder *Collect files* copies into (next to the project file).
DEFAULT_MEDIA_DIR = "media"
COPY_CHUNK = 4 * 1024 * 1024


class Cancelled(Exception):
    pass


def norm(path) -> str:
    """Absolute, normalised path (case kept)."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def same_path(a, b) -> bool:
    return os.path.normcase(norm(a)) == os.path.normcase(norm(b))


def path_params(type_id: str) -> list[str]:
    """Names of the file parameters (``kind == "path"``) of a generator."""
    from ..generators import get_generator   # lazy: populates the registry
    try:
        gen = get_generator(type_id)
    except KeyError:
        return []
    return [s.key for s in gen.param_specs() if s.kind == "path"]


# ── references in a live project ────────────────────────────────────────────
@dataclass
class MediaRef:
    track_id: str
    clip_id: str
    #: "path" (the clip source's own file) or "param:<name>" (a generator parameter)
    key: str
    path: str
    kind: str          # "video" | "audio" | "generator"
    clip_name: str = ""


def clip_refs(track_id: str, clip: Clip) -> list[MediaRef]:
    src = clip.source
    if isinstance(src, (AudioFileSource, VideoFileSource)):
        if not src.path:
            return []
        kind = "video" if isinstance(src, VideoFileSource) else "audio"
        return [MediaRef(track_id, clip.id, "path", src.path, kind, clip.name)]
    if isinstance(src, GeneratorSource):
        out = []
        for name in path_params(src.type_id):
            value = str(src.params.get(name) or "").strip()
            if value:
                out.append(MediaRef(track_id, clip.id, f"param:{name}", value, "generator",
                                    clip.name))
        return out
    return []


def media_refs(project: Project) -> list[MediaRef]:
    return [r for t in project.tracks for c in t.clips for r in clip_refs(t.id, c)]


def with_path(clip: Clip, key: str, new_path: str):
    """A copy of the clip's source with the file at *key* replaced."""
    src = clip.source.clone()
    if key == "path":
        src.path = str(new_path)
    elif key.startswith("param:"):
        src.params[key[6:]] = str(new_path)
    else:
        raise KeyError(key)
    return src


def relinked_sources(project: Project, mapping: dict[str, str]) -> dict[str, object]:
    """``clip id → new source`` for every clip whose file appears in *mapping*
    (old absolute path → new path; keys compared case-insensitively on Windows)."""
    lookup = {os.path.normcase(norm(k)): v for k, v in mapping.items()}
    out: dict[str, object] = {}
    for t in project.tracks:
        for c in t.clips:
            src = None
            for ref in clip_refs(t.id, c):
                new = lookup.get(os.path.normcase(norm(ref.path)))
                if new is None or same_path(new, ref.path):
                    continue
                tmp = Clip(src if src is not None else c.source)
                src = with_path(tmp, ref.key, norm(new))
            if src is not None:
                out[c.id] = src
    return out


# ── the project file: relative on save, resolved on load ────────────────────
def _clip_dicts(data: dict):
    """Every clip dict of a project dict, pattern pool entries included."""
    def walk(clips):
        for clip in clips:
            yield clip
            src = clip.get("source") or {}
            if src.get("kind") == "pattern":
                yield from walk(e["clip"] for e in src.get("pool", []) if e.get("clip"))
    for track in data.get("tracks", []):
        yield from walk(track.get("clips", []))


def _dict_refs(data: dict):
    """``(source dict, key, holder, field)`` for every file path in a project dict;
    ``holder[field]`` is the stored path."""
    for clip in _clip_dicts(data):
        src = clip.get("source") or {}
        kind = src.get("kind")
        if kind in ("audio", "video"):
            if src.get("path"):
                yield src, "path", src, "path"
        elif kind == "generator":
            params = src.get("params") or {}
            for name in path_params(str(src.get("type", ""))):
                if str(params.get(name) or "").strip():
                    yield src, f"param:{name}", params, name


def relative_path(path: str, base_dir) -> str | None:
    """*path* relative to *base_dir* with POSIX slashes, or None (another drive)."""
    try:
        return Path(os.path.relpath(norm(path), norm(base_dir))).as_posix()
    except ValueError:
        return None


def store_paths(data: dict, base_dir) -> dict:
    """Rewrite a ``Project.to_dict()`` result in place for saving next to *base_dir*.
    ``base_dir=None`` keeps every path absolute (autosave)."""
    for src, key, holder, fld in _dict_refs(data):
        absolute = norm(holder[fld])
        src.setdefault("abs_paths", {})[key] = absolute
        rel = relative_path(absolute, base_dir) if base_dir is not None else None
        holder[fld] = rel if rel is not None else absolute
    return data


def resolve_paths(data: dict, base_dir) -> dict:
    """The inverse of :func:`store_paths` (in place): every path absolute again.

    Relative paths are tried against *base_dir* first, then the stored absolute
    path; if neither exists the absolute one (the original location) is kept,
    so the file shows as missing and can be relinked by name.
    """
    for src, key, holder, fld in list(_dict_refs(data)):
        stored = str(holder[fld])
        absolute = (src.get("abs_paths") or {}).get(key)
        if os.path.isabs(stored):
            candidates = [stored]
        else:
            candidates = [os.path.join(os.fspath(base_dir), stored)] if base_dir else []
        if absolute:
            candidates.append(absolute)
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found is None:
            found = absolute or (candidates[0] if candidates else stored)
        holder[fld] = norm(found)
    for clip in _clip_dicts(data):
        (clip.get("source") or {}).pop("abs_paths", None)
    return data


# ── status, relink ──────────────────────────────────────────────────────────
@dataclass
class MediaFile:
    """One distinct file of the project with every clip that uses it."""

    path: str
    kind: str
    refs: list[MediaRef] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)

    @property
    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def inside(self, folder) -> bool:
        """True if the file lies in *folder* (or below it)."""
        try:
            rel = os.path.relpath(norm(self.path), norm(folder))
        except ValueError:
            return False
        return not rel.startswith("..") and not os.path.isabs(rel)


def media_files(project: Project) -> list[MediaFile]:
    files: dict[str, MediaFile] = {}
    for ref in media_refs(project):
        key = os.path.normcase(norm(ref.path))
        mf = files.get(key)
        if mf is None:
            mf = files[key] = MediaFile(norm(ref.path), ref.kind)
        mf.refs.append(ref)
    return sorted(files.values(), key=lambda m: (m.kind != "video", Path(m.path).name.lower()))


def missing_files(project: Project) -> list[MediaFile]:
    return [m for m in media_files(project) if not m.exists]


def search_folder(folder, names, cancel=None) -> dict[str, str]:
    """Find files by name below *folder*: ``lower-case name → path`` (first hit,
    the shallowest one wins)."""
    wanted = {n.lower() for n in names}
    found: dict[str, tuple[int, str]] = {}
    root = norm(folder)
    for dirpath, dirnames, filenames in os.walk(root):
        if cancel is not None and cancel():
            raise Cancelled()
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        depth = dirpath.count(os.sep)
        for f in filenames:
            low = f.lower()
            if low in wanted and (low not in found or depth < found[low][0]):
                found[low] = (depth, os.path.join(dirpath, f))
        if len(found) == len(wanted):
            break
    return {k: v for k, (_d, v) in found.items()}


def relink_by_name(files, folder, cancel=None) -> dict[str, str]:
    """``old path → new path`` for the *files* (MediaFile or paths) found in *folder*."""
    paths = [f.path if isinstance(f, MediaFile) else str(f) for f in files]
    hits = search_folder(folder, [Path(p).name for p in paths], cancel)
    return {p: hits[Path(p).name.lower()] for p in paths if Path(p).name.lower() in hits}


# ── collect files ───────────────────────────────────────────────────────────
COPY, INSIDE, REUSE, MISSING = "copy", "inside", "reuse", "missing"


@dataclass
class CollectItem:
    file: MediaFile
    action: str           # COPY / INSIDE (already in the project folder) / REUSE / MISSING
    target: str = ""      # where the file ends up ("" for MISSING)


def _fingerprint(path, chunk=COPY_CHUNK) -> tuple:
    """Size plus a hash of the head and the tail — fast for large videos."""
    size = os.path.getsize(path)
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return size, h.hexdigest()


def same_content(a, b) -> bool:
    try:
        return _fingerprint(a) == _fingerprint(b)
    except OSError:
        return False


def plan_collect(project: Project, project_dir, media_dir) -> list[CollectItem]:
    """What *Collect files* would do. Files already inside *project_dir* stay
    where they are; the others get a unique name in *media_dir*."""
    media_dir = norm(media_dir)
    taken: set[str] = set()
    items = []
    for mf in media_files(project):
        if not mf.exists:
            items.append(CollectItem(mf, MISSING))
            continue
        if mf.inside(project_dir):
            items.append(CollectItem(mf, INSIDE, mf.path))
            taken.add(os.path.normcase(mf.path))
            continue
        stem, ext = Path(mf.path).stem, Path(mf.path).suffix
        n = 1
        while True:
            name = f"{stem}{ext}" if n == 1 else f"{stem} ({n}){ext}"
            target = os.path.join(media_dir, name)
            key = os.path.normcase(target)
            if key not in taken:
                if not os.path.exists(target):
                    items.append(CollectItem(mf, COPY, target))
                    break
                if same_content(mf.path, target):
                    items.append(CollectItem(mf, REUSE, target))
                    break
            n += 1
        taken.add(os.path.normcase(items[-1].target))
    return items


def copy_file(src, dst, progress=None, cancel=None) -> None:
    """Copy in chunks through ``dst + ".part"``; a cancelled copy leaves nothing.
    *progress(done_bytes)* is called after every chunk."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    part = dst + ".part"
    done = 0
    try:
        with open(src, "rb") as fi, open(part, "wb") as fo:
            while True:
                if cancel is not None and cancel():
                    raise Cancelled()
                buf = fi.read(COPY_CHUNK)
                if not buf:
                    break
                fo.write(buf)
                done += len(buf)
                if progress is not None:
                    progress(done)
        shutil.copystat(src, part)
        os.replace(part, dst)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


def run_collect(items: list[CollectItem], progress=None, cancel=None) -> dict[str, str]:
    """Copy the COPY items; returns ``old path → new path`` for every file that
    moved (COPY and REUSE). *progress(name, fraction)* reports overall progress.
    On cancel the files copied so far stay (they are complete) and Cancelled is
    raised."""
    todo = [i for i in items if i.action == COPY]
    total = sum(i.file.size for i in todo) or 1
    finished = 0
    for item in todo:
        name = Path(item.file.path).name

        def step(done, name=name, base=finished):
            if progress is not None:
                progress(name, min(1.0, (base + done) / total))

        copy_file(item.file.path, item.target, step, cancel)
        finished += item.file.size
    return {i.file.path: i.target for i in items if i.action in (COPY, REUSE)}
