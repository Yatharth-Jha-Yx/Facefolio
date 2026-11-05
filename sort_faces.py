#!/usr/bin/env python3
"""
sort_faces.py — offline photo/video sorter by person.

Usage:
    python sort_faces.py <input_folder> <output_folder> [--eps 0.45] [--video-step 2] [--symlink] [--review]

Pipeline: detect faces -> 512-d embedding (InsightFace) -> match known people (names.json)
→ DBSCAN the rest -> copy/link files per person.

Incremental runs: <output>/names.json keeps one centroid embedding per person. Rename a person
(edit the key in names.json, or use the --review UI) and the next run renames the folder and
auto-files new photos of them under that name.
"""
import argparse
import json
import os
import pickle
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from tqdm import tqdm

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEIF_EXT = {".heif", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".gif"}  # OpenCV reads GIFs as videos

MODEL_ROOT = Path(__file__).resolve().parent  # models land in ./models/buffalo_l
NAMES_FILE = "names.json"
MANIFEST_FILE = "people.json"
CACHE_FILE = ".embeddings_cache.pkl"
THUMBS_DIR = "_thumbs"
UNKNOWN = "_unknown"
BAD_NAME = re.compile(r'[\\/:*?"<>|]')
CLUSTER_CHUNK = 8000  # faces per DBSCAN call; bounds memory when one person has thousands of faces
MIN_DET_SCORE = 0.6   # detector confidence below this = blurry/partial face, not used
MAX_EXEMPLARS = 60    # user-confirmed face embeddings kept per person (most recent)
RUN_FILE = "run.json"  # input folder + options of the last run, so the UI can re-sort
FILE_TIMEOUT = 300    # seconds before a single file counts as hung
HANG_EXIT = 3         # exit code after skipping a hung file; run_sort.ps1 restarts on it


def load_model(gpu=None):
    """InsightFace downloads 'buffalo_l' (~300 MB) once to ./models, then works offline.
    gpu: DirectML adapter index (0, 1, ...) or None for CPU."""
    from insightface.app import FaceAnalysis
    if gpu is None:
        providers, options = ["CPUExecutionProvider"], None
    else:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        options = [{"device_id": str(gpu)}, {}]
    app = FaceAnalysis(name="buffalo_l", root=str(MODEL_ROOT),
                       allowed_modules=["detection", "recognition"],
                       providers=providers, provider_options=options)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def read_image(path):
    """BGR image or None. HEIF/HEIC goes through pillow-heif (OpenCV can't decode it)."""
    if path.suffix.lower() in HEIF_EXT:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            register_heif_opener()
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    # cv2.imread can't open non-ASCII paths on Windows; imdecode can
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def embed_image(app, img, t_ms=None):
    """Return one face record per detected face in a BGR image."""
    out = []
    for f in app.get(img):
        e = f.normed_embedding
        if e is not None:
            out.append({"e": e.astype(np.float32), "bbox": [float(v) for v in f.bbox],
                        "score": float(f.det_score), "t": t_ms})
    return out


def embed_video(app, path, step_seconds):
    """Sample one frame every `step_seconds`, return face records (deduped within the video)."""
    cap = cv2.VideoCapture(str(path))
    # no decodable video stream (seen on broken snaps): grab() can block forever on these
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) <= 0 or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) <= 0:
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 < fps < 241 else 25
    step = max(int(fps * step_seconds), 1)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # broken files can report no frame count and "grab" forever; cap at the stated length or 1 hour
    max_frames = int(n_frames) + 10 if n_frames > 0 else int(fps * 3600)
    faces, idx, failures = [], 0, 0
    while idx < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            failures = 0 if ok else failures + 1
            if failures > 5:
                break
            if ok:
                for r in embed_image(app, frame, t_ms=idx * 1000.0 / fps):
                    # skip if nearly identical to one we already have from this video
                    if not any(np.dot(r["e"], x["e"]) > 0.85 for x in faces):
                        faces.append(r)
        idx += 1
    cap.release()
    return faces


def collect_files(root, skip_dir=None):
    """Media files under root (skipping skip_dir) and a count of other files by extension."""
    files, skipped = [], defaultdict(int)
    root = Path(root).resolve()  # absolute paths in manifests, whatever form the input was typed in
    for dirpath, dirnames, filenames in os.walk(root):
        if skip_dir:
            dirnames[:] = [d for d in dirnames if Path(dirpath, d).resolve() != skip_dir]
        for fn in filenames:
            p = Path(dirpath, fn)
            if p.suffix.lower() in IMAGE_EXT | HEIF_EXT | VIDEO_EXT:
                files.append(p)
            else:
                skipped[p.suffix.lower() or "(none)"] += 1
    return sorted(files), dict(skipped)


def load_cache(path):
    """Append-only stream of (key, records) pickles, so an interrupted scan resumes where it stopped."""
    cache, good = {}, 0
    if path.exists():
        with open(path, "r+b") as fh:
            while True:
                try:
                    k, v = pickle.load(fh)
                except EOFError:
                    break
                except Exception:
                    break  # truncated last record from a crash, or an old cache format
                cache[k], good = v, fh.tell()
            fh.truncate(good)  # drop any broken tail so new records append cleanly
    return cache


def load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def validate_name(name):
    if not name or name.startswith("_") or BAD_NAME.search(name) or name != name.strip():
        raise ValueError(f"Invalid person name {name!r} (no leading '_', no \\ / : * ? \" < > |)")


def _move_person(out, old, new, manifest):
    """Rename (or merge) person folder, thumbnail and manifest entry `old` -> `new`."""
    src, dst = out / old, out / new
    if src.exists():
        if not dst.exists():
            src.rename(dst)
        else:
            for f in src.iterdir():
                target = dst / f.name
                if target.exists():
                    f.unlink()  # same file already present in the merged folder
                else:
                    f.rename(target)
            src.rmdir()
    old_thumb, new_thumb = out / THUMBS_DIR / f"{old}.png", out / THUMBS_DIR / f"{new}.png"
    if old_thumb.exists():
        if new_thumb.exists():
            old_thumb.unlink()
        else:
            old_thumb.rename(new_thumb)
    if old in manifest:
        entry = manifest.pop(old)
        if new in manifest:
            manifest[new]["files"] = sorted(set(manifest[new]["files"]) | set(entry["files"]))
        else:
            entry["thumbnail"] = f"{THUMBS_DIR}/{new}.png" if new_thumb.exists() else None
            manifest[new] = entry


def apply_renames(out, mapping):
    """Rename people in an output folder: {old_name: new_name}. Renaming onto an existing name merges."""
    out = Path(out)
    names = load_json(out / NAMES_FILE, {})
    manifest = load_json(out / MANIFEST_FILE, {})
    mapping = {old: new.strip() for old, new in mapping.items() if old in names and old != new.strip()}
    for new in mapping.values():  # validate everything before touching the disk
        validate_name(new)
    for old, new in mapping.items():
        b = names.pop(old)
        if new in names:
            a = names[new]
            c = np.array(a["centroid"]) * a["count"] + np.array(b["centroid"]) * b["count"]
            a["centroid"] = normalize(c).tolist()
            a["count"] += b["count"]
            a["exemplars"] = (a.get("exemplars", []) + b.get("exemplars", []))[-MAX_EXEMPLARS:]
            a["include"] = sorted(set(a.get("include", [])) | set(b.get("include", [])))
            # a file the user pulled out of one half but confirmed in the other is confirmed
            a["exclude"] = sorted((set(a.get("exclude", [])) | set(b.get("exclude", []))) - set(a["include"]))
            print(f"Merged {old} into {new}")
        else:
            names[new] = b
            print(f"Renamed {old} -> {new}")
        names[new]["folder"] = new
        _move_person(out, old, new, manifest)
    save_json(out / NAMES_FILE, names)
    save_json(out / MANIFEST_FILE, manifest)


def path_key(p):
    """Canonical form of a source path for comparing manifest entries with scanned files.
    No filesystem access (abspath/normpath only), so it is cheap for 100k entries."""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def dedupe_paths(paths):
    """Drop entries that name the same file (e.g. relative vs absolute), keeping the first spelling."""
    seen, out = set(), []
    for p in paths:
        k = path_key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def apply_move(out, src, link_names, dst, recs_for):
    """Move files (by their name inside out/src) from person `src` to person `dst`, and learn from it.

    `dst` may be a new name, an existing person, or UNKNOWN. `recs_for(source_path)` returns the face
    records of a scanned file (from the embeddings cache) or None.
    Learning: the face `src` claimed is removed from src's centroid and the file is excluded from src
    for good; the same face is added to dst's centroid and kept as a confirmed exemplar, and the file is
    pinned to dst. The next sort run applies all of this.
    """
    out = Path(out)
    names = load_json(out / NAMES_FILE, {})
    manifest = load_json(out / MANIFEST_FILE, {})
    if src not in names and src != UNKNOWN:
        raise ValueError(f"Unknown person {src!r}")
    if dst == src:
        raise ValueError("Source and destination are the same person")
    if dst != UNKNOWN:
        validate_name(dst)
    src_dir, dst_dir = (out / src).resolve(), out / dst
    dst_dir.mkdir(exist_ok=True)
    src_entry = manifest.setdefault(src, {"files": [], "thumbnail": None})
    dst_entry = manifest.setdefault(dst, {"files": [], "thumbnail": None})
    by_name = {Path(p).name: p for p in src_entry["files"]}
    src_c = normalize(names[src]["centroid"]) if src in names and names[src].get("centroid") else None

    moved_sources, moved_embs, first_face = [], [], None
    for ln in link_names:
        f = src_dir / ln
        if not f.is_file() or f.resolve().parent != src_dir:
            continue
        source = by_name.get(ln)
        if source is None:  # link was renamed name_N.ext to avoid a clash
            m = re.fullmatch(r"(.*)_\d+(\.[^.]+)", ln)
            source = by_name.get(m.group(1) + m.group(2)) if m else None
        target = dst_dir / ln
        if target.exists() or target.is_symlink():
            f.unlink()
        else:
            f.rename(target)
        if source is None:
            continue
        moved_sources.append(source)
        recs = recs_for(source) or []
        if recs:
            # the face src claimed is the one most like src; from _unknown, the clearest face
            best = max(recs, key=(lambda r: float(np.dot(r["e"], src_c))) if src_c is not None else (lambda r: r["score"]))
            moved_embs.append(np.asarray(best["e"], np.float32))
            first_face = first_face or (source, best)

    moved = set(moved_sources)
    src_entry["files"] = [p for p in src_entry["files"] if p not in moved]
    dst_entry["files"] = sorted(dedupe_paths(dst_entry["files"] + moved_sources))

    if src in names:
        info = names[src]
        info["exclude"] = sorted(set(info.get("exclude", [])) | moved)
        info["include"] = [p for p in info.get("include", []) if p not in moved]
        if moved_embs:  # a confirmed exemplar that was really this face goes with it
            info["exemplars"] = [e for e in info.get("exemplars", [])
                                 if max(float(np.dot(normalize(e), m)) for m in moved_embs) < 0.98]
        if moved_embs and info.get("centroid") is not None:
            c = np.array(info["centroid"]) * info["count"] - np.sum(moved_embs, axis=0)
            info["count"] = max(info["count"] - len(moved_embs), 1)
            info["centroid"] = normalize(c).tolist()
    if dst != UNKNOWN:
        info = names.setdefault(dst, {"centroid": None, "count": 0, "folder": dst})
        info["include"] = sorted(set(info.get("include", [])) | moved)
        info["exclude"] = [p for p in info.get("exclude", []) if p not in moved]
        if moved_embs:
            info["exemplars"] = (info.get("exemplars", []) + [e.tolist() for e in moved_embs])[-MAX_EXEMPLARS:]
            base = np.array(info["centroid"]) * info["count"] if info.get("centroid") is not None else 0
            info["centroid"] = normalize(base + np.sum(moved_embs, axis=0)).tolist()
            info["count"] += len(moved_embs)
        if not dst_entry.get("thumbnail") and first_face:
            thumb = out / THUMBS_DIR / f"{dst}.png"
            if save_thumbnail(Path(first_face[0]), first_face[1], thumb):
                dst_entry["thumbnail"] = f"{THUMBS_DIR}/{thumb.name}"

    # an unnamed group that is now empty has nothing left to show
    removed_src = False
    if src in names and not src_entry["files"] and re.fullmatch(r"Person_\d+", src):
        names.pop(src)
        manifest.pop(src)
        if src_dir.is_dir() and not any(src_dir.iterdir()):
            src_dir.rmdir()
        thumb = out / THUMBS_DIR / f"{src}.png"
        if thumb.exists():
            thumb.unlink()
        removed_src = True
    save_json(out / NAMES_FILE, names)
    save_json(out / MANIFEST_FILE, manifest)
    return {"moved": len(moved_sources), "learned": len(moved_embs), "removed_src": removed_src}


def apply_add(out, name, sources, link=True, dismiss=False):
    """Suggestions: pin face-less files to person `name` (dismiss=False), or rule them out for that
    person (dismiss=True). Pinned files are linked into the person's folder and leave _unknown/_no_faces."""
    out = Path(out)
    names = load_json(out / NAMES_FILE, {})
    manifest = load_json(out / MANIFEST_FILE, {})
    if name not in names:
        raise ValueError(f"Unknown person {name!r}")
    info = names[name]
    if dismiss:
        info["exclude"] = sorted(set(info.get("exclude", [])) | set(sources))
        save_json(out / NAMES_FILE, names)
        return {"dismissed": len(sources)}
    d = out / name
    d.mkdir(exist_ok=True)
    added = []
    for s in sources:
        src = Path(s)
        if not src.is_file():
            continue
        place_file(src, d, link)
        stale = out / UNKNOWN / src.name
        if stale.exists() or stale.is_symlink():
            stale.unlink()
        added.append(s)
    keys = {path_key(s) for s in added}
    info["include"] = sorted(dedupe_paths(info.get("include", []) + added))
    info["exclude"] = [p for p in info.get("exclude", []) if path_key(p) not in keys]
    entry = manifest.setdefault(name, {"files": [], "thumbnail": None})
    entry["files"] = sorted(dedupe_paths(entry["files"] + added))
    for other in (UNKNOWN, "_no_faces"):
        if other in manifest:
            manifest[other]["files"] = [p for p in manifest[other]["files"] if path_key(p) not in keys]
    save_json(out / NAMES_FILE, names)
    save_json(out / MANIFEST_FILE, manifest)
    return {"added": len(added)}


def discard_unnamed(out, names, manifest, only=None):
    """--recluster: drop unnamed Person_* groups and _unknown (their folders only hold copies/links).
    only: drop just these unnamed groups (and keep _unknown)."""
    drop = [n for n in names if re.fullmatch(r"Person_\d+", n)] + [UNKNOWN]
    if only is not None:
        drop = [n for n in drop if n in set(only) and n != UNKNOWN]
    for n in drop:
        names.pop(n, None)
        manifest.pop(n, None)
        d = out / n
        if d.is_dir() and d.resolve().parent == out.resolve():
            shutil.rmtree(d)
        thumb = out / THUMBS_DIR / f"{n}.png"
        if thumb.exists():
            thumb.unlink()
    n_groups = sum(1 for n in drop if n != UNKNOWN)
    print(f"Discarded {n_groups} unnamed groups; {len(names)} people remain.")


def reconcile_manual_renames(out, names, manifest):
    """names.json keys edited by hand: move the old folder (recorded in 'folder') to the new key."""
    for name, info in names.items():
        folder = info.get("folder", name)
        if folder != name:
            validate_name(name)
            print(f"Renamed {folder} -> {name} (from {NAMES_FILE})")
            _move_person(out, folder, name, manifest)
            info["folder"] = name


def place_file(src, dst_dir, link):
    """Copy or link `src` into `dst_dir`. Returns a status string for logging."""
    dst = dst_dir / src.name
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve() or dst.stat().st_size == src.stat().st_size:
                return "present"
        except OSError:
            pass
        n = 1
        while (dst_dir / f"{src.stem}_{n}{src.suffix}").exists():
            n += 1
        dst = dst_dir / f"{src.stem}_{n}{src.suffix}"
    if link:
        try:
            os.symlink(src.resolve(), dst)
            return "linked"
        except OSError:
            try:  # symlinks need Developer Mode on Windows; hardlink also avoids duplication
                os.link(src.resolve(), dst)
                return "hardlinked"
            except OSError:
                pass
    shutil.copy2(src, dst)
    return "copied"


def save_thumbnail(path, face, dst):
    """Crop the face (with margin) from an image or video frame and save a 160px PNG."""
    if face["t"] is None:
        img = read_image(path)
    else:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, face["t"])
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        return False
    x1, y1, x2, y2 = face["bbox"]
    m = 0.25 * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    crop = img[max(int(y1 - m), 0):min(int(y2 + m), h), max(int(x1 - m), 0):min(int(x2 + m), w)]
    if crop.size == 0:
        return False
    scale = 160 / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), max(int(crop.shape[0] * scale), 1)))
    dst.parent.mkdir(exist_ok=True)
    ok, buf = cv2.imencode(".png", crop)  # imwrite can't handle non-ASCII paths on Windows
    if ok:
        buf.tofile(str(dst))
    return ok


def file_key(path, video_step):
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return key + f"|step={video_step}" if path.suffix.lower() in VIDEO_EXT else key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--eps", type=float, default=0.45, help="DBSCAN cosine distance threshold")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--merge", type=float, default=0.5,
                    help="merge clusters whose centroids are within this cosine distance (0 = off)")
    ap.add_argument("--min-files", type=int, default=3, help="a person must appear in at least this many files")
    ap.add_argument("--min-face", type=int, default=50, help="ignore faces smaller than this many pixels")
    ap.add_argument("--new-group-min-files", type=int, default=10,
                    help="on a re-sort, a new unnamed group needs at least this many files")
    ap.add_argument("--no-classifier", action="store_true",
                    help="ignore <output>/classifier.pkl from train_classifier.py")
    ap.add_argument("--recluster", action="store_true",
                    help="discard the unnamed Person_* groups in output and cluster again (scan cache is kept)")
    ap.add_argument("--video-step", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--deep-videos", type=float, default=None, metavar="STEP",
                    help="re-check videos where no face was found, sampling every STEP seconds (e.g. 0.5)")
    ap.add_argument("--symlink", action="store_true",
                    help="link instead of copy (symlink, else hardlink, else copy)")
    ap.add_argument("--review", action="store_true", help="open the naming UI after sorting")
    ap.add_argument("--gpu", type=int, default=None, metavar="N",
                    help="run the model on DirectML GPU adapter N (default: CPU)")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = load_json(out / NAMES_FILE, {})
    manifest = load_json(out / MANIFEST_FILE, {})
    reconcile_manual_renames(out, names, manifest)
    if args.recluster:
        discard_unnamed(out, names, manifest)
        save_json(out / NAMES_FILE, names)
        save_json(out / MANIFEST_FILE, manifest)
    had_people = bool(names)

    print("Reading the folder and saved scan data (a minute or two for large folders)...", flush=True)
    files, skipped = collect_files(args.input, skip_dir=out.resolve())
    if not files:
        sys.exit("No images/videos found.")
    cache = load_cache(out / CACHE_FILE)
    print(f"Found {len(files)} media files ({sum(k in cache for k in map(lambda p: file_key(p, args.video_step), files))}"
          f" already scanned), {len(names)} known people.")
    if skipped:
        print("Not media, skipped: " + ", ".join(f"{n} {ext}" for ext, n in sorted(skipped.items())))
    print(f"Loading model on {'GPU ' + str(args.gpu) if args.gpu is not None else 'CPU'}...")
    app = load_model(args.gpu)

    faces = []  # face records + "file" (index into files) + "new" (not seen in a previous run)
    unreadable, no_faces, n_cached = [], [], 0
    n_deep = n_deep_hits = 0
    current = {"key": None, "since": 0.0}
    with open(out / CACHE_FILE, "ab") as cache_fh:

        def watchdog():
            # a broken file can hang inside the decoder where Python can't interrupt it: record it as
            # unreadable and exit with HANG_EXIT so the wrapper (run_sort.ps1) restarts past it
            while True:
                time.sleep(10)
                key, since = current["key"], current["since"]
                if key and time.time() - since > FILE_TIMEOUT:
                    tqdm.write(f"\n  stuck > {FILE_TIMEOUT}s on {key.split('|')[0]}; marking unreadable, restarting")
                    pickle.dump((key, None), cache_fh)
                    cache_fh.flush()
                    os._exit(HANG_EXIT)

        threading.Thread(target=watchdog, daemon=True).start()
        for i, path in enumerate(tqdm(files, desc="Scanning", smoothing=0.02)):
            key = file_key(path, args.video_step)
            deep_key = file_key(path, args.deep_videos) if args.deep_videos else None
            if key in cache and cache[key] == [] and deep_key and deep_key not in cache \
                    and path.suffix.lower() in VIDEO_EXT:
                # --deep-videos: a video with no face at the normal step gets a frame-by-frame look.
                # The result replaces the normal-step entry (so later runs keep it) and deep_key
                # records that this file has had its deep scan.
                current["key"], current["since"] = deep_key, time.time()
                try:
                    recs = embed_video(app, path, args.deep_videos) or []
                except Exception as e:
                    tqdm.write(f"  error in {path}: {e}")
                    recs = []
                current["key"] = None
                cache[key] = cache[deep_key] = recs
                pickle.dump((deep_key, recs), cache_fh)
                pickle.dump((key, recs), cache_fh)
                cache_fh.flush()
                n_deep += 1
                n_deep_hits += bool(recs)
                new = True
            elif key in cache:
                recs, new = cache[key], False
                n_cached += 1
                if recs is None:
                    unreadable.append(str(path))
                    continue
            else:
                current["key"], current["since"] = key, time.time()
                try:
                    if path.suffix.lower() in VIDEO_EXT:
                        recs = embed_video(app, path, args.video_step)
                    else:
                        img = read_image(path)
                        recs = embed_image(app, img) if img is not None else None
                except Exception as e:  # one corrupt file must not kill a multi-hour scan
                    tqdm.write(f"  error in {path}: {e}")
                    recs = None
                current["key"] = None
                cache[key], new = recs, True
                pickle.dump((key, recs), cache_fh)
                cache_fh.flush()
                if recs is None:
                    unreadable.append(str(path))
                    continue
            if not recs:
                no_faces.append(str(path))
            for r in recs:
                faces.append({**r, "file": i, "new": new})
    print(f"Scanned {len(files)} files ({n_cached} from cache): {len(faces)} faces, "
          f"{len(no_faces)} files without faces, {len(unreadable)} unreadable or audio-only "
          f"(listed in {MANIFEST_FILE} under _unreadable).")
    if n_deep:
        print(f"Deep video rescan: {n_deep} face-less videos re-checked every {args.deep_videos}s, "
              f"faces found in {n_deep_hits}.")

    E = np.vstack([f["e"] for f in faces]) if faces else np.zeros((0, 512), np.float32)
    assigned = [None] * len(faces)

    # user corrections (apply_move): files pinned to / excluded from a person, by scanned-file index
    index_of = {path_key(p): i for i, p in enumerate(files)}
    excluded_people = defaultdict(set)  # file index -> people it must never match
    for n, info in names.items():
        for p in info.get("exclude", []):
            if (i := index_of.get(path_key(p))) is not None:
                excluded_people[i].add(n)

    def match_known(js):
        """Assign faces js to the nearest known person (centroid or confirmed exemplar) within eps."""
        people = list(names)
        cols, owner = [], []
        for k, n in enumerate(people):
            info = names[n]
            if info.get("centroid") is not None:
                cols.append(normalize(info["centroid"]))
                owner.append(k)
            for e in info.get("exemplars", []):
                cols.append(normalize(e))
                owner.append(k)
        if not cols:
            return 0
        C, owner = np.vstack(cols), np.array(owner)
        person_cols = {n: np.flatnonzero(owner == k) for k, n in enumerate(people)}
        n_hit = 0
        for s in range(0, len(js), 20000):
            block = js[s:s + 20000]
            sims = E[block] @ C.T
            for row, j in enumerate(block):
                for n in excluded_people.get(faces[j]["file"], ()):
                    sims[row, person_cols[n]] = -1.0
            best = sims.argmax(axis=1)
            for j, b, sim in zip(block, best, sims[np.arange(len(block)), best]):
                if 1.0 - sim < args.eps:
                    assigned[j] = people[owner[b]]
                    n_hit += 1
        return n_hit

    # tiny or low-confidence faces give unreliable embeddings: they neither match nor form clusters
    good = [j for j, f in enumerate(faces)
            if f["score"] >= MIN_DET_SCORE and max(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= args.min_face]
    print(f"{len(good)} faces are large/clear enough to use (min {args.min_face}px, score {MIN_DET_SCORE}).")

    # 1) match against known people
    n_matched = match_known(good)

    # 1b) the classifier trained on named people (train_classifier.py) overrides when it is confident
    clf_path = out / "classifier.pkl"
    if clf_path.exists() and not args.no_classifier and good:
        with open(clf_path, "rb") as fh:
            bundle = pickle.load(fh)
        clf, need = bundle["model"], bundle["confidence"]
        proba = clf.predict_proba(E[good])
        refs = {n: np.vstack([normalize(names[n]["centroid"])] + [normalize(e) for e in names[n].get("exemplars", [])])
                for n in clf.classes_ if n in names and names[n].get("centroid") is not None}
        n_clf = 0
        for row, j in enumerate(good):
            k = int(proba[row].argmax())
            n = clf.classes_[k]
            if proba[row, k] < need or n not in refs or n in excluded_people.get(faces[j]["file"], ()):
                continue
            # sanity: still has to look broadly like them, so a stranger isn't forced onto a known person
            if float((refs[n] @ E[j]).max()) < 1.0 - args.eps - 0.15:
                continue
            if assigned[j] != n:
                n_clf += 1
            assigned[j] = n
        n_matched = sum(a is not None for a in assigned)
        print(f"Classifier ({len(clf.classes_)} people): changed {n_clf} face assignments.")

    # 2) cluster the rest into new people, a chunk at a time: each chunk is first matched against
    #    clusters from earlier chunks, then DBSCAN'd. Shuffled so a chunk isn't one date range.
    rest = [j for j in good if assigned[j] is None]
    rest = [rest[k] for k in np.random.default_rng(0).permutation(len(rest))]
    groups, group_centroids = [], []  # provisional clusters: lists of face indices
    gid = [None] * len(faces)
    chunks = range(0, len(rest), CLUSTER_CHUNK)

    def match_groups(js):
        if not groups:
            return
        C = np.vstack(group_centroids)
        for s in range(0, len(js), 20000):
            block = js[s:s + 20000]
            sims = E[block] @ C.T
            best = sims.argmax(axis=1)
            for j, b, sim in zip(block, best, sims[np.arange(len(block)), best]):
                if 1.0 - sim < args.eps:
                    gid[j] = int(b)
                    groups[b].append(j)

    for s in tqdm(chunks, desc="Clustering", disable=len(chunks) < 2):
        chunk = rest[s:s + CLUSTER_CHUNK]
        match_groups(chunk)
        chunk = [j for j in chunk if gid[j] is None]
        if len(chunk) < args.min_samples:
            continue
        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine",
                        algorithm="brute", n_jobs=-1).fit_predict(E[chunk])
        for lbl in sorted(set(labels) - {-1}):
            members = [j for j, l in zip(chunk, labels) if l == lbl]
            for j in members:
                gid[j] = len(groups)
            groups.append(members)
            group_centroids.append(normalize(E[members].sum(axis=0)))
    # noise from early chunks may belong to clusters first formed in a later chunk
    match_groups([j for j in rest if gid[j] is None])

    # 3) merge clusters of the same person (split by lighting, filters, age...): average-linkage
    #    on cluster centroids, then drop clusters seen in fewer than --min-files files
    if len(groups) > 1 and args.merge > 0:
        merged = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=args.merge).fit_predict(np.vstack(group_centroids))
        by_label = defaultdict(list)
        for g, lbl in zip(groups, merged):
            by_label[lbl].extend(g)
        print(f"Merged {len(groups)} clusters into {len(by_label)}.")
        groups = list(by_label.values())
    # on a re-sort (people already exist) leftovers are mostly strangers: only someone who turns up
    # often gets a new group, so each re-sort doesn't add hundreds of tiny ones
    need = args.min_files if not had_people else max(args.min_files, args.new_group_min_files)
    groups = [g for g in groups if len({faces[j]["file"] for j in g}) >= need]
    groups.sort(key=lambda g: -len({faces[j]["file"] for j in g}))  # Person_0 = most files
    used = [int(m.group(1)) for n in names for m in [re.fullmatch(r"Person_(\d+)", n)] if m]
    next_id = max(used, default=-1) + 1
    for g in groups:
        name = f"Person_{next_id}"
        next_id += 1
        names[name] = {"centroid": None, "count": 0, "folder": name}
        for j in g:
            assigned[j] = name
    new_people = len(groups)

    # 3) update centroids with faces not seen in earlier runs (all faces for brand-new people)
    by_person = defaultdict(list)
    for j, a in enumerate(assigned):
        if a is not None:
            by_person[a].append(j)
    for name, js in by_person.items():
        info = names[name]
        add = [faces[j]["e"] for j in js if faces[j]["new"] or info["centroid"] is None]
        if not add:
            continue
        base = np.array(info["centroid"]) * info["count"] if info["centroid"] is not None else 0
        info["centroid"] = normalize(base + np.sum(add, axis=0)).tolist()
        info["count"] += len(add)

    # 4) place files: each person they appear in; _unknown only if no face was recognised
    person_files = defaultdict(set)
    for j, a in enumerate(assigned):
        person_files[a].add(faces[j]["file"])
    unrecognised = person_files.pop(None, set())
    for n, info in names.items():  # files the user moved to a person stay there whatever the match says
        for p in info.get("include", []):
            if (i := index_of.get(path_key(p))) is not None:
                person_files[n].add(i)
    unknown_files = unrecognised - set().union(*person_files.values())
    if unknown_files:
        person_files[UNKNOWN] = unknown_files
    # a scanned file no longer assigned to a person (corrections, a changed profile) leaves their folder
    scanned = {path_key(p) for p in files}
    for name, entry in manifest.items():
        if name.startswith("_") and name != UNKNOWN:
            continue
        keep = {path_key(files[i]) for i in person_files.get(name, ())}
        stale = [p for p in entry["files"] if path_key(p) in scanned and path_key(p) not in keep]
        for p in stale:
            link = out / name / Path(p).name
            if link.exists() or link.is_symlink():
                link.unlink()
        if stale:
            entry["files"] = [p for p in entry["files"] if p not in set(stale)]

    stats = defaultdict(int)
    for name, idxs in sorted(person_files.items()):
        d = out / name
        d.mkdir(exist_ok=True)
        entry = manifest.setdefault(name, {"files": [], "thumbnail": None})
        for i in sorted(idxs):
            stats[place_file(files[i], d, args.symlink)] += 1
        entry["files"] = sorted(dedupe_paths(entry["files"] + [str(files[i]) for i in idxs]))

    # 5) thumbnails: best-scoring face of each person in this run
    for name, js in by_person.items():
        best = max(js, key=lambda j: faces[j]["score"])
        thumb = out / THUMBS_DIR / f"{name}.png"
        if save_thumbnail(files[faces[best]["file"]], faces[best], thumb):
            manifest[name]["thumbnail"] = f"{THUMBS_DIR}/{thumb.name}"

    # files without a face: this run's, plus earlier ones not scanned now; minus files the user pinned
    # to someone (suggestions) or that now have a face (deep video rescan)
    pinned = {path_key(p) for info in names.values() for p in info.get("include", [])}
    earlier = [p for p in manifest.get("_no_faces", {}).get("files", []) if path_key(p) not in scanned]
    nf = [p for p in dedupe_paths(earlier + no_faces) if path_key(p) not in pinned]
    if nf or "_no_faces" in manifest:
        manifest["_no_faces"] = {"files": sorted(nf), "thumbnail": None}
    if unreadable:
        manifest["_unreadable"] = {"files": unreadable, "thumbnail": None}
    save_json(out / NAMES_FILE, names)
    save_json(out / MANIFEST_FILE, manifest)
    save_json(out / RUN_FILE, {"input": str(Path(args.input).resolve()), "eps": args.eps,
                               "min_samples": args.min_samples, "merge": args.merge, "min_files": args.min_files,
                               "min_face": args.min_face, "video_step": args.video_step,
                               "deep_videos": args.deep_videos, "symlink": args.symlink, "gpu": args.gpu})

    n_people = sum(1 for k in person_files if k != UNKNOWN)
    print(f"Done. {n_people} people in this run ({new_people} new, {n_matched} faces matched known people), "
          f"{len(unknown_files)} files unassigned.")
    print("Files: " + ", ".join(f"{v} {k}" for k, v in sorted(stats.items())))
    print(f"Manifest: {out / MANIFEST_FILE}   Names: {out / NAMES_FILE}")

    if args.review:
        from review_ui import run_ui
        run_ui(out)


if __name__ == "__main__":
    main()
