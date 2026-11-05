#!/usr/bin/env python3
"""
review_ui.py — local web UI to review people and assign names (opens in your browser).

Usage:
    python review_ui.py <output_folder> [--port 8765] [--no-browser]
    (or: python sort_faces.py <in> <out> --review)

Runs only on 127.0.0.1; nothing leaves the machine. Giving two people the same name merges them.
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from appearance import APPEAR_CACHE, appear_key, load_appear_cache
from sort_faces import (CACHE_FILE, IMAGE_EXT, MANIFEST_FILE, NAMES_FILE, RUN_FILE, THUMBS_DIR, UNKNOWN,
                        VIDEO_EXT, apply_add, apply_move, apply_renames, load_cache, load_json, path_key)

HERE = Path(__file__).resolve().parent
PAGE_HTML = HERE / "review_ui.html"
MEDIA_EXT = IMAGE_EXT | VIDEO_EXT | {".heif", ".heic"}
TOP_PEOPLE = 3  # a face-less photo is suggested only to the 3 people it resembles most
lock = threading.Lock()  # renames and moves change folders and json; one at a time


class Embeddings:
    """Face records of every scanned file, loaded from the cache on first use (a few hundred MB)."""

    def __init__(self, out):
        self.out, self.index, self._lock = out, None, threading.Lock()

    def __call__(self, source_path):
        with self._lock:
            if self.index is None:
                cache = load_cache(self.out / CACHE_FILE)
                self.index = {path_key(k.split("|")[0]): v for k, v in cache.items() if v}
        return self.index.get(path_key(source_path))

    def reset(self):
        with self._lock:
            self.index = None


class Suggester:
    """Ranks face-less photos (_no_faces + _unknown) by how much they look like a person's photos,
    using the whole-image embeddings from appearance.py. Rebuilt lazily after every change."""

    def __init__(self, out):
        self.out, self._lock, self.state = out, threading.Lock(), None
        self.vec_by_key = None  # appearance cache, loaded once

    def reset(self):
        with self._lock:
            self.state = None

    def ready(self):
        return (self.out / APPEAR_CACHE).exists()

    def _vec(self, path):
        try:
            return self.vec_by_key.get(appear_key(path))
        except OSError:
            return None

    def _build(self):
        if self.vec_by_key is None:
            self.vec_by_key = load_appear_cache(self.out / APPEAR_CACHE)
        manifest = load_json(self.out / MANIFEST_FILE, {})
        names = load_json(self.out / NAMES_FILE, {})
        pinned = {path_key(p) for i in names.values() for p in i.get("include", [])}
        cands, vecs = [], []
        for p in dict.fromkeys(manifest.get("_no_faces", {}).get("files", []) + manifest.get(UNKNOWN, {}).get("files", [])):
            v = self._vec(p)
            if v is not None and path_key(p) not in pinned:
                cands.append(p)
                vecs.append(v)
        C = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 768), np.float32)
        person = {}
        for n, entry in manifest.items():
            if n.startswith("_"):
                continue
            pv = [v for v in (self._vec(p) for p in entry["files"]) if v is not None]
            if pv:
                person[n] = np.vstack(pv).astype(np.float32)
        excl = {n: {path_key(p) for p in i.get("exclude", [])} for n, i in names.items()}
        # each candidate is only suggested to the few people it looks most like overall, so one
        # strong look-alike doesn't show up on every page
        top_people = [set() for _ in cands]
        if person and len(cands):
            people = list(person)
            cent = np.vstack([P.mean(axis=0) / (np.linalg.norm(P.mean(axis=0)) or 1) for P in person.values()])
            for s in range(0, len(cands), 20000):
                sims = C[s:s + 20000] @ cent.T
                top = np.argsort(-sims, axis=1)[:, :TOP_PEOPLE]
                for r, row in enumerate(top):
                    top_people[s + r] = {people[k] for k in row}
        self.state = {"cands": cands, "C": C, "person": person, "excl": excl, "top": top_people,
                      "cand_keys": [path_key(p) for p in cands]}

    def get(self, name, limit=60):
        with self._lock:
            if self.state is None:
                self._build()
            s = self.state
        P = s["person"].get(name)
        if P is None or not len(s["cands"]):
            return [], s
        sims = s["C"] @ P.T  # candidates x person photos
        k = min(3, P.shape[0])
        score = np.sort(sims, axis=1)[:, -k:].mean(axis=1)  # close to several of their photos, not one
        excl = s["excl"].get(name, set())
        order = [i for i in np.argsort(-score)
                 if name in s["top"][i] and s["cand_keys"][i] not in excl][:limit]
        return [(int(i), float(score[i])) for i in order], s


class Resort:
    """Runs sort_faces.py on the last run's input/options, then appearance.py (look-alike data for
    files that changed people), keeping the latest progress line."""

    def __init__(self, out):
        self.out, self.proc, self.line, self.code = out, None, "", None
        self.busy = False

    def running(self):
        return self.busy

    def start(self):
        run = load_json(self.out / RUN_FILE, None)
        if not run:
            raise ValueError("No previous run recorded for this folder; run sort_faces.py once first.")
        cmd = [sys.executable, str(HERE / "sort_faces.py"), run["input"], str(self.out),
               "--eps", str(run["eps"]), "--min-samples", str(run["min_samples"]), "--merge", str(run["merge"]),
               "--min-files", str(run["min_files"]), "--min-face", str(run["min_face"]),
               "--video-step", str(run["video_step"])]
        if run.get("symlink"):
            cmd.append("--symlink")
        if run.get("deep_videos"):
            cmd += ["--deep-videos", str(run["deep_videos"])]
        gpu = ["--gpu", str(run["gpu"])] if run.get("gpu") is not None else []
        cmds = [cmd + gpu]
        if (self.out / APPEAR_CACHE).exists():
            cmds.append([sys.executable, str(HERE / "appearance.py"), str(self.out)] + gpu)
        self.line, self.code, self.busy = "Starting...", None, True
        threading.Thread(target=self._run, args=(cmds,), daemon=True).start()

    def _run(self, cmds):
        try:
            for cmd in cmds:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                             encoding="utf-8", errors="replace", bufsize=1, cwd=str(HERE))
                buf = ""
                while True:
                    ch = self.proc.stdout.read(1)
                    if not ch:
                        break
                    if ch in "\r\n":
                        if buf.strip() and "Warning" not in buf and "tform.estimate" not in buf:
                            self.line = buf.strip()
                        buf = ""
                    else:
                        buf += ch
                self.code = self.proc.wait()
                if self.code != 0:
                    self.line = f"Failed (exit {self.code}): {self.line}"
                    return
        finally:
            self.busy = False
            if self.on_done:
                self.on_done()

    on_done = None


def make_handler(out):
    out = Path(out).resolve()
    embeddings = Embeddings(out)
    resort = Resort(out)
    suggester = Suggester(out)

    def after_resort():  # new look-alike vectors and a new manifest: rebuild both views
        suggester.vec_by_key = None
        suggester.reset()
        embeddings.reset()
    resort.on_done = after_resort

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # the page and API must never be cached (an old page hides new buttons); photos may be
            self.send_header("Cache-Control", "max-age=3600" if ctype.startswith(("image/", "video/")) else "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def json(self, data, code=200):
            self.send(code, json.dumps(data).encode())

        def person_dir(self, name):
            """Folder of a known person (or _unknown), or None. Never escapes the output folder."""
            names = load_json(out / NAMES_FILE, {})
            if name not in names and name != UNKNOWN:
                return None
            d = (out / name).resolve()
            return d if d.parent == out and d.is_dir() else None

        def do_GET(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/")]
            if url.path == "/":
                return self.send(200, PAGE_HTML.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/people":
                names = load_json(out / NAMES_FILE, {})
                manifest = load_json(out / MANIFEST_FILE, {})
                people = [{"name": n, "files": len(manifest.get(n, {}).get("files", [])),
                           "named": not n.startswith("Person_"),
                           "thumb": bool(manifest.get(n, {}).get("thumbnail"))} for n in names]
                people.sort(key=lambda p: -p["files"])
                corrections = sum(len(i.get("include", [])) for i in names.values())
                return self.json({"folder": str(out), "people": people,
                                  "unknown": len(manifest.get(UNKNOWN, {}).get("files", [])),
                                  "no_faces": len(manifest.get("_no_faces", {}).get("files", [])),
                                  "corrections": corrections, "can_resort": (out / RUN_FILE).exists(),
                                  "classifier": load_json(out / "classifier.json", None),
                                  "resort": {"running": resort.running(), "line": resort.line, "code": resort.code}})
            if url.path == "/api/resort/status":
                return self.json({"running": resort.running(), "line": resort.line, "code": resort.code})
            if len(parts) == 2 and parts[0] == "thumb":
                f = out / THUMBS_DIR / f"{parts[1]}.png"
                if f.resolve().parent == (out / THUMBS_DIR).resolve() and f.is_file():
                    return self.send(200, f.read_bytes(), "image/png")
                return self.send(404)
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "person":
                d = self.person_dir(parts[2])
                if not d:
                    return self.json({"error": "unknown person"}, 404)
                q = parse_qs(url.query)
                offset, limit = int(q.get("offset", [0])[0]), int(q.get("limit", [120])[0])
                files = sorted((f for f in os.listdir(d) if Path(f).suffix.lower() in MEDIA_EXT), reverse=True)
                return self.json({"total": len(files), "files": [
                    {"name": f, "video": Path(f).suffix.lower() in VIDEO_EXT - {".gif"}}
                    for f in files[offset:offset + limit]]})
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "suggest":
                if not suggester.ready():
                    return self.json({"ready": False, "items": []})
                ranked, s = suggester.get(parts[2])
                return self.json({"ready": True, "items": [
                    {"id": i, "name": Path(s["cands"][i]).name, "score": round(sc, 3),
                     "video": Path(s["cands"][i]).suffix.lower() in VIDEO_EXT - {".gif"}} for i, sc in ranked]})
            if len(parts) == 2 and parts[0] == "cand":
                s = suggester.state
                if not s or not parts[1].isdigit() or int(parts[1]) >= len(s["cands"]):
                    return self.send(404)
                f = Path(s["cands"][int(parts[1])])
                return self.send_file(f) if f.is_file() else self.send(404)
            if len(parts) == 3 and parts[0] == "media":
                d = self.person_dir(parts[1])
                f = (d / parts[2]).resolve() if d else None
                if not f or f.parent != d or not f.is_file():
                    return self.send(404)
                return self.send_file(f)
            self.send(404)

        def send_file(self, f):
            """Serve a file, honouring a single Range header so videos can seek."""
            if f.suffix.lower() in {".heif", ".heic"}:  # browsers can't show HEIF: convert to JPEG
                import cv2
                from sort_faces import read_image
                img = read_image(f)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88]) if img is not None else (False, None)
                return self.send(200, buf.tobytes(), "image/jpeg") if ok else self.send(404)
            size = f.stat().st_size
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range", "")
            start, end = 0, size - 1
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(size - int(b), 0)
                end = min(int(b), size - 1) if a and b else size - 1
            with open(f, "rb") as fh:
                fh.seek(start)
                body = fh.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if rng:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send(206 if rng else 200, body, ctype, extra)

        def do_POST(self):
            # only accept requests from our own page (blocks other websites posting to localhost)
            origin = self.headers.get("Origin", "")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.json({"error": "forbidden"}, 403)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/rename":
                try:
                    with lock:
                        apply_renames(out, {str(k): str(v) for k, v in body.get("mapping", {}).items()})
                except (ValueError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
                return self.json({"ok": True})
            if self.path == "/api/open":
                d = self.person_dir(body.get("name", "")) if body.get("name") else out
                if d and sys.platform == "win32":
                    os.startfile(d)
                return self.json({"ok": bool(d)})
            if self.path == "/api/move":
                src, dst = str(body.get("from", "")), str(body.get("to", "")).strip()
                files = [str(f) for f in body.get("files", [])][:5000]
                if not self.person_dir(src) or not dst or not files:
                    return self.json({"error": "bad request"}, 400)
                if resort.running():
                    return self.json({"error": "Wait for the re-sort to finish first."}, 409)
                try:
                    with lock:
                        result = apply_move(out, src, files, dst, embeddings)
                except (ValueError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
                suggester.reset()
                return self.json(result)
            if self.path in ("/api/suggest/add", "/api/suggest/dismiss"):
                s = suggester.state
                name, ids = str(body.get("name", "")), body.get("ids", [])
                if not s or not ids or resort.running():
                    return self.json({"error": "Suggestions changed; reload the page." if not resort.running()
                                      else "Wait for the re-sort to finish first."}, 409)
                sources = [s["cands"][int(i)] for i in ids if 0 <= int(i) < len(s["cands"])]
                link = bool(load_json(out / RUN_FILE, {}).get("symlink", True))
                try:
                    with lock:
                        result = apply_add(out, name, sources, link, dismiss=self.path.endswith("dismiss"))
                except (ValueError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
                suggester.reset()
                return self.json(result)
            if self.path == "/api/train":
                if resort.running():
                    return self.json({"error": "Wait for the re-sort to finish first."}, 409)
                from train_classifier import train
                try:
                    with lock:
                        report = train(out)
                except Exception as e:
                    return self.json({"error": f"Training failed: {e}"}, 500)
                return self.json(report, 400 if "error" in report else 200)
            if self.path == "/api/resort":
                if resort.running():
                    return self.json({"error": "Already running"}, 409)
                try:
                    with lock:
                        resort.start()
                        embeddings.reset()  # the run may scan new files
                        suggester.reset()
                except (ValueError, OSError) as e:
                    return self.json({"error": str(e)}, 400)
                return self.json({"ok": True})
            self.send(404)

    return Handler


def run_ui(out, port=8765, browser=True):
    for p in range(port, port + 20):  # next free port if one is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(out))
            break
        except OSError:
            continue
    else:
        sys.exit("No free port found.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Review UI: {url}   (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    run_ui(a.output, a.port, not a.no_browser)
