#!/usr/bin/env python3
"""
appearance.py — whole-image "looks like" embeddings, for photos where no usable face was found
(mirror selfies with the phone over the face, backs of heads...).

Usage:
    python appearance.py <output_folder> [--gpu 1]

Embeds, with DINOv2-small (./models/dinov2-small.onnx):
  - every file in _no_faces and _unknown (the candidates), and
  - a sample of each person's files plus everything pinned to them (their "look").
The review UI then suggests candidates that look like a person's photos: same phone case, room,
mirror, hair, clothes. Suggestions are only shown, never filed without the user confirming.
Results are cached in <output>/.appearance_cache.pkl (append-only, resumable like the face scan).
"""
import argparse
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sort_faces import MANIFEST_FILE, NAMES_FILE, UNKNOWN, VIDEO_EXT, load_json, normalize, read_image

MODEL_FILE = Path(__file__).resolve().parent / "models" / "dinov2-small.onnx"
APPEAR_CACHE = ".appearance_cache.pkl"
SAMPLES_PER_PERSON = 40
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def appear_key(path):
    st = Path(path).stat()
    return f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}"


def load_appear_cache(path):
    """{key: 768-d vector or None}; same append-only format as the face cache."""
    cache, good = {}, 0
    if path.exists():
        with open(path, "r+b") as fh:
            while True:
                try:
                    k, v = pickle.load(fh)
                except EOFError:
                    break
                except Exception:
                    break
                cache[k], good = v, fh.tell()
            fh.truncate(good)
    return cache


def frames_of(path):
    """1 frame for an image, 3 (at 25/50/75 %) for a video. BGR arrays; [] if unreadable."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXT:
        img = read_image(path)
        return [img] if img is not None else []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) <= 0:
        cap.release()
        return []
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = []
    for q in (0.25, 0.5, 0.75) if n > 3 else (0,):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * q))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def preprocess(img):
    # squash to 224x224 so the whole frame (phone, mirror, room) is seen, not a centre crop
    rgb = cv2.cvtColor(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


class Embedder:
    def __init__(self, gpu=None):
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        if gpu is None:
            providers, options = ["CPUExecutionProvider"], None
        else:
            providers, options = ["DmlExecutionProvider", "CPUExecutionProvider"], [{"device_id": str(gpu)}, {}]
        self.sess = ort.InferenceSession(str(MODEL_FILE), providers=providers, provider_options=options)

    def __call__(self, imgs):
        """Normalised [CLS ; mean patch] descriptor per image (768-d)."""
        x = np.stack([preprocess(i) for i in imgs])
        h = self.sess.run(None, {"pixel_values": x})[0]
        cls, patches = h[:, 0], h[:, 1:].mean(axis=1)
        d = np.concatenate([cls / np.linalg.norm(cls, axis=1, keepdims=True),
                            patches / np.linalg.norm(patches, axis=1, keepdims=True)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def files_to_embed(out):
    """Candidates (no usable face) and per-person samples, as absolute path strings."""
    manifest = load_json(out / MANIFEST_FILE, {})
    names = load_json(out / NAMES_FILE, {})
    rng = random.Random(0)
    cands = manifest.get("_no_faces", {}).get("files", []) + manifest.get(UNKNOWN, {}).get("files", [])
    people = []
    for n, entry in manifest.items():
        if n.startswith("_"):
            continue
        pinned = names.get(n, {}).get("include", [])
        rest = [f for f in entry["files"] if f not in set(pinned)]
        people += pinned + rng.sample(rest, min(SAMPLES_PER_PERSON, len(rest)))
    return list(dict.fromkeys(cands + people))


def build(out, gpu=None, batch=16):
    out = Path(out)
    cache = load_appear_cache(out / APPEAR_CACHE)
    todo = []
    for p in files_to_embed(out):
        try:
            if appear_key(p) not in cache:
                todo.append(p)
        except OSError:
            pass
    print(f"Appearance: {len(todo)} files to embed ({len(cache)} already cached).")
    if not todo:
        return
    emb = Embedder(gpu)
    with open(out / APPEAR_CACHE, "ab") as fh, ThreadPoolExecutor(6) as pool:
        bar = tqdm(total=len(todo), desc="Appearance", smoothing=0.02)
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            frames = list(pool.map(frames_of, chunk))
            flat = [f for fs in frames for f in fs]
            vecs = emb(flat) if flat else np.zeros((0, 768), np.float32)
            k = 0
            for p, fs in zip(chunk, frames):
                v = normalize(vecs[k:k + len(fs)].mean(axis=0)).astype(np.float16) if fs else None
                k += len(fs)
                pickle.dump((appear_key(p), v), fh)
            fh.flush()
            bar.update(len(chunk))
        bar.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--gpu", type=int, default=None, metavar="N")
    a = ap.parse_args()
    build(a.output, a.gpu)
