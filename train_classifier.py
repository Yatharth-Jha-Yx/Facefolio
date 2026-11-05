#!/usr/bin/env python3
"""
train_classifier.py — learn the people you named from their face embeddings.

Usage:
    python train_classifier.py <output_folder>

Trains a small classifier (logistic regression on the 512-d InsightFace embeddings, seconds on CPU)
on every *named* person: the faces in their folder plus confirmed exemplars from your corrections.
Faces in files you removed from a person are never used as that person's examples.
The face network itself is not changed. sort_faces.py uses <output>/classifier.pkl on the next run;
delete that file to switch it off. Writes classifier.json with an accuracy report.
"""
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sort_faces import (CACHE_FILE, MANIFEST_FILE, MIN_DET_SCORE, NAMES_FILE, load_cache, load_json,
                        normalize, path_key)

MODEL_FILE = "classifier.pkl"
REPORT_FILE = "classifier.json"
MIN_FACES = 5        # a person needs this many example faces to be learned
MIN_PEOPLE = 2
CONFIDENCE = 0.80    # predictions below this probability are left to the normal matching
MAX_PER_PERSON = 3000  # keep training balanced and fast


def is_named(n):
    return not re.fullmatch(r"Person_\d+", n) and not n.startswith("_")


def collect_examples(out):
    """(embeddings, labels, from_correction) for named people."""
    names = load_json(out / NAMES_FILE, {})
    manifest = load_json(out / MANIFEST_FILE, {})
    cache = load_cache(out / CACHE_FILE)
    recs_by = {path_key(k.split("|")[0]): v for k, v in cache.items() if v}
    X, y, corr = [], [], []
    for n, info in names.items():
        if not is_named(n) or info.get("centroid") is None:
            continue
        c = normalize(info["centroid"])
        refs = [c] + [normalize(e) for e in info.get("exemplars", [])]
        excluded = {path_key(p) for p in info.get("exclude", [])}
        included = {path_key(p) for p in info.get("include", [])}
        rows = []
        for p in manifest.get(n, {}).get("files", []):
            k = path_key(p)
            if k in excluded:
                continue
            recs = [r for r in recs_by.get(k, []) if r["score"] >= MIN_DET_SCORE]
            if not recs:
                continue
            # the face in this file that is this person: closest to their centroid/exemplars
            sims = [max(float(np.dot(r["e"], ref)) for ref in refs) for r in recs]
            b = int(np.argmax(sims))
            if sims[b] < 0.3 and k not in included:  # nobody here really looks like them
                continue
            rows.append((np.asarray(recs[b]["e"], np.float32), k in included))
        rows += [(normalize(e), True) for e in info.get("exemplars", [])]
        if len(rows) > MAX_PER_PERSON:
            rng = np.random.default_rng(0)
            keep = [r for r in rows if r[1]]
            rest = [r for r in rows if not r[1]]
            rows = keep + [rest[i] for i in rng.choice(len(rest), MAX_PER_PERSON - len(keep), replace=False)]
        for e, is_corr in rows:
            X.append(e)
            y.append(n)
            corr.append(is_corr)
    return (np.vstack(X) if X else np.zeros((0, 512), np.float32)), np.array(y), np.array(corr, bool)


def train(out):
    out = Path(out)
    t0 = time.time()
    X, y, corr = collect_examples(out)
    counts = {n: int((y == n).sum()) for n in sorted(set(y))}
    skipped = {n: c for n, c in counts.items() if c < MIN_FACES}
    keep = np.isin(y, [n for n, c in counts.items() if c >= MIN_FACES])
    X, y, corr = X[keep], y[keep], corr[keep]
    people = sorted(set(y))
    report = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "people": len(people), "faces": int(len(y)),
              "skipped_too_few_faces": skipped, "confidence": CONFIDENCE}
    if len(people) < MIN_PEOPLE:
        report["error"] = (f"Name at least {MIN_PEOPLE} people with {MIN_FACES}+ photos each first "
                           f"(have {len(people)}).")
        (out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
    # honest accuracy: predict each face with a model that never saw it
    folds = min(5, min(int((y == n).sum()) for n in people))
    if folds >= 2:
        proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                                  method="predict_proba")
        classes = np.array(sorted(people))
        pred, conf = classes[proba.argmax(1)], proba.max(1)
        sure = conf >= CONFIDENCE
        report["accuracy_all"] = round(float((pred == y).mean()), 4)
        report["confident_share"] = round(float(sure.mean()), 4)
        report["accuracy_when_confident"] = round(float((pred[sure] == y[sure]).mean()), 4) if sure.any() else None
        if corr.any():
            report["corrections_tested"] = int(corr.sum())
            report["accuracy_on_corrections"] = round(float((pred[corr] == y[corr]).mean()), 4)
        per = {}
        for n in people:
            m = y == n
            per[n] = {"faces": int(m.sum()), "accuracy": round(float((pred[m] == n).mean()), 3)}
        report["per_person"] = per
        confused = defaultdict(int)
        for a, b in zip(y, pred):
            if a != b:
                confused[f"{a} -> {b}"] += 1
        report["most_confused"] = dict(sorted(confused.items(), key=lambda kv: -kv[1])[:10])
    clf.fit(X, y)
    with open(out / MODEL_FILE, "wb") as fh:
        pickle.dump({"model": clf, "confidence": CONFIDENCE, "people": people}, fh)
    report["seconds"] = round(time.time() - t0, 1)
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train_classifier.py <output_folder>")
    r = train(sys.argv[1])
    if "error" in r:
        sys.exit(r["error"])
    print(f"Trained on {r['faces']} faces of {r['people']} people in {r['seconds']}s.")
    if "accuracy_all" in r:
        print(f"Accuracy (unseen faces): {r['accuracy_all']:.1%}; when confident ({r['confident_share']:.0%} of faces): "
              f"{r['accuracy_when_confident']:.1%}")
    if "accuracy_on_corrections" in r:
        print(f"Accuracy on your {r['corrections_tested']} corrections: {r['accuracy_on_corrections']:.1%}")
    if r.get("most_confused"):
        print("Most confused: " + ", ".join(f"{k} ({v})" for k, v in r["most_confused"].items()))
    if r["skipped_too_few_faces"]:
        print(f"Not learned yet (fewer than {MIN_FACES} photos): {', '.join(r['skipped_too_few_faces'])}")
