# Facefolio

Offline photo & video sorting by person. Facefolio sorts a folder of photos and videos into one
folder per person, entirely on your machine. You then name
people and fix mistakes in a local web app styled like the People view in Apple Photos.

- **Finds faces** in images (JPG, PNG, WebP, HEIF/HEIC) and videos (MP4, MOV, MKV, GIF…) with
  [InsightFace](https://github.com/deepinsight/insightface), then groups them with DBSCAN.
- **Learns from your corrections**: naming, merging, moving photos between people and "not this
  person" all update each person's profile, so a re-sort doesn't repeat the mistake. An optional
  classifier (`train_classifier.py`) learns the people you've named.
- **Finds photos without a visible face**, such as mirror selfies with the phone in front of the face. A
  frame-by-frame video rescan catches brief face appearances, and look-alike suggestions (DINOv2,
  whole-image similarity) offer face-less photos that resemble a person, for you to confirm.
- **Scales to large folders**: tested on ~118,000 files. Scans are cached and resumable, a watchdog
  skips files that hang the decoder, and `--symlink` links files instead of copying them.

Nothing leaves your machine: the web UI is served on `127.0.0.1` only.

## Setup

Python 3.10–3.12.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows (use .venv/bin/pip elsewhere)
```

GPU on Windows: `pip uninstall onnxruntime` then `pip install onnxruntime-directml`, and pass
`--gpu N` (the DirectML adapter index; on laptops the discrete GPU is often `1`).

Models:
- **buffalo_l** (faces) downloads automatically to `models/` on first run (~280 MB).
- **DINOv2-small** (look-alikes, optional), placed at `models/dinov2-small.onnx`:
  ```bash
  curl -L -o models/dinov2-small.onnx https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx
  ```

## Usage

```bash
# 1. sort (copies files; --symlink links them instead)
python sort_faces.py <photos_folder> <output_folder> --symlink

# 2. review: name people, merge, move photos, re-sort
python review_ui.py <output_folder>

# optional extras
python sort_faces.py <photos_folder> <output_folder> --deep-videos 0.25   # look harder in face-less videos
python appearance.py <output_folder>                                      # enable look-alike suggestions
python train_classifier.py <output_folder>                                # learn named people
```

On Windows, `run_sort.ps1` wraps `sort_faces.py` and restarts it if a corrupt file hangs the
decoder. Any run can be stopped with Ctrl+C and resumed by running the same command again.

Useful options: `--eps` (match strictness, default 0.45), `--merge` (merge similar groups, 0.5),
`--min-files` (min files per person, 3), `--min-face` (ignore faces under N px, 50),
`--recluster` (regroup unnamed people from the cached scan), `--gpu N`. See `--help`.

## Output folder

| Path | What |
| --- | --- |
| `<Name>/`, `Person_N/` | the person's photos and videos (copies or links) |
| `_unknown/` | files with faces nobody was matched to |
| `people.json` | person → files, plus `_no_faces` and `_unreadable` lists |
| `names.json` | person → face profile and your corrections |
| `_thumbs/`, `.embeddings_cache.pkl`, `.appearance_cache.pkl` | thumbnails and caches |

Deleting the output folder never touches your original files.

## Licenses

Facefolio's code is [MIT](LICENSE). The models it uses are licensed separately: the InsightFace
**buffalo_l** models are for **non-commercial research use only** (check InsightFace's terms before
any commercial use), and DINOv2 is Apache-2.0.
