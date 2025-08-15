# Face Recognition Pipeline 🔍👤

A modular, research-driven face **detection & recognition** system with plug-and-play detectors (OpenCV, MTCNN, Dlib, `face_recognition`) and a slick **Gradio** web UI. Built for fast iteration, real-world reliability, and clean code vibes.

---

## TL;DR
- Detect faces → generate embeddings → match identities.
- Swap detectors on the fly (Haar, MTCNN, Dlib HOG/CNN, `face_recognition`).
- Run **real-time** (webcam/video) or **batch** (folders).
- Test in a browser with a **Gradio** app (image upload + live camera + detector dropdown).

---

## Features
- **Multi-detector support**: OpenCV Haar, **MTCNN**, **Dlib HOG/CNN**, `face_recognition`.
- **Embeddings** via Dlib ResNet (128D) with cosine similarity / Euclidean distance.
- **Real-time** inference from webcam or video files.
- **Batch** processing for datasets.
- **Gradio UI**: upload image, toggle camera, pick detector, view detections + identities.
- **Modular architecture**: clean interfaces for detectors, encoders, matchers, I/O.
- **Config-first**: YAML/CLI switches, easy to extend.

---

## Demo (Gradio)
```bash
python app/gradio_app.py
# then open the printed local URL in your browser
```
**You can:**
- Upload an image or start webcam
- Choose detector from a dropdown
- See bounding boxes + names/similarity

---

## Installation
> Python 3.9–3.11 recommended. GPU optional (Dlib CUDA build speeds things up but CPU works).

### 1) Create env & install deps
```bash
# clone
git clone https://github.com/amitesh-maurya/face-recognition-pipeline.git
cd face-recognition-pipeline

# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install
pip install -U pip wheel
pip install -r requirements.txt
```

### 2) Optional: Dlib with CUDA (speed-up)
- Install CUDA/cuDNN for your GPU
- Build dlib with CUDA flags or install a prebuilt wheel compatible with your setup

> If that’s too extra, stick to CPU — it’s fine for testing.

---

## Quickstart
### Detect & recognize from an image
```bash
python scripts/recognize_image.py \
  --image assets/samples/group.jpg \
  --detector mtcnn \
  --db data/embeddings/known_faces.pkl \
  --threshold 0.6
```

### Run on webcam (real-time)
```bash
python scripts/recognize_stream.py \
  --source 0 \
  --detector dlib_cnn \
  --db data/embeddings/known_faces.pkl
```

### Build embeddings from a folder
```bash
# expects folder structure: data/known/<person_name>/*.jpg
python scripts/build_embeddings.py \
  --images data/known \
  --output data/embeddings/known_faces.pkl \
  --detector face_recognition
```

---

## Project Structure
```
face-recognition-pipeline/
│  app.py
│  requirements.txt
│
├─ core/
│   ├─ __init__.py
│   ├─ config.py
│   ├─ logging_crypto.py
│   ├─ gallery.py
│   ├─ face_utils.py
│   ├─ match.py
│   └─ pipeline.py
│
└─ ui/
    ├─ __init__.py
    └─ gradio_ui.py

```

---

## Architecture (How it flows)
```
[Input (image/video/webcam)]
        │
        ▼
 [Detector]  ──▶ face boxes + landmarks
        │
        ▼
 [Encoder]   ──▶ 128D embedding per face (Dlib ResNet)
        │
        ▼
 [Matcher]   ──▶ identity + similarity score vs. embeddings DB
        │
        ▼
[Output]     ──▶ annotated frames + JSON results
```

### Swappable Detectors
- `opencv_haar` – lightweight, decent on frontal faces
- `mtcnn` – solid accuracy, slower on CPU
- `dlib_hog` – CPU-friendly, classical HOG + SVM
- `dlib_cnn` – accurate but needs more compute (benefits from GPU)
- `face_recognition` – convenience wrapper around dlib

---

## Configuration
All scripts accept flags; you can also use a YAML config.

**Example CLI flags:**
```bash
--detector {opencv_haar,mtcnn,dlib_hog,dlib_cnn,face_recognition}
--threshold 0.6
--min-size 40
--upsample 1
--nms 0.3
--source 0|/path/to/video.mp4
--db data/embeddings/known_faces.pkl
--save-vis runs/vis.jpg
```

**YAML (optional):** `configs/default.yaml`
```yaml
detector: mtcnn
threshold: 0.6
min_size: 40
upsample: 1
nms: 0.3
encoder: dlib_resnet
matcher: cosine
```

---

## Output Formats
- **Annotated image/video** with boxes, names, scores
- **JSON** results for programmatic use
- **Pickle** database of embeddings for fast lookups

---

## Benchmarks (placeholder)
> Plug in your machine + dataset results here for transparency.

| Detector        | FPS (CPU) | FPS (GPU) | Precision | Recall |
|-----------------|-----------|-----------|-----------|--------|
| OpenCV Haar     |           |           |           |        |
| MTCNN           |           |           |           |        |
| Dlib HOG        |           |           |           |        |
| Dlib CNN        |           |           |           |        |
| face_recognition|           |           |           |        |

---

## Tips & Gotchas
- Lighting + face angle matter; consider `--upsample` & detector choice.
- Keep your **known faces** clean (centered, sharp, multiple angles).
- Tune **threshold** per detector/encoder combo.
- If webcam lags, drop resolution or switch to HOG/Haar.

---

## Roadmap
- [ ] ONNX/TensorRT export for the encoder
- [ ] Face alignment before encoding
- [ ] Face tracking across frames (Deep SORT)
- [ ] Enrollment via Gradio (add new identities from UI)
- [ ] SQLite/FAISS backend for embeddings

---

## Contributing
PRs are welcome! Please:
1. Run tests: `pytest -q`
2. Use type hints & docstrings
3. Pre-commit hooks for linting/formatting

---

## License
MIT — do your thing, just keep the notice.

---

## Citation
If this repo helped your research/product, consider citing:
```bibtex
@software{amitesh_face_recognition_pipeline,
  author = {Maurya, Amitesh},
  title = {Modular Face Recognition Pipeline with Gradio UI},
  year = {2025},
  url = {https://github.com/amitesh-maurya/face-recognition-pipeline}
}
```

---

## Acknowledgements
- Dlib, OpenCV, `face_recognition`, MTCNN authors & maintainers — absolute legends.
- Gradio team for making fast UIs stupid-simple.

---

> Got feature ideas? Open an issue. Want a quick tweak for your setup? PRs welcome.

