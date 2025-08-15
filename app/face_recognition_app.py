# app
import os
import cv2
import faiss
import json
import time
import queue
import numpy as np
import logging
import datetime
import threading
import face_recognition
import gradio as gr
from cryptography.fernet import Fernet

# -----------------------------
# Config
# -----------------------------
LOG_FILE = "face_recog_audit.log"
EMBED_DIM = 128
# Cosine-style similarity: we L2-normalize and use inner-product index
SIM_THRESHOLD = 0.55  # tweak 0.50–0.65 depending on your gallery quality

GALLERY_EMB_PATH = "gallery_embeddings.npy"
GALLERY_LBL_PATH = "gallery_labels.npy"

# Gradio defaults
TITLE = "Real-Time Face Recognition"
DESC = "FAISS-backed cosine matching + face_recognition encodings. Upload or use webcam."

# -----------------------------
# Logging & Crypto (optional)
# -----------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
# NOTE: key is ephemeral per run; persist it if you want to decrypt later
fernet = Fernet(Fernet.generate_key())

def enc_log(d: dict):
    try:
        encrypted = fernet.encrypt(json.dumps(d, default=str).encode()).decode()
        logging.info(f"ENCRYPTED_LOG: {encrypted}")
    except Exception as e:
        logging.exception(f"Audit encryption failed: {e}")

# -----------------------------
# FAISS Gallery (cosine via IP)
# -----------------------------
def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms

def build_index() -> faiss.Index:
    # Inner-product index for cosine sims on normalized embeddings
    return faiss.IndexFlatIP(EMBED_DIM)

def load_gallery():
    if os.path.exists(GALLERY_EMB_PATH) and os.path.exists(GALLERY_LBL_PATH):
        emb = np.load(GALLERY_EMB_PATH).astype("float32")
        lbls = np.load(GALLERY_LBL_PATH, allow_pickle=True).tolist()
        # Normalize for cosine
        emb = _normalize_rows(emb)
        idx = build_index()
        if emb.shape[0] > 0:
            idx.add(emb)
        return idx, emb, lbls
    # empty
    idx = build_index()
    return idx, np.empty((0, EMBED_DIM), dtype="float32"), []

def save_gallery(embeddings: np.ndarray, labels: list):
    np.save(GALLERY_EMB_PATH, embeddings.astype("float32"))
    np.save(GALLERY_LBL_PATH, np.array(labels, dtype=object))

faiss_index, gallery_embeddings, gallery_labels = load_gallery()

# -----------------------------
# Face Utils
# -----------------------------
def encode_faces_rgb(rgb_img: np.ndarray, locations):
    # face_recognition expects RGB
    enc = face_recognition.face_encodings(rgb_img, known_face_locations=locations)
    if len(enc) == 0:
        return np.empty((0, EMBED_DIM), dtype="float32")
    return np.array(enc, dtype="float32")

def draw_box_rgb(rgb_img, loc, text, color=(0, 255, 0)):
    top, right, bottom, left = loc
    cv2.rectangle(rgb_img, (left, top), (right, bottom), color, 2)
    cv2.putText(
        rgb_img, text, (left, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
    )

# -----------------------------
# Matching
# -----------------------------
def match_embedding(emb: np.ndarray):
    """
    emb: (128,) float32 (NOT normalized yet)
    returns (label, score, idx)
    """
    if faiss_index.ntotal == 0:
        return "Unknown", 0.0, -1
    emb = emb.reshape(1, -1).astype("float32")
    emb = _normalize_rows(emb)  # cosine via IP
    scores, inds = faiss_index.search(emb, 1)  # higher is better
    score = float(scores[0][0])
    idx = int(inds[0][0])
    if idx < 0:
        return "Unknown", 0.0, -1
    label = gallery_labels[idx] if 0 <= idx < len(gallery_labels) else "Unknown"
    if score >= SIM_THRESHOLD:
        return label, score, idx
    return "Unknown", score, idx

# -----------------------------
# Ingest (optional threaded pipeline)
# -----------------------------
ingest_q = queue.Queue(maxsize=32)
detect_q = queue.Queue(maxsize=32)
embed_q = queue.Queue(maxsize=32)
match_q = queue.Queue(maxsize=32)

def ingest_frame(frame_rgb: np.ndarray):
    ingest_q.put((datetime.datetime.utcnow(), frame_rgb))

def detect_worker():
    while True:
        ts, rgb = ingest_q.get()
        try:
            locs = face_recognition.face_locations(rgb, model="hog")
            detect_q.put((ts, rgb, locs))
        except Exception as e:
            logging.exception(f"detect_worker error: {e}")

def embed_worker():
    while True:
        ts, rgb, locs = detect_q.get()
        try:
            encs = encode_faces_rgb(rgb, locs)
            for loc, emb in zip(locs, encs):
                embed_q.put((ts, rgb, loc, emb))
        except Exception as e:
            logging.exception(f"embed_worker error: {e}")

def match_worker():
    while True:
        ts, rgb, loc, emb = embed_q.get()
        try:
            label, score, idx = match_embedding(emb)
            match_q.put((ts, rgb, loc, label, score, idx))
        except Exception as e:
            logging.exception(f"match_worker error: {e}")

def audit_worker():
    while True:
        ts, _, loc, label, score, idx = match_q.get()
        top, right, bottom, left = loc
        enc_log({
            "timestamp": ts.isoformat(),
            "status": "match" if label != "Unknown" else "no_match",
            "label": label,
            "score": score,
            "bbox": [left, top, right, bottom],
            "idx": idx
        })

# spin up threads (daemon)
for target in (detect_worker, embed_worker, match_worker, audit_worker):
    threading.Thread(target=target, daemon=True).start()

# -----------------------------
# Gradio Handlers (simple sync)
# -----------------------------
def recognize_image(rgb_img):
    """
    rgb_img: numpy RGB from Gradio
    Do a synchronous path (simpler + less latency for UI).
    """
    if rgb_img is None:
        return None

    # Detect
    locations = face_recognition.face_locations(rgb_img, model="hog")  # or "cnn" if installed
    if len(locations) == 0:
        return rgb_img

    # Encode
    encs = encode_faces_rgb(rgb_img, locations)

    # Match + Draw
    for loc, emb in zip(locations, encs):
        label, score, _ = match_embedding(emb)
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        txt = f"{label if label!='Unknown' else 'no_match'} ({score:.2f})"
        draw_box_rgb(rgb_img, loc, txt, color=color)

        # Audit
        top, right, bottom, left = loc
        enc_log({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "status": "match" if label != "Unknown" else "no_match",
            "label": label,
            "score": score,
            "bbox": [left, top, right, bottom]
        })

    return rgb_img

def enroll_image(rgb_img, person_id: str):
    """
    Adds all faces found in the image to the FAISS gallery under person_id.
    Returns how many faces enrolled.
    """
    if rgb_img is None or not person_id:
        return "Provide an image and a person id.", None

    locs = face_recognition.face_locations(rgb_img, model="hog")
    if len(locs) == 0:
        return "No face found to enroll.", rgb_img

    encs = encode_faces_rgb(rgb_img, locs)
    if encs.shape[0] == 0:
        return "Failed to encode face(s).", rgb_img

    # normalize for cosine
    encs_norm = _normalize_rows(encs.astype("float32"))
    global faiss_index, gallery_embeddings, gallery_labels
    # extend in-memory arrays
    if gallery_embeddings.shape[0] == 0:
        gallery_embeddings = encs_norm
    else:
        gallery_embeddings = np.vstack([gallery_embeddings, encs_norm])
    gallery_labels.extend([person_id] * encs_norm.shape[0])

    # re-create index to keep it simple (or faiss_index.add for incremental)
    faiss_index = build_index()
    if gallery_embeddings.shape[0] > 0:
        faiss_index.add(gallery_embeddings)

    save_gallery(gallery_embeddings, gallery_labels)

    # draw boxes for feedback
    for loc in locs:
        draw_box_rgb(rgb_img, loc, f"enrolled:{person_id}", color=(255, 255, 0))

    return f"Enrolled {encs_norm.shape[0]} face(s) for '{person_id}'.", rgb_img

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESC}")

    with gr.Tab("Recognize"):
        inp = gr.Image(type="numpy", label="Image / Webcam", sources=["webcam", "upload"], stream=False)
        out = gr.Image(type="numpy", label="Output")
        btn = gr.Button("Detect & Match")
        btn.click(fn=recognize_image, inputs=inp, outputs=out)

    with gr.Tab("Enroll"):
        enroll_img = gr.Image(type="numpy", label="Image with face", sources=["upload", "webcam"], stream=False)
        person = gr.Textbox(label="Person ID (name/label)", placeholder="e.g., amitesh")
        enroll_btn = gr.Button("Enroll")
        status = gr.Textbox(label="Status")
        enroll_out = gr.Image(type="numpy", label="Preview")
        enroll_btn.click(fn=enroll_image, inputs=[enroll_img, person], outputs=[status, enroll_out])

if __name__ == "__main__":
    # optional: warm up a bit so first request is snappy
    time.sleep(0.1)
    demo.launch()
