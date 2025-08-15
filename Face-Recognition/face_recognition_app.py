import cv2
import numpy as np
import face_recognition
import faiss
import threading
import queue
import logging
import datetime
import os
from cryptography.fernet import Fernet
import gradio as gr

# -----------------------------
# Configuration & Setup
# -----------------------------
logging.basicConfig(
    filename='face_recog_audit.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

fernet = Fernet(Fernet.generate_key())

EMBED_DIM = 128
index = faiss.IndexFlatL2(EMBED_DIM)
id_to_metadata = []

ingest_queue = queue.Queue(maxsize=50)
detect_queue = queue.Queue(maxsize=50)
embed_queue = queue.Queue(maxsize=50)
match_queue = queue.Queue(maxsize=50)

# -----------------------------
# Pipeline Functions
# -----------------------------
def ingest_frame(frame):
    timestamp = datetime.datetime.utcnow()
    ingest_queue.put((timestamp, frame))
    return frame

def detect_faces_worker():
    while True:
        timestamp, frame = ingest_queue.get()
        rgb = frame[:, :, ::-1]
        locations = face_recognition.face_locations(rgb, model='hog')
        detect_queue.put((timestamp, frame, locations))

def extract_embeddings_worker():
    while True:
        timestamp, frame, locations = detect_queue.get()
        encodings = face_recognition.face_encodings(frame, locations)
        for loc, emb in zip(locations, encodings):
            embed_queue.put((timestamp, emb, loc))

def match_embeddings_worker():
    while True:
        timestamp, emb, loc = embed_queue.get()
        D, I = index.search(np.array([emb], dtype='float32'), k=1)
        dist, idx = D[0][0], I[0][0]
        status = 'match' if dist < 0.6 else 'no_match'
        metadata = id_to_metadata[idx] if status=='match' else None
        match_queue.put((status, metadata, float(dist), loc, timestamp))

def audit_and_draw():
    item = match_queue.get()
    status, metadata, dist, loc, timestamp = item
    x1, y1, x2, y2 = loc[3], loc[0], loc[1], loc[2]
    log_entry = {
        'timestamp': timestamp.isoformat(),
        'status': status,
        'distance': dist,
        'bbox': [x1, y1, x2, y2],
        'id': metadata.get('id') if metadata else None
    }
    encrypted = fernet.encrypt(str(log_entry).encode()).decode()
    logging.info(f"ENCRYPTED_LOG: {encrypted}")
    return status, dist, (x1, y1, x2, y2)

# -----------------------------
# Threading Start
# -----------------------------
for fn in (detect_faces_worker, extract_embeddings_worker, match_embeddings_worker):
    threading.Thread(target=fn, daemon=True).start()
threading.Thread(target=audit_and_draw, daemon=True).start()

# -----------------------------
# Gradio Interface
# -----------------------------
def process(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = datetime.datetime.utcnow()
    rgb = frame[:, :, ::-1]
    locations = face_recognition.face_locations(rgb, model='hog')
    if len(locations) == 0:
        return frame
    encodings = face_recognition.face_encodings(frame, locations)
    for loc, emb in zip(locations, encodings):
        D, I = index.search(np.array([emb], dtype='float32'), k=1)
        dist, idx = D[0][0], I[0][0]
        status = 'match' if dist < 0.6 else 'no_match'
        x1, y1, x2, y2 = loc[3], loc[0], loc[1], loc[2]
        color = (0,255,0) if status=='match' else (0,0,255)
        label = f"{status} ({dist:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # (Optional) Log result
    return frame


iface = gr.Interface(
    fn=process,
    inputs=gr.Image(),
    outputs=gr.Image(),
    live=True,
    title="Real-Time Face Recognition",
    description="Desktop surveillance & social media ingestion pipeline"
)


if __name__ == "__main__":
    # Preload gallery if available
    if os.path.exists('gallery_embeddings.npy'):
        gallery: object = np.load('gallery_embeddings.npy').astype('float32')
        index.add(gallery)
        # id_to_metadata load logic...
    iface.launch()
