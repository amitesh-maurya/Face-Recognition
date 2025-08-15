import os
import numpy as np
import faiss
from .config import EMBED_DIM, GALLERY_EMB_PATH

index = faiss.IndexFlatL2(EMBED_DIM)
id_to_metadata = []

def load_gallery():
    if os.path.exists(GALLERY_EMB_PATH):
        gallery = np.load(GALLERY_EMB_PATH).astype("float32")
        index.add(gallery)
        print(f"[Gallery] Loaded {gallery.shape[0]} embeddings.")
    else:
        print("[Gallery] No embeddings found.")

def save_gallery():
    if index.ntotal > 0:
        faiss.write_index(index, GALLERY_EMB_PATH)
        print("[Gallery] Saved embeddings.")
