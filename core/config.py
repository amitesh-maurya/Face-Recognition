import os
from cryptography.fernet import Fernet

# App metadata
TITLE = "Real-Time Face Recognition"
DESC = "Desktop surveillance & social media ingestion pipeline"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GALLERY_EMB_PATH = os.path.join(BASE_DIR, "gallery_embeddings.npy")

# Embedding config
EMBED_DIM = 128
DIST_THRESHOLD = 0.6

# Encryption key
FERNET = Fernet(Fernet.generate_key())
STREAMING_MODE: bool = False  # True for webcam streaming, False for upload+webcam
