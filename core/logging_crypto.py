import logging
from .config import FERNET

logging.basicConfig(
    filename="face_recog_audit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def audit_log(entry: dict):
    encrypted = FERNET.encrypt(str(entry).encode()).decode()
    logging.info(f"ENCRYPTED_LOG: {encrypted}")
