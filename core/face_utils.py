import cv2
import numpy as np
import face_recognition
from .gallery import index, id_to_metadata
from .config import DIST_THRESHOLD
from .logging_crypto import audit_log

def recognize_image(frame):
    # Convert to RGB for processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    for loc, emb in zip(locations, encodings):
        emb = np.array([emb], dtype="float32")
        D, I = index.search(emb, k=1)
        dist, idx = D[0][0], I[0][0]

        status = "match" if dist < DIST_THRESHOLD else "no_match"
        color = (0, 255, 0) if status == "match" else (0, 0, 255)

        # Draw bounding box & label
        top, right, bottom, left = loc
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{status} ({dist:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Log the recognition attempt
        audit_log({
            "status": status,
            "distance": float(dist),
            "bbox": [left, top, right, bottom],
            "id": id_to_metadata[idx] if status == "match" else None
        })

    return frame

def enroll_image(frame, person_id):
    # Convert to RGB for processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect and encode
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    if not encodings:
        return "No face found", frame

    # Add first detected face to the index
    enc = np.array(encodings, dtype="float32")
    index.add(enc)
    id_to_metadata.append(person_id)

    return f"Enrolled {person_id}", frame
