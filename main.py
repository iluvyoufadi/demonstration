"""
Face Recognition API  —  deployment version
Loads best_model.pkl (dict with X_train / y_train / pipeline / threshold / size).
Pipeline key drives preprocessing — same logic as deployment_code.py / notebook.
Threshold: Euclidean 1-NN distance in raw 128-D dlib space (τ = 0.4495, Chapter 4).

Best model from notebook: pipeline="align_gamma"  ACC=97.62%  FRR=2.22%  FAR=1.63%
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import dlib
import cv2
import joblib
import io
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load dlib models ──────────────────────────────────────────────────────────
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec  = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ── Load best_model.pkl (dict format from notebook Section 8) ─────────────────
# Keys: X_train, y_train, pipeline, threshold, size
_model    = joblib.load("best_model.pkl")
X_TRAIN   = np.array(_model["X_train"], dtype=np.float64)
Y_TRAIN   = np.array(_model["y_train"])
PIPELINE  = _model["pipeline"]          # e.g. "align_gamma"
THRESHOLD = float(_model["threshold"])  # 0.4495
SIZE      = int(_model.get("size", 150))

print(f"[API] Pipeline : {PIPELINE}")
print(f"[API] Threshold: {THRESHOLD}")
print(f"[API] Train size: {len(X_TRAIN)} embeddings")

# ── LFW integer / string label → display name ─────────────────────────────────
# Y_TRAIN contains string folder names (person names) from the dataset structure.
# No integer mapping needed — we match the nearest neighbour directly.


# ── Preprocessing helpers (mirrors deployment_code.py exactly) ───────────────
def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def get_face_chip(img_rgb: np.ndarray, shape, pipeline: str, size: int):
    """Return the preprocessed face chip for the given pipeline key."""
    det = None  # shape already computed by caller

    if pipeline == "align_only":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.25)

    elif pipeline == "bbox":
        # shape_predictor rect is carried via shape.rect
        r = shape.rect
        x1 = max(r.left(), 0);  y1 = max(r.top(), 0)
        x2 = min(r.right(), img_rgb.shape[1]);  y2 = min(r.bottom(), img_rgb.shape[0])
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        face = cv2.resize(crop, (size, size))

    elif pipeline == "align_tight":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.0)

    elif pipeline == "align_padding":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.50)

    elif pipeline == "align_gamma":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.25)
        face = apply_gamma(face)

    elif pipeline == "align_clahe":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.25)
        face = apply_clahe(face)

    elif pipeline == "align_gamma_clahe":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.25)
        face = apply_gamma(face)
        face = apply_clahe(face)

    elif pipeline == "align_tight_gamma_clahe":
        face = dlib.get_face_chip(img_rgb, shape, size=size, padding=0.0)
        face = apply_gamma(face)
        face = apply_clahe(face)

    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    return np.ascontiguousarray(face, dtype=np.uint8)


def compute_descriptor(face: np.ndarray) -> np.ndarray:
    """128-D embedding averaged over original + horizontally flipped chip."""
    emb1 = np.array(face_rec.compute_face_descriptor(face))
    emb2 = np.array(face_rec.compute_face_descriptor(cv2.flip(face, 1)))
    return (emb1 + emb2) / 2.0


# ── Response schema ───────────────────────────────────────────────────────────
class RecognitionResponse(BaseModel):
    recognized:   bool
    label:        str   # display label (person name or "N/A")
    name:         str   # same as label for consistency with Flutter model
    confidence:   float # nearest-neighbour distance expressed as % closeness
    distance:     float # NN Euclidean distance — best pipeline
    raw_distance: float # NN Euclidean distance — bbox baseline (for comparison)
    face_found:   bool


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(file: UploadFile = File(...)):
    # ── Decode ────────────────────────────────────────────────────────────────
    contents = await file.read()
    img_rgb  = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # ── Detect ────────────────────────────────────────────────────────────────
    dets = detector(img_rgb, 1)
    if not dets:
        return RecognitionResponse(
            recognized=False, label="N/A", name="No Face",
            confidence=0.0, distance=0.0, raw_distance=0.0, face_found=False,
        )

    d     = dets[0]
    shape = predictor(img_rgb, d)

    # ── Best-pipeline embedding ───────────────────────────────────────────────
    chip = get_face_chip(img_rgb, shape, PIPELINE, SIZE)
    if chip is None:
        return RecognitionResponse(
            recognized=False, label="N/A", name="No Face",
            confidence=0.0, distance=0.0, raw_distance=0.0, face_found=False,
        )
    best_emb = compute_descriptor(chip).reshape(1, -1)

    # ── Bbox baseline embedding (for Flutter comparison card) ─────────────────
    h, w   = img_rgb.shape[:2]
    x1, y1 = max(0, d.left()), max(0, d.top())
    x2, y2 = min(w, d.right()), min(h, d.bottom())
    raw_crop = np.ascontiguousarray(cv2.resize(img_rgb[y1:y2, x1:x2], (SIZE, SIZE)))
    raw_emb  = compute_descriptor(raw_crop).reshape(1, -1)

    # ── 1-NN in raw 128-D space (mirrors deployment_code.py recognize_face) ───
    dists_best = np.linalg.norm(X_TRAIN - best_emb, axis=1)
    dists_raw  = np.linalg.norm(X_TRAIN - raw_emb,  axis=1)

    best_idx   = int(np.argmin(dists_best))
    dist_best  = float(dists_best[best_idx])
    dist_raw   = float(np.min(dists_raw))
    pred_label = str(Y_TRAIN[best_idx])

    # ── Threshold gate (τ = 0.4495, Chapter 4) ───────────────────────────────
    is_rec = dist_best <= THRESHOLD

    # Confidence: invert distance into 0-100 range for the UI bar
    # conf = 100 × max(0, 1 − dist / (τ × 2))  so that dist=0 → 100%, dist=τ → 50%
    conf = float(max(0.0, (1.0 - dist_best / (THRESHOLD * 2.0))) * 100)

    display_name  = pred_label if is_rec else "Unknown Person"
    display_label = pred_label if is_rec else "N/A"

    return RecognitionResponse(
        recognized=is_rec,
        label=display_label,
        name=display_name,
        confidence=round(conf, 1),
        distance=round(dist_best, 4),
        raw_distance=round(dist_raw, 4),
        face_found=True,
    )
