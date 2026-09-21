from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from database import FaceDatabase
from evaluate import evaluate_dataset
from face_model import FaceModel
from matcher import FaceMatcher

ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = ROOT_DIR / "faces.db"
DATASET_ROOT = ROOT_DIR / "facedata"

app = FastAPI(title="FaceID AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = FaceModel()
DATABASE = FaceDatabase(DATABASE_PATH)


def _roster_count() -> int:
    return sum(path.is_dir() for path in DATASET_ROOT.iterdir()) if DATASET_ROOT.exists() else 0


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "FaceID AI API",
        "status": "online",
        "health": "/api/health",
        "docs": "/docs",
    }


def _safe_person_name(name: str) -> str:
    cleaned = name.strip().replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Person name is required.")
    return cleaned


def _normalize_source(value: str | None) -> str:
    if value is None:
        return "upload"

    normalized = str(value).strip().lower()
    if normalized in {"upload", "image upload", "uploaded image", "image_upload", "uploaded", "file upload", "manual upload"}:
        return "upload"
    if normalized in {"webcam", "camera", "webcam recognition", "live webcam", "web source", "web-source", "web_source", "webcam source"}:
        return "webcam"
    if "web" in normalized or "camera" in normalized or "live" in normalized:
        return "webcam"
    if "upload" in normalized or "file" in normalized or "manual" in normalized:
        return "upload"
    return "upload" if normalized.startswith("upload") else "webcam"


def _save_enrollment_image(person_name: str, original_name: str, contents: bytes) -> Path:
    person_dir = DATASET_ROOT / _safe_person_name(person_name)
    person_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(original_name).name or "face.jpg"
    safe_stem = Path(filename).stem or "face"
    suffix = Path(filename).suffix.lower() or ".jpg"
    destination = person_dir / f"{safe_stem}{suffix}"

    index = 1
    while destination.exists():
        destination = person_dir / f"{safe_stem}_{index}{suffix}"
        index += 1

    destination.write_bytes(contents)
    return destination


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return {
        "status": "online",
        "scrfd": "Active",
        "arcface": "Active",
        "sqlite": "Connected",
        "recognition_engine": "Ready",
        "database": str(DATABASE_PATH),
        "threshold": DATABASE.get_threshold(),
        "people_count": DATABASE.count_people(),
        "roster_count": _roster_count(),
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "threshold": DATABASE.get_threshold(),
        "min_threshold": 0.3,
        "max_threshold": 0.8,
        "default_threshold": 0.45,
        "supports_webcam": True,
        "supports_file_upload": True,
        "supported_formats": ["jpg", "jpeg", "png", "bmp"],
        "max_image_size_mb": 10,
    }


@app.post("/api/recognize")
async def recognize_image(
    file: UploadFile = File(...),
    threshold: float | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    source_name = _normalize_source(source)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No image file was provided.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG, PNG, or BMP.")

    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    faces = MODEL.detect_and_embed(image)
    if not faces:
        threshold_value = DATABASE.get_threshold() if threshold is None else threshold
        DATABASE.add_recognition_log(
            "Unknown",
            0.0,
            threshold_value,
            "Unknown",
            source=source_name,
            message="No face detected.",
        )
        return {
            "detected": False,
            "identity": "Unknown",
            "similarity": 0.0,
            "threshold": threshold_value,
            "status": "Unknown",
            "source": source_name,
            "message": "No face detected.",
        }

    if len(faces) > 1:
        threshold_value = DATABASE.get_threshold() if threshold is None else threshold
        DATABASE.add_recognition_log(
            "Unknown",
            0.0,
            threshold_value,
            "Unknown",
            source=source_name,
            message="Multiple faces detected.",
        )
        return {
            "detected": True,
            "multiple_faces": True,
            "identity": "Unknown",
            "similarity": 0.0,
            "threshold": threshold_value,
            "status": "Unknown",
            "source": source_name,
            "message": "Multiple faces detected. Please upload an image with one clear face.",
        }

    matcher = FaceMatcher(DATABASE, threshold=DATABASE.get_threshold() if threshold is None else threshold)
    result = matcher.match(faces[0].embedding)
    result_status = "Recognized" if not result.is_unknown else "Unknown"
    result_message = "Recognition complete." if not result.is_unknown else "Face was rejected as unknown."
    DATABASE.add_recognition_log(
        result.name,
        result.similarity,
        matcher.threshold,
        result_status,
        source=source_name,
        message=result_message,
    )
    return {
        "detected": True,
        "multiple_faces": False,
        "identity": result.name,
        "similarity": float(result.similarity),
        "threshold": matcher.threshold,
        "status": result_status,
        "source": source_name,
        "message": result_message,
    }


@app.post("/api/enroll")
async def enroll_person(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Person name is required.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required for enrollment.")

    valid_embeddings: list[np.ndarray] = []
    rejected_count = 0
    rejected_messages: list[str] = []
    saved_files: list[Path] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            rejected_count += 1
            rejected_messages.append(f"{file.filename or 'unknown'}: unsupported format")
            continue

        contents = await file.read()
        if len(contents) == 0:
            rejected_count += 1
            rejected_messages.append(f"{file.filename or 'unknown'}: empty image")
            continue

        array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            rejected_count += 1
            rejected_messages.append(f"{file.filename or 'unknown'}: invalid image")
            continue

        detected_faces = MODEL.detect_and_embed(image)
        if len(detected_faces) != 1:
            rejected_count += 1
            rejected_messages.append(
                f"{file.filename or 'unknown'}: {len(detected_faces)} faces detected"
            )
            continue

        saved_files.append(_save_enrollment_image(cleaned_name, file.filename, contents))
        valid_embeddings.append(detected_faces[0].embedding)

    if not valid_embeddings:
        raise HTTPException(
            status_code=400,
            detail="No valid enrollment images were accepted. Each image must contain exactly one clear face.",
        )

    reference = np.mean(np.vstack(valid_embeddings), axis=0)
    DATABASE.save(cleaned_name, reference, sample_count=len(valid_embeddings))

    summary = {
        "person": cleaned_name,
        "processed_count": len(files),
        "accepted_count": len(valid_embeddings),
        "rejected_count": rejected_count,
        "message": f"{len(valid_embeddings)} images processed successfully.",
    }
    if rejected_messages:
        summary["rejections"] = rejected_messages
    summary["saved_in_dataset"] = [str(path.relative_to(ROOT_DIR)) for path in saved_files]
    return summary


@app.get("/api/people")
def list_people() -> dict[str, Any]:
    people = DATABASE.list_people()
    return {"people": people}


@app.get("/api/recognition/history")
def recognition_history(limit: int = 100) -> dict[str, Any]:
    return {"records": DATABASE.list_recognition_logs(limit)}


@app.delete("/api/people/{person}")
def delete_person(person: str) -> dict[str, Any]:
    deleted = DATABASE.delete_person(person)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Person '{person}' was not found.")
    return {"status": "deleted", "person": person}


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    data = evaluate_dataset("facedata", threshold=DATABASE.get_threshold())
    return data


@app.patch("/api/config/threshold")
async def update_threshold(value: float) -> dict[str, Any]:
    threshold = float(value)
    if not 0.3 <= threshold <= 0.8:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.30 and 0.80.")
    DATABASE.set_threshold(threshold)
    return {"threshold": DATABASE.get_threshold(), "message": "Threshold updated successfully."}


@app.get("/api/health")
def healthcheck() -> dict[str, Any]:
    return {"status": "ok"}


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
