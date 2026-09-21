"""
config.py
=========
Central configuration for the Face Recognition Identification System.

All tunable constants live here so you never need to hunt for magic
numbers buried inside application code.  Import this module wherever
you need a setting.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

# Root of the project (the directory that contains this file).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# SQLite database that stores enrolled embeddings at runtime.
DB_PATH: Path = PROJECT_ROOT / "faces.db"

# Directory tree written during enrollment (human-readable exports).
DATABASE_DIR: Path = PROJECT_ROOT / "database"
EMBEDDINGS_DIR: Path = DATABASE_DIR / "embeddings"   # .npy files, one per person
METADATA_PATH: Path = DATABASE_DIR / "metadata.json" # JSON registry of enrolled persons

# Directories for images.
DATA_DIR: Path = PROJECT_ROOT / "data"
ENROLLED_DIR: Path = DATA_DIR / "enrolled"   # Enrollment images (person sub-folders)
TEST_DIR: Path = DATA_DIR / "test"           # Evaluation / test images

# Evaluation output.
EVALUATION_DIR: Path = PROJECT_ROOT / "evaluation"
RESULTS_PATH: Path = EVALUATION_DIR / "results.json"

# ---------------------------------------------------------------------------
# InsightFace / ArcFace model settings
# ---------------------------------------------------------------------------

# Model pack to load.  "buffalo_l" is the large ArcFace pack (SCRFD + ArcFace-R100).
# It is downloaded automatically by InsightFace on first use (~300 MB).
MODEL_NAME: str = "buffalo_l"

# Detection input resolution.  Larger values improve small-face detection at
# the cost of speed.  640×640 is a good default for webcam frames.
DET_SIZE: tuple[int, int] = (640, 640)

# ONNX Runtime execution provider.  Use "CUDAExecutionProvider" for GPU.
EXECUTION_PROVIDER: str = "CPUExecutionProvider"

# ---------------------------------------------------------------------------
# Matching / recognition
# ---------------------------------------------------------------------------

# Minimum cosine similarity required to accept a face as a known person.
#
# This value is NOT universally optimal.  It was chosen as a reasonable
# starting point for the ArcFace buffalo_l model.  You should calibrate it
# against your own validation set:
#   - A lower threshold (e.g. 0.35) → more matches, higher false-acceptance.
#   - A higher threshold (e.g. 0.65) → fewer matches, higher false-rejection.
#
# Run `python evaluation/evaluate.py` to see a threshold-sweep table and
# select the value that best balances FAR and FRR for your dataset.
MATCH_THRESHOLD: float = 0.45

# Range accepted by the API's /api/config/threshold endpoint.
MIN_THRESHOLD: float = 0.20
MAX_THRESHOLD: float = 0.90

# Thresholds tested during the evaluation sweep.
THRESHOLD_SWEEP: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

# Supported image file extensions for enrollment and evaluation.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# Maximum raw image upload size (bytes).  10 MB is generous for face photos.
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

# Default camera index (0 = first camera on the system).
DEFAULT_CAMERA: int = 0

# OpenCV window title for the webcam feed.
WEBCAM_WINDOW_TITLE: str = "Face Recognition — press Q to quit"

# Font scale and thickness used when drawing labels on webcam frames.
LABEL_FONT_SCALE: float = 0.75
LABEL_THICKNESS: int = 2

# Bounding-box colours (BGR).
COLOR_KNOWN: tuple[int, int, int] = (0, 200, 0)    # Green  → recognized
COLOR_UNKNOWN: tuple[int, int, int] = (0, 0, 220)  # Red    → unknown
COLOR_TEXT: tuple[int, int, int] = (255, 255, 255) # White  → label text
