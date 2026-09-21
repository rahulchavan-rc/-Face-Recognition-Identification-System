"""
main.py
-------
Command-line interface for the Face Recognition Identification System.

Usage examples
--------------
Enroll one person (one or more images):
    python main.py enroll --name "Alice" --image data/enrolled/alice/alice1.jpg
    python main.py enroll --name "Alice" --image data/enrolled/alice/alice1.jpg data/enrolled/alice/alice2.jpg

Bulk-enroll a dataset folder (one sub-folder per person):
    python main.py enroll-dataset --dataset data/enrolled

Identify faces in an image:
    python main.py recognize --image data/test/test1.jpg
    python main.py recognize --image data/test/test1.jpg --threshold 0.50

Start real-time webcam recognition:
    python main.py webcam
    python main.py webcam --threshold 0.50 --camera 0

Inspect raw detection/embedding output (diagnostic):
    python main.py inspect --image data/test/test1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DB_PATH, DEFAULT_CAMERA, MATCH_THRESHOLD
from database import FaceDatabase
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.enrollment import FaceEnroller, enroll_dataset_folder
from src.recognition import identify_image, run_webcam
from src.utils import eprint, format_result


# ---------------------------------------------------------------------------
# Shared model initialisation (lazy, happens once per command)
# ---------------------------------------------------------------------------

def _load_models() -> tuple[FaceDetector, FaceEmbedder]:
    """Load and return the face detector and embedder."""
    print("Loading face detection and recognition models…")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    print("Models ready.")
    return detector, embedder


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_enroll(args: argparse.Namespace) -> None:
    """Enroll a single person from one or more image files."""
    db = FaceDatabase(args.database)
    detector, embedder = _load_models()
    enroller = FaceEnroller(db, detector, embedder)

    image_paths = [Path(p) for p in args.image]
    print(f"\nEnrolling '{args.name}' from {len(image_paths)} image(s)…")

    try:
        summary = enroller.enroll_from_paths(args.name, image_paths)
    except Exception as exc:  # noqa: BLE001
        eprint(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  ✓ Enrolled '{summary['person']}'")
    print(f"  Accepted images : {summary['accepted']}")
    print(f"  Rejected images : {len(summary['rejected'])}")
    if summary["rejected"]:
        for msg in summary["rejected"]:
            print(f"    - {msg}")
    print(f"  Embedding saved : {summary['npy_path']}")
    print(f"  Embedding dim   : {summary['embedding_dim']}")


def cmd_enroll_dataset(args: argparse.Namespace) -> None:
    """Bulk-enroll all person folders found in a dataset directory."""
    db = FaceDatabase(args.database)
    detector, embedder = _load_models()

    print(f"\nBulk-enrolling dataset from: {args.dataset}")
    try:
        results = enroll_dataset_folder(args.dataset, db, detector, embedder)
    except (FileNotFoundError, ValueError) as exc:
        eprint(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\nDataset enrollment complete.  Enrolled {len(results)} person(s).")


def cmd_recognize(args: argparse.Namespace) -> None:
    """Identify faces in an image file."""
    db = FaceDatabase(args.database)
    detector, embedder = _load_models()

    print(f"\nRecognizing faces in: {args.image}")
    print(f"Threshold: {args.threshold:.3f}")

    try:
        results = identify_image(args.image, db, detector, embedder, args.threshold)
    except (FileNotFoundError, ValueError) as exc:
        eprint(f"ERROR: {exc}")
        sys.exit(1)

    if not results:
        print("\nNo face detected in the image.")
        return

    print(f"\nDetected {len(results)} face(s):")
    for i, face in enumerate(results, start=1):
        line = format_result(face.match.name, face.match.similarity, args.threshold)
        print(f"  Face {i}: {line}")


def cmd_webcam(args: argparse.Namespace) -> None:
    """Launch real-time webcam recognition."""
    db = FaceDatabase(args.database)
    detector, embedder = _load_models()

    try:
        run_webcam(
            database=db,
            detector=detector,
            embedder=embedder,
            threshold=args.threshold,
            camera_index=args.camera,
        )
    except RuntimeError as exc:
        eprint(f"ERROR: {exc}")
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    """Diagnostic: show raw detection scores and embedding dimensions."""
    detector, embedder = _load_models()

    try:
        faces = detector.detect_file(args.image)
    except FileNotFoundError as exc:
        eprint(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\nImage: {args.image}")
    print(f"Detected faces: {len(faces)}")

    for i, face in enumerate(faces, start=1):
        try:
            emb = embedder.embed(face)
            dim = emb.shape[0]
            emb_info = f"dim={dim}, norm={float(emb @ emb):.4f}"
        except Exception as exc:  # noqa: BLE001
            emb_info = f"embedding failed: {exc}"

        left, top, right, bottom = (int(v) for v in face.bbox)
        print(
            f"  Face {i}: bbox=({left},{top},{right},{bottom})  "
            f"score={face.detection_score:.3f}  {emb_info}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Face Recognition Identification System  —  ArcFace + SCRFD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--database",
        default=str(DB_PATH),
        help="Path to the SQLite database (default: faces.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- enroll -----------------------------------------------------------
    enroll_p = sub.add_parser("enroll", help="Enroll a person from face image(s)")
    enroll_p.add_argument("--name", required=True, help="Person's display name")
    enroll_p.add_argument(
        "--image", nargs="+", required=True,
        metavar="IMAGE", help="One or more face image paths"
    )
    enroll_p.add_argument("--database", default=str(DB_PATH))

    # ---- enroll-dataset ---------------------------------------------------
    ds_p = sub.add_parser(
        "enroll-dataset",
        help="Bulk-enroll a dataset folder (one sub-folder per person)",
    )
    ds_p.add_argument("--dataset", required=True, help="Root dataset directory")
    ds_p.add_argument("--database", default=str(DB_PATH))

    # ---- recognize --------------------------------------------------------
    rec_p = sub.add_parser("recognize", help="Identify faces in an image")
    rec_p.add_argument("--image", required=True, help="Path to the input image")
    rec_p.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Match threshold (default: {MATCH_THRESHOLD})",
    )
    rec_p.add_argument("--database", default=str(DB_PATH))

    # ---- webcam -----------------------------------------------------------
    web_p = sub.add_parser("webcam", help="Real-time webcam recognition")
    web_p.add_argument(
        "--camera", type=int, default=DEFAULT_CAMERA,
        help=f"Camera index (default: {DEFAULT_CAMERA})",
    )
    web_p.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Match threshold (default: {MATCH_THRESHOLD})",
    )
    web_p.add_argument("--database", default=str(DB_PATH))

    # ---- inspect ----------------------------------------------------------
    ins_p = sub.add_parser("inspect", help="Diagnostic: show raw face detection output")
    ins_p.add_argument("--image", required=True, help="Path to the input image")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Route to the correct handler.
    dispatch = {
        "enroll": cmd_enroll,
        "enroll-dataset": cmd_enroll_dataset,
        "recognize": cmd_recognize,
        "webcam": cmd_webcam,
        "inspect": cmd_inspect,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()