"""
evaluation/evaluate.py
======================
Evaluation framework for the Face Recognition Identification System.

Measures:
  - Identification accuracy on known persons (leave-one-out)
  - False Rejection Rate (FRR) — known persons classified as Unknown
  - Unknown Rejection Rate — unknown persons correctly rejected
  - False Acceptance Rate (FAR) — unknown persons incorrectly accepted
  - Threshold sweep across a configurable range

Usage
-----
# Evaluate known persons only:
    python evaluation/evaluate.py --dataset data/enrolled --threshold 0.45

# With a separate unknown dataset (persons NOT enrolled):
    python evaluation/evaluate.py --dataset data/enrolled \\
        --unknown-dataset data/test --threshold 0.45

# Run threshold sweep (saves results.json automatically):
    python evaluation/evaluate.py --dataset data/enrolled --sweep

Output
------
Results are printed to stdout and saved to evaluation/results.json.

IMPORTANT
---------
This script ONLY reports results it can measure from actual images.
If no dataset is provided, or the dataset is empty, it prints:
    "Not evaluated — requires local test dataset"
No evaluation numbers are fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config import RESULTS_PATH, SUPPORTED_EXTENSIONS, THRESHOLD_SWEEP
from face_model import FaceModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = SUPPORTED_EXTENSIONS


def load_dataset_embeddings(
    model: FaceModel, dataset: Path
) -> dict[str, list[np.ndarray]]:
    """
    Generate ArcFace embeddings for every supported image under ``dataset``.

    Expected layout::

        dataset/
            Alice/
                img1.jpg
                img2.jpg
            Bob/
                img1.jpg

    Returns a dict mapping person name → list of unit-normalized embeddings.
    Images where detection fails or returns != 1 face are skipped.
    """
    embeddings: dict[str, list[np.ndarray]] = {}

    person_folders = sorted(p for p in dataset.iterdir() if p.is_dir())
    if not person_folders:
        return {}

    for folder in person_folders:
        vecs: list[np.ndarray] = []
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for img_path in image_paths:
            try:
                faces = model.detect_and_embed_file(img_path)
            except Exception:  # noqa: BLE001
                continue
            if len(faces) == 1:
                vecs.append(faces[0].embedding)
        if vecs:
            embeddings[folder.name] = vecs

    return embeddings


def cosine_identify(
    query: np.ndarray,
    references: dict[str, list[np.ndarray]],
    threshold: float,
) -> tuple[str, float]:
    """Return (predicted_name, best_score) for a query embedding."""
    scores: list[tuple[str, float]] = []
    for name, vecs in references.items():
        for ref in vecs:
            scores.append((name, float(np.dot(query, ref))))
    if not scores:
        return "Unknown", 0.0
    best_name, best_score = max(scores, key=lambda x: x[1])
    return (best_name if best_score >= threshold else "Unknown"), best_score


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_known(
    embeddings: dict[str, list[np.ndarray]], threshold: float
) -> dict[str, object]:
    """
    Leave-one-out evaluation on known persons.

    For each image of a known person, use all other images as references
    and attempt to identify that image.

    Returns
    -------
    dict with keys:
        known_samples, correct_identifications, accuracy,
        rejected_as_unknown, false_rejection_rate, details
    """
    total = correct = rejected = 0
    details: list[dict[str, object]] = []

    for expected, vecs in embeddings.items():
        for i, query in enumerate(vecs):
            # Build references excluding the current query image.
            refs: dict[str, list[np.ndarray]] = {}
            for name, name_vecs in embeddings.items():
                remaining = [v for j, v in enumerate(name_vecs) if not (name == expected and j == i)]
                if remaining:
                    refs[name] = remaining

            predicted, score = cosine_identify(query, refs, threshold)
            total += 1
            if predicted == expected:
                correct += 1
            if predicted == "Unknown":
                rejected += 1

            details.append({
                "expected": expected,
                "predicted": predicted,
                "similarity": round(float(score), 4),
                "correct": predicted == expected,
            })

    accuracy = correct / total if total else 0.0
    frr = rejected / total if total else 0.0

    return {
        "known_samples": total,
        "correct_identifications": correct,
        "accuracy": round(float(accuracy), 4),
        "rejected_as_unknown": rejected,
        "false_rejection_rate": round(float(frr), 4),
        "details": details,
    }


def evaluate_unknown(
    unknown_embeddings: dict[str, list[np.ndarray]],
    known_references: dict[str, list[np.ndarray]],
    threshold: float,
) -> dict[str, object]:
    """
    Evaluate how well the system rejects unknown persons.

    Parameters
    ----------
    unknown_embeddings:
        Embeddings of persons NOT enrolled in ``known_references``.
    known_references:
        The enrolled database.
    threshold:
        Decision threshold.

    Returns
    -------
    dict with keys:
        unknown_samples, correctly_rejected, unknown_rejection_rate,
        false_acceptance_rate, details
    """
    total = correctly_rejected = 0
    details: list[dict[str, object]] = []

    for _person, vecs in unknown_embeddings.items():
        for query in vecs:
            predicted, score = cosine_identify(query, known_references, threshold)
            total += 1
            if predicted == "Unknown":
                correctly_rejected += 1
            details.append({
                "expected": "Unknown",
                "predicted": predicted,
                "similarity": round(float(score), 4),
                "correctly_rejected": predicted == "Unknown",
            })

    rejection_rate = correctly_rejected / total if total else 0.0
    far = (total - correctly_rejected) / total if total else 0.0

    return {
        "unknown_samples": total,
        "correctly_rejected": correctly_rejected,
        "unknown_rejection_rate": round(float(rejection_rate), 4),
        "false_acceptance_rate": round(float(far), 4),
        "details": details,
    }


def threshold_sweep(
    known_embeddings: dict[str, list[np.ndarray]],
    unknown_embeddings: dict[str, list[np.ndarray]] | None,
    thresholds: list[float],
) -> list[dict[str, object]]:
    """
    Evaluate the system at multiple threshold values.

    Returns a list of result dicts, one per threshold, sorted ascending.
    """
    rows: list[dict[str, object]] = []
    for t in thresholds:
        known_result = evaluate_known(known_embeddings, t)
        row: dict[str, object] = {
            "threshold": round(t, 4),
            "known_accuracy": known_result["accuracy"],
            "false_rejection_rate": known_result["false_rejection_rate"],
        }
        if unknown_embeddings:
            unk_result = evaluate_unknown(unknown_embeddings, known_embeddings, t)
            row["unknown_rejection_rate"] = unk_result["unknown_rejection_rate"]
            row["false_acceptance_rate"] = unk_result["false_acceptance_rate"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_known_results(result: dict[str, object], threshold: float) -> None:
    print("\n" + "=" * 58)
    print("  KNOWN-PERSON IDENTIFICATION RESULTS")
    print("=" * 58)
    print(f"  Threshold                 : {threshold:.3f}")
    print(f"  Known samples evaluated   : {result['known_samples']}")
    print(f"  Correct identifications   : {result['correct_identifications']}")
    print(f"  Accuracy                  : {result['accuracy']:.2%}")
    print(f"  Rejected as Unknown       : {result['rejected_as_unknown']}")
    print(f"  False Rejection Rate      : {result['false_rejection_rate']:.2%}")
    print("=" * 58)


def print_unknown_results(result: dict[str, object]) -> None:
    print("\n" + "=" * 58)
    print("  UNKNOWN-PERSON REJECTION RESULTS")
    print("=" * 58)
    print(f"  Unknown samples tested    : {result['unknown_samples']}")
    print(f"  Correctly rejected        : {result['correctly_rejected']}")
    print(f"  Unknown Rejection Rate    : {result['unknown_rejection_rate']:.2%}")
    print(f"  False Acceptance Rate     : {result['false_acceptance_rate']:.2%}")
    print("=" * 58)


def print_sweep_table(rows: list[dict[str, object]]) -> None:
    has_unk = "unknown_rejection_rate" in rows[0] if rows else False

    if has_unk:
        header = f"{'Threshold':>10}  {'Known Acc':>10}  {'FRR':>8}  {'URR':>8}  {'FAR':>8}"
        sep = "-" * 54
    else:
        header = f"{'Threshold':>10}  {'Known Acc':>10}  {'FRR':>8}"
        sep = "-" * 36

    print("\n" + "=" * len(sep))
    print("  THRESHOLD SWEEP")
    print("=" * len(sep))
    print(header)
    print(sep)
    for row in rows:
        if has_unk:
            print(
                f"  {row['threshold']:>8.2f}"
                f"  {row['known_accuracy']:>9.2%}"
                f"  {row['false_rejection_rate']:>7.2%}"
                f"  {row['unknown_rejection_rate']:>7.2%}"
                f"  {row['false_acceptance_rate']:>7.2%}"
            )
        else:
            print(
                f"  {row['threshold']:>8.2f}"
                f"  {row['known_accuracy']:>9.2%}"
                f"  {row['false_rejection_rate']:>7.2%}"
            )
    print(sep)
    if has_unk:
        print("  Columns: Threshold | Known Accuracy | FRR | Unknown Rejection Rate | FAR")
    else:
        print("  Columns: Threshold | Known Accuracy | False Rejection Rate")
    print(
        "\n  NOTE: The optimal threshold balances FRR and FAR for your use-case.\n"
        "  Select it using a held-out validation set, not this training dataset."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", required=True,
        help="Directory containing one sub-folder per enrolled person.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.45,
        help="Cosine similarity threshold for a single evaluation run (default: 0.45).",
    )
    parser.add_argument(
        "--unknown-dataset", default=None,
        help="Optional directory of persons NOT enrolled in --dataset.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run threshold sweep across a range of values and print table.",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_PATH),
        help=f"Path to save the JSON results file (default: {RESULTS_PATH}).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading face models (ArcFace buffalo_l)…")
    model = FaceModel()

    print(f"\nLoading embeddings from: {dataset_path}")
    known_embeddings = load_dataset_embeddings(model, dataset_path)

    if not known_embeddings:
        print(
            "\nNot evaluated — requires local test dataset.\n"
            "Place face images in sub-folders under the --dataset directory:\n"
            f"  {dataset_path}/Alice/img1.jpg\n"
            f"  {dataset_path}/Bob/img1.jpg\n"
            "Then re-run this script."
        )
        sys.exit(0)

    print(f"Loaded embeddings for {len(known_embeddings)} person(s).")

    # Optional unknown dataset.
    unknown_embeddings: dict[str, list[np.ndarray]] | None = None
    if args.unknown_dataset:
        unk_path = Path(args.unknown_dataset)
        if not unk_path.is_dir():
            print(f"WARNING: Unknown dataset directory not found: {unk_path}", file=sys.stderr)
        else:
            unknown_embeddings = load_dataset_embeddings(model, unk_path)
            print(f"Loaded unknown embeddings for {len(unknown_embeddings)} person(s).")

    output: dict[str, object] = {}

    if args.sweep:
        print("\nRunning threshold sweep…")
        rows = threshold_sweep(known_embeddings, unknown_embeddings, THRESHOLD_SWEEP)
        print_sweep_table(rows)
        output["threshold_sweep"] = rows

    # Single-threshold evaluation.
    print(f"\nRunning evaluation at threshold={args.threshold:.3f}…")
    known_result = evaluate_known(known_embeddings, args.threshold)
    print_known_results(known_result, args.threshold)
    output["threshold"] = args.threshold
    output["known_evaluation"] = {k: v for k, v in known_result.items() if k != "details"}

    if unknown_embeddings:
        unk_result = evaluate_unknown(unknown_embeddings, known_embeddings, args.threshold)
        print_unknown_results(unk_result)
        output["unknown_evaluation"] = {k: v for k, v in unk_result.items() if k != "details"}

    # Save results.json.
    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
