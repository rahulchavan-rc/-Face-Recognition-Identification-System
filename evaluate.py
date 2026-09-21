"""Evaluate identification accuracy on a folder of person-labeled images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from face_model import FaceModel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_embeddings(model: FaceModel, dataset: Path) -> dict[str, list[np.ndarray]]:
    """Generate embeddings for valid images in dataset/person/images."""
    embeddings: dict[str, list[np.ndarray]] = {}
    for person_folder in sorted(path for path in dataset.iterdir() if path.is_dir()):
        person_embeddings: list[np.ndarray] = []
        for image_path in sorted(person_folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            faces = model.detect_and_embed_file(image_path)
            if len(faces) == 1:
                person_embeddings.append(faces[0].embedding)
        if person_embeddings:
            embeddings[person_folder.name] = person_embeddings
    return embeddings


def identify(
    query: np.ndarray,
    references: dict[str, list[np.ndarray]],
    threshold: float,
) -> tuple[str, float]:
    """Find the closest reference embedding using cosine similarity."""
    scores = [
        (name, float(np.dot(query, reference)))
        for name, person_references in references.items()
        for reference in person_references
    ]
    if not scores:
        return "Unknown", 0.0
    name, score = max(scores, key=lambda item: item[1])
    return (name if score >= threshold else "Unknown"), score


def evaluate_dataset(dataset: str, threshold: float = 0.45, unknown_dataset: str | None = None) -> dict[str, object]:
    model = FaceModel()
    all_embeddings = load_embeddings(model, Path(dataset))
    total = correct = unknown = 0
    details: list[dict[str, object]] = []

    for expected_name, person_embeddings in all_embeddings.items():
        for query_index, query in enumerate(person_embeddings):
            references = {
                name: [
                    embedding
                    for index, embedding in enumerate(items)
                    if not (name == expected_name and index == query_index)
                ]
                for name, items in all_embeddings.items()
            }
            references = {name: items for name, items in references.items() if items}
            predicted_name, score = identify(query, references, threshold)
            total += 1
            correct += predicted_name == expected_name
            unknown += predicted_name == "Unknown"
            details.append(
                {
                    "expected": expected_name,
                    "predicted": predicted_name,
                    "similarity": round(float(score), 3),
                }
            )

    accuracy = correct / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0

    result: dict[str, object] = {
        "known_samples": total,
        "correct_identifications": correct,
        "accuracy": round(float(accuracy), 4),
        "rejected_as_unknown": unknown,
        "unknown_rate": round(float(unknown_rate), 4),
        "false_rejection_rate": round(float(unknown_rate), 4),
        "threshold": float(threshold),
        "details": details,
    }

    if unknown_dataset:
        unknown_embeddings = load_embeddings(model, Path(unknown_dataset))
        unknown_total = 0
        correctly_rejected = 0
        unknown_details: list[dict[str, object]] = []
        for person_embeddings in unknown_embeddings.values():
            for query in person_embeddings:
                predicted_name, score = identify(query, all_embeddings, threshold)
                unknown_total += 1
                correctly_rejected += predicted_name == "Unknown"
                unknown_details.append(
                    {
                        "expected": "Unknown",
                        "predicted": predicted_name,
                        "similarity": round(float(score), 3),
                    }
                )
        rejection_rate = correctly_rejected / unknown_total if unknown_total else 0.0
        result["unknown_samples"] = unknown_total
        result["correctly_rejected_unknown_samples"] = correctly_rejected
        result["unknown_rejection_rate"] = round(float(rejection_rate), 4)
        result["false_acceptance_rate"] = round(
            float((unknown_total - correctly_rejected) / unknown_total)
            if unknown_total
            else 0.0,
            4,
        )
        result["unknown_details"] = unknown_details

    return result


def evaluate(dataset: str, threshold: float, unknown_dataset: str | None = None) -> None:
    result = evaluate_dataset(dataset, threshold, unknown_dataset)
    print(f"Known samples: {result['known_samples']}")
    print(f"Correct identifications: {result['correct_identifications']}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Rejected as Unknown: {result['rejected_as_unknown']} ({result['unknown_rate']:.2%})")
    print(f"False rejection rate: {result['false_rejection_rate']:.2%}")
    print(f"Threshold: {result['threshold']:.3f}")
    if unknown_dataset:
        print(f"Unknown samples: {result['unknown_samples']}")
        print(f"Correctly rejected Unknown samples: {result['correctly_rejected_unknown_samples']}")
        print(f"Unknown rejection rate: {result['unknown_rejection_rate']:.2%}")
        print(f"False acceptance rate: {result['false_acceptance_rate']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Folder containing one folder per person")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--unknown-dataset",
        help="Optional folder of people not enrolled in the known dataset",
    )
    args = parser.parse_args()
    evaluate(args.dataset, args.threshold, args.unknown_dataset)


if __name__ == "__main__":
    main()