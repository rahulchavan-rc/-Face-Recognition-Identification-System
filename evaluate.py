"""Evaluate identification accuracy on a folder of person-labeled images."""

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


def evaluate(dataset: str, threshold: float, unknown_dataset: str | None = None) -> None:
    model = FaceModel()
    all_embeddings = load_embeddings(model, Path(dataset))
    total = correct = unknown = 0

    for expected_name, person_embeddings in all_embeddings.items():
        for query_index, query in enumerate(person_embeddings):
            # Leave the query out of the reference set to avoid testing an image against itself.
            references = {
                name: [embedding for index, embedding in enumerate(items) if not (name == expected_name and index == query_index)]
                for name, items in all_embeddings.items()
            }
            references = {name: items for name, items in references.items() if items}
            predicted_name, score = identify(query, references, threshold)
            total += 1
            correct += predicted_name == expected_name
            unknown += predicted_name == "Unknown"
            print(f"expected={expected_name}, predicted={predicted_name}, similarity={score:.3f}")

    accuracy = correct / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0
    print(f"Known samples: {total}")
    print(f"Correct identifications: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Rejected as Unknown: {unknown} ({unknown_rate:.2%})")
    print(f"Threshold: {threshold:.3f}")

    if unknown_dataset:
        unknown_embeddings = load_embeddings(model, Path(unknown_dataset))
        unknown_total = 0
        correctly_rejected = 0
        for person_embeddings in unknown_embeddings.values():
            for query in person_embeddings:
                predicted_name, score = identify(query, all_embeddings, threshold)
                unknown_total += 1
                correctly_rejected += predicted_name == "Unknown"
                print(f"expected=Unknown, predicted={predicted_name}, similarity={score:.3f}")

        rejection_rate = correctly_rejected / unknown_total if unknown_total else 0.0
        print(f"Unknown samples: {unknown_total}")
        print(f"Correctly rejected Unknown samples: {correctly_rejected}")
        print(f"Unknown rejection rate: {rejection_rate:.2%}")


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