"""Enroll faces or inspect SCRFD detections and ArcFace embeddings."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from database import FaceDatabase
from face_model import FaceModel
from matcher import FaceMatcher


def inspect_image(model: FaceModel, image: str) -> None:
    faces = model.detect_and_embed_file(image)

    print(f"Detected faces: {len(faces)}")
    for index, face in enumerate(faces, start=1):
        print(
            f"Face {index}: detection_score={face.detection_score:.3f}, "
            f"embedding_dimensions={face.embedding.shape[0]}"
        )


def enroll_person(model: FaceModel, database: FaceDatabase, name: str, images: list[str]) -> None:
    embeddings: list[np.ndarray] = []
    for image in images:
        faces = model.detect_and_embed_file(image)
        if len(faces) != 1:
            raise ValueError(
                f"Enrollment image '{image}' must contain exactly one face; found {len(faces)}."
            )
        embeddings.append(faces[0].embedding)

    # Averaging several clear photos creates a more stable reference for this person.
    reference = np.mean(np.vstack(embeddings), axis=0)
    database.save(name, reference)
    print(f"Enrolled '{name}' using {len(images)} image(s).")


def enroll_dataset(model: FaceModel, database: FaceDatabase, dataset: str) -> None:
    """Enroll each person from a dataset organized as dataset/person/images."""
    dataset_path = Path(dataset)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    people = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    if not people:
        raise ValueError(f"No person folders found in dataset: {dataset}")

    for person_folder in people:
        image_paths = sorted(
            path for path in person_folder.iterdir() if path.suffix.lower() in image_suffixes
        )
        if not image_paths:
            print(f"Skipping '{person_folder.name}': no supported images found.")
            continue

        valid_embeddings: list[np.ndarray] = []
        failed_images: list[str] = []
        for image_path in image_paths:
            faces = model.detect_and_embed_file(image_path)
            if len(faces) == 1:
                valid_embeddings.append(faces[0].embedding)
            else:
                failed_images.append(f"{image_path.name} ({len(faces)} faces)")

        if not valid_embeddings:
            print(f"Skipping '{person_folder.name}': no valid face images.")
            continue

        # Multiple images reduce the effect of pose and lighting in one photo.
        reference = np.mean(np.vstack(valid_embeddings), axis=0)
        database.save(person_folder.name, reference)
        print(
            f"Enrolled '{person_folder.name}' with "
            f"{len(valid_embeddings)}/{len(image_paths)} image(s)."
        )
        if failed_images:
            print(f"  Skipped: {', '.join(failed_images)}")


def identify_image(
    model: FaceModel,
    database: FaceDatabase,
    image: str,
    threshold: float,
) -> None:
    faces = model.detect_and_embed_file(image)
    matcher = FaceMatcher(database, threshold=threshold)

    if not faces:
        print("No face detected.")
        return

    for index, face in enumerate(faces, start=1):
        result = matcher.match(face.embedding)
        print(
            f"Face {index}: {result.name} "
            f"(similarity={result.similarity:.3f}, threshold={threshold:.3f})"
        )


def identify_webcam(model: FaceModel, database: FaceDatabase, camera: int, threshold: float) -> None:
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera}.")

    matcher = FaceMatcher(database, threshold=threshold)
    print("Webcam started. Press q to quit.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            for face in model.detect_and_embed(frame):
                result = matcher.match(face.embedding)
                left, top, right, bottom = (int(value) for value in face.bbox)
                color = (0, 180, 0) if not result.is_unknown else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{result.name} {result.similarity:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (left, max(25, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.imshow("ArcFace Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Detect faces in one image")
    inspect_parser.add_argument("image")

    enroll_parser = subparsers.add_parser("enroll", help="Store a person's face embedding")
    enroll_parser.add_argument("name")
    enroll_parser.add_argument("images", nargs="+", help="One or more clear face images")
    enroll_parser.add_argument("--database", default="faces.db")

    dataset_parser = subparsers.add_parser(
        "enroll-dataset", help="Enroll person folders from a dataset directory"
    )
    dataset_parser.add_argument("dataset", help="Folder containing one folder per person")
    dataset_parser.add_argument("--database", default="faces.db")

    identify_parser = subparsers.add_parser(
        "identify", help="Identify faces or reject them as Unknown"
    )
    identify_parser.add_argument("image")
    identify_parser.add_argument("--database", default="faces.db")
    identify_parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Minimum cosine similarity required for identification",
    )

    webcam_parser = subparsers.add_parser(
        "webcam", help="Identify faces using a live webcam"
    )
    webcam_parser.add_argument("--camera", type=int, default=0)
    webcam_parser.add_argument("--database", default="faces.db")
    webcam_parser.add_argument("--threshold", type=float, default=0.45)

    args = parser.parse_args()
    model = FaceModel()

    if args.command == "inspect":
        inspect_image(model, args.image)
    elif args.command == "enroll":
        enroll_person(model, FaceDatabase(args.database), args.name, args.images)
    elif args.command == "enroll-dataset":
        enroll_dataset(model, FaceDatabase(args.database), args.dataset)
    elif args.command == "identify":
        identify_image(
            model,
            FaceDatabase(args.database),
            args.image,
            args.threshold,
        )
    else:
        identify_webcam(
            model,
            FaceDatabase(args.database),
            args.camera,
            args.threshold,
        )


if __name__ == "__main__":
    main()