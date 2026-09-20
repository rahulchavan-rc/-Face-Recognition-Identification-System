# ArcFace Face Recognition System

This project detects faces with SCRFD, creates ArcFace embeddings, matches them with cosine similarity, and rejects low-scoring matches as `Unknown`.

## Model and Tools

- Python 3.14 virtual environment
- InsightFace `buffalo_l` model pack
- SCRFD face detector
- ArcFace face embeddings
- SQLite embedding database
- OpenCV image and webcam input
- CPU execution through ONNX Runtime

## Setup

```bash
cd /Users/rahulchavan3134gmail.com/Desktop/face
source venv/bin/activate
```

The dependencies are listed in [requirements.txt](requirements.txt). The `buffalo_l` model downloads automatically the first time the application starts.

## Dataset Format

Put one folder per enrolled person inside `facedata`:

```text
facedata/
  rahul/photo1.jpeg
  rahul/photo2.jpeg
  sanvi/photo1.jpeg
```

Each enrollment image should contain exactly one clear face. Images with zero or multiple faces are skipped.

## Commands

Enroll all person folders:

```bash
./venv/bin/python main.py enroll-dataset facedata --database faces.db
```

Identify a face in an image:

```bash
./venv/bin/python main.py identify facedata/rahul/rahul1.jpeg --database faces.db --threshold 0.45
```

Run live webcam recognition and press `q` to quit:

```bash
./venv/bin/python main.py webcam --database faces.db --threshold 0.45
```

Inspect detection and embedding dimensions:

```bash
./venv/bin/python main.py inspect image.jpeg
```

## Matching and Unknown Rejection

ArcFace embeddings are normalized before storage. For normalized vectors, their dot product is cosine similarity. The application selects the highest similarity and identifies the person only when it is at least the threshold; otherwise it returns `Unknown`.

The current starting threshold is `0.45`. It must be tuned using separate known and unknown validation images. A higher threshold reduces false matches but may reject genuine people.

## Evaluation

Run leave-one-image-out evaluation on known people:

```bash
./venv/bin/python evaluate.py facedata --threshold 0.45
```

Current result on this dataset:

```text
Known samples: 47
Correct identifications: 45
Accuracy: 95.74%
Rejected as Unknown: 2 (4.26%)
Threshold: 0.450
```

For unknown rejection testing, create a separate folder containing people who are not enrolled:

```bash
./venv/bin/python evaluate.py facedata --unknown-dataset unknown_faces --threshold 0.45
```

The command reports the unknown rejection rate and any false acceptances.

## Failure Cases

- Very dark, blurry, or low-resolution images
- Faces turned too far sideways
- Masks, sunglasses, or heavy occlusion
- Multiple faces in an enrollment image
- No face detected
- Similar-looking people
- Threshold too low, causing false matches
- Threshold too high, causing genuine people to be rejected

## Possible Improvements

- Add more varied enrollment images per person.
- Tune the threshold using a held-out validation set.
- Add image-quality checks before generating embeddings.
- Use GPU execution for faster webcam processing.
- Add face tracking and temporal voting to stabilize webcam labels.
- Encrypt embeddings and obtain consent before storing biometric data.