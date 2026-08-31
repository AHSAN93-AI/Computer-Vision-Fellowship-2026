# AI Visual Quality Inspection Platform

> **Fill-in flags:** sections marked `> FILL IN —` need numbers, files, or
> media that only exist on your machine (dataset stats, training logs,
> accuracy/confusion-matrix numbers, screenshots, demo video). Everything
> else below is written directly from the code in this repo, so it's
> accurate as-is.

## 1. Project Title

**AI Visual Quality Inspection Platform** — Week 6, Computer Vision
Engineering Track fellowship challenge.

## 2. Inspection Problem

Automated pass/fail quality control for a manufacturing line: every unit
that passes the camera station must be classified as **PASS**, **FAIL**
(with a named defect and severity), or **INVALID** (image not usable for
a decision — too dark, too bright, or too blurred) — without a human
inspector reviewing each item.

## 3. Product / Domain

Fabric / textile patches (`product_type` defaults to `fabric_patch` in
the API, but the field is free text so any single-object-per-frame
product line can reuse the same pipeline).

## 4. Defect Types

Six classes, trained into `defect_cnn.pt` (`CLASS_NAMES` in
`app/config.py`):

| Class | Meaning | Treated as |
|---|---|---|
| `good` | No defect | Normal |
| `color` | Color/dye defect | Defect |
| `cut` | Cut/tear | Defect — **critical class** |
| `hole` | Hole | Defect — **critical class**, floor severity Major |
| `metal_contamination` | Embedded metal contamination | Defect — floor severity Major |
| `thread` | Loose/stray thread | Defect |

`hole` and `cut` are configured as **critical classes**
(`critical_classes` in `DEFAULT_DECISION_THRESHOLDS`) — a confirmed hit
on either fails the part regardless of measured defect area.

## 5. Dataset

> FILL IN — you ran dataset prep and training yourself, so this repo
> only ships the two trained weight files, not the dataset. Please add:
> - Source (public dataset name/link, or self-collected + how)
> - Total image count, and the per-class split (`good` / `color` / `cut`
>   / `hole` / `metal_contamination` / `thread`)
> - Train / validation / test split sizes or ratios
> - Any augmentation used during training
> - Image resolution/format the raw images were captured at, before your
>   training pipeline resized them to the model's 64×64 grayscale input

## 6. Architecture

```
upload (image / webcam frame / video frame)
  → preprocessing.quality   (brightness + Laplacian-variance blur check → INVALID short-circuit)
  → preprocessing.transforms (decode → grayscale, resize to 64×64, normalize to [0,1])
  → vision.classifier  (DefectCNN)        → predicted class + per-class confidence
  → vision.anomaly     (ConvAutoencoder)  → anomaly score + per-pixel reconstruction-error heatmap
  → defect_area_ratio = anomalous_pixels / total_pixels   (from the thresholded error map)
  → inspection.severity  (Minor / Major / Critical, area-ratio bands + per-class floors)
  → inspection.decision  (critical-class / defect-count / defect-area rule engine → PASS / FAIL)
  → inspection.evidence  (original.png / heatmap.png / annotated.png → evidence/<inspection_id>/)
  → inspection.result    (structured record) → database.db (SQLite)
```

**Stack:** Flask backend, vanilla HTML/CSS/JS frontend (no build step),
SQLite storage, PyTorch for both models, OpenCV for image quality checks
and evidence rendering.

**Pages:**

| Route | Purpose |
|---|---|
| `/` | Inspect a sample — four input tabs (image upload, webcam/USB camera, recorded conveyor video with auto-inspect interval, uploaded video with manual frame capture), all funneling through the same `/api/inspect` endpoint |
| `/history` | Filterable table of past inspections, evidence viewer, CSV export |
| `/analytics` | Pass/defect/invalid rate, defects-by-category, defect-rate-over-time, severity mix, latency |
| `/settings` | Live-editable decision + severity thresholds (persisted to SQLite, take effect without a restart) |

## 7. Models

**Classifier — `classifier_v1` (`models/defect_cnn.pt`)**
`DefectCNN`: 3 conv blocks (32→64→128 channels, each `Conv2d → BatchNorm2d
→ ReLU → MaxPool2d`), `AdaptiveAvgPool2d(1) → Flatten → Linear(128, 6)`
head, 6-way softmax over `CLASS_NAMES`. Input: 1×64×64 grayscale, values
in [0, 1].

**Anomaly model — `anomaly_v1` (`models/anomaly_autoencoder.pt`)**
`ConvAutoencoder`: stride-2 conv encoder (64×64 → 32×32 → 16×16 → 8×8,
64 channels = 4096 features) → linear bottleneck to a 16-d latent →
mirrored `ConvTranspose2d` decoder back to 1×64×64 with a sigmoid output.
Trained primarily on `good` samples, so it reconstructs normal texture
well and reconstructs defects poorly — the per-pixel reconstruction
error is both the anomaly score and the localization heatmap.

Both loaders (`app/vision/classifier.py`, `app/vision/anomaly.py`)
reconstruct the exact architecture from the checkpoint's `state_dict`
keys, so retraining just means dropping a new `.pt` in `models/` with
the same shape — no other code changes needed unless the architecture
itself changes.

> FILL IN — training hyperparameters (optimizer, learning rate, batch
> size, epochs, loss function per model) and hardware used, if you want
> them documented here rather than only in your training notebook.

## 8. Installation

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `flask`, `torch`, `numpy`, `pillow`,
`opencv-python-headless`, `pytest`.

Trained weights already ship in `models/defect_cnn.pt` and
`models/anomaly_autoencoder.pt` — no download step needed.

## 9. Training

> FILL IN — this repo contains the inference/serving pipeline and the
> two resulting `.pt` files, not the training script(s). Add here:
> - Where the training code lives (separate notebook/repo, or paste a
>   summary)
> - How each model was trained (classifier: standard supervised
>   cross-entropy on the 6 classes; autoencoder: reconstruction loss,
>   e.g. MSE, on `good`-only or mostly-`good` samples — confirm which)
> - Number of epochs and how you selected the final checkpoint (best
>   val loss/accuracy, early stopping, etc.)
> - Training/validation loss curves if you have plots to include

## 10. Running the Application

```bash
python3 -m app.main
```

Then open `http://localhost:5000`. The database (`database/inspections.db`)
and evidence folder (`evidence/`) are created automatically on first run.

## 11. Inspection Logic

Every `/api/inspect` call runs the full pipeline synchronously and
returns one of three statuses:

- **INVALID** — decided first, before any model runs. `preprocessing.quality`
  rejects the frame if mean brightness is below 25 or above 230, or if
  the Laplacian variance (blur measure) is below 30.
- **FAIL** — decided by `inspection.decision.decide()`:
  - `hole` or `cut` predicted with confidence ≥ 0.50 → automatic FAIL
    (critical class), regardless of measured area.
  - Otherwise, FAIL if the number of detected defects exceeds 0
    (`max_allowed_defect_count`), or the defect area ratio exceeds 0.02
    (`max_allowed_area_ratio`).
  - A defect is "detected" if either the classifier names a non-`good`
    class above the confidence threshold, or the autoencoder's anomaly
    score is ≥ 0.62 — these are independent signals, so an unfamiliar
    defect the classifier misses can still be caught by the anomaly
    model, and vice versa.
- **PASS** — everything else. On PASS, `defects_detected` is always
  cleared, even if the classifier had low-confidence noise below the
  threshold.

All five thresholds above (`classifier_confidence_threshold`,
`anomaly_score_threshold`, `max_allowed_defect_count`,
`max_allowed_area_ratio`, `critical_classes`) are live-editable from
`/settings` and persisted to SQLite — no restart or retraining needed to
retune the decision boundary.

Severity (`inspection.severity`) is a separate rule layer: `defect_area_ratio`
is banded into Minor (≤0.01) / Major (≤0.05) / Critical (>0.05), with a
per-class floor — `hole`, `cut`, and `metal_contamination` can never be
rated below Major even if the measured area is tiny.

## 12. Evaluation Results

> FILL IN — no metrics/evaluation artifacts shipped with this package
> (checked: no `.json`/`.csv` results file in the repo). Add for the
> classifier, on your held-out test set:
> - Overall accuracy, and precision/recall/F1 per class
> - A confusion matrix (image or table) — this matters most here since
>   `hole`/`cut` misclassified as `good` is a much worse failure mode
>   than confusing `color` and `thread`
> - Any class imbalance you had to correct for

## 13. Anomaly Detection

The autoencoder is unsupervised and trained mostly on `good` samples, so
it doesn't predict a class — it flags "this doesn't reconstruct like a
normal sample," which is deliberately independent of the classifier's
six trained classes. In the decision engine this means:

- A sample the classifier confidently calls `good` can still fail if the
  anomaly score (mean per-pixel reconstruction error) is ≥ 0.62 —
  useful for defect types or defect *instances* that don't look like
  anything the classifier was trained on.
- The same reconstruction-error map is reused as the localization
  signal: pixels above `mean + 1·std` of the error map are marked
  anomalous, and their fraction of total pixels becomes
  `defect_area_ratio`, which both the decision engine and the severity
  bands read from.
- The error map is rendered as a JET colormap heatmap and saved as one
  of the three evidence images per inspection, blended into the
  `annotated.png` so a reviewer sees the verdict and *where it came
  from* in one image.

> FILL IN — whether the anomaly model catches anything the classifier
> alone missed on your validation set (see also Builder Journal, Q7).

## 14. Performance

`processing_time_ms` is measured end-to-end per inspection (quality
check → both model forward passes → evidence rendering → DB write) and
returned in every `/api/inspect` response and shown per-row in
`/history`; average/latency-over-time is also charted on `/analytics`.

> FILL IN — typical latency numbers on your hardware (CPU vs GPU if you
> tested both), and whether that's fast enough for your target line
> speed.

## 15. Screenshots

> FILL IN — add screenshots of `/`, `/history`, `/analytics`, and
> `/settings`, plus one PASS and one FAIL evidence image (original /
> heatmap / annotated) here.

## 16. Demo

> FILL IN — link or embed a short screen recording walking through:
> upload/camera capture → PASS and FAIL results → history → analytics →
> adjusting a threshold in settings and seeing it change a verdict.

## 17. Known Limitations

- Defect localization is anomaly-heatmap based (thresholded
  reconstruction error), not a bounding-box detector or segmentation
  mask — pixel-accurate boxes would need a detector/segmentation model
  under `vision/`, feeding its mask into `defect_area_ratio`.
- Single-process SQLite — fine for a demo, not for concurrent
  production-line writers.
- Both models run on 64×64 grayscale input, so any defect that only
  shows up in color or at finer resolution than that downsampling
  preserves won't be visible to either model.
- No authentication/access control on any route or API endpoint.
- `/settings` changes apply immediately and globally with no audit log
  of who changed a threshold or when.

## 18. Future Improvements

- Swap in a detector/segmentation model (YOLO/U-Net) for pixel-accurate
  defect localization instead of the anomaly heatmap proxy.
- Add per-camera/per-line calibration for the quality-check thresholds
  (brightness/blur), since a fixed global threshold may not suit every
  camera setup.
- Batch inspection endpoint for reprocessing a backlog of stored images
  after a threshold or model change.
- Role-based access and an audit trail on `/settings` changes.
- Move from single-process SQLite to a server-based DB if concurrent
  writers become a real requirement.

## Tests

```bash
pytest tests/ -v
```

22 tests cover image-quality checks, severity bands, the decision
engine, the result schema, evidence path generation, and a real SQLite
round-trip — none of them load a model, per the "business logic only"
test requirement.
