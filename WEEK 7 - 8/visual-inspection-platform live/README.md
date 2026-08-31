# AI Visual Quality Inspection Platform

Flask + SQLite quality-control console for fabric/textile inspection.
Runs a supervised defect classifier (`classifier_v1` / `defect_cnn.pt`) and
an unsupervised anomaly autoencoder (`anomaly_v1` / `anomaly_autoencoder.pt`)
on every uploaded sample, combines both signals in a configurable decision
engine, and logs every inspection with evidence images to SQLite.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 -m app.main
```

Then open `http://localhost:5000`.

Your two trained weight files already sit in `models/`:
- `models/defect_cnn.pt` — 3-conv-block CNN, BatchNorm + AdaptiveAvgPool head, 6-way softmax over `CLASS_NAMES` in `app/config.py`.
- `models/anomaly_autoencoder.pt` — conv encoder (64→32→16→8, 16-d latent) / decoder autoencoder, trained mostly on `good` samples.

If you retrain either model, just drop the new `.pt` in place — the loader
in `app/vision/classifier.py` / `app/vision/anomaly.py` reconstructs the
exact same class shape from the state_dict, so no other code needs to change
unless the architecture itself changes.

## Pages

| Route | What it does |
|---|---|
| `/` | Inspect a sample from four input sources (image upload, webcam/USB camera, recorded conveyor video, uploaded video), run the pipeline, see the stamped PASS/FAIL/INVALID result |
| `/history` | Filterable table of past inspections, evidence viewer, CSV export |
| `/analytics` | Pass/defect/invalid rate, defects-by-category, defect-rate-over-time, severity mix, latency |
| `/settings` | Live-editable decision + severity thresholds |

## Input sources (Inspect page)

The Sample upload panel has four tabs, all funneling through the same
`/api/inspect` single-image endpoint client-side — the backend never needs
to know where a frame came from:

- **Image upload** — the original dropzone; drop or browse a JPG/PNG.
- **Webcam / USB camera** — `getUserMedia` opens a live preview. Any
  UVC-compatible camera (including a USB industrial camera) shows up in the
  device dropdown once connected and permitted; "Capture frame & inspect"
  grabs the current frame onto a canvas and submits it.
- **Conveyor feed** — upload a recorded video of a conveyor line; while it
  plays, "Start auto-inspection" grabs a frame at a configurable interval
  (default 3s) and inspects each one automatically, logging PASS/FAIL per
  frame in a running list — simulating a fixed camera watching a moving line.
- **Video upload** — upload any video, scrub/pause on a frame, and
  "Capture current frame & inspect" it manually.

All four paths reuse the same canvas-capture helper (`grabFrame`) and the
same `runInspection()` call, so evidence, severity, decision logic, history,
and analytics behave identically no matter which source produced the frame.

## Pipeline

```
upload → preprocessing.quality (blur/brightness → INVALID short-circuit)
       → preprocessing.transforms (resize 64×64, normalize)
       → vision.classifier (DefectCNN)         → predicted class + confidence
       → vision.anomaly (ConvAutoencoder)      → anomaly score + error heatmap
       → defect_area_ratio (anomalous_pixels / total_pixels)
       → inspection.severity  (Minor/Major/Critical rules)
       → inspection.decision  (PASS / FAIL rule engine)
       → inspection.evidence  (original/heatmap/annotated → evidence/<id>/)
       → inspection.result    (structured record) → database.db (SQLite)
```

## Tests

```bash
pytest tests/ -v
```

22 tests covering image-quality checks, severity bands, the decision engine,
the result schema, evidence path generation, and a real SQLite round-trip —
none of them load a model, per the spec's "business logic only" requirement.

## Known limitations / next steps

- Defect localization here is anomaly-heatmap based (thresholded reconstruction
  error), not a bounding-box detector or segmentation mask — swap in YOLO/U-Net
  under `vision/` and feed its mask into `defect_area_ratio` if you need pixel-
  accurate boxes for Requirement 7's detection/segmentation option.
- `anomaly_autoencoder.pt` currently points at your `best_autoencoderv5.pt`
  checkpoint (the other supplied file, `anomaly_autoencoderv5.pt`, has the
  identical architecture — swap it in `models/` if that's actually the one
  you want serving).
- Single-process SQLite; fine for a fellowship demo, not for concurrent
  production line writers.
