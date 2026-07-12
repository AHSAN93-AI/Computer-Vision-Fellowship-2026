# Model Training & Evaluation — Hand Gesture Detection

**AI Summer Fellowship 2026 — Computer Vision Track, Week 2**
**Assignment 3: Model Training | Assignment 4: Evaluation**

---

## Assignment 3 — Model Training

### Model & Setup

| Setting | Value |
|---|---|
| Base Model | YOLOv8n (Nano), pretrained (`yolov8n.pt`) |
| Training Type | Transfer Learning |
| Framework | Ultralytics 8.4.92 |
| Hardware | Google Colab — Tesla T4 GPU |
| Classes | 5 (fist, open palm, peace, thumbs down, thumbs up) |

Two final training runs were completed on the 5-class dataset (after `call_me` was removed) to directly compare **non-augmented** vs **augmented** data, continuing the experiment series from earlier runs.

### Hyperparameters (both runs — identical config, different dataset export)

| Parameter | Value |
|---|---|
| Epochs (max) | 100 |
| Image Size (imgsz) | 640 |
| Batch Size | 16 |
| Learning Rate (lr0) | 0.01 |
| Patience (early stopping) | 20 |
| Optimizer | Ultralytics default (SGD/Auto) |

### Run 6 — No Augmentation (5 classes)

- Dataset: direct Roboflow YOLOv8 export, no augmentation applied
- Training stopped early via `patience=20`
- **Epochs completed: 69** (best weights from epoch 49)
- Training time: ~0.118 hours

### Run 7 — Augmented (5 classes)

- Dataset: Roboflow YOLOv8 export with augmentation applied at export time
- Training stopped early via `patience=20`
- **Epochs completed: 58** (best weights from epoch 38)
- Training time: ~0.201 hours

### What Was Experimented With (Assignment 7 overlap)

- **Epochs:** capped at 100, but early stopping consistently triggered in the 38–69 range across runs
- **Image Size:** earlier runs used imgsz=512; final 5-class runs standardized on imgsz=640
- **Batch Size:** held at 16 across all recorded runs
- **Learning Rate:** held at 0.01 (default) across all recorded runs
- **Data variant:** the key variable tested in this pair of runs — augmented vs. non-augmented export, holding all other hyperparameters constant

---

## Assignment 4 — Evaluation

### Run 6 — No Augmentation — Validation Results

**Overall (72 validation images, 67 instances):**

| Metric | Value |
|---|---|
| Precision | 0.937 |
| Recall | 0.907 |
| mAP50 | 0.969 |
| mAP50-95 | 0.668 |

**Per-Class Breakdown:**

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Fist | 14 | 14 | 1.000 | 0.856 | 0.954 | 0.658 |
| Open Palm | 12 | 12 | 0.929 | 0.917 | 0.984 | 0.682 |
| Peace | 10 | 10 | 0.904 | 0.900 | 0.978 | 0.588 |
| Thumbs Down | 9 | 9 | 0.872 | 1.000 | 0.995 | 0.833 |
| Thumbs Up | 22 | 22 | 0.982 | 0.864 | 0.934 | 0.578 |

![Confusion Matrix - No Augmentation](assets/noaug_confusion_matrix.png)
![Confusion Matrix Normalized - No Augmentation](assets/noaug_confusion_matrix_normalized.png)

**Sample predictions:** run on the held-out test set (39 images) at `conf=0.4`, `imgsz=640`. Most test images returned a single correct-class detection (e.g. thumbs up, fist, thumbs down, open palm, peace all present); 3 of 39 test images returned no detections.

---

### Run 7 — Augmented — Validation Results

**Overall (72 validation images, 67 instances):**

| Metric | Value |
|---|---|
| Precision | 0.865 |
| Recall | 0.911 |
| mAP50 | 0.942 |
| mAP50-95 | 0.674 |

**Per-Class Breakdown:**

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Fist | 14 | 14 | 0.770 | 0.959 | 0.943 | 0.639 |
| Open Palm | 12 | 12 | 0.913 | 0.917 | 0.943 | 0.640 |
| Peace | 10 | 10 | 0.888 | 0.900 | 0.962 | 0.672 |
| Thumbs Down | 9 | 9 | 0.808 | 1.000 | 0.984 | 0.831 |
| Thumbs Up | 22 | 22 | 0.945 | 0.781 | 0.880 | 0.588 |

![Confusion Matrix - Augmented](assets/aug_confusion_matrix.png)
![Confusion Matrix Normalized - Augmented](assets/aug_confusion_matrix_normalized.png)

**Sample predictions:** run on the same 39-image test set at `conf=0.4`, `imgsz=640`. Same overall detection pattern as Run 6, including a couple of images with 2 fists correctly detected in one frame; 3 of 39 test images returned no detections.

---

## Comparison — Run 6 (No Augmentation) vs Run 7 (Augmented)

| Metric | No Augmentation | Augmented | Difference |
|---|---|---|---|
| Precision | **0.937** | 0.865 | −0.072 |
| Recall | 0.907 | **0.911** | +0.004 |
| mAP50 | **0.969** | 0.942 | −0.027 |
| mAP50-95 | 0.668 | **0.674** | +0.006 |
| Epochs to converge | 69 (best@49) | 58 (best@38) | augmented converged faster |

### Analysis — Strengths and Weaknesses

**No Augmentation (Run 6):**
- Strengths: Highest precision (0.937) and mAP50 (0.969) of the two runs — very few false positives overall. `Fist` reached perfect precision (1.000). `Thumbs down` reached perfect recall (1.000) and near-perfect mAP50 (0.995).
- Weaknesses: Slightly lower recall on `fist` (0.856) and `thumbs up` (0.864) — a small number of missed detections on those classes. mAP50-95 (0.668) indicates bounding boxes are accurate but not pixel-perfect at stricter IoU thresholds.

**Augmented (Run 7):**
- Strengths: Slightly better recall (0.911) and mAP50-95 (0.674) — marginally tighter boxes and fewer missed detections overall. Converged in fewer epochs (38 vs 49), suggesting augmentation helped the model generalize faster from the same underlying images.
- Weaknesses: Precision dropped noticeably (0.865 vs 0.937), driven mainly by `fist` (0.770) — augmentation appears to have introduced more false positives for this class, and `thumbs up` recall dropped to 0.781, meaning more missed detections for that gesture specifically.

**Overall conclusion:** Consistent with the earlier 6-class experiments in the project log, augmentation continues to trade precision for a small gain in recall/mAP50-95 rather than improving both — at this dataset size (~300 images), it does not deliver a clean win. **No Augmentation (Run 6) is the stronger candidate for deployment** based on mAP50 and precision, while the augmented run remains a useful comparison point for the Assignment 7 experiment report and shows augmentation is not free of trade-offs even after class cleanup.

### Known Weak Points (both runs)

- `thumbs up` has the lowest mAP50 of any class in both runs (0.934 / 0.880) despite having the most training instances (22) — likely visual overlap with `peace`/`open palm` in some hand orientations.
- mAP50-95 sits noticeably below mAP50 across all classes in both runs, meaning bounding box localization is good but not tight — expected at this dataset size and something to flag honestly in the report rather than something to hide.

---

## Files Referenced

- `hand_gesture_No_AUGMENTATION.ipynb` — Run 6 training notebook
- `hand_gesture_final_AUGMENTED.ipynb` — Run 7 training notebook
- Confusion matrices saved under `assets/` alongside this file
