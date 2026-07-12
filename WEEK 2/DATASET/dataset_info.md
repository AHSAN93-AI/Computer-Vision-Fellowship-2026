# Dataset Info — Hand Gesture Detection

**AI Summer Fellowship 2026 — Computer Vision Track, Week 2**
**Assignment 2: Annotation**

---

## 1. Overview

The dataset was annotated using **Roboflow** (browser-based, free tier, direct YOLOv8 export). All images were manually labeled with bounding boxes around the hand performing each gesture.

The **Call Me** class (thumb + pinky gesture) was dropped after initial annotation — it was visually too close to other gestures and was removed to keep the class set clean and reduce misclassification risk. The dataset now covers **5 gesture classes**.

---

## 2. Class List (5 Classes)

| # | Class Name | Description |
|---|---|---|
| 1 | Fist | Closed hand |
| 2 | Open Palm | Open hand / stop gesture |
| 3 | Peace | Peace sign (V) |
| 4 | Thumbs Down | Thumb pointing down |
| 5 | Thumbs Up | Thumb pointing up |

*(Call Me class removed from the original 6-class plan.)*

---

## 3. Annotation Counts (per class)

| Class | Label Count |
|---|---|
| Fist | 71 |
| Open Palm | 59 |
| Peace | 56 |
| Thumbs Down | 53 |
| Thumbs Up | 77 |
| **Total Labels** | **316** |

---

## 4. Dataset Summary

| Metric | Value |
|---|---|
| Number of Classes | 5 |
| Number of Labels (bounding boxes) | 316 |
| Number of Images | *(fill in — total annotated images used to produce the 316 labels above; one box per image unless multi-hand frames exist)* |
| Annotation Tool | Roboflow |
| Export Format | YOLOv8 (Ultralytics-compatible) |

> Note: label count and image count are usually equal for this project since each image contains one hand/one gesture. If any image has more than one bounding box, image count will be slightly lower than label count — confirm the exact image total from the Roboflow dataset health tab before finalizing.

---

## 5. Annotation Requirements Checklist

- [x] Accurate bounding boxes (tight around the hand region)
- [x] Correct class labels assigned per gesture
- [x] No missing objects — every hand in every image has a corresponding box
- [x] Extra/duplicate 7th class (typo/case-mismatch) identified and deleted before export
- [x] Invisible/zero-size bounding box bug identified and fixed (caused by click-release instead of click-hold-drag-release)

---

## 6. Dataset Split

| Split | Ratio (Target) |
|---|---|
| Train | 70% |
| Validation | 20% |
| Test | 10% |

---

## 7. Notes

- Format chosen: **YOLOv8** export (not COCO) — matches Ultralytics training pipeline directly with no conversion step needed.
- YOLO detection format does not use per-class image folders; all images live in `images/`, with matching `.txt` label files (class index + normalized bounding box coordinates) in `labels/`.
- Class balance is reasonable across the 5 remaining classes, with Thumbs Up (77) as the largest class and Thumbs Down (53) as the smallest.
