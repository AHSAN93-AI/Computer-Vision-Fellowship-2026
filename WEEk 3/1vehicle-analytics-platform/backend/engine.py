"""
engine.py
Core Vehicle Video Analytics Engine.

Implements exactly the required feature set:
  1. Real-Time Detection      (YOLOv8, pretrained)
  2. Object Tracking          (ByteTrack, persistent IDs)
  3. Object Counter           (total / active / entered / exited)
       -> Entered = object appears in camera view (new track ID)
       -> Exited  = object has not been seen for a short grace period
                     (track lost / left the camera view)
  4. Virtual Counting Line    (user-drawn, additional line-crossing IN / OUT)
  5. Region of Interest (ROI) (user-drawn polygon, occupancy count)
  6. Movement Analytics       (trajectory / path / direction)
  7. Analytics Dashboard      (current objects, total, avg confidence, FPS, processing time)
  8. Video Recording          (annotated MP4 saved to disk)
  Bonus: Heatmap
  Plus: still-image detection, per-class ID numbering series, remove/reset source
"""

import os
import time
import threading
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ---------------------------------------------------------------- config -
MODEL_PATH = "yolov8n.pt"          # swap to a fine-tuned .pt if you have one
TRAJECTORY_LENGTH = 40             # points kept per track for path drawing
RECORDING_DIR = "../recordings"

# --- "lock" system tuning ---
# YOLO is always run at this low base confidence so the tracker gets enough
# raw detections to keep matching an object through brief confidence dips
# (this is what was causing boxes to flicker / drop and reappear as new IDs).
DETECTION_BASE_CONF = 0.10
# A track only "locks" (gets a stable, displayed ID) after being matched for
# this many consecutive frames — filters out one-off false-positive blips
# that were previously being miscounted as separate objects.
MINIMUM_CONSECUTIVE_FRAMES = 5
# Once locked, a track survives this many frames of being fully undetected
# (e.g. briefly occluded) before it's dropped and treated as "exited".
LOST_TRACK_BUFFER = 60
MISSING_FRAMES_THRESHOLD = 50      # kept in sync with LOST_TRACK_BUFFER below

# COCO class ids treated as "vehicles"
CLASS_NAME = {2: "car", 3: "motorcycle", 7: "truck", 5: "bus"}

# Per-class ID series: each class gets its own numbering series so IDs are
# never ambiguous across classes.
#   car        -> prefix "0"  -> 002, 004, 006, ...
#   motorcycle -> prefix "1"  -> 102, 104, 106, ...
#   truck      -> prefix "2"  -> 202, 204, 206, ...
#   bus        -> prefix "3"  -> 302, 304, 306, ...
CLASS_PREFIX = {2: "0", 3: "1", 7: "2", 5: "3"}


class VehicleAnalyticsEngine:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        # track_activation_threshold controls how confident a NEW object must
        # be to start a track ("lock on"); lost_track_buffer keeps a locked
        # track alive through brief detection gaps instead of dropping it and
        # starting a new ID; minimum_consecutive_frames stops single-frame
        # false positives from ever becoming a counted track.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.30,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=0.8,
            frame_rate=25,
            minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
        )

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(trace_length=TRAJECTORY_LENGTH, thickness=2)

        # source
        self.source_type = None        # "webcam" | "file" | None
        self.source_path = 0
        self.cap = None

        # runtime state
        self.running = False
        self.lock = threading.Lock()

        # Only ONE thread is ever allowed to touch cv2.VideoCapture (open,
        # read, release, reassign). Flask can serve multiple requests
        # concurrently (start/stop/remove_source/video_feed), and OpenCV's
        # FFmpeg backend is not safe to read from one thread while another
        # thread releases/reassigns it — doing so crashes with
        # "Assertion fctx->async_lock failed". A single background capture
        # thread + this lock fully serializes all camera/file access.
        self._capture_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self.latest_jpeg = None
        self._shutdown_event = threading.Event()

        # line zone (virtual counting line) — Feature 4
        self.line_zone = None
        self.line_points = None        # ((x1,y1),(x2,y2))

        # ROI zone — Feature 5
        self.roi_zone = None
        self.roi_points = None

        # analytics — Features 3, 6, 7
        self.seen_ids = set()
        self.trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.frame_count = 0
        self.fps = 0.0
        self.processing_time_ms = 0.0
        self.active_objects = 0
        self.roi_count = 0
        self.confidence_threshold = 0.3

        # appearance-based entered / exited counting (Feature 3 fix)
        self.entered_count = 0
        self.exited_count = 0
        self.exited_ids = set()          # ids already counted as exited (no double count)
        self.last_seen_frame = {}        # tracker_id -> frame_count when last detected

        # per-class ID series (Feature: custom class-based ID numbering)
        self.custom_id_map = {}                 # tracker_id (int) -> "002" style string
        self.class_seq = {"0": 0, "1": 0, "2": 0, "3": 0}
        self.id_class_name = {}                  # tracker_id -> class name

        # heatmap (bonus)
        self.heatmap_enabled = False
        self.heatmap_accum = None                 # np.float32 array, same H,W as frame

        # recording — Feature 8
        self.recording = False
        self.video_writer = None

        # background capture thread — started only once everything above
        # exists, since it immediately starts reading these attributes.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def shutdown(self):
        """Call on process exit for a clean stop of the background capture
        thread (avoids noisy native cleanup messages on Ctrl+C)."""
        self.running = False
        self._shutdown_event.set()
        with self._capture_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

    # ---------------------------------------------------------- source ----
    def set_source(self, source_type, path=None):
        with self._capture_lock:
            self.source_type = source_type
            self.source_path = 0 if source_type == "webcam" else path
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.source_path)
        self.tracker.reset()
        with self.lock:
            self._reset_stats()

    def remove_source(self):
        """Stops and fully releases the current source so a new one can be loaded."""
        self.running = False
        with self._capture_lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.source_type = None
            self.source_path = 0
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
        self.tracker.reset()
        with self._frame_lock:
            self.latest_jpeg = None
        with self.lock:
            self._reset_stats()

    def _reset_stats(self):
        self.seen_ids = set()
        self.trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.frame_count = 0
        self.active_objects = 0
        self.entered_count = 0
        self.exited_count = 0
        self.exited_ids = set()
        self.last_seen_frame = {}
        self.custom_id_map = {}
        self.class_seq = {"0": 0, "1": 0, "2": 0, "3": 0}
        self.id_class_name = {}
        self.heatmap_accum = None
        if self.line_points is not None:
            self.set_line(*self.line_points[0], *self.line_points[1])

    # ------------------------------------------------------------- line ---
    def set_line(self, x1, y1, x2, y2):
        with self.lock:
            self.line_points = ((x1, y1), (x2, y2))
            self.line_zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))

    def clear_line(self):
        with self.lock:
            self.line_zone = None
            self.line_points = None

    # -------------------------------------------------------------- roi ---
    def set_roi(self, points):
        with self.lock:
            self.roi_points = points
            self.roi_zone = sv.PolygonZone(polygon=np.array(points, dtype=np.int32))

    def clear_roi(self):
        with self.lock:
            self.roi_zone = None
            self.roi_points = None

    def set_confidence_threshold(self, value):
        with self.lock:
            self.confidence_threshold = float(value)
            # controls how confident a brand-new object must be to lock on;
            # detection itself always runs at a low base confidence (see
            # DETECTION_BASE_CONF) so the tracker can still use fainter boxes
            # to keep matching an object it has already locked onto.
            self.tracker.track_activation_threshold = float(value)

    def set_heatmap(self, enabled):
        with self.lock:
            self.heatmap_enabled = bool(enabled)

    # --------------------------------------------------------- control ----
    def start(self):
        if self.cap is None:
            self.set_source("webcam")
        self.running = True

    def stop(self):
        self.running = False

    def toggle_recording(self, on, frame_size=None):
        with self.lock:
            if on and not self.recording:
                os.makedirs(RECORDING_DIR, exist_ok=True)
                filename = f"{RECORDING_DIR}/recording_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w, h = frame_size or (640, 480)
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                self.recording = True
                self.current_recording_path = filename
            elif not on and self.recording:
                self.recording = False
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

    # ------------------------------------------------------- custom ids ---
    def _get_custom_id(self, tracker_id, class_id):
        """Assigns each tracker_id a persistent, class-scoped display ID.
        Each class keeps its own series, incrementing by 2:
          car -> 002, 004, 006 ...   motorcycle -> 102, 104, 106 ...
          truck -> 202, 204, 206 ... bus -> 302, 304, 306 ...
        """
        if tracker_id in self.custom_id_map:
            return self.custom_id_map[tracker_id]
        prefix = CLASS_PREFIX.get(int(class_id), "9")
        self.class_seq[prefix] = self.class_seq.get(prefix, 0) + 2
        custom_id = f"{prefix}{self.class_seq[prefix]:02d}"
        self.custom_id_map[tracker_id] = custom_id
        self.id_class_name[tracker_id] = CLASS_NAME.get(int(class_id), "vehicle")
        return custom_id

    # ------------------------------------------------------------ frames --
    def _capture_loop(self):
        """The ONLY place cv2.VideoCapture is read from. Runs for the life
        of the process; produces annotated JPEG frames into a shared buffer
        that generate_frames() (one per browser connection) simply reads.
        """
        while True:
            if self._shutdown_event.is_set():
                return
            try:
                self._capture_loop_step()
            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _capture_loop_step(self):
        if not self.running:
            time.sleep(0.05)
            return

        t0 = time.time()
        with self._capture_lock:
            cap = self.cap
            if cap is None:
                ok, frame = False, None
            else:
                ok, frame = cap.read()
                if not ok and self.source_type == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()

        if not ok or frame is None:
            time.sleep(0.05)
            return

        annotated = self._process_frame(frame)

        processing_time_ms = (time.time() - t0) * 1000.0
        with self.lock:
            self.processing_time_ms = processing_time_ms
            self.frame_count += 1
            if processing_time_ms > 0:
                self.fps = 1000.0 / processing_time_ms
            recording = self.recording
            writer = self.video_writer

        if recording and writer is not None:
            writer.write(annotated)

        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with self._frame_lock:
                self.latest_jpeg = buf.tobytes()

    def generate_frames(self):
        """Generator yielding MJPEG-encoded annotated frames. Safe to call
        from any number of concurrent Flask requests — it never touches
        cv2.VideoCapture directly, only reads the shared latest_jpeg buffer.
        """
        last_sent = None
        while True:
            with self._frame_lock:
                jpeg = self.latest_jpeg
            if jpeg is None or jpeg is last_sent:
                time.sleep(0.03)
                continue
            last_sent = jpeg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

    def _process_frame(self, frame):
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            classes=list(CLASS_NAME.keys()),
            conf=DETECTION_BASE_CONF,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # ---- Feature 3: counter bookkeeping ----
        self.active_objects = len(detections)
        if detections.confidence is not None:
            for conf in detections.confidence:
                self.confidence_sum += float(conf)
                self.confidence_count += 1

        current_ids = set()
        custom_labels = []
        if detections.tracker_id is not None:
            for tid, box, cls_id, conf in zip(
                detections.tracker_id, detections.xyxy, detections.class_id, detections.confidence
            ):
                tid = int(tid)
                current_ids.add(tid)

                # --- Entered: object appears in camera view for the first time ---
                if tid not in self.seen_ids:
                    self.seen_ids.add(tid)
                    self.entered_count += 1
                    # if it was previously marked exited (re-appeared), un-mark it
                    self.exited_ids.discard(tid)

                self.last_seen_frame[tid] = self.frame_count

                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

                # Feature 6: trajectory / movement path
                self.trajectories[tid].append((cx, cy))

                # per-class ID series
                custom_id = self._get_custom_id(tid, cls_id)
                cname = CLASS_NAME.get(int(cls_id), "vehicle")
                custom_labels.append(f"🔒#{custom_id} {cname}")

        # --- Exited: object hasn't been seen for MISSING_FRAMES_THRESHOLD frames ---
        for tid, last_frame in list(self.last_seen_frame.items()):
            if tid in current_ids:
                continue
            if tid in self.exited_ids:
                continue
            if self.frame_count - last_frame > MISSING_FRAMES_THRESHOLD:
                self.exited_ids.add(tid)
                self.exited_count += 1

        # ---- Feature 4: virtual counting line (additional line-crossing stat) ----
        if self.line_zone is not None:
            self.line_zone.trigger(detections)

        # ---- Feature 5: ROI occupancy ----
        if self.roi_zone is not None:
            mask = self.roi_zone.trigger(detections)
            self.roi_count = int(mask.sum()) if mask is not None else 0
        else:
            self.roi_count = 0

        # ---- Bonus: heatmap accumulation ----
        if self.heatmap_enabled:
            if self.heatmap_accum is None or self.heatmap_accum.shape != (h, w):
                self.heatmap_accum = np.zeros((h, w), dtype=np.float32)
            self.heatmap_accum *= 0.98  # slow decay so older activity fades
            if detections.xyxy is not None:
                for box in detections.xyxy:
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    cv2.circle(self.heatmap_accum, (cx, cy), 22, 1.0, -1)

        # ---- draw annotations ----
        annotated = frame.copy()

        if self.heatmap_enabled and self.heatmap_accum is not None:
            norm = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.7, colored, 0.4, 0)

        annotated = self.trace_annotator.annotate(annotated, detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=custom_labels)

        if self.line_points is not None:
            (x1, y1), (x2, y2) = self.line_points
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)

        if self.roi_points is not None:
            pts = np.array(self.roi_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (255, 180, 0), 2)

        return annotated

    # ------------------------------------------------------- still image --
    def detect_image(self, image):
        """Runs detection (no tracking) on a single uploaded image.
        Assigns the same class-based ID series, scoped to this image only.
        Returns (annotated_image, stats_dict).
        """
        results = self.model(
            image,
            classes=list(CLASS_NAME.keys()),
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        local_seq = {"0": 0, "1": 0, "2": 0, "3": 0}
        labels = []
        per_class_count = defaultdict(int)
        conf_values = []

        if detections.class_id is not None:
            for cls_id, conf in zip(detections.class_id, detections.confidence):
                prefix = CLASS_PREFIX.get(int(cls_id), "9")
                local_seq[prefix] = local_seq.get(prefix, 0) + 2
                custom_id = f"{prefix}{local_seq[prefix]:02d}"
                cname = CLASS_NAME.get(int(cls_id), "vehicle")
                labels.append(f"#{custom_id} {cname} {conf:.2f}")
                per_class_count[cname] += 1
                conf_values.append(float(conf))

        annotated = image.copy()
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        stats = {
            "total_objects": len(detections),
            "per_class_count": dict(per_class_count),
            "avg_confidence": round(sum(conf_values) / len(conf_values), 3) if conf_values else 0.0,
        }
        return annotated, stats

    # ------------------------------------------------------------- stats --
    def get_stats(self):
        avg_conf = (
            self.confidence_sum / self.confidence_count
            if self.confidence_count > 0 else 0.0
        )
        line_in = self.line_zone.in_count if self.line_zone is not None else 0
        line_out = self.line_zone.out_count if self.line_zone is not None else 0
        return {
            "running": self.running,
            "active_objects": self.active_objects,
            "total_unique_objects": len(self.seen_ids),
            "entered": self.entered_count,
            "exited": self.exited_count,
            "line_in": int(line_in),
            "line_out": int(line_out),
            "roi_count": self.roi_count,
            "avg_confidence": round(avg_conf, 3),
            "fps": round(self.fps, 1),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "recording": self.recording,
            "confidence_threshold": self.confidence_threshold,
            "frame_count": self.frame_count,
            "heatmap_enabled": self.heatmap_enabled,
        }
