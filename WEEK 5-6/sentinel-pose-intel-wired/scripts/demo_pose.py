"""
Phase 2 Verification Script — Pose Estimation Demo

Runs YOLO-Pose on a video source (webcam 0 by default), draws skeleton
overlays, and displays the result in an OpenCV window.

Usage:
    python -m scripts.demo_pose                     # webcam
    python -m scripts.demo_pose path/to/video.mp4   # video file

Press 'q' to quit, 's' to toggle skeleton overlay, 'b' to toggle bboxes.
"""

from __future__ import annotations

import sys
import time
import logging

import cv2

from app import setup_logging
from app.config import get_settings
from app.vision.video_source import VideoSource
from app.vision.pose_estimator import YoloPoseEstimator
from app.vision.skeleton_renderer import draw_skeletons

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    source_str = sys.argv[1] if len(sys.argv) > 1 else settings.video_source

    logger.info("Starting pose demo with source: %s", source_str)

    video = VideoSource(source_str)
    if not video.open():
        logger.error("Cannot open video source: %s", source_str)
        sys.exit(1)

    estimator = YoloPoseEstimator()

    draw_skel = True
    draw_bbox = True
    frame_num = 0

    try:
        while True:
            frame = video.read_frame()
            if frame is None:
                break

            # Run pose estimation
            persons = estimator.estimate(frame)

            # Draw overlays
            if draw_skel or draw_bbox:
                draw_skeletons(
                    frame,
                    persons,
                    draw_bbox=draw_bbox,
                    draw_keypoints=draw_skel,
                    draw_bones=draw_skel,
                    draw_labels=True,
                )

            # HUD: FPS + person count + inference time
            fps_text = f"FPS: {video.fps:.1f}"
            inf_text = f"Inference: {estimator.inference_time_ms:.0f}ms"
            ppl_text = f"People: {len(persons)}"
            vis_kps = [p.visible_count for p in persons]
            kp_text = f"Keypoints: {vis_kps}" if persons else "Keypoints: -"

            y = 24
            for text in [fps_text, inf_text, ppl_text, kp_text]:
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 1, cv2.LINE_AA)
                y += 22

            cv2.imshow("Sentinel Pose Intel — Phase 2 Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                draw_skel = not draw_skel
                logger.info("Skeleton overlay: %s", "ON" if draw_skel else "OFF")
            elif key == ord("b"):
                draw_bbox = not draw_bbox
                logger.info("Bounding boxes: %s", "ON" if draw_bbox else "OFF")

            frame_num += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        video.release()
        cv2.destroyAllWindows()

    logger.info("Demo finished — %d frames processed", frame_num)


if __name__ == "__main__":
    main()
