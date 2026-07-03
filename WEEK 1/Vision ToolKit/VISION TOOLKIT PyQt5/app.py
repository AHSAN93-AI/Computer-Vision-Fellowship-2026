"""
Vision Toolkit — Desktop Edition
A Computer Vision desktop application built with PyQt5 + OpenCV.

Run with:
    python app.py
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QComboBox, QLineEdit, QColorDialog, QFileDialog, QVBoxLayout,
    QHBoxLayout, QFormLayout, QStackedWidget, QGroupBox, QMessageBox,
    QScrollArea, QGridLayout, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert a BGR or grayscale OpenCV image to a QPixmap."""
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:  # grayscale
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # .copy() detaches from numpy buffer


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


class ImageLabel(QLabel):
    """A QLabel that scales its pixmap to fit while keeping aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self._pixmap = QPixmap()

    def setImage(self, img: np.ndarray):
        self._pixmap = cv_to_qpixmap(img)
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if not self._pixmap.isNull():
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


# ----------------------------------------------------------------------
# Camera Capture Dialog
# ----------------------------------------------------------------------
class CameraCaptureDialog(QDialog):
    """Shows a live webcam feed with a button to capture a single frame."""

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.setWindowTitle("Capture from Camera")
        self.resize(720, 560)

        self.captured_frame = None
        self.cap = cv2.VideoCapture(camera_index)

        layout = QVBoxLayout(self)

        self.preview_label = ImageLabel()
        layout.addWidget(self.preview_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ccc;")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self._capture)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.capture_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if not self.cap.isOpened():
            self.status_label.setText(
                "⚠️ Could not access the camera. Check permissions or that it's not in use by another app."
            )
            self.capture_btn.setEnabled(False)
        else:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_preview)
            self.timer.start(30)  # ~33 fps

    def _update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            self._last_frame = frame
            self.preview_label.setImage(frame)

    def _capture(self):
        if hasattr(self, "_last_frame") and self._last_frame is not None:
            self.captured_frame = self._last_frame.copy()
            self.accept()

    def closeEvent(self, event):
        self._release_camera()
        super().closeEvent(event)

    def reject(self):
        self._release_camera()
        super().reject()

    def accept(self):
        self._release_camera()
        super().accept()

    def _release_camera(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ----------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------
class VisionToolkit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Toolkit — Desktop Edition")
        self.resize(1200, 750)

        self.original_img = None      # untouched, loaded once
        self.working_img = None       # base image drawing tools commit onto
        self.current_result = None    # currently displayed processed image
        self.file_path = None
        self.draw_color = "#00FF00"

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---------------- Left: image display ----------------
        image_panel = QVBoxLayout()
        images_row = QHBoxLayout()

        orig_box = QVBoxLayout()
        orig_box.addWidget(QLabel("<b>Original</b>"))
        self.original_label = ImageLabel()
        orig_box.addWidget(self.original_label)

        result_box = QVBoxLayout()
        result_box.addWidget(QLabel("<b>Processed</b>"))
        self.result_label = ImageLabel()
        result_box.addWidget(self.result_label)

        images_row.addLayout(orig_box)
        images_row.addLayout(result_box)
        image_panel.addLayout(images_row)

        # Info bar
        self.info_label = QLabel("No image loaded.")
        self.info_label.setStyleSheet("color: #ccc; padding: 6px;")
        image_panel.addWidget(self.info_label)

        main_layout.addLayout(image_panel, stretch=3)

        # ---------------- Right: control panel ----------------
        control_panel = QVBoxLayout()
        control_panel.setAlignment(Qt.AlignTop)

        # Upload / Save buttons
        io_group = QGroupBox("File")
        io_layout = QVBoxLayout()
        open_btn = QPushButton("📂 Upload Image")
        open_btn.clicked.connect(self.open_image)
        capture_btn = QPushButton("📷 Capture from Camera")
        capture_btn.clicked.connect(self.capture_from_camera)
        save_btn = QPushButton("💾 Save Processed Image")
        save_btn.clicked.connect(self.save_image)
        reset_btn = QPushButton("↩️ Reset to Original")
        reset_btn.clicked.connect(self.reset_image)
        io_layout.addWidget(open_btn)
        io_layout.addWidget(capture_btn)
        io_layout.addWidget(save_btn)
        io_layout.addWidget(reset_btn)
        io_group.setLayout(io_layout)
        control_panel.addWidget(io_group)

        # Tool selector
        tool_group = QGroupBox("Tool")
        tool_layout = QVBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_combo.addItems([
            "Image Info",
            "Grayscale",
            "Edge Detection (Canny)",
            "Blur Filters",
            "Thresholding",
            "Drawing Tools",
            "Brightness / Contrast",
            "Rotate / Resize",
            "Histogram",
        ])
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tool_layout.addWidget(self.tool_combo)
        tool_group.setLayout(tool_layout)
        control_panel.addWidget(tool_group)

        # Stacked pages for each tool's parameters
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_info_page())          # 0
        self.stack.addWidget(self._build_empty_page())         # 1 grayscale (no params)
        self.stack.addWidget(self._build_canny_page())         # 2
        self.stack.addWidget(self._build_blur_page())          # 3
        self.stack.addWidget(self._build_threshold_page())     # 4
        self.stack.addWidget(self._build_drawing_page())       # 5
        self.stack.addWidget(self._build_brightness_page())    # 6
        self.stack.addWidget(self._build_rotate_resize_page()) # 7
        self.stack.addWidget(self._build_empty_page())         # 8 histogram (no params)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.stack)
        params_group.setLayout(params_layout)
        control_panel.addWidget(params_group)

        control_panel.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_widget)
        main_layout.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------
    # Tool parameter pages
    # ------------------------------------------------------------------
    def _build_empty_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("No parameters for this tool."))
        return w

    def _build_info_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Image metadata is shown below the canvas."))
        return w

    def _build_canny_page(self):
        w = QWidget()
        layout = QFormLayout(w)
        self.canny_low = QSlider(Qt.Horizontal)
        self.canny_low.setRange(0, 500)
        self.canny_low.setValue(100)
        self.canny_low.valueChanged.connect(self.process)
        self.canny_high = QSlider(Qt.Horizontal)
        self.canny_high.setRange(0, 500)
        self.canny_high.setValue(200)
        self.canny_high.valueChanged.connect(self.process)
        layout.addRow("Lower Threshold", self.canny_low)
        layout.addRow("Upper Threshold", self.canny_high)
        return w

    def _build_blur_page(self):
        w = QWidget()
        layout = QFormLayout(w)
        self.blur_type = QComboBox()
        self.blur_type.addItems(["Gaussian Blur", "Median Blur"])
        self.blur_type.currentIndexChanged.connect(self.process)
        self.blur_ksize = QSlider(Qt.Horizontal)
        self.blur_ksize.setRange(1, 25)
        self.blur_ksize.setSingleStep(2)
        self.blur_ksize.setValue(5)
        self.blur_ksize.valueChanged.connect(self.process)
        layout.addRow("Blur Type", self.blur_type)
        layout.addRow("Kernel Size", self.blur_ksize)
        return w

    def _build_threshold_page(self):
        w = QWidget()
        layout = QFormLayout(w)
        self.thresh_type = QComboBox()
        self.thresh_type.addItems(["Binary Threshold", "Adaptive Threshold"])
        self.thresh_type.currentIndexChanged.connect(self.process)
        self.thresh_val = QSlider(Qt.Horizontal)
        self.thresh_val.setRange(0, 255)
        self.thresh_val.setValue(127)
        self.thresh_val.valueChanged.connect(self.process)
        self.block_size = QSlider(Qt.Horizontal)
        self.block_size.setRange(3, 51)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(11)
        self.block_size.valueChanged.connect(self.process)
        layout.addRow("Type", self.thresh_type)
        layout.addRow("Binary Value", self.thresh_val)
        layout.addRow("Adaptive Block Size", self.block_size)
        return w

    def _build_drawing_page(self):
        w = QWidget()
        layout = QFormLayout(w)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Rectangle", "Circle", "Line", "Text"])
        self.shape_combo.currentIndexChanged.connect(self.process)

        self.color_btn = QPushButton("Pick Color")
        self.color_btn.clicked.connect(self._pick_color)

        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 10)
        self.thickness_slider.setValue(2)
        self.thickness_slider.valueChanged.connect(self.process)

        self.x1_slider = QSlider(Qt.Horizontal)
        self.y1_slider = QSlider(Qt.Horizontal)
        self.x2_slider = QSlider(Qt.Horizontal)
        self.y2_slider = QSlider(Qt.Horizontal)
        for s in (self.x1_slider, self.y1_slider, self.x2_slider, self.y2_slider):
            s.setRange(0, 1000)
            s.valueChanged.connect(self.process)

        self.text_input = QLineEdit("Vision Toolkit")
        self.text_input.textChanged.connect(self.process)
        self.font_scale_slider = QSlider(Qt.Horizontal)
        self.font_scale_slider.setRange(5, 50)
        self.font_scale_slider.setValue(10)
        self.font_scale_slider.valueChanged.connect(self.process)

        apply_btn = QPushButton("✅ Apply Drawing to Image")
        apply_btn.clicked.connect(self._apply_drawing)

        layout.addRow("Shape", self.shape_combo)
        layout.addRow("Color", self.color_btn)
        layout.addRow("Thickness", self.thickness_slider)
        layout.addRow("X1 / Center X / Start X", self.x1_slider)
        layout.addRow("Y1 / Center Y / Start Y", self.y1_slider)
        layout.addRow("X2 / Radius / End X", self.x2_slider)
        layout.addRow("Y2 / End Y", self.y2_slider)
        layout.addRow("Text", self.text_input)
        layout.addRow("Font Scale", self.font_scale_slider)
        layout.addRow(apply_btn)
        return w

    def _build_brightness_page(self):
        w = QWidget()
        layout = QFormLayout(w)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.process)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.process)
        layout.addRow("Brightness", self.brightness_slider)
        layout.addRow("Contrast", self.contrast_slider)
        return w

    def _build_rotate_resize_page(self):
        w = QWidget()
        layout = QFormLayout(w)
        self.rr_action = QComboBox()
        self.rr_action.addItems(["Rotate", "Resize"])
        self.rr_action.currentIndexChanged.connect(self.process)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self.process)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.process)
        layout.addRow("Action", self.rr_action)
        layout.addRow("Angle", self.angle_slider)
        layout.addRow("Scale (%)", self.scale_slider)
        return w

    def _pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.draw_color = color.name()
        self.process()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Could not read this image file.")
            return
        self.file_path = path
        self.original_img = img.copy()
        self.working_img = img.copy()
        self.original_label.setImage(self.original_img)
        self.process()

    def capture_from_camera(self):
        dialog = CameraCaptureDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.captured_frame is not None:
            img = dialog.captured_frame
            self.file_path = None  # no file on disk yet
            self.original_img = img.copy()
            self.working_img = img.copy()
            self.original_label.setImage(self.original_img)
            self.process()

    def save_image(self):
        if self.current_result is None:
            QMessageBox.information(self, "No Image", "Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "vision_toolkit_output.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self.current_result)
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")

    def reset_image(self):
        if self.original_img is None:
            return
        self.working_img = self.original_img.copy()
        self.process()

    # ------------------------------------------------------------------
    # Core processing dispatcher
    # ------------------------------------------------------------------
    def _on_tool_changed(self, index):
        self.stack.setCurrentIndex(index)
        self.process()

    def process(self):
        if self.working_img is None:
            return

        img = self.working_img
        h, w = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        tool = self.tool_combo.currentText()
        result = img.copy()

        if tool == "Image Info":
            file_size_kb = 0
            if self.file_path and os.path.exists(self.file_path):
                file_size_kb = round(os.path.getsize(self.file_path) / 1024, 2)
            self.info_label.setText(
                f"Width: {w}px | Height: {h}px | Resolution: {w}x{h} | "
                f"File Size: {file_size_kb} KB | Channels: {channels}"
            )
            result = img

        elif tool == "Grayscale":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif tool == "Edge Detection (Canny)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = cv2.Canny(gray, self.canny_low.value(), self.canny_high.value())

        elif tool == "Blur Filters":
            k = self.blur_ksize.value()
            if k % 2 == 0:
                k += 1  # kernel size must be odd
            if self.blur_type.currentText() == "Gaussian Blur":
                result = cv2.GaussianBlur(img, (k, k), 0)
            else:
                result = cv2.medianBlur(img, k)

        elif tool == "Thresholding":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.thresh_type.currentText() == "Binary Threshold":
                _, result = cv2.threshold(gray, self.thresh_val.value(), 255, cv2.THRESH_BINARY)
            else:
                block = self.block_size.value()
                if block % 2 == 0:
                    block += 1
                if block < 3:
                    block = 3
                result = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block, 2
                )

        elif tool == "Drawing Tools":
            result = self._render_drawing_preview(img)

        elif tool == "Brightness / Contrast":
            alpha = (self.contrast_slider.value() + 100) / 100.0
            beta = self.brightness_slider.value()
            result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        elif tool == "Rotate / Resize":
            if self.rr_action.currentText() == "Rotate":
                angle = self.angle_slider.value()
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                result = cv2.warpAffine(img, matrix, (w, h))
            else:
                scale = self.scale_slider.value()
                new_w = max(1, int(w * scale / 100))
                new_h = max(1, int(h * scale / 100))
                result = cv2.resize(img, (new_w, new_h))

        elif tool == "Histogram":
            result = self._render_histogram(img)

        self.current_result = result
        self.result_label.setImage(result)

        if tool != "Image Info":
            self.info_label.setText(
                f"Width: {w}px | Height: {h}px | Channels: {channels}"
            )

    def _render_drawing_preview(self, img):
        result = img.copy()
        h, w = img.shape[:2]
        shape = self.shape_combo.currentText()
        color_bgr = hex_to_bgr(self.draw_color)
        thickness = self.thickness_slider.value()

        # Keep sliders in sensible bounds relative to image size
        for s in (self.x1_slider, self.x2_slider):
            s.setMaximum(w)
        for s in (self.y1_slider, self.y2_slider):
            s.setMaximum(h)

        x1, y1 = self.x1_slider.value(), self.y1_slider.value()
        x2, y2 = self.x2_slider.value(), self.y2_slider.value()

        if shape == "Rectangle":
            cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, thickness)
        elif shape == "Circle":
            radius = max(1, x2)  # x2 slider doubles as radius
            cv2.circle(result, (x1, y1), radius, color_bgr, thickness)
        elif shape == "Line":
            cv2.line(result, (x1, y1), (x2, y2), color_bgr, thickness)
        elif shape == "Text":
            font_scale = self.font_scale_slider.value() / 10.0
            cv2.putText(
                result, self.text_input.text(), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA
            )
        return result

    def _apply_drawing(self):
        if self.working_img is None:
            return
        self.working_img = self._render_drawing_preview(self.working_img)
        self.process()

    def _render_histogram(self, img):
        fig, ax = plt.subplots(figsize=(5, 4))
        if len(img.shape) == 2:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            ax.plot(hist, color="black")
        else:
            colors = ("b", "g", "r")
            for i, c in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax.plot(hist, color=c)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())
        hist_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return hist_img


# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VisionToolkit()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
