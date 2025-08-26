import sys
import os
import cv2
import easyocr
import numpy as np
import os
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer

# Định nghĩa các hằng số toàn cục
GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

# Khởi tạo EasyOCR và YOLO
reader = easyocr.Reader(['en'], gpu=True)  # Thay 'en' bằng 'vi' nếu cần tiếng Việt
# Đường dẫn gốc dự án, model và UI
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo11_trained_v2.pt")
UI_PATH = os.path.join(ROOT_DIR, "ui", "giaodien_camera.ui")
model = YOLO(MODEL_PATH)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Tải giao diện từ file XML
        loadUi(UI_PATH, self)

        # Kết nối các nút bấm với hàm xử lý
        # self.pushButton_them.clicked.connect(self.load_image)
        # self.pushButton_XL.clicked.connect(self.process_image)
        # self.pushButton_huy.clicked.connect(self.clear_view)
        self.pushButton_camera.clicked.connect(self.toggle_camera)

        # Biến để lưu trữ ảnh gốc, ảnh đã xử lý và trạng thái camera
        self.original_image = None
        self.processed_image = None
        self.camera_active = False
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def preprocess(self, imgOriginal):
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        _, _, imgValue = cv2.split(imgHSV)
        imgMaxContrastGrayscale = self.maximizeContrast(imgValue)
        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
        return imgBlurred

    def maximizeContrast(self, imgGrayscale):
        height, width = imgGrayscale.shape
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        return cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    def process_frame(self, frame):
        confidence_threshold = 0.8
        results = model(frame)
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    license_plate_crop = frame[y1:y2, x1:x2]
                    processed_plate = self.preprocess(license_plate_crop)
                    text_result = reader.readtext(processed_plate, detail=0)
                    detected_text = " ".join(text_result).strip()

                    custom_label = f"vehicle_plate: {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    height, width, _ = annotated_frame.shape
                    if y1 < 50:
                        label_pos = (x1, y2 + 20)
                        text_pos = (x1, y2 + 40)
                    else:
                        label_pos = (x1, y1 - 10)
                        text_pos = (x1, y1 - 30)

                    cv2.putText(annotated_frame, custom_label, label_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(annotated_frame, detected_text, text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        self.processed_image = annotated_frame
        return annotated_frame

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)  # Mở camera mặc định (0)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.critical(self, "Lỗi", "Không thể mở camera!")
                return
            self.camera_active = True
            self.timer.start(30)  # Cập nhật khung hình mỗi 30ms (~33 FPS)
            self.pushButton_camera.setText("Tắt Camera")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.camera_active = False
            self.pushButton_camera.setText("Camera")
            self.clear_view()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            processed_frame = self.process_frame(frame)
            self.display_image(processed_frame)

    def clear_view(self):
        if self.camera_active:
            self.toggle_camera()
        self.graphicsView.setScene(QtWidgets.QGraphicsScene(self))
        self.original_image = None
        self.processed_image = None

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()