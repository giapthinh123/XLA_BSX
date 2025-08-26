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
import matplotlib.pyplot as plt

# Định nghĩa các hằng số toàn cục
GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)

# Khởi tạo EasyOCR và YOLO
reader = easyocr.Reader(['en'], gpu=True)
# Thiết lập đường dẫn thư mục gốc dự án, mô hình và UI
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo11_trained_v2.pt")
UI_PATH = os.path.join(ROOT_DIR, "ui", "giaodien_image.ui")
model = YOLO(MODEL_PATH)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Tải giao diện từ file XML
        loadUi(UI_PATH, self)

        # Kết nối các nút bấm với hàm xử lý
        self.pushButton_them.clicked.connect(self.load_image)
        self.pushButton_XL.clicked.connect(self.process_image)
        self.pushButton_huy.clicked.connect(self.clear_view)

        # Biến để lưu trữ ảnh gốc và ảnh đã xử lý
        self.original_image = None
        self.processed_image = None

    def load_image(self):
        # Mở hộp thoại để chọn file ảnh
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            # Đọc ảnh bằng OpenCV
            self.original_image = cv2.imread(file_name)
            if self.original_image is None:
                QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh từ {file_name}.")
                return

            # Chuyển đổi ảnh sang định dạng RGB để hiển thị
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Hiển thị ảnh trong QGraphicsView
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def preprocess(self, imgOriginal):
        # Trích xuất kênh Value từ HSV
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        _, _, imgValue = cv2.split(imgHSV)

        # Tăng độ tương phản
        imgMaxContrastGrayscale = self.maximizeContrast(imgValue)

        # Áp dụng Gaussian Blur
        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
        return imgBlurred

    def maximizeContrast(self, imgGrayscale):
        height, width = imgGrayscale.shape
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        return cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    def process_image(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Vui lòng tải ảnh trước khi xử lý!")
            return

        # Thực hiện phát hiện bằng YOLO
        confidence_threshold = 0.8
        results = model(self.original_image)
        annotated_frame = self.original_image.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    license_plate_crop = self.original_image[y1:y2, x1:x2]

                    # Tiền xử lý và OCR
                    processed_plate = self.preprocess(license_plate_crop)
                    text_result = reader.readtext(processed_plate, detail=0)
                    detected_text = " ".join(text_result).strip()

                    # Vẽ hộp bao và nhãn
                    custom_label = f"vehicle_plate: {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    # Điều chỉnh vị trí nhãn
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

        # Lưu ảnh đã xử lý
        self.processed_image = annotated_frame

        # Hiển thị ảnh đã xử lý
        image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        QtWidgets.QMessageBox.information(self, "Thông báo", "Xử lý hoàn tất!")

    def clear_view(self):
        # Xóa nội dung trong QGraphicsView
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