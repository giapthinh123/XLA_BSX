## Ứng dụng nhận diện biển số xe (YOLO + EasyOCR + PyQt5)

Ứng dụng hỗ trợ phát hiện vùng biển số bằng YOLO và nhận dạng ký tự bằng EasyOCR, chạy trên giao diện PyQt5. Dự án gồm các phần: huấn luyện, suy luận trên ảnh/camera, và các script tăng cường dữ liệu.

### Mục lục
- **Giới thiệu**
- **Yêu cầu cài đặt**
- **Chuẩn bị mô hình (.pt)**
- **Cấu trúc thư mục**
- **Chạy ứng dụng giao diện**
- **Huấn luyện mô hình**
- **Tổ chức dữ liệu và tăng cường dữ liệu**
- **Đổi tên dữ liệu**
- **Đóng góp & License**
- **Mẹo và lỗi thường gặp**

## Giới thiệu
- **Phát hiện**: YOLO (Ultralytics) xác định hộp biển số trên ảnh/khung hình.
- **Nhận dạng**: EasyOCR đọc ký tự từ vùng biển số đã trích xuất và tiền xử lý.
- **Giao diện**: PyQt5 với 2 màn hình mẫu: xử lý ảnh (`giaodien_image.ui`) và camera (`giaodien_camera.ui`).

## Yêu cầu cài đặt
- Python 3.8–3.11 (khuyến nghị 3.10+)
- Windows 10/11, GPU CUDA (tùy chọn)

Cài các thư viện chính:
```bash
pip install ultralytics opencv-python easyocr pyqt5 numpy matplotlib
```
Lưu ý: `ultralytics` phụ thuộc `torch`. Nếu gặp lỗi về Torch/CUDA, cài đặt thủ công theo hướng dẫn của PyTorch.

## Cấu trúc thư mục
```text
XLA_BSX/
  README.md
  requirements.txt
  .gitignore
  models/
    yolo11_trained_v2.pt
  ui/
    giaodien_image.ui
    giaodien_camera.ui
  data/
    raw/
      images/
      labels/
    processed/
      images/
      labels/
  src/
    test_image.py
    test_camera.py
    train.py
    augment/
      rotate.py
      Scale.py
      Translation.py
      noise.py
    tools/
      change_name.py
  configs/
    data.yaml.example
```
Bạn có thể giữ nguyên layout cũ, nhưng khuyến nghị sử dụng cấu trúc trên để dễ bảo trì.

## Chuẩn bị mô hình (.pt)
Đặt file trọng số YOLO (ví dụ: `yolo11_trained_v2.pt` hoặc `best.pt`) vào thư mục gốc dự án. Cập nhật đường dẫn trong các file suy luận nếu cần:

Trong `test_image.py`:
```python
model = YOLO("yolo11_trained_v2.pt")  # hoặc đường dẫn tới .pt của bạn
```

Trong `test_camera.py`:
```python
model = YOLO("yolo11_trained_v2.pt")
```

## Chạy ứng dụng giao diện
### 1) Suy luận trên ảnh tĩnh
```bash
python src/test_image.py
```
- Nút "Thêm" để chọn ảnh, "Xử lý" để phát hiện + OCR, "Hủy" để xóa hiển thị.
- Kết quả sẽ vẽ hộp biển số, confidence và chuỗi ký tự OCR.

### 2) Suy luận bằng camera
```bash
python src/test_camera.py
```
- Nút "Camera" để bật/tắt camera, luồng hiển thị kèm hộp và OCR theo thời gian thực.
- Nếu không mở được camera, đổi chỉ số `cv2.VideoCapture(0)` thành `1` hoặc `2`.

## Huấn luyện mô hình
Mặc định `train.py` tải mô hình và gọi `model.train(...)`. Hãy cập nhật đường dẫn `data.yaml` cho phù hợp cấu trúc dữ liệu của bạn:
```python
results = model.train(
    data="path/to/data.yaml",  # cập nhật lại
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,      # GPU 0; dùng -1 hoặc bỏ nếu không có GPU
    workers=2,
)
```
Chạy huấn luyện:
```bash
python src/train.py
```
Trọng số sau huấn luyện có thể lưu thành `yolo11_trained_v2.pt` ở thư mục gốc.

## Tổ chức dữ liệu và tăng cường dữ liệu
Thư mục mặc định trong dự án:
- `images/`, `labels/`: dữ liệu đã sẵn sàng cho YOLO (định dạng nhãn YOLO: `class x_center y_center width height`, chuẩn hóa 0–1).
- `image_origin/`, `labels_origin/`: dữ liệu gốc để chạy các script tăng cường.

Các script tăng cường (chạy tại thư mục gốc dự án):
- `src/augment/rotate.py`: xoay ảnh và cập nhật nhãn. Xuất ảnh vào `data/processed/images` và nhãn vào `data/processed/labels`.
  ```bash
  python src/augment/rotate.py
  ```
- `src/augment/Scale.py`: thay đổi tỷ lệ (scale) ảnh, điều chỉnh nhãn. Xuất ảnh vào `data/processed/images` và nhãn vào `data/processed/labels`.
  ```bash
  python src/augment/Scale.py
  ```
- `src/augment/Translation.py`: tịnh tiến (dịch chuyển) ảnh, điều chỉnh nhãn. Xuất ảnh vào `data/processed/images` và nhãn vào `data/processed/labels`.
  ```bash
  python src/augment/Translation.py
  ```
- `src/augment/noise.py`: thêm nhiễu muối tiêu cho ảnh, ghi ra `data/processed`.
  ```bash
  python src/augment/noise.py
  ```

Ghi chú:
- Các script chọn ngẫu nhiên một tập ảnh từ `image_origin/`. Chỉnh tham số số lượng nếu cần.
- Đảm bảo thư mục đích `images/` và `labels/` tồn tại trước khi chạy.

## Đổi tên dữ liệu
`src/tools/change_name.py` đổi tên đồng bộ ảnh và nhãn trong `data/processed/images` và `data/processed/labels` về dạng tuần tự `image_1.jpg`, `image_1.txt`, ...
```bash
python src/tools/change_name.py
```
## Đóng góp & License
- Đóng góp qua Pull Request, tuân thủ Conventional Commits và mô tả rõ thay đổi.
- License: MIT (hoặc cập nhật theo tệp LICENSE nếu có).

Lưu ý: Sao lưu trước khi đổi tên để tránh ghi đè không mong muốn.

## Mẹo và lỗi thường gặp
- **Torch/CUDA**: nếu lỗi cài `torch`, cài theo hướng dẫn chính thức của PyTorch cho đúng phiên bản CUDA/CPU.
- **EasyOCR không có GPU**: đổi `easyocr.Reader(['en'], gpu=True)` thành `gpu=False`.
- **Không mở được camera**: kiểm tra quyền camera, đổi chỉ số thiết bị `0 -> 1/2`.
- **Đường dẫn tuyệt đối trong train.py**: cập nhật `data=...` cho phù hợp máy của bạn.

---
Nếu bạn cần thêm hướng dẫn tích hợp dữ liệu mới hoặc đóng gói ứng dụng, vui lòng mở issue hoặc liên hệ trực tiếp.
