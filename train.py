from ultralytics import YOLO

# Hàm huấn luyện (nếu bạn muốn tách riêng)
def train_yolo():
    # Tải mô hình YOLOv11
    model = YOLO("yolo11_trained_v1.pt")

    # Huấn luyện mô hình
    results = model.train(
        data="C:/Users/thinh/Desktop/project_XLA_V11/data.yaml",  # Đường dẫn file cấu hình
        epochs=100,            # Số epoch
        imgsz=640,             # Kích thước ảnh
        batch=8,              # Kích thước batch
        device=0,              # GPU 0
        workers=2,             # Số luồng xử lý dữ liệu
        project="runs/train",  # Thư mục lưu kết quả
        name="exp",            # Tên thử nghiệm
        patience=50            # Dừng sớm sau 50 epoch
    )

    # Lưu mô hình đã huấn luyện
    model.save("yolo11_trained_v2.pt")

# Bảo vệ code trong khối main để tránh lỗi multiprocessing trên Windows
if __name__ == '__main__':
    train_yolo()