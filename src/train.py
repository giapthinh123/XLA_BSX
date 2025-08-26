from ultralytics import YOLO
import os

# Hàm huấn luyện (nếu bạn muốn tách riêng)
def train_yolo():
    # Thiết lập đường dẫn
    base_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
    model_init_path = os.path.join(root_dir, "models", "yolo11_trained_v1.pt")
    data_yaml_path = os.path.join(root_dir, "configs", "data.yaml")

    # Tải mô hình YOLO khởi tạo
    model = YOLO(model_init_path)

    # Huấn luyện mô hình
    results = model.train(
        data=data_yaml_path,  # Cập nhật đường dẫn file cấu hình
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
    output_model_path = os.path.join(root_dir, "models", "yolo11_trained_v2.pt")
    model.save(output_model_path)

# Bảo vệ code trong khối main để tránh lỗi multiprocessing trên Windows
if __name__ == '__main__':
    train_yolo()