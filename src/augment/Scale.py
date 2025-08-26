import os
import cv2
import numpy as np
import random

# Đường dẫn chuẩn theo cấu trúc mới
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
raw_image_dir = os.path.join(ROOT_DIR, "data", "raw", "images")
raw_label_dir = os.path.join(ROOT_DIR, "data", "raw", "labels")
output_image_dir = os.path.join(ROOT_DIR, "data", "processed", "images")
output_label_dir = os.path.join(ROOT_DIR, "data", "processed", "labels")

# Định dạng file
image_ext = ".jpg"  # Thay đổi nếu cần (.png, .jpeg, ...)
label_ext = ".txt"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Hàm scale ảnh và điều chỉnh nhãn
def scale_image_and_labels(image, label_lines, scale_factor):
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    
    # Scale ảnh
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Điều chỉnh từng dòng nhãn
    new_label_lines = []
    for line in label_lines:
        try:
            class_id, x_center, y_center, width, height = map(float, line.split())
        except ValueError:
            print(f"Lỗi định dạng nhãn: {line}. Bỏ qua dòng này.")
            continue
        
        # Vì tọa độ YOLO đã chuẩn hóa (0-1), chỉ cần giữ nguyên x_center, y_center
        # Nhưng width và height cần được điều chỉnh theo scale factor
        new_width = width / scale_factor
        new_height = height / scale_factor
        
        # Đảm bảo tọa độ vẫn nằm trong phạm vi [0, 1]
        if 0 <= x_center <= 1 and 0 <= y_center <= 1:
            new_label_line = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {new_width:.6f} {new_height:.6f}"
            new_label_lines.append(new_label_line)
    
    return scaled_image, new_label_lines

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(raw_image_dir) if f.endswith(image_ext)]

# Chọn ngẫu nhiên 1000 ảnh (hoặc ít hơn nếu thư mục có ít file)
num_samples = min(1000, len(image_files))
selected_images = random.sample(image_files, num_samples)

# Xử lý từng ảnh
for i, old_image_name in enumerate(selected_images, 1):
    old_image_path = os.path.join(raw_image_dir, old_image_name)
    old_label_name = old_image_name.replace(image_ext, label_ext)
    old_label_path = os.path.join(raw_label_dir, old_label_name)
    
    # Đọc ảnh
    image = cv2.imread(old_image_path)
    if image is None:
        print(f"Không đọc được ảnh: {old_image_path}, bỏ qua.")
        continue
    
    # Đọc nhãn (có thể nhiều dòng)
    if not os.path.exists(old_label_path):
        print(f"Nhãn không tồn tại: {old_label_path}, bỏ qua.")
        continue
    
    with open(old_label_path, 'r') as f:
        label_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not label_lines:
        print(f"File nhãn rỗng: {old_label_path}, bỏ qua.")
        continue
    
    # Hệ số scale ngẫu nhiên (ví dụ: từ 0.5 đến 1.5)
    scale_factor = random.uniform(0.5, 1.5)  # Có thể thay bằng giá trị cố định, ví dụ: 1.2
    
    # Scale ảnh và nhãn
    scaled_image, new_label_lines = scale_image_and_labels(image, label_lines, scale_factor)
    
    # Tạo tên mới
    new_image_name = f"image_Scale_{i}{image_ext}"
    new_label_name = f"image_Scale_{i}{label_ext}"
    new_image_path = os.path.join(output_image_dir, new_image_name)
    new_label_path = os.path.join(output_label_dir, new_label_name)
    
    # Lưu ảnh và nhãn
    cv2.imwrite(new_image_path, scaled_image)
    with open(new_label_path, 'w') as f:
        f.write("\n".join(new_label_lines))
    
    print(f"Đã xử lý: {new_image_name} với scale factor {scale_factor:.2f}")

print("Hoàn tất!")