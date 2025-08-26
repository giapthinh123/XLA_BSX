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

# Hàm xoay ảnh và điều chỉnh nhãn
def rotate_image_and_label(image, label, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Xoay ảnh
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    # Điều chỉnh nhãn
    class_id, x_center, y_center, width, height = map(float, label.split())
    
    # Chuyển sang tọa độ pixel
    x_pixel = x_center * w
    y_pixel = y_center * h
    
    # Xoay tọa độ trung tâm
    x_shifted = x_pixel - w / 2
    y_shifted = y_pixel - h / 2
    theta = np.radians(angle)
    x_new = x_shifted * np.cos(theta) + y_shifted * np.sin(theta)
    y_new = -x_shifted * np.sin(theta) + y_shifted * np.cos(theta)
    x_pixel_new = x_new + w / 2
    y_pixel_new = y_new + h / 2
    
    # Chuẩn hóa lại
    x_center_new = x_pixel_new / w
    y_center_new = y_pixel_new / h
    
    # Giữ nguyên width, height
    new_label = f"{int(class_id)} {x_center_new:.6f} {y_center_new:.6f} {width:.6f} {height:.6f}"
    
    return rotated_image, new_label

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(raw_image_dir) if f.endswith(image_ext)]

# Chọn ngẫu nhiên 1000 ảnh (hoặc ít hơn nếu thư mục có ít file)
num_samples = min(1000, len(image_files))
selected_images = random.sample(image_files, num_samples)

# Xử lý từng ảnh
for i, old_image_name in enumerate(selected_images, 1):
    # Đường dẫn file cũ
    old_image_path = os.path.join(raw_image_dir, old_image_name)
    old_label_name = old_image_name.replace(image_ext, label_ext)
    old_label_path = os.path.join(raw_label_dir, old_label_name)
    
    # Đọc ảnh và nhãn
    image = cv2.imread(old_image_path)
    if not os.path.exists(old_label_path):
        print(f"Nhãn không tồn tại: {old_label_path}, bỏ qua.")
        continue
    
    with open(old_label_path, 'r') as f:
        label = f.read().strip()
    
    # Góc xoay ngẫu nhiên (ví dụ: từ -45 đến 45 độ)
    angle = random.uniform(-45, 45)  # Có thể thay bằng góc cố định, ví dụ: 30
    
    # Xoay ảnh và nhãn
    rotated_image, new_label = rotate_image_and_label(image, label, angle)
    
    # Tạo tên mới
    new_image_name = f"image_rotate_{i}{image_ext}"
    new_label_name = f"image_rotate_{i}{label_ext}"
    new_image_path = os.path.join(output_image_dir, new_image_name)
    new_label_path = os.path.join(output_label_dir, new_label_name)
    
    # Lưu ảnh và nhãn
    cv2.imwrite(new_image_path, rotated_image)
    with open(new_label_path, 'w') as f:
        f.write(new_label)
    
    print(f"Đã xử lý: {new_image_name} với góc xoay {angle:.2f} độ")

print("Hoàn tất!")