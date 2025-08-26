import os
import cv2
import numpy as np
import random

# Đường dẫn chuẩn theo cấu trúc mới
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
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

# Hàm thêm nhiễu muối tiêu
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    
    # Thêm nhiễu "muối" (trắng)
    num_salt = int(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Thêm nhiễu "tiêu" (đen)
    num_pepper = int(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

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
    
    # Đọc ảnh
    image = cv2.imread(old_image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {old_image_path}, bỏ qua.")
        continue
    
    # Đọc nhãn
    if not os.path.exists(old_label_path):
        print(f"Nhãn không tồn tại: {old_label_path}, bỏ qua.")
        continue
    with open(old_label_path, 'r') as f:
        label = f.read().strip()
    
    # Thêm nhiễu muối tiêu (tỷ lệ ngẫu nhiên để đa dạng)
    salt_prob = random.uniform(0.005, 0.02)  # Tỷ lệ muối từ 0.5% đến 2%
    pepper_prob = random.uniform(0.005, 0.02)  # Tỷ lệ tiêu từ 0.5% đến 2%
    noisy_image = add_salt_and_pepper_noise(image, salt_prob=salt_prob, pepper_prob=pepper_prob)
    
    # Tạo tên mới
    new_image_name = f"image_noise_{i}{image_ext}"
    new_label_name = f"image_noise_{i}{label_ext}"
    new_image_path = os.path.join(output_image_dir, new_image_name)
    new_label_path = os.path.join(output_label_dir, new_label_name)
    
    # Lưu ảnh và nhãn (nhãn giữ nguyên vì nhiễu không ảnh hưởng tọa độ)
    cv2.imwrite(new_image_path, noisy_image)
    with open(new_label_path, 'w') as f:
        f.write(label)
    
    print(f"Đã xử lý: {new_image_name} với salt_prob={salt_prob:.4f}, pepper_prob={pepper_prob:.4f}")

print("Hoàn tất!")