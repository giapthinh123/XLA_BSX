import os
import cv2
import numpy as np
import random

# Đường dẫn đến thư mục chứa ảnh và nhãn
image_dir = "image_origin"  # Thay bằng đường dẫn thư mục ảnh
label_dir = "labels_origin"  # Thay bằng đường dẫn thư mục nhãn
output_dir = "images"  # Lưu ảnh và nhãn đã di chuyển vào thư mục 

image_ext = ".jpg"
label_ext = ".txt"

os.makedirs(output_dir, exist_ok=True)

# Hàm kiểm tra xem tất cả 4 điểm của bounding box có nằm ngoài ảnh không
def is_box_outside_image(x_center, y_center, width, height, w, h, tx, ty):
    x1 = (x_center - width / 2) * w
    y1 = (y_center - height / 2) * h
    x2 = (x_center + width / 2) * w
    y2 = (y_center + height / 2) * h
    
    x1_new = x1 + tx
    y1_new = y1 + ty
    x2_new = x2 + tx
    y2_new = y2 + ty
    
    return (x2_new < 0 or x1_new > w or y2_new < 0 or y1_new > h)

# Hàm dịch chuyển ảnh và điều chỉnh nhãn
def translate_image_and_labels(image, label_lines, tx, ty):
    h, w = image.shape[:2]
    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (w, h))
    
    new_label_lines = []
    for line in label_lines:
        try:
            class_id, x_center, y_center, width, height = map(float, line.split())
        except ValueError:
            print(f"Lỗi định dạng nhãn: {line}. Bỏ qua dòng này.")
            continue
        
        if is_box_outside_image(x_center, y_center, width, height, w, h, tx, ty):
            new_label_line = f"{int(class_id)} 0 0 0 0"
        else:
            x_center_new = x_center + (tx / w)
            y_center_new = y_center + (ty / h)
            
            # Giới hạn tọa độ trong [0, 1]
            x_center_new = max(0, min(1, x_center_new))
            y_center_new = max(0, min(1, y_center_new))
            
            new_label_line = f"{int(class_id)} {x_center_new:.6f} {y_center_new:.6f} {width:.6f} {height:.6f}"
        
        new_label_lines.append(new_label_line)
    
    return translated_image, new_label_lines

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]

num_samples = min(1000, len(image_files))
selected_images = random.sample(image_files, num_samples)

for i, old_image_name in enumerate(selected_images, 1):
    old_image_path = os.path.join(image_dir, old_image_name)
    old_label_name = old_image_name.replace(image_ext, label_ext)
    old_label_path = os.path.join(label_dir, old_label_name)
    
    image = cv2.imread(old_image_path)
    if image is None:
        print(f"Không đọc được ảnh: {old_image_path}, bỏ qua.")
        continue
    
    if not os.path.exists(old_label_path):
        print(f"Nhãn không tồn tại: {old_label_path}, bỏ qua.")
        continue
    
    with open(old_label_path, 'r') as f:
        label_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not label_lines:
        print(f"File nhãn rỗng: {old_label_path}, bỏ qua.")
        continue
    
    h, w = image.shape[:2]
    tx = random.uniform(-w * 0.5, w * 0.5)
    ty = random.uniform(-h * 0.5, h * 0.5)
    
    translated_image, new_label_lines = translate_image_and_labels(image, label_lines, tx, ty)
    
    new_image_name = f"image_Translation_{i}{image_ext}"
    new_label_name = f"image_Translation_{i}{label_ext}"
    new_image_path = os.path.join(output_dir, new_image_name)
    new_label_path = os.path.join("labels", new_label_name)
    
    cv2.imwrite(new_image_path, translated_image)
    with open(new_label_path, 'w') as f:
        f.write("\n".join(new_label_lines))
    
    print(f"Đã xử lý: {new_image_name} với dịch chuyển tx={tx:.2f}, ty={ty:.2f}")

print("Hoàn tất!")