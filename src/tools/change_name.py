import os

# Đường dẫn đến thư mục chứa ảnh và nhãn
image_dir = "images"  # Thay bằng đường dẫn thư mục chứa ảnh
label_dir = "labels"  # Thay bằng đường dẫn thư mục chứa nhãn (nếu riêng)

# Định dạng file ảnh (có thể thay đổi: .jpg, .png, ...)
image_ext = ".jpg"
label_ext = ".txt"

# Lấy danh sách file ảnh
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])

# Đếm bắt đầu từ 1
counter = 1

# Duyệt qua từng file ảnh
for old_image_name in image_files:
    # Tạo tên mới cho ảnh
    new_image_name = f"image_{counter}{image_ext}"
    
    # Tạo tên cũ và mới cho file nhãn (loại bỏ phần mở rộng của ảnh để tìm nhãn tương ứng)
    old_label_name = old_image_name.replace(image_ext, label_ext)
    new_label_name = f"image_{counter}{label_ext}"
    
    # Đường dẫn đầy đủ
    old_image_path = os.path.join(image_dir, old_image_name)
    new_image_path = os.path.join(image_dir, new_image_name)
    old_label_path = os.path.join(label_dir, old_label_name)
    new_label_path = os.path.join(label_dir, new_label_name)
    
    # Đổi tên file ảnh
    if os.path.exists(old_image_path):
        os.rename(old_image_path, new_image_path)
        print(f"Renamed: {old_image_path} -> {new_image_path}")
    
    # Đổi tên file nhãn (nếu tồn tại)
    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)
        print(f"Renamed: {old_label_path} -> {new_label_path}")
    
    # Tăng biến đếm
    counter += 1

print("Đổi tên hoàn tất!")