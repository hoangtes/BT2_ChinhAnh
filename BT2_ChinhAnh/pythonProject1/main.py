import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh
def show_images(images, titles):
    n = len(images)
    plt.figure(figsize=(15, 5))  # Thiết lập kích thước khung hình

    for i in range(n):
        plt.subplot(1, n, i + 1)  # Tạo lưới 1 hàng và n cột
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Chuyển đổi từ BGR sang RGB
        plt.title(titles[i])  # Đặt tiêu đề cho ảnh
        plt.axis('off')  # Tắt trục

    plt.tight_layout()  # Tự động điều chỉnh khoảng cách giữa các ô
    plt.show()  # Hiển thị ảnh

# Đường dẫn tới tệp đầu vào và đầu ra
input_path = 'input/anh_hoathinh.jpg'  # Ví dụ: 'input.jpg'
output_path_sobel = 'output/output_sobel.jpg'  # Kết quả của Sobel
output_path_log = 'output/output_log.jpg'  # Kết quả của LoG

# Đọc ảnh từ file đầu vào với chế độ màu
image = cv2.imread(input_path)  # Đọc ảnh màu

# Kiểm tra nếu ảnh không tồn tại
if image is None:
    print(f"Không thể mở file {input_path}")
else:
    # Dò biên bằng toán tử Sobel
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang thang độ xám
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Dò biên bằng Laplacian of Gaussian (LoG)
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    log = cv2.Laplacian(gaussian_blur, cv2.CV_64F)

    # Chuẩn hóa kết quả LoG
    log_normalized = cv2.convertScaleAbs(log)  # Chuyển đổi sang uint8

    # Chuyển đổi kết quả Sobel sang dạng uint8 để lưu file
    sobel_output = np.uint8(sobel_combined)

    # Lưu kết quả ra file đầu ra
    cv2.imwrite(output_path_sobel, sobel_output)  # Lưu kết quả Sobel
    cv2.imwrite(output_path_log, log_normalized)  # Lưu kết quả LoG

    print(f"Đã lưu ảnh sau xử lý Sobel tại: {output_path_sobel}")
    print(f"Đã lưu ảnh sau xử lý LoG tại: {output_path_log}")

    # Hiển thị ảnh gốc, ảnh Sobel và ảnh LoG trên cùng một khung hình
    show_images([image, sobel_output, log_normalized], ['Ảnh gốc', 'Kết quả Sobel', 'Kết quả LoG'])
