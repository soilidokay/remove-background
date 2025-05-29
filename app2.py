import cv2
import numpy as np

def crop_solid_color_padding_optimized(image_path, padding_color_rgb=None, tolerance=15):
    """
    Cắt bỏ các dải padding màu đồng nhất ở các cạnh của hình ảnh một cách tối ưu.
    Sử dụng các thao tác mảng của NumPy để tăng tốc độ.

    Args:
        image_path (str): Đường dẫn đến file hình ảnh có padding.
        padding_color_rgb (tuple, optional): Màu của padding (R, G, B).
                                             Nếu None, sẽ cố gắng tự động phát hiện.
                                             Mặc định None.
        tolerance (int): Ngưỡng sai số màu cho phép để xác định pixel thuộc padding.
                         Mặc định 15.

    Returns:
        numpy.ndarray: Hình ảnh đã được cắt, hoặc None nếu lỗi.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc hình ảnh từ đường dẫn: {image_path}")
            return None

        h, w, _ = img.shape

        # Tự động phát hiện màu padding nếu không được cung cấp
        if padding_color_rgb is None:
            # Lấy màu của 4 góc để ước tính màu padding
            corners = [
                img[0, 0],         # Top-left
                img[0, w-1],       # Top-right
                img[h-1, 0],       # Bottom-left
                img[h-1, w-1]      # Bottom-right
            ]
            # Tính trung bình màu của các góc. Chuyển sang RGB từ BGR của OpenCV
            avg_color_bgr = np.mean(corners, axis=0).astype(int)
            padding_color_rgb = (avg_color_bgr[2], avg_color_bgr[1], avg_color_bgr[0])
            print(f"Tự động phát hiện màu padding: {padding_color_rgb}")

        # Chuyển đổi màu padding sang định dạng BGR của OpenCV
        padding_color_bgr = np.array(padding_color_rgb[::-1])

        # Tạo mask cho các pixel không phải padding (nội dung chính)
        # Bằng cách này, chúng ta chỉ cần một phép tính duy nhất cho toàn bộ ảnh
        # np.any(..., axis=2) sẽ kiểm tra nếu bất kỳ kênh màu nào khác biệt quá tolerance
        # (True = không phải padding)
        non_padding_mask = np.any(np.abs(img - padding_color_bgr) > tolerance, axis=2)

        # Tìm các tọa độ của các pixel không phải padding
        # np.argwhere trả về các cặp (row, col)
        coords = np.argwhere(non_padding_mask)

        if coords.size == 0:
            print("Không tìm thấy nội dung chính (toàn bộ ảnh là padding hoặc màu tương tự).")
            return None

        # Tìm bounding box của nội dung chính
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Cắt ảnh
        cropped_img = img[y_min : y_max + 1, x_min : x_max + 1]
        return cropped_img

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None

# --- Cách sử dụng ---
if __name__ == "__main__":
    image_file = "48.jpg" # Đường dẫn đến ảnh của bạn
    image_file = "images/Screenshot 2025-05-29 210148.png"

    # Sử dụng hàm tối ưu hóa
    result_image_optimized = crop_solid_color_padding_optimized(image_file, tolerance=15)

    if result_image_optimized is not None:
        cv2.imshow("Original Image (48.jpg)", cv2.imread(image_file))
        cv2.imshow("Cropped Image (Optimized)", result_image_optimized)
        cv2.imwrite("content_from_48_optimized.png", result_image_optimized)
        print("Đã lưu ảnh nội dung chính tối ưu hóa tại: content_from_48_optimized.png")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không thể cắt ảnh.")