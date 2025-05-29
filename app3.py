import cv2
import numpy as np

def crop_screenshot_ui(image_path, top_pixels=0, bottom_pixels=0, left_pixels=0, right_pixels=0):
    """
    Cắt bỏ các pixel ở các cạnh của hình ảnh chụp màn hình dựa trên số lượng pixel cố định.
    Thích hợp cho việc loại bỏ các thanh UI cố định.

    Args:
        image_path (str): Đường dẫn đến file hình ảnh.
        top_pixels (int): Số lượng pixel muốn cắt từ cạnh trên.
        bottom_pixels (int): Số lượng pixel muốn cắt từ cạnh dưới.
        left_pixels (int): Số lượng pixel muốn cắt từ cạnh trái.
        right_pixels (int): Số lượng pixel muốn cắt từ cạnh phải.

    Returns:
        numpy.ndarray: Hình ảnh đã được cắt, hoặc None nếu lỗi.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc hình ảnh từ đường dẫn: {image_path}")
            return None

        h, w, _ = img.shape

        # Tính toán các tọa độ cắt
        y_start = top_pixels
        y_end = h - bottom_pixels
        x_start = left_pixels
        x_end = w - right_pixels

        # Đảm bảo các tọa độ hợp lệ
        if y_start >= y_end or x_start >= x_end:
            print("Kích thước cắt không hợp lệ. Có thể bạn đang cắt quá nhiều.")
            return None

        cropped_img = img[y_start:y_end, x_start:x_end]
        return cropped_img

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None

# --- Cách sử dụng ---
if __name__ == "__main__":
    screenshot_file = "images/Screenshot 2025-05-29 210148.png"

    # Ước lượng số pixel cần cắt (bạn có thể phải điều chỉnh các giá trị này)
    # Dựa trên ảnh của bạn:
    #   - Thanh thông báo/điều khiển ở trên cùng: Khoảng 30-40 pixel
    #   - Thanh đen ở hai bên: Có vẻ đồng nhất cho các video dọc trên nền ngang
    #   - Phần dưới có thể có thanh điều khiển khác hoặc chỉ là padding đen
    # Ví dụ ban đầu:
    cut_top = 35 # Ước lượng từ ảnh
    cut_bottom = 0 # Hoặc 10-20 nếu có thanh cuộn hoặc điều khiển dưới
    cut_left = 330 # Ước lượng từ ảnh
    cut_right = 330 # Ước lượng từ ảnh (giả sử đối xứng)

    result_screenshot_cropped = crop_screenshot_ui(
        screenshot_file,
        top_pixels=cut_top,
        bottom_pixels=cut_bottom,
        left_pixels=cut_left,
        right_pixels=cut_right
    )

    if result_screenshot_cropped is not None:
        cv2.imshow("Original Screenshot", cv2.imread(screenshot_file))
        cv2.imshow("Cropped Screenshot (UI Removed)", result_screenshot_cropped)
        cv2.imwrite("cropped_screenshot.png", result_screenshot_cropped)
        print("Đã lưu ảnh chụp màn hình đã cắt tại: cropped_screenshot.png")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không thể cắt ảnh chụp màn hình.")