import cv2
import numpy as np

def crop_solid_color_padding(image_path, padding_color_rgb=None, tolerance=10):
    """
    Cắt bỏ các dải padding màu đồng nhất ở các cạnh của hình ảnh.
    Nếu padding_color_rgb không được cung cấp, nó sẽ cố gắng tự động phát hiện màu padding
    từ các góc của hình ảnh.

    Args:
        image_path (str): Đường dẫn đến file hình ảnh có padding.
        padding_color_rgb (tuple, optional): Màu của padding (R, G, B).
                                             Nếu None, sẽ cố gắng tự động phát hiện.
                                             Mặc định None.
        tolerance (int): Ngưỡng sai số màu cho phép để xác định pixel thuộc padding.
                         Giá trị thấp hơn sẽ nghiêm ngặt hơn. Mặc định 10.

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
            # Chúng ta giả định padding ở các góc
            corners = [
                img[0, 0],         # Top-left
                img[0, w-1],       # Top-right
                img[h-1, 0],       # Bottom-left
                img[h-1, w-1]      # Bottom-right
            ]
            # Tính trung bình màu của các góc. Cần chuyển sang RGB từ BGR của OpenCV
            avg_color_bgr = np.mean(corners, axis=0).astype(int)
            padding_color_rgb = (avg_color_bgr[2], avg_color_bgr[1], avg_color_bgr[0])
            print(f"Tự động phát hiện màu padding: {padding_color_rgb}")

        # Chuyển đổi màu padding sang định dạng BGR của OpenCV
        padding_color_bgr = np.array(padding_color_rgb[::-1])

        # Tìm biên trái
        left_bound = 0
        for x in range(w):
            # Kiểm tra cột pixel. Nếu bất kỳ pixel nào trong cột đó không phải padding,
            # thì đây là biên trái.
            column = img[:, x]
            # cv2.absdiff(column, padding_color_bgr) tính độ lệch.
            # np.all(...) kiểm tra nếu TẤT CẢ các kênh màu đều nằm trong tolerance.
            # ~np.all(...) là NGƯỢC LẠI: nếu BẤT KỲ kênh màu nào vượt quá tolerance.
            if np.any(np.any(cv2.absdiff(column, padding_color_bgr) > tolerance, axis=1)):
                left_bound = x
                break

        # Tìm biên phải
        right_bound = w - 1
        for x in range(w - 1, -1, -1):
            column = img[:, x]
            if np.any(np.any(cv2.absdiff(column, padding_color_bgr) > tolerance, axis=1)):
                right_bound = x
                break

        # Tìm biên trên
        top_bound = 0
        for y in range(h):
            row = img[y, :]
            if np.any(np.any(cv2.absdiff(row, padding_color_bgr) > tolerance, axis=1)):
                top_bound = y
                break

        # Tìm biên dưới
        bottom_bound = h - 1
        for y in range(h - 1, -1, -1):
            row = img[y, :]
            if np.any(np.any(cv2.absdiff(row, padding_color_bgr) > tolerance, axis=1)):
                bottom_bound = y
                break

        # Kiểm tra xem có tìm thấy biên nào không
        if left_bound >= right_bound or top_bound >= bottom_bound:
            print("Không tìm thấy nội dung chính hoặc ảnh quá nhỏ sau khi cắt.")
            return None

        # Cắt ảnh
        cropped_img = img[top_bound : bottom_bound + 1, left_bound : right_bound + 1]
        return cropped_img

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None

# --- Cách sử dụng ---
if __name__ == "__main__":
    # Thay đổi đường dẫn này thành ảnh '48.jpg' của bạn
    image_file = "images/Screenshot 2025-05-29 210148.png"
    # image_file = "images/48.jpg"

    # Lưu ý: Với ảnh '48.jpg', màu padding là một màu be/trắng ngà.
    # Bạn có thể thử không truyền padding_color_rgb để hàm tự động phát hiện,
    # hoặc cung cấp chính xác màu đó nếu biết.
    # Ví dụ, nếu bạn biết chính xác màu padding là (230, 227, 221) (RGB), bạn có thể dùng:
    # result_image = crop_solid_color_padding(image_file, padding_color_rgb=(230, 227, 221), tolerance=15)

    # Thử với tự động phát hiện màu padding (khuyến nghị cho trường hợp này)
    result_image = crop_solid_color_padding(image_file, tolerance=60) # Tăng tolerance một chút nếu màu padding không hoàn hảo đồng nhất

    if result_image is not None:
        cv2.imshow("Original Image (48.jpg)", cv2.imread(image_file))
        cv2.imshow("Cropped Image (Content Only)", result_image)
        cv2.imwrite("content_from_48.png", result_image)
        print("Đã lưu ảnh nội dung chính tại: content_from_48.png")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không thể cắt ảnh.")