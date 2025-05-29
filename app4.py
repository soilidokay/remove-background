from rembg import remove
from PIL import Image # Thư viện Pillow để làm việc với ảnh

def remove_background_with_ai(input_path, output_path):
    """
    Tách nền khỏi ảnh bằng thư viện rembg (AI).

    Args:
        input_path (str): Đường dẫn đến ảnh gốc.
        output_path (str): Đường dẫn để lưu ảnh đã tách nền.
    """
    try:
        input_image = Image.open(input_path)
        output_image = remove(input_image) # Dùng remove() để tách nền
        output_image.save(output_path)
        print(f"Đã tách nền và lưu ảnh tại: {output_path}")
    except Exception as e:
        print(f"Lỗi khi tách nền bằng rembg: {e}")

# --- Cách sử dụng ---
if __name__ == "__main__":
    image_with_complex_padding = "images/Screenshot 2025-05-29 210148.png" # Đặt ảnh của bạn ở đây
    output_image_no_bg = "image_no_background.png"

    remove_background_with_ai(image_with_complex_padding, output_image_no_bg)

    # Hiển thị ảnh kết quả (tùy chọn)
    try:
        import cv2
        cv2.imshow("Original Image", cv2.imread(image_with_complex_padding))
        cv2.imshow("Image without Background", cv2.imread(output_image_no_bg))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ImportError:
        print("Không tìm thấy OpenCV. Không thể hiển thị ảnh trực tiếp.")