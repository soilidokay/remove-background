import cv2
import numpy as np

def remove_multicolor_padding_with_stages(image_path, output_path=None):
    # Giai đoạn 1: Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        return None
    cv2.imshow("Giai đoạn 1: Ảnh gốc", img)
    cv2.waitKey(0) # Chờ người dùng nhấn phím bất kỳ

    # Giai đoạn 2: Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Giai đoạn 2: Ảnh xám", gray)
    cv2.waitKey(0)

    # Giai đoạn 3: Dò cạnh bằng Canny
    edges = cv2.Canny(gray, 30, 100)
    cv2.imshow("Giai đoạn 3: Cạnh (Canny)", edges)
    cv2.waitKey(0)

    # Giai đoạn 4: Đóng các lỗ nhỏ (Dilation)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow("Giai đoạn 4: Cạnh sau Dilation", edges_dilated)
    cv2.waitKey(0)

    # Giai đoạn 5: Đóng các lỗ nhỏ (Erosion)
    edges_closed = cv2.erode(edges_dilated, kernel, iterations=1)
    cv2.imshow("Giai đoạn 5: Cạnh sau Erosion (Đóng)", edges_closed)
    cv2.waitKey(0)

    # Giai đoạn 6: Tìm các điểm có cạnh và tính toán khung bao
    coords = cv2.findNonZero(edges_closed)
    if coords is None:
        print("Không tìm thấy bất kỳ cạnh nào sau khi xử lý. Không thể cắt ảnh.")
        return img # Trả về ảnh gốc nếu không tìm thấy cạnh
        
    x, y, w, h = cv2.boundingRect(coords)

    # Giai đoạn 7: Cắt ảnh
    cropped = img[y:y+h, x:x+w]
    cv2.imshow("Giai đoạn 7: Ảnh đã cắt", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows() # Đóng tất cả cửa sổ hình ảnh

    if output_path:
        cv2.imwrite(output_path, cropped)
        print(f"Ảnh đã cắt được lưu tại: {output_path}")

    return cropped

# Ví dụ sử dụng:
# Thay 'your_image.jpg' bằng đường dẫn đến ảnh của bạn
# và 'output_cropped_image.jpg' bằng tên file đầu ra mong muốn
remove_multicolor_padding_with_stages('images/51.png', 'output_cropped_image.jpg')