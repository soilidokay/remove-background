import cv2
import numpy as np

def calculate_bounding_box_and_stretch_height(image_path, output_path=None, min_line_length=100, max_line_gap=10):
    """
    Tính toán khung bao (bounding rectangle) bao quanh các đường thẳng dọc dài nhất
    ở phía bên trái và bên phải của ảnh, sau đó điều chỉnh chiều cao của khung bao
    để bằng với chiều cao của ảnh gốc.

    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        output_path (str, optional): Đường dẫn để lưu ảnh đã cắt. Mặc định là None.
        min_line_length (int): Độ dài tối thiểu của một đường thẳng để được coi là hợp lệ (đơn vị pixel).
        max_line_gap (int): Khoảng cách tối đa giữa các điểm để được coi là cùng một đường thẳng.

    Returns:
        numpy.ndarray: Ảnh đã cắt với chiều cao bằng ảnh gốc.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        return None

    height, width = img.shape[:2]
    
    # Giai đoạn 1: Đọc ảnh gốc
    cv2.imshow("Giai đoạn 1: Ảnh gốc", img)
    cv2.waitKey(0)

    # Giai đoạn 2: Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Giai đoạn 2: Ảnh xám", gray)
    cv2.waitKey(0)

    # Giai đoạn 3: Lọc Gaussian để làm mượt ảnh và giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Giai đoạn 3: Ảnh đã làm mượt (Gaussian Blur)", blurred)
    cv2.waitKey(0)

    # Giai đoạn 4: Dò cạnh bằng Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    cv2.imshow("Giai đoạn 4: Cạnh (Canny)", edges)
    cv2.waitKey(0)

    # Giai đoạn 5: Phát hiện đường thẳng bằng HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    lines_drawn_img = img.copy()
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính toán góc của đường thẳng
            if x2 - x1 == 0:
                angle_rad = np.pi / 2 
            else:
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
            
            angle_deg = np.abs(np.degrees(angle_rad))

            # Nếu là đường thẳng dọc (góc gần 90 độ)
            if 80 <= angle_deg <= 100: 
                vertical_lines.append(line[0])
                cv2.line(lines_drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vẽ đường thẳng dọc màu xanh lá
            else:
                 cv2.line(lines_drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Vẽ đường thẳng không dọc màu đỏ
        
    cv2.imshow("Giai đoạn 5: Đường thẳng đã phát hiện (Xanh: dọc, Đỏ: khác)", lines_drawn_img)
    cv2.waitKey(0)

    # Giai đoạn 6: Tìm đường thẳng dọc dài nhất bên trái và bên phải
    leftmost_vertical_line_x = None
    rightmost_vertical_line_x = None
    
    max_left_length = 0
    max_right_length = 0

    if vertical_lines:
        # Chúng ta chỉ cần tọa độ x cho việc cắt theo chiều ngang
        # Tìm x nhỏ nhất từ các đường thẳng ở nửa trái và x lớn nhất từ các đường thẳng ở nửa phải
        # Để đảm bảo đây là các đường thẳng "chính" và "dài nhất"
        
        # Sắp xếp các đường thẳng dọc theo tọa độ x trung bình
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

        # Lọc đường thẳng dọc trái nhất dài nhất
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            current_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if (x1 + x2) / 2 < width / 2: # Đường thẳng ở nửa trái ảnh
                if current_length > max_left_length:
                    max_left_length = current_length
                    leftmost_vertical_line_x = min(x1, x2) # Lấy tọa độ x nhỏ nhất của đường thẳng này
            elif (x1 + x2) / 2 >= width / 2: # Đường thẳng ở nửa phải ảnh
                if current_length > max_right_length:
                    max_right_length = current_length
                    rightmost_vertical_line_x = max(x1, x2) # Lấy tọa độ x lớn nhất của đường thẳng này

        print(f"X của đường thẳng dọc dài nhất bên trái: {leftmost_vertical_line_x}")
        print(f"X của đường thẳng dọc dài nhất bên phải: {rightmost_vertical_line_x}")
            
    # Giai đoạn 7: Tính toán khung bao cuối cùng
    # Chiều cao của khung bao sẽ bằng chiều cao của ảnh gốc
    final_y_min = 0
    final_y_max = height

    # Xác định x_min và x_max dựa trên các đường thẳng dọc dài nhất
    # Sử dụng x_min/x_max của ảnh gốc làm giá trị mặc định
    final_x_min = 0
    final_x_max = width

    if leftmost_vertical_line_x is not None:
        final_x_min = max(0, leftmost_vertical_line_x - 5) # Thêm lề 5px về bên trái
    
    if rightmost_vertical_line_x is not None:
        final_x_max = min(width, rightmost_vertical_line_x + 5) # Thêm lề 5px về bên phải

    # Đảm bảo khung bao hợp lệ
    if final_x_max <= final_x_min:
        print("Không tìm thấy đủ đường thẳng dọc hợp lệ để tạo khung bao. Trả về ảnh gốc.")
        return img
        
    # Giai đoạn 8: Cắt ảnh
    cropped = img[final_y_min:final_y_max, final_x_min:final_x_max]

    # Vẽ khung bao cuối cùng lên ảnh gốc để hiển thị
    result_img = img.copy()
    # Vẽ khung bao với chiều cao toàn bộ ảnh
    cv2.rectangle(result_img, (final_x_min, final_y_min), (final_x_max, final_y_max), (255, 0, 0), 2)
    cv2.imshow("Giai đoạn 8: Ảnh với khung bao cuối cùng (Chiều cao ảnh gốc)", result_img)
    cv2.waitKey(0)
    
    cv2.imshow("Giai đoạn 9: Ảnh đã cắt (Chiều cao ảnh gốc)", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, cropped)
        print(f"Ảnh đã cắt được lưu tại: {output_path}")

    return cropped

# --- Ví dụ sử dụng ---
# Thay 'your_image.jpg' bằng đường dẫn đến ảnh của bạn
calculate_bounding_box_and_stretch_height('images/48.jpg', 'output_stretched_cropped_image.jpg')