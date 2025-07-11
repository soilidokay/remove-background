import cv2
import numpy as np

def calculate_bounding_box_around_longest_straightest_vertical_lines_sobel(image_path, output_path=None, min_line_length=50, max_line_gap=30, angle_tolerance=10, gaussian_ksize=(7, 7), sobel_ksize=3, sobel_threshold=50):
    """
    Tính toán khung bao (bounding rectangle) bao quanh các đường thẳng dọc dài nhất và thẳng nhất
    ở phía bên trái và bên phải của ảnh, sau đó điều chỉnh chiều cao của khung bao
    để bằng với chiều cao của ảnh gốc, sử dụng Sobel để dò cạnh.

    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        output_path (str, optional): Đường dẫn để lưu ảnh đã cắt. Mặc định là None.
        min_line_length (int): Độ dài tối thiểu của một đường thẳng để được coi là hợp lệ (pixel).
        max_line_gap (int): Khoảng cách tối đa giữa các điểm để được coi là cùng một đường thẳng.
        angle_tolerance (int): Độ lệch tối đa (độ) so với 90 độ để được coi là đường thẳng dọc.
        gaussian_ksize (tuple): Kích thước kernel cho Gaussian Blur (ví dụ: (5, 5)).
        sobel_ksize (int): Kích thước kernel cho Sobel (phải là số lẻ, ví dụ: 3, 5, 7).
        sobel_threshold (int): Ngưỡng để nhị phân hóa ảnh cạnh Sobel.

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
    # Đây là bước tiền xử lý quan trọng cho Sobel
    blurred = cv2.GaussianBlur(gray, gaussian_ksize, 0)
    cv2.imshow(f"Giai đoạn 3: Ảnh đã làm mượt (Gaussian Blur {gaussian_ksize})", blurred)
    cv2.waitKey(0)

    # Giai đoạn 4: Dò cạnh bằng Sobel
    # Tính gradient theo chiều ngang (dx=1, dy=0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    # Tính gradient theo chiều dọc (dx=0, dy=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # Chuyển đổi về giá trị tuyệt đối và kiểu 8-bit để hiển thị
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    sobely_abs = cv2.convertScaleAbs(sobely)

    cv2.imshow(f"Giai đoạn 4a: Sobel X (ksize={sobel_ksize})", sobelx_abs)
    cv2.waitKey(0)
    cv2.imshow(f"Giai đoạn 4b: Sobel Y (ksize={sobel_ksize})", sobely_abs)
    cv2.waitKey(0)

    # Kết hợp gradient x và y để có ảnh cạnh tổng thể
    # Có thể dùng cv2.addWeighted hoặc cv2.magnitude
    # edges_sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)
    edges_sobel_combined = cv2.magnitude(sobelx, sobely) # Sử dụng magnitude để kết hợp tốt hơn
    edges_sobel_combined = cv2.convertScaleAbs(edges_sobel_combined) # Chuyển đổi về 8-bit

    cv2.imshow(f"Giai đoạn 4c: Cạnh Sobel tổng hợp", edges_sobel_combined)
    cv2.waitKey(0)

    # Ngưỡng hóa kết quả Sobel để tạo ảnh cạnh nhị phân
    # Các giá trị lớn hơn ngưỡng sẽ trở thành 255 (cạnh), còn lại là 0
    _, edges = cv2.threshold(edges_sobel_combined, sobel_threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow(f"Giai đoạn 4d: Cạnh (Sobel) - Ngưỡng {sobel_threshold}", edges)
    cv2.waitKey(0)

    # Giai đoạn 5: Phát hiện đường thẳng bằng HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    lines_drawn_img = img.copy()
    
    # Lưu trữ các đường thẳng dọc cùng với độ dài và độ lệch góc
    # Dạng: [(x1, y1, x2, y2, length, angle_deviation)]
    candidate_vertical_lines = [] 

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính toán góc của đường thẳng (tính bằng radian)
            if x2 - x1 == 0:
                angle_rad = np.pi / 2 
            else:
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
            
            angle_deg = np.abs(np.degrees(angle_rad))
            
            # Tính độ lệch góc so với 90 độ
            angle_deviation = np.abs(angle_deg - 90)

            # Nếu là đường thẳng dọc (trong khoảng dung sai góc)
            if angle_deviation <= angle_tolerance: 
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                candidate_vertical_lines.append((x1, y1, x2, y2, length, angle_deviation))
                cv2.line(lines_drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vẽ đường thẳng dọc màu xanh lá
            else:
                 cv2.line(lines_drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Vẽ đường thẳng không dọc màu đỏ
        
    cv2.imshow("Giai đoạn 5: Đường thẳng đã phát hiện (Xanh: dọc, Đỏ: khác)", lines_drawn_img)
    cv2.waitKey(0)

    # Giai đoạn 6: Tìm đường thẳng dọc dài nhất VÀ thẳng nhất bên trái và bên phải
    leftmost_best_line_x = None
    rightmost_best_line_x = None
    
    best_left_score = -1.0 # Score sẽ là length - angle_deviation * weight (ưu tiên dài hơn, ít lệch góc hơn)
    best_right_score = -1.0
    
    # Trọng số cho độ lệch góc. Bạn có thể điều chỉnh để ưu tiên độ dài hay độ thẳng hơn.
    # Ví dụ: 1.0 nghĩa là 1 độ lệch góc ảnh hưởng bằng 1 pixel độ dài.
    # Nếu muốn ưu tiên độ thẳng hơn, tăng trọng số này.
    angle_weight = 2.0 

    if candidate_vertical_lines:
        for line_data in candidate_vertical_lines:
            x1, y1, x2, y2, length, angle_deviation = line_data
            
            # Tính điểm số cho đường thẳng: ưu tiên độ dài và ít lệch góc
            current_score = length - (angle_deviation * angle_weight)
            
            # Xác định nửa ảnh
            mid_x_line = (x1 + x2) / 2

            if mid_x_line < width / 2: # Đường thẳng ở nửa trái ảnh
                if current_score > best_left_score:
                    best_left_score = current_score
                    leftmost_best_line_x = min(x1, x2) # Lấy tọa độ x nhỏ nhất của đường này
            else: # Đường thẳng ở nửa phải ảnh
                if current_score > best_right_score:
                    best_right_score = current_score
                    rightmost_best_line_x = max(x1, x2) # Lấy tọa độ x lớn nhất của đường này
        
        print(f"X của đường thẳng dọc dài nhất & thẳng nhất bên trái: {leftmost_best_line_x}")
        print(f"X của đường thẳng dọc dài nhất & thẳng nhất bên phải: {rightmost_best_line_x}")
            
    # Giai đoạn 7: Tính toán khung bao cuối cùng với chiều cao ảnh gốc
    final_y_min = 0
    final_y_max = height

    final_x_min = 0
    final_x_max = width

    # Nếu tìm thấy đường thẳng bên trái, xác định x_min
    if leftmost_best_line_x is not None:
        final_x_min = max(0, leftmost_best_line_x - 5) # Thêm lề 5px
    
    # Nếu tìm thấy đường thẳng bên phải, xác định x_max
    if rightmost_best_line_x is not None:
        final_x_max = min(width, rightmost_best_line_x + 5) # Thêm lề 5px

    # Đảm bảo khung bao hợp lệ
    if final_x_max <= final_x_min:
        print("Không tìm thấy đủ đường thẳng dọc hợp lệ để tạo khung bao. Trả về ảnh gốc.")
        return img
        
    # Giai đoạn 8: Cắt ảnh
    cropped = img[final_y_min:final_y_max, final_x_min:final_x_max]

    # Vẽ khung bao cuối cùng lên ảnh gốc để hiển thị
    result_img = img.copy()
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
count = 54
for i in range(48,54):
# --- Ví dụ sử dụng ---
# Thay 'your_image.jpg' bằng đường dẫn đến ảnh của bạn
# Đặt các tham số mới cho Sobel
# Lưu ý: sobel_ksize và sobel_threshold cần được điều chỉnh tùy theo ảnh của bạn
    calculate_bounding_box_around_longest_straightest_vertical_lines_sobel(
        f'images/{i}.png', 
        'output_cropped_sobel_edges.jpg',
        gaussian_ksize=(5, 5),       # Kích thước kernel Gaussian
        sobel_ksize=3,               # Kích thước kernel Sobel (thường là 3, 5, 7)
        sobel_threshold=50           # Ngưỡng để nhị phân hóa cạnh Sobel
    )

# Ví dụ với vòng lặp như bạn đã có, nhưng sử dụng hàm mới
# for i in range(48, 54):
#     calculate_bounding_box_around_longest_straightest_vertical_lines_sobel(f'images/{i}.png', f'outputs_sobel/{i}.jpg')
