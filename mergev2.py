import cv2
import numpy as np

def detect_seam_boundaries_v2(image, threshold=0.4, max_width=1200, show=True, min_seam_width=50, max_seam_width=300):
    """
    Phát hiện ranh giới seam giữa hai vùng trong ảnh panorama bằng cách phân tích độ biến đổi theo chiều ngang.
    
    Tham số:
        image: ảnh đầu vào dạng NumPy (BGR)
        threshold: ngưỡng xác định seam
        max_width: resize ảnh khi hiển thị
        show: True để hiển thị từng bước bằng cv2.imshow
        min_seam_width: khoảng cách nhỏ nhất giữa 2 biên seam
        max_seam_width: khoảng cách lớn nhất giữa 2 biên seam

    Trả về:
        {
            "left_edge": int hoặc None,
            "right_edge": int hoặc None,
            "seam_indices": list[int],
            "seam_region": ảnh cắt vùng seam hoặc None,
            "result_image": ảnh có vẽ seam + biên
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    max_grad = np.max(gradient_mag)
    if max_grad == 0:
        max_grad = 1e-10  # tránh chia 0
    gradient_mag = np.uint8(255 * gradient_mag / max_grad)

    column_energy = gradient_mag.sum(axis=0)
    energy_norm = (column_energy - column_energy.min()) / np.ptp(column_energy)

    binary_mask = (energy_norm > threshold).astype(np.uint8)
    kernel = np.ones((1, 15), np.uint8)
    closed = cv2.morphologyEx(binary_mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
    seam_indices = np.where(closed > 0)[0]

    seam_vis = image.copy()
    for x in seam_indices:
        cv2.line(seam_vis, (x, 0), (x, seam_vis.shape[0]), (0, 0, 255), 1)

    result = seam_vis.copy()
    left_edge, right_edge = None, None
    seam_region = None

    if len(seam_indices) > 0:
        left_edge = int(seam_indices.min())
        right_edge = int(seam_indices.max())
        seam_width = right_edge - left_edge
        
        if min_seam_width <= seam_width <= max_seam_width:
            cv2.line(result, (left_edge, 0), (left_edge, result.shape[0]), (0, 255, 0), 2)
            cv2.line(result, (right_edge, 0), (right_edge, result.shape[0]), (0, 255, 0), 2)
            seam_region = image[:, left_edge:right_edge]
        else:
            # Không thỏa khoảng cách, bỏ chọn seam
            left_edge, right_edge = None, None
            seam_indices = []
            seam_region = None

    def resize(img):
        h, w = img.shape[:2]
        if w > max_width:
            ratio = max_width / w
            return cv2.resize(img, (int(w * ratio), int(h * ratio)))
        return img

    if show:
        cv2.imshow("1. Original Image", resize(image))
        cv2.imshow("2. Gradient Magnitude", resize(gradient_mag))
        cv2.imshow("3. Column Energy", resize(np.tile((energy_norm * 255).astype(np.uint8), (image.shape[0], 1))))
        cv2.imshow("4. Seam Lines (Red)", resize(seam_vis))
        cv2.imshow("5. Seam Edges (Green)", resize(result))
        if seam_region is not None:
            cv2.imshow("6. Cropped Seam Region", resize(seam_region))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "seam_indices": seam_indices.tolist() if len(seam_indices) > 0 else [],
        "seam_region": seam_region,
        "result_image": result
    }
def detect_seam_boundaries_v3(image, threshold=0.4, max_width=1200,
                              min_seam_width=0.1, max_seam_width=0.3, seam_width_is_ratio=True,
                              ignore_edges=False, edge_margin=30,
                              show=True):
    """
    Phát hiện ranh giới seam giữa hai vùng trong ảnh panorama.

    Thêm:
        ignore_edges: bool, nếu True thì bỏ seam nếu biên nằm quá gần mép ảnh
        edge_margin: khoảng cách (pixel) từ mép ảnh để xét ignore nếu ignore_edges=True
    """
    h, w = image.shape[:2]

    if seam_width_is_ratio:
        max_seam_width_px = int(w * max_seam_width)
        min_seam_width = int(w * min_seam_width)
    else:
        max_seam_width_px = max_seam_width
        min_seam_width = min_seam_width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    max_grad = np.max(gradient_mag)
    if max_grad == 0:
        max_grad = 1e-10
    gradient_mag = np.uint8(255 * gradient_mag / max_grad)

    column_energy = gradient_mag.sum(axis=0)
    energy_norm = (column_energy - column_energy.min()) / np.ptp(column_energy)

    binary_mask = (energy_norm > threshold).astype(np.uint8)
    kernel = np.ones((1, 15), np.uint8)
    closed = cv2.morphologyEx(binary_mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
    seam_indices = np.where(closed > 0)[0]

    seam_vis = image.copy()
    for x in seam_indices:
        cv2.line(seam_vis, (x, 0), (x, seam_vis.shape[0]), (0, 0, 255), 1)

    result = seam_vis.copy()
    left_edge, right_edge = None, None
    seam_region = None

    if len(seam_indices) > 0:
        left_edge = int(seam_indices.min())
        right_edge = int(seam_indices.max())
        seam_width = right_edge - left_edge

        # Bỏ seam nếu width ko hợp lệ hoặc nếu biên quá gần mép ảnh và ignore_edges = True
        if min_seam_width <= seam_width <= max_seam_width_px:
            if ignore_edges:
                if (left_edge < edge_margin) or (right_edge > w - edge_margin):
                    # Bỏ qua seam này vì quá gần mép ảnh
                    left_edge, right_edge = None, None
                    seam_indices = []
                    seam_region = None
                else:
                    cv2.line(result, (left_edge, 0), (left_edge, result.shape[0]), (0, 255, 0), 2)
                    cv2.line(result, (right_edge, 0), (right_edge, result.shape[0]), (0, 255, 0), 2)
                    seam_region = image[:, left_edge:right_edge]
            else:
                cv2.line(result, (left_edge, 0), (left_edge, result.shape[0]), (0, 255, 0), 2)
                cv2.line(result, (right_edge, 0), (right_edge, result.shape[0]), (0, 255, 0), 2)
                seam_region = image[:, left_edge:right_edge]
        else:
            left_edge, right_edge = None, None
            seam_indices = []
            seam_region = None

    def resize(img):
        h, w = img.shape[:2]
        if w > max_width:
            ratio = max_width / w
            return cv2.resize(img, (int(w * ratio), int(h * ratio)))
        return img

    if show:
        cv2.imshow("1. Original Image", resize(image))
        cv2.imshow("2. Gradient Magnitude", resize(gradient_mag))
        cv2.imshow("3. Column Energy", resize(np.tile((energy_norm * 255).astype(np.uint8), (image.shape[0], 1))))
        cv2.imshow("4. Seam Lines (Red)", resize(seam_vis))
        cv2.imshow("5. Seam Edges (Green)", resize(result))
        if seam_region is not None:
            cv2.imshow("6. Cropped Seam Region", resize(seam_region))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "seam_indices": seam_indices.tolist() if len(seam_indices) > 0 else [],
        "seam_region": seam_region,
        "result_image": result
    }

def detect_seam_boundaries_v4(image, threshold=0.4, max_width=1200,
                              min_seam_width=50, max_seam_width=0.3, max_seam_width_is_ratio=True,
                              ignore_edges=False, edge_margin=30,
                              contiguous_groups=True,
                              show=True):
    """
    Tìm ranh giới seam, ưu tiên chọn cặp biên nằm gần tâm ảnh nhất.
    
    Nếu contiguous_groups = True thì tìm nhóm các điểm liền kề.
    Nếu False thì tìm đoạn cửa sổ tối ưu trên toàn bộ điểm seam.
    """

    h, w = image.shape[:2]
    if max_seam_width_is_ratio:
        max_seam_width_px = int(w * max_seam_width)
    else:
        max_seam_width_px = max_seam_width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    max_grad = np.max(gradient_mag)
    if max_grad == 0:
        max_grad = 1e-10
    gradient_mag = np.uint8(255 * gradient_mag / max_grad)

    column_energy = gradient_mag.sum(axis=0)
    energy_norm = (column_energy - column_energy.min()) / np.ptp(column_energy)

    binary_mask = (energy_norm > threshold).astype(np.uint8)
    kernel = np.ones((1, 15), np.uint8)
    closed = cv2.morphologyEx(binary_mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
    seam_indices = np.where(closed > 0)[0]

    center_x = w // 2
    best_left = None
    best_right = None
    best_score = -1
    best_dist = float('inf')
    best_indices = []

    if contiguous_groups:
        # Nhóm các điểm liền kề như trước
        groups = []
        if len(seam_indices) > 0:
            group = [seam_indices[0]]
            for idx in seam_indices[1:]:
                if idx == group[-1] + 1:
                    group.append(idx)
                else:
                    groups.append(group)
                    group = [idx]
            groups.append(group)
        else:
            groups = []

        for g in groups:
            left_edge = g[0]
            right_edge = g[-1]
            seam_width = right_edge - left_edge
            mid = (left_edge + right_edge) // 2

            if seam_width < min_seam_width or seam_width > max_seam_width_px:
                continue
            if ignore_edges:
                if left_edge < edge_margin or right_edge > w - edge_margin:
                    continue
            dist = abs(mid - center_x)

            # Ưu tiên nhóm gần tâm, nếu bằng thì chọn nhóm rộng hơn
            score = len(g)
            if dist < best_dist or (dist == best_dist and score > best_score):
                best_dist = dist
                best_score = score
                best_left = left_edge
                best_right = right_edge
                best_indices = g

    else:
        # Tìm đoạn cửa sổ trượt trong toàn bộ seam_indices
        # Mình sẽ dùng phương pháp chạy sliding window trên mảng các cột
        # Nhưng seam_indices có thể không liên tục, nên ta xử lý windows theo vị trí cột

        # Nếu không đủ điểm thì trả về None
        if len(seam_indices) == 0:
            pass
        else:
            seam_indices_sorted = np.array(sorted(seam_indices))
            n = len(seam_indices_sorted)

            # Duyệt tất cả các đoạn con có độ dài >= min_seam_width và <= max_seam_width_px
            # Ở đây mình duyệt bằng chỉ số mảng (start,end) và so sánh vị trí cột

            for start_i in range(n):
                for end_i in range(start_i, n):
                    left_edge = seam_indices_sorted[start_i]
                    right_edge = seam_indices_sorted[end_i]
                    seam_width = right_edge - left_edge
                    if seam_width < min_seam_width:
                        continue
                    if seam_width > max_seam_width_px:
                        break  # đoạn tiếp theo sẽ rộng hơn nên thoát vòng lặp end_i

                    if ignore_edges:
                        if left_edge < edge_margin or right_edge > w - edge_margin:
                            continue

                    # Tính điểm: số điểm seam trong đoạn
                    score = (end_i - start_i + 1)
                    mid = (left_edge + right_edge) // 2
                    dist = abs(mid - center_x)

                    # Ưu tiên score cao rồi đến gần tâm
                    if score > best_score or (score == best_score and dist < best_dist):
                        best_score = score
                        best_dist = dist
                        best_left = left_edge
                        best_right = right_edge
                        best_indices = list(range(left_edge, right_edge+1))

    # Vẽ kết quả
    seam_vis = image.copy()
    for x in seam_indices:
        cv2.line(seam_vis, (x, 0), (x, h), (0, 0, 255), 1)

    result = seam_vis.copy()
    seam_region = None

    if best_left is not None and best_right is not None:
        cv2.line(result, (best_left, 0), (best_left, h), (0, 255, 0), 2)
        cv2.line(result, (best_right, 0), (best_right, h), (0, 255, 0), 2)
        seam_region = image[:, best_left:best_right]

    def resize(img):
        ih, iw = img.shape[:2]
        if iw > max_width:
            ratio = max_width / iw
            return cv2.resize(img, (int(iw * ratio), int(ih * ratio)))
        return img

    if show:
        cv2.imshow("1. Original Image", resize(image))
        cv2.imshow("2. Gradient Magnitude", resize(gradient_mag))
        cv2.imshow("3. Column Energy", resize(np.tile((energy_norm * 255).astype(np.uint8), (h,1))))
        cv2.imshow("4. Seam Lines (Red)", resize(seam_vis))
        cv2.imshow("5. Seam Edges (Green)", resize(result))
        if seam_region is not None:
            cv2.imshow("6. Cropped Seam Region", resize(seam_region))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "left_edge": best_left,
        "right_edge": best_right,
        "seam_indices": best_indices,
        "seam_region": seam_region,
        "result_image": result
    }
# Ví dụ dùng cho nhiều ảnh
start = 48
count = 56
debug = True

for i in range(start, count):
    image_path = f'images/{i}.png'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        continue

    result = detect_seam_boundaries_v3(img, threshold=0.4,min_seam_width=0.3,max_seam_width=0.9, show=debug)

    if result["seam_region"] is not None:
        output_path = f'output_seam_region_{i}.png'
        cv2.imwrite(output_path, result["seam_region"])
        print(f"Ảnh vùng seam đã được lưu tại: {output_path}")
    else:
        print(f"Ảnh {i} không tìm được seam phù hợp trong khoảng {50} - {300}px.")
