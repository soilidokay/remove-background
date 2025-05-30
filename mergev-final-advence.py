import cv2
import numpy as np


def detect_seam_boundaries_v3(image, thresholds=[0.4], max_width=1200,
                              min_seam_width=0.1, max_seam_width=0.3, seam_width_is_ratio=True,
                              ignore_edges=False, edge_margin=30,
                              show=True):
    """
    Phát hiện ranh giới seam giữa hai vùng trong ảnh panorama.

    Thêm:
        thresholds: list, danh sách các ngưỡng để thử lần lượt
        ignore_edges: bool, nếu True thì bỏ seam nếu biên nằm quá gần mép ảnh
        edge_margin: khoảng cách (pixel) từ mép ảnh để xét ignore nếu ignore_edges=True
    """
    h, w = image.shape[:2]
    if ignore_edges:
        image = image[:, edge_margin: w - edge_margin]
        w = image.shape[1]  # cập nhật lại width mới sau khi crop
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

    left_edge, right_edge = None, None
    seam_indices = []
    seam_region = None
    result = image.copy()
    seam_vis = image.copy()

    # Thử từng ngưỡng trong danh sách thresholds
    for threshold in thresholds:
        binary_mask = (energy_norm > threshold).astype(np.uint8)
        kernel = np.ones((1, 15), np.uint8)
        closed = cv2.morphologyEx(binary_mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
        seam_indices = np.where(closed > 0)[0]

        # Kiểm tra số lượng seam
        if len(seam_indices) > 0:
            left_edge = int(seam_indices.min())
            right_edge = int(seam_indices.max())
            seam_width = right_edge - left_edge

            # Kiểm tra seam width và điều kiện mép ảnh
            if min_seam_width <= seam_width <= max_seam_width_px:
                # Seam hợp lệ, vẽ các đường biên
                for x in seam_indices:
                    cv2.line(seam_vis, (x, 0), (x, seam_vis.shape[0]), (0, 0, 255), 1)
                cv2.line(result, (left_edge, 0), (left_edge, result.shape[0]), (0, 255, 0), 2)
                cv2.line(result, (right_edge, 0), (right_edge, result.shape[0]), (0, 255, 0), 2)
                seam_region = image[:, left_edge:right_edge]
                break  # Thoát vòng lặp khi tìm được seam hợp lệ

            else:
                left_edge, right_edge = None, None
                seam_indices = []
                seam_region = None
        else:
            # Không tìm thấy seam, thử ngưỡng tiếp theo
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


# Ví dụ dùng cho nhiều ảnh
start = 44
count = 56
debug = True
thresholds = [0.7, 0.6, 0.5, 0.4]  # Danh sách các ngưỡng để thử

for i in range(start, count):
    image_path = f'images/{i}.png'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        continue

    result = detect_seam_boundaries_v3(img, thresholds=thresholds, min_seam_width=0.3, max_seam_width=0.9, show=debug,ignore_edges=False,edge_margin=50)

    if result["seam_region"] is not None:
        output_path = f'output_seam_region_{i}.png'
        cv2.imwrite(output_path, result["seam_region"])
        print(f"Ảnh vùng seam đã được lưu tại: {output_path}")
    else:
        print(f"Ảnh {i} không tìm được seam phù hợp với các ngưỡng {thresholds}.")
