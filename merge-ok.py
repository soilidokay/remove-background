import cv2
import numpy as np

def detect_seam_boundaries_v2(image, threshold=0.4, max_width=1200, show=True):
    """
    Phát hiện ranh giới seam giữa hai vùng trong ảnh panorama bằng cách phân tích độ biến đổi theo chiều ngang.
    
    Tham số:
        image: ảnh đầu vào dạng NumPy (BGR)
        threshold: ngưỡng xác định seam
        max_width: resize ảnh khi hiển thị
        show: True để hiển thị từng bước bằng cv2.imshow

    Trả về:
        {
            "left_edge": int,
            "right_edge": int,
            "seam_indices": list[int],
            "seam_region": ảnh cắt vùng seam hoặc None,
            "result_image": ảnh có vẽ seam + biên
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Làm mờ nhẹ để giảm nhiễu

    # Tính gradient magnitude (Sobel kết hợp X và Y)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = np.uint8(255 * gradient_mag / np.max(gradient_mag))

    # Tổng độ biến thiên theo cột
    column_energy = gradient_mag.sum(axis=0)
    energy_norm = (column_energy - column_energy.min()) / np.ptp(column_energy)

    # Nhị phân hóa + đóng lỗ để nối seam liền kề
    binary_mask = (energy_norm > threshold).astype(np.uint8)
    kernel = np.ones((1, 15), np.uint8)
    closed = cv2.morphologyEx(binary_mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
    seam_indices = np.where(closed > 0)[0]

    seam_vis = image.copy()
    for x in seam_indices:
        cv2.line(seam_vis, (x, 0), (x, seam_vis.shape[0]), (0, 0, 255), 1)

    result = seam_vis.copy()
    seam_region = None
    left_edge, right_edge = None, None

    if len(seam_indices) > 0:
        left_edge = int(seam_indices.min())
        right_edge = int(seam_indices.max())
        cv2.line(result, (left_edge, 0), (left_edge, result.shape[0]), (0, 255, 0), 2)
        cv2.line(result, (right_edge, 0), (right_edge, result.shape[0]), (0, 255, 0), 2)
        seam_region = image[:, left_edge:right_edge]

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
        "seam_indices": seam_indices.tolist(),
        "seam_region": seam_region,
        "result_image": result
    }
start =48
count = 56
debug = True
for i in range(start, count):
    image_path = f'images/{i}.png'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        continue

    result = detect_seam_boundaries_v2(img, show=debug)

    if result["seam_region"] is not None:
        output_path = f'output_seam_region_{i}.png'
        cv2.imwrite(output_path, result["seam_region"])
        print(f"Ảnh vùng seam đã được lưu tại: {output_path}")