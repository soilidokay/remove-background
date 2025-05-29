import cv2
import numpy as np

def remove_multicolor_padding(image_path, output_path=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dò cạnh
    edges = cv2.Canny(gray, 30, 100)
    
    # Đóng các lỗ nhỏ
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Tìm các điểm có cạnh
    coords = cv2.findNonZero(edges)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    if output_path:
        cv2.imwrite(output_path, cropped)
    return cropped

# Sử dụng:
remove_multicolor_padding('images/51.png', 'output.51.png')
# remove_multicolor_padding('images/50.png', 'output.50.png')
# remove_multicolor_padding('images/49.png', 'output.49.png')
# remove_multicolor_padding('images/48.jpg', 'output.48.jpg')
# remove_multicolor_padding('images/Screenshot 2025-05-29 210148.png', 'output.jpg')