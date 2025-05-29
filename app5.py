import cv2
import numpy as np

def crop_largest_contour(image_path, output_path=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("Không tìm thấy vùng nội dung.")
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y+h, x:x+w]
    if output_path:
        cv2.imwrite(output_path, cropped)
    return cropped

# Sử dụng:
crop_largest_contour('images/Screenshot 2025-05-29 210148.png', 'output.jpg')