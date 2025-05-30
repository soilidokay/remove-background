import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và làm mờ
img = cv2.imread('images/52.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Tính đạo hàm theo X và Y bằng Sobel
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Tính gradient magnitude
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude_int = magnitude.astype(int)
np.savetxt("csv/gradient_magnitude_int52.csv", magnitude_int, fmt='%d', delimiter=",")

# Chuẩn hóa về [0, 255] để dễ xem
magnitude_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude_display = magnitude_display.astype(np.uint8)

# Hiển thị hoặc lưu
cv2.imshow('Gradient Magnitude', magnitude_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
