import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "D:/OtherProjects/n8n-ai/n8n-image/data/resouces/query/input.mp4"

cap = cv2.VideoCapture(video_path)

ret,f = cap.read()
cv2.imwrite("test.png",f)