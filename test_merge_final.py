import os

import cv2

# from final.mergev_final_ok import detect_seam_boundaries_v3
from final.mergev_final_advence_ok import detect_seam_boundaries_v3

folder_path = 'D:/OtherProjects/n8n-ai/n8n-image/data/resouces/frames'
# get all file in folder
count = 0
for f in os.listdir(folder_path):
    file = os.path.join(folder_path, f)
    if os.path.isfile(file):
        frame = cv2.imread(file)
        result = detect_seam_boundaries_v3(frame, min_seam_width=0.3, max_seam_width=0.9, show=False)
        cropped_frame = frame
        if result["seam_region"] is not None:
            cropped_frame = result["seam_region"]
        cv2.imwrite(f'outputs/frames/frame{count}.jpg', cropped_frame)
        count += 1
