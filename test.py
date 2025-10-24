import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "D:/OtherProjects/n8n-ai/n8n-image/data/resouces/query/input.mp4"

# cap = cv2.VideoCapture(video_path)

# ret,f = cap.read()
# cv2.imwrite("test.png",f)


import ffmpeg
import numpy as np
def extract_video_frames(uri, start_time=None, end_time=None, fps=1):
    probe = ffmpeg.probe(uri)
    width = int(probe['streams'][0]['width'])
    height = int(probe['streams'][0]['height'])

    input_kwargs = {}
    if start_time is not None:
        input_kwargs['ss'] = start_time
    if end_time is not None:
        input_kwargs['to'] = end_time

    process = (
        ffmpeg
        .input(uri, **input_kwargs)
        .filter('fps', fps=fps)   # ✅ đặt filter fps ở đây
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame
        
for i, frame in enumerate(extract_video_frames(video_path, fps=1)):
    plt.imshow(frame)
    plt.axis('off')
    plt.savefig(f'frame_{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close()