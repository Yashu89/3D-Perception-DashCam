import cv2 as cv
import numpy as np
import time
from PIL import Image
from transformers import pipeline

import torch

depth_model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device = 0
)

vid = cv.VideoCapture("videos/driving_1.mp4")

if not vid.isOpened():
    raise RuntimeError("Couldn't open video")

depth_width = 640
depth_height = 360

video_fps = vid.get(cv.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 30

while True:

    ret, frame = vid.read()

    if not ret:
        break

    depth_input = cv.resize(frame, (depth_width, depth_height), interpolation=cv.INTER_AREA)

    rgb_frame = cv.cvtColor(depth_input, cv.COLOR_BGR2RGB)

    rgb_image = Image.fromarray(rgb_frame)
    start_time = time.perf_counter()
   
    result = depth_model(rgb_image)

    inference_time = time.perf_counter() - start_time
   
    depth = result["depth"]
   
    depth = np.array(depth)
   
    depth = cv.resize(depth, (frame.shape[1], frame.shape[0]))

    depth_display = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX).astype("uint8")

    inference_ms = inference_time * 1000

    processing_fps = (1 / inference_time if inference_time > 0 else 0)

    cv.putText(frame,f"Depth inference: {inference_ms:.1f} ms",(20, 30),cv.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)

    cv.putText(frame,f"Depth FPS: {processing_fps:.1f}",(20, 60),cv.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)

    cv.imshow("Original",frame)
    cv.imshow("Depth",depth_display)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv.destroyAllWindows()