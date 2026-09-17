import cv2 as cv
from ultralytics import YOLO
import time
import numpy as np
from PIL import Image
from transformers import pipeline
import sys

model = YOLO("yolo11n.pt")
depth_model = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", device = 0)

Target_Classes=["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "wall", "barricade"]
confidence_threshold = 0.70
depth_width = 640
depth_height = 360

#temporary approximates of camera and in real world we get these 4 values by calibrating the cam
fx = 700.0
fy=700.0
cx_camera = 384.0
cy_camera = 216.0

# Below input, output, vid of input use these only when processing the recorded videos with this file otherwise comment it 
# input_video = sys.argv[1]
# output_video = sys.argv[2]
# vid = cv.VideoCapture(input_video)

#when using the webcam directly uncomment below commands
vid = cv.VideoCapture(0)
if not vid.isOpened():
    raise RuntimeError("Could not open the video")

fps = vid.get(cv.CAP_PROP_FPS)
if fps<=0:
    fps = 30

# Uncomment these ony when using the recorded video
# frame_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
# frame_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

# cc = cv.VideoWriter_fourcc(*'mp4v')

# writer = cv.VideoWriter(output_video, cc, float(fps), (frame_width, frame_height))

# if not writer.isOpened():
#     raise RuntimeError("could not process")

track_history = {}
max_history = 30
depth_history = {}
max_depth_history = 10
position_history = {}
max_position_history = 10
risk_state = {}
safe_counter = {}
warning_counter = {}
high_counter = {}
velocity_history = {}

dt = 1 / fps

def calculate_risk(track_id, z, ttc, closing_speed, in_collision_path, risk_state, safe_counter, warning_counter, high_counter, confirm_frames=3):
    instant_risk = "SAFE"  
    if not in_collision_path:
        instant_risk = "SAFE"
    elif in_collision_path:
        if z > 7.0:
            instant_risk = "SAFE"
        elif z < 3.0 or ttc < 1.0:
            instant_risk = "HIGH"
        elif z <= 7.0 or ttc < 2.0 or closing_speed > 0.5:
            instant_risk = "WARNING"
        else:
            instant_risk = "SAFE"

    #History
    if track_id not in risk_state:
        risk_state[track_id] = "SAFE"

    if track_id not in safe_counter:
        safe_counter[track_id] = 0
    if track_id not in warning_counter:
            warning_counter[track_id] = 0
    if track_id not in high_counter:
            high_counter[track_id] = 0

    if instant_risk == "HIGH":
        high_counter[track_id] += 1
        warning_counter[track_id] = 0
        safe_counter[track_id]=0

        if high_counter[track_id] >= confirm_frames:
            risk_state[track_id] = "HIGH"

    elif instant_risk == "WARNING":
        high_counter[track_id] = 0
        warning_counter[track_id] += 1
        safe_counter[track_id]=0
        
        if warning_counter[track_id] >= confirm_frames:
            risk_state[track_id] = "WARNING"
    else:
        safe_counter[track_id] += 1
        warning_counter[track_id] = 0
        high_counter[track_id] = 0

        if safe_counter[track_id] >= confirm_frames:
            risk_state[track_id] = "SAFE"
    return risk_state[track_id]

def overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersec = iw * ih

    area_a = max(1, (ax2 - ax1)*(ay2 - ay1))
    area_b = max(1, (bx2 - bx1)*(by2 - by1))

    overlap_a = intersec / area_a
    overlap_b = intersec / area_b

    return overlap_a, overlap_b

while True:
    ret, frame=vid.read()

    if not ret:
        break

    if frame.shape[0] > 0:
        print("Frame:", frame.shape[1], "x", frame.shape[0])

    frame_start_time = time.perf_counter()

    #Depth
    depth_input = cv.resize(frame, (depth_width, depth_height), interpolation=cv.INTER_AREA)
    
    rgb_frame = cv.cvtColor(depth_input, cv.COLOR_BGR2RGB)
    
    rgb_image = Image.fromarray(rgb_frame)
    
    start_time = time.perf_counter()
       
    result = depth_model(rgb_image)
    
    depth_inference_time = time.perf_counter() - start_time
       
    depth = result["predicted_depth"]

    depth = depth.squeeze().detach().cpu().numpy()
       
    depth = np.array(depth)
       
    depth = cv.resize(depth, (frame.shape[1], frame.shape[0]), cv.INTER_LINEAR)

    #Yolo + Byte Track
    start_time = time.perf_counter()
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    inference_time = time.perf_counter() - start_time

    result = results[0]

    boxes = result.boxes

    if boxes.id is None:
        continue

    track_ids = boxes.id.int().cpu().tolist()
    
    detections = []
    detection_count = 0

    for i, box in enumerate(boxes):

        confidence = float(box.conf[0])

        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name not in Target_Classes:
            continue

        # Position of box left top corner to bottom down corner
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        track_id = track_ids[i]
        detection_count += 1
        
        detections.append({
            "index" : i,
            "track_id" : track_id,
            "class_name" : class_name,
            "confidence" : confidence,
            "box" : (x1, y1, x2, y2)            
        })


    # Removal of highly occluded objs
    # overlap_threshold = 0.50
    # suppressed_ids = set()

    # for i in range(len(detections)):
    #     for j in range(i + 1, len(detections)):
    #         if detections[i]["track_id"] == detections[j]["track_id"]:
    #             continue
    #         box_a = detections[i]["box"]
    #         box_b = detections[j]["box"]

    #         overlap_a, overlap_b = overlap_ratio(box_a, box_b)

    #         if overlap_a > overlap_threshold:
    #             suppressed_ids.add(detections[i]["track_id"])
    #         elif overlap_b > overlap_threshold:
    #             suppressed_ids.add(detections[j]["track_id"])

    # detections = [detection for detection in detections 
    #               if detection["track_id"] not in suppressed_ids]
    # detection_count = len(detections)

    # Depth + 3d + ttc + risk
    for detection in detections:
        i = detection["index"]
        track_id = detection["track_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = detection["box"]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        #Depth
        box_width = x2 - x1
        box_height = y2 - y1

        s_x1 = int(x1 + 0.35 * box_width)
        s_x2 = int(x1 + 0.65 * box_width)

        s_y1 = int(y1 + 0.60 * box_height)
        s_y2 = int(y1 + 0.90 * box_height)

        s_x1 = max(0, s_x1)
        s_x2 = min(frame.shape[1], s_x2)

        s_y1 = max(0, s_y1)
        s_y2 = min(frame.shape[0], s_y2)

        depth_region = depth[s_y1:s_y2, s_x1:s_x2]

        if depth_region.size > 0:
            valid_depth = depth_region[np.isfinite(depth_region) & (depth_region > 0.1) & (depth_region < 80.0)]
            if valid_depth.size > 0.0:
                object_depth = float(np.median(valid_depth))
            else:
                object_depth = 0.0
        else:
            object_depth = 0.0

        if track_id not in depth_history:
            depth_history[track_id]=[]

        depth_history[track_id].append(object_depth)

        if len(depth_history[track_id]) > max_depth_history:
            depth_history[track_id].pop(0)

        stable_depth = float(np.median(depth_history[track_id]))

        z = stable_depth
        x = (cx - cx_camera) * z / fx
        y = (cy - cy_camera) * z / fy

        # Position History (x, y, z) of one obj
        if track_id not in position_history:
            position_history[track_id] = []

        position_history[track_id].append((x, y, z))

        if len(position_history[track_id]) > max_position_history:
            position_history[track_id].pop(0)

        # Obj center position (cx, cy)
        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        if len(track_history[track_id]) > max_history:
            track_history[track_id].pop(0)

        if len(position_history[track_id]) >= 2:
            x_prev, y_prev, z_prev = position_history[track_id][-2]

            vx_prev = (x - x_prev) / dt
            vy_prev = (y - y_prev) / dt
            vz_prev = (z - z_prev) / dt

            if track_id not in velocity_history:
                velocity_history[track_id] = {
                    "vx" : 0.0,
                    "vy" : 0.0,
                    "vz" : 0.0
                }

            alpha = 0.45

            vx_3d = alpha * vx_prev + (1 - alpha) * velocity_history[track_id]["vx"]
            vy_3d = alpha * vy_prev + (1 - alpha) * velocity_history[track_id]["vy"]
            vz_3d = alpha * vz_prev + (1 - alpha) * velocity_history[track_id]["vz"]

            velocity_history[track_id]["vx"] = vx_3d
            velocity_history[track_id]["vy"] = vy_3d
            velocity_history[track_id]["vz"] = vz_3d

            speed_3d = np.sqrt(vx_3d**2 + vy_3d**2 + vz_3d**2)

            ttc = float("inf")

            if vz_3d < 0 and z > 0:
                closing_speed = -vz_3d
                ttc = z / closing_speed
            if ttc > 20:
                ttc = float("inf")

        else:
            vx_3d = 0.0
            vy_3d = 0.0
            vz_3d = 0.0
            speed_3d = 0.0
            ttc = np.inf
        #Above velocities are m/s

        closing_speed = max(0.0, -vz_3d)

        # Collision Risk
        lane_half_width = 1.5
        in_collision_path = (z > 0 and abs(x) < lane_half_width)

        risk = calculate_risk(track_id, z, ttc, closing_speed, in_collision_path, risk_state, safe_counter, warning_counter, high_counter)

        if risk == "HIGH":
            box_color = (0, 0, 255)
        elif risk == "WARNING":
            box_color = (0, 255, 255)
        else:
            box_color = (0, 255, 0)

        cv.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        #Box of depth_region
        #cv.rectangle(frame, (s_x1, s_y1), (s_x2, s_y2), (0, 255, 255), 2)
        
        if np.isfinite(ttc):
            ttc_label = f"TTC: {ttc:.1f} s"
        else:
            ttc_label = f"TTC: --"
        label = f"{class_name} D:{stable_depth:.2f}m {ttc_label}"
        #position_label = f"X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f} m"
        closing_label = f"closing:{closing_speed:.1f}m/s"
        risk_label = f"Risk: {risk}"

        cv.putText(frame, closing_label, (x1, max(y1 - 25, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv.putText(frame, label, (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        #cv.putText(frame, position_label, (x1, min(y1 + 40, frame.shape[0] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv.putText(frame, risk_label, (x1, min(y2 - 20, frame.shape[0] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)   

        # velocity_label = f"Vx:{vx_3d:.1f}, Vy:{vy_3d:.1f}, Vz:{vz_3d:.1f} m/s"
        # cv.putText(frame, velocity_label, (x1, min(y1 + 20, frame.shape[0] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
    inference_ms = inference_time * 1000
    total_time = time.perf_counter() - frame_start_time
    pipeline_fps = 1 / total_time if total_time > 0 else 0

    cv.putText(frame, f"Inference: {inference_ms:.1f}ms", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    cv.putText(frame, f"FPS: {pipeline_fps:.1f}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    cv.putText(frame, f"Objects: {detection_count}", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    #Use writer only when using recorded video otherwise keep it in comment
    #writer.write(frame)
    cv.imshow("YOLO + IoU", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
#Use writer only when using recorded video otherwise keep it in comment
#writer.release()
cv.destroyAllWindows()