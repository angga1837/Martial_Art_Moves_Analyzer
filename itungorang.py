import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_pose = mp.solutions.pose
_mp_detector = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

_person_counter = YOLO("yolov8n.pt")        # deteksi untuk hitung berapa orang
_pose_yolo = YOLO("yolov8n-pose.pt")        # multi-person keypoints, pretrained COCO-17

# Mapping 33 titik Mediapipe untuk 17 titik COCO YOLOv8-pose 
MEDIAPIPE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
HIP_L, HIP_R, SHO_L, SHO_R = 11, 12, 5, 6   

def normalize_keypoints(kpts_xy_px, kpts_vis, img_w, img_h):
    xy = kpts_xy_px.astype(np.float32).copy()
    xy[:, 0] /= img_w
    xy[:, 1] /= img_h
    cx = (xy[HIP_L, 0] + xy[HIP_R, 0]) / 2
    cy = (xy[HIP_L, 1] + xy[HIP_R, 1]) / 2
    scale = np.hypot(xy[SHO_L, 0] - xy[SHO_R, 0], xy[SHO_L, 1] - xy[SHO_R, 1]) + 1e-6
    feats = []
    for i in range(17):
        feats.extend([(xy[i, 0] - cx) / scale, (xy[i, 1] - cy) / scale, float(kpts_vis[i])])
    return np.array(feats, dtype=np.float32)  # shape (51,)

def extract_mediapipe(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _mp_detector.process(rgb)
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    xy = np.array([[lm[i].x * w, lm[i].y * h] for i in MEDIAPIPE_TO_COCO17])
    vis = np.array([lm[i].visibility for i in MEDIAPIPE_TO_COCO17])
    return normalize_keypoints(xy, vis, w, h)

def extract_yolo_pose_all(frame):
    h, w = frame.shape[:2]
    results = _pose_yolo(frame, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return []
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else None
    kxy = results.keypoints.xy.cpu().numpy()
    kconf = (results.keypoints.conf.cpu().numpy()
             if results.keypoints.conf is not None else np.ones(kxy.shape[:2]))
    people = []
    for i in range(len(kxy)):
        area = 0.0
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1)
        people.append((area, normalize_keypoints(kxy[i], kconf[i], w, h)))
    people.sort(key=lambda p: -p[0])
    return [p[1] for p in people]

def count_people(frame):
    results = _person_counter(frame, classes=[0], verbose=False)[0]
    return len(results.boxes) if results.boxes is not None else 0


class PoseSwitcher:
    def __init__(self, check_interval=8, hysteresis=3):
        self.check_interval = check_interval
        self.hysteresis = hysteresis
        self.frame_count = 0
        self.mode = "SINGLE"
        self._pending_mode = None
        self._pending_count = 0

    def _update_mode(self, frame):
        self.frame_count += 1
        if self.frame_count % self.check_interval != 0:
            return self.mode
        n_people = count_people(frame)
        proposed = "SINGLE" if n_people <= 1 else "MULTI"
        if proposed == self._pending_mode:
            self._pending_count += 1
        else:
            self._pending_mode = proposed
            self._pending_count = 1
        if self._pending_count >= self.hysteresis and proposed != self.mode:
            self.mode = proposed
        return self.mode
    def get_people(self, frame):
        mode = self._update_mode(frame)
        if mode == "SINGLE":
            feat = extract_mediapipe(frame)
            return mode, ([feat] if feat is not None else [])
        feats = extract_yolo_pose_all(frame)
        return mode, feats[:2]

def isolate_and_extract_mediapipe(frame, padding=40):
    results = _person_counter(frame, classes=[0], verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return extract_mediapipe(frame)
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[np.argmax(areas)]
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1) - padding), max(0, int(y1) - padding)
    x2, y2 = min(w, int(x2) + padding), min(h, int(y2) + padding)
    cropped = frame[y1:y2, x1:x2]
    return extract_mediapipe(cropped) if cropped.size > 0 else None