from collections import deque
import cv2
import numpy as np
import torch
import asyncio
import gc
import requests
import mediapipe as mp
from ultralytics import YOLO
from goCam import *
from deepface import DeepFace

LINE_TOKEN = "owva2Uxp1YB1BeKxE31Ji8E1gy7DFwyZwQYd0UKsPRV"
LSTM_MODEL_PATH = "/Users/trispark/summer2024/sweet_guard/server/transformer_augment.keras"
yolo_model = YOLO("yolo11n.pt")
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# LINE 알림 전송 함수
def send_line_notify(message, token=LINE_TOKEN):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': message}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print('메시지 전송 성공')
    else:
        print(f'메시지 전송 실패: {response.status_code}')


# YOLO 객체 탐지 및 포즈 데이터 처리 함수들
def process_frame(frame_bytes):
    frame = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)

def detect_bounding_boxes(frame):
    with torch.no_grad():
        results = yolo_model(frame, verbose=False)
    return [r.boxes.xywh[j].cpu().numpy() for r in results if 0 
            in r.boxes.cls.cpu().numpy() for j, c 
            in enumerate(r.boxes.cls.cpu().numpy()) if c == 0]

def process_pose_data(frame, bounding_boxes):
    frame_to_pose = FrameToPoseArray(frame, bounding_boxes)
    pose_data = frame_to_pose.get_pose_data()
    return pose_data if pose_data.shape == (99, 4) else None

def extract_face_with_mediapipe(cropped_person):
    rgb_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb_person)

    if not results.detections:
        return None
    
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = cropped_person.shape
    x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
    x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
    return cropped_person[y1:y2, x1:x2]

def identify_owner(cropped_person):
    """
    얼굴 인식 로직을 통해 집주인 여부 확인.
    """
    # Mediapipe로 얼굴 탐지
    cropped_face = extract_face_with_mediapipe(cropped_person)
    if cropped_face is None:
        print("얼굴 탐지 실패")
        return False

    # DeepFace로 얼굴 인식
    try:
        result = DeepFace.verify(img1_path=cropped_face, img2_path="owner_image.jpg", model_name="ArcFace")
        return result["verified"]  # 집주인 여부 반환
    except Exception as e:
        print(f"얼굴 인식 실패: {e}")
        return False

class FrameToPoseArray:
    def __init__(self, frame, bounding_boxes):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.frame = frame
        self.bounding_boxes = bounding_boxes
        self.cropped_images = []
        self.pose_array = []

    def crop_images(self):
        for bbox in self.bounding_boxes:
            cx, cy, w, h = map(int, bbox)
            x1, y1 = cx - w // 2, cy - h // 2 # 좌상단
            x2, y2 = cx + w // 2, cy + h // 2 # 우하단
            self.cropped_images.append((self.frame[y1:y2, x1:x2], (x1, y1, w, h))) # cropped frame & bbox 좌표

    def extract_pose_landmarks(self):
        for cropped_image, bbox in self.cropped_images:
            x1, y1, w, h = bbox
            result = self.pose.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                # normalized landmarks -> convert to original landmarks (using bbox)
                original_landmarks = []
                for lm in result.pose_landmarks.landmark:
                    x_original = lm.x * w + x1
                    y_original = lm.y * h + y1
                    z_original = lm.z  # depth는 same
                    original_landmarks.append([x_original, y_original, z_original, lm.visibility])
                self.pose_array.append(np.array(original_landmarks))
            else:
                self.pose_array.append(np.zeros((33, 4)))

    def pad_pose_data(self):
        while len(self.pose_array) < 3:
            self.pose_array.append(np.zeros((33, 4)))
        return np.array(self.pose_array[:3])

    def get_pose_data(self):
        self.crop_images()
        self.extract_pose_landmarks()
        self.pad_pose_data()
        return np.concatenate(self.pose_array)