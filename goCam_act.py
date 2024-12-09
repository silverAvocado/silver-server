import cv2
from fastapi import FastAPI, BackgroundTasks
from threading import Thread
from collections import deque
from video_utils_act import * 
from activity_analysis import *

import numpy as np
import gc
import torch
import requests
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os

app = FastAPI()
is_streaming = False
streaming_task = None
is_scheduled_stream = False  # 활동성 분석을 위한 로직인지

load_dotenv() 
MAIN_SERVER_IP = os.getenv("MAIN_SERVER_IP")
MAIN_SERVER_URL = f"http://{MAIN_SERVER_IP}:8000/get_predict"

SEQUENCE_LENGTH = 80
LABELS = ["daily", "violence", "falldown"]
SUSPICION_THRESHOLD = 5
EMERGENCY_THRESHOLD = 10

TRANSFORMER_MODEL_PATH = "./guard_model.keras"
transformer_model = load_model(TRANSFORMER_MODEL_PATH)

recent_labels = deque(maxlen=100)
notification_sent = {"violence": False, "falldown": False}

def reset_notifications():
    global notification_sent
    notification_sent = {key: False for key in notification_sent.keys()}

# 모델 및 예측 함수
def predict_with_model(input):
    pred_idx = np.argmax(transformer_model.predict([input], verbose=0))
    return LABELS[pred_idx]

def run_stream():
    global is_streaming, is_scheduled_stream
    cap = cv2.VideoCapture(1)  # 컴퓨터와 연결된 USB 카메라
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    prediction_label = "Frame incoming"
    start_time = time.time()
    activity_frames = [] # 5초마다 저장할 프레임 리스트
    last_saved_time = start_time  # 마지막으로 저장한 프레임 시간

    while is_streaming and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 3분 간의 데이터로 활동성 분석
        if is_scheduled_stream and time.time() - start_time > 180:
            print("3분 경과: 스트림 종료")
            break

        # 바운딩 박스 및 포즈 데이터 추출
        bounding_boxes = detect_bounding_boxes(frame)
        if bounding_boxes:
            pose_data = process_pose_data(frame, bounding_boxes)
            if pose_data is not None:
                flattened_pose = pose_data.reshape(-1)
                sequence_data.append(flattened_pose)

            if len(sequence_data) == SEQUENCE_LENGTH:
                input_data = np.array(sequence_data).reshape(1, SEQUENCE_LENGTH, 396)

                prediction_label = predict_with_model(input_data)
                recent_labels.append(prediction_label)

                try:
                    requests.post(MAIN_SERVER_URL, json={"prediction": prediction_label})
                except Exception as e:
                    print(f"Error sending prediction: {e}")

                # 상황 처리 로직 및 노티 전송 관리
                if prediction_label in ["violence", "falldown"]:
                    handle_emergency(prediction_label)

        # 5초마다 프레임 선택
        current_time = time.time()
        if is_scheduled_stream and current_time - last_saved_time >= 5:  # 5초 경과
            activity_frames.append(frame.copy())
            last_saved_time = current_time

        # 프레임에 예측 결과 표시
        if frame is not None:
            cv2.putText(
                frame,
                f"Prediction: {prediction_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("수신된 영상", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("영상 표시 종료")
                print("영상 표시 종료")
                break

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

     # 활동성 분석 실행
    if is_scheduled_stream:
        if notification_sent["violence"]:  # 폭력 알림이 전송된 경우
            print("폭력 알림이 전송되어 활동성 분석을 생략합니다.")
        else:  # 폭력 알림이 아닌 경우 (낙상 포함)
            print("폭력 알림이 없으므로 활동성 분석을 실행합니다.")
            analyze_saved_frames(activity_frames)

def handle_emergency(prediction_label):
    if not notification_sent[prediction_label]:  
        if recent_labels.count(prediction_label) >= SUSPICION_THRESHOLD:
            send_line_notify(f"{prediction_label} 의심 상황 발생")
            notification_sent[prediction_label] = True
        if recent_labels.count(prediction_label) >= EMERGENCY_THRESHOLD:
            send_line_notify(f"{prediction_label} 상황 발생. 즉시 신고")
            notification_sent[prediction_label] = True

# 활동성 분석을 위한 함수
def analyze_saved_frames(activity_frames):
    owner_pose_data = []  # 집주인의 관절 데이터
    social_activity_weight = 0  # 사회적 활동 가중치

    for frame in activity_frames:
        bounding_boxes = detect_bounding_boxes(frame)
        if not bounding_boxes:
            print("사람 없음: 분석 제외")
            continue

        # 얼굴 인식 및 집주인 확인
        for bbox in bounding_boxes:
            cx, cy, w, h = map(int, bbox)
            x1, y1 = cx - w // 2, cy - h // 2 # 좌상단
            x2, y2 = cx + w // 2, cy + h // 2 # 우하단
            cropped_person = frame[y1:y2, x1:x2]

            if identify_owner(cropped_person):  # 집주인 식별
                pose_data = process_pose_data(frame, [bbox])
                if(len(bounding_boxes) > 1):
                    social_activity_weight += 0.1

                if pose_data is not None:
                    owner_pose_data.append(pose_data)
                    break

    # 활동성 점수 계산
    if owner_pose_data:
        activity_score = calculate_activity_score(owner_pose_data)
        if activity_score is None:
            print("활동성 점수 계산 불가. 데이터 부족.")
            return

        activity_score += social_activity_weight  # 가중치 추가
        print(f"활동성 점수: {activity_score:.2f}")

        # 날짜와 시간대별로 저장
        current_date = datetime.now().strftime("%Y-%m-%d")  
        current_time_slot = datetime.now().strftime("%H:%M")

        # 점수 저장 (점수가 없을 경우 None 전달)
        save_daily_score(current_date, current_time_slot, activity_score)
        
        if activity_score == 0:
            send_loneliness_alert(current_date)  # 고독사 위험 경고

        post_activity_score(activity_score)
    else:
        print("집주인을 찾지 못해 활동성 분석을 수행할 수 없습니다.")
def send_loneliness_alert(date):
    """
    고독사 위험 경고를 LINE 메시지로 전송.
    """
    message = f"[경고] {date}일 하루 동안 활동성이 전혀 감지되지 않았습니다. 고독사 위험이 우려됩니다."
    try:
        send_line_notify(message)
    except Exception as e:
        print(f"LINE 메시지 전송 실패: {e}")
def post_activity_score(activity_score):
    """
    활동성 점수를 서버에 전송.
    """
    try:
        url = "http://example.com/api/activity_score"  # 서버 URL
        data = {"activity_score": activity_score}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            print("활동성 점수 전송 성공")
        else:
            print(f"활동성 점수 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"서버 전송 중 오류 발생: {e}")

@app.post("/start")
def start_stream(background_tasks: BackgroundTasks, source="voice"):
    global is_streaming, streaming_task, sequence_data, is_scheduled_stream

    if not is_streaming:
        is_streaming = True
        is_scheduled_stream = (source == "scheduled")
        reset_notifications()  # 스트림 시작 시 노티 플래그 초기화

        # deque 초기화
        sequence_data = deque(maxlen=SEQUENCE_LENGTH)

        # 스트림을 새로 시작
        streaming_task = Thread(target=run_stream)
        streaming_task.start()
        return {"message": "스트림 시작됨"}
    else:
        return {"message": "스트림이 이미 실행 중입니다."}

# 스트림 중지 API 엔드포인트
@app.post("/stop")
def stop_stream():
    global is_streaming
    if is_streaming:
        is_streaming = False
        return {"message": "스트림 중지됨"}
    else:
        return {"message": "스트림이 실행 중이 아닙니다."}

    
@app.get("/check")
def check():
    if is_streaming:
        return {"state": True }
    else:
        return {"state":False}

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)