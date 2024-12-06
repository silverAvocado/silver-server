# 필요한 라이브러리 임포트
from fastapi import FastAPI, WebSocket, UploadFile, File, APIRouter, WebSocketDisconnect
from collections import deque
import numpy as np
import requests
from tensorflow.keras.models import load_model
import librosa
from router.video import *
import speech_recognition as sr
from resemblyzer import VoiceEncoder
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import cv2
import os

router = APIRouter()

# 모델 및 설정 로드
AUDIO_MODEL_PATH = "./sweet_model.h5"
audio_model = load_model(AUDIO_MODEL_PATH)
SUSPICION_THRESHOLD = 3

# 전역 변수 설정
load_dotenv() 
CAM_SERVER_IP = os.getenv("CAM_SERVER_IP")
CAM_SERVER_URL = f"http://{CAM_SERVER_IP}:9000"
BUFFER_DURATION = 5  
SLIDING_INTERVAL = 1  
SAMPLE_RATE = 16000  

audio_buffer: deque[np.ndarray] = deque(maxlen=int(BUFFER_DURATION * SAMPLE_RATE))
recent_labels = deque(maxlen=30)


# WebSocket 엔드포인트로 모든 음성 처리 통합
@router.websocket("/ws/audio")
async def audio_processing(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨: 실시간 음성 데이터 처리 시작")

    try:
        while True:
            # 클라이언트로부터 음성 데이터 수신
            audio_chunk = await websocket.receive_bytes()
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_buffer.extend(audio_array)

            # 5초 분량의 오디오 데이터가 쌓이면 예측 수행
            if len(audio_buffer) == BUFFER_DURATION * SAMPLE_RATE:
                buffer_array = np.array(audio_buffer)
                
                # MFCC 추출 및 변환
                segments = extract_mfcc_segments(buffer_array, sr=SAMPLE_RATE)
                
                # 각 세그먼트를 모델의 input shape에 맞게 변환하고 예측
                for segment in segments:
                    data_resized = cv2.resize(segment, (128, 128))
                    data_resized = np.expand_dims(data_resized, axis=0)  
                    data_resized = np.expand_dims(data_resized, axis=-1) 
                    
                    # 모델 예측
                    prediction = audio_model.predict(data_resized, verbose=0)
                    predicted_class = int(np.argmax(prediction))
                    prediction_label = "일상" if predicted_class == 0 else "위험"
                    print(f"Prediction class at auidio1: {predicted_class}")

                    # 예측 결과에 따른 처리
                    recent_labels.append(prediction_label)

                    if prediction_label == "위험":
                        label_count = recent_labels.count(prediction_label)
                        if label_count == SUSPICION_THRESHOLD:
                            await websocket.send_json({"status": 200,  "code": 200100, "message": "위험 상황 발생. 연결 종료"})
                            print("위험 상황 발생 - WebSocket 연결 종료")
                            return

                # 슬라이딩 윈도우 방식으로 버퍼 일부 제거 (1초 분량)
                if len(audio_buffer) == int(BUFFER_DURATION * SAMPLE_RATE):
                    del list(audio_buffer)[:int(SLIDING_INTERVAL * SAMPLE_RATE)]

    except WebSocketDisconnect:
        print("WebSocket 연결이 닫혔습니다.")
    finally:
        await websocket.close()

def extract_mfcc_segments(audio_data, sr=16000, duration=5):
    """
    오디오 데이터를 5초씩 잘라서 MFCC 추출
    """
    segments = []
    total_length = librosa.get_duration(y=audio_data, sr=sr)
    
    for start in range(0, int(total_length), duration):
        end = start + duration
        if end <= total_length:
            # 5초 구간 오디오 추출
            y_segment = audio_data[start * sr:end * sr]
            
            # MFCC 추출 (157개 MFCC, 3채널)
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=157)
            mfcc_3ch = np.stack([mfcc] * 3, axis=-1)  
            segments.append(mfcc_3ch)
    
    return segments

# VoiceEncoder 초기화 및 저장 경로 설정
encoder = VoiceEncoder()
REGISTERED_AUDIO_PATH = "registered_user_audio/"
REGISTERED_EMBEDDING_PATH = "registered_user_embedding.npy"

# 사용자 등록 API
@app.post("/register_user_voice")
async def register_user_voice(file: UploadFile = File(...)):
    # 폴더가 없으면 생성
    os.makedirs(REGISTERED_AUDIO_PATH, exist_ok=True)
    
    # 음성을 저장
    audio_path = os.path.join(REGISTERED_AUDIO_PATH, file.filename)
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    # Voice Embedding 생성 및 저장
    wav = encoder.load_audio(audio_path)
    embedding = encoder.embed_utterance(wav)
    np.save(REGISTERED_EMBEDDING_PATH, embedding)
    
    return JSONResponse(
        content={"message": "Voice registered successfully"}, status_code=200
    )

# 위험 상황 처리 API
@app.post("/handle_abnormal_situation_file")
async def handle_abnormal_situation_file(file: UploadFile = File(...)):
    # Voice Embedding 로드
    if not os.path.exists(REGISTERED_EMBEDDING_PATH):
        return JSONResponse(
            content={"code": 200105, "message": "No registered user voice found"},
            status_code=400
        )
    registered_user_embedding = np.load(REGISTERED_EMBEDDING_PATH)
    
    # 녹음 파일 읽기
    audio_path = os.path.join("temp_audio", file.filename)
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    recognizer = sr.Recognizer()
    audio_data = encoder.load_audio(audio_path)
    recognized = False
    try:
        # 음성 인식
        audio = sr.AudioFile(audio_path)
        with audio as source:
            recorded_audio = recognizer.record(source)
        text = recognizer.recognize_google(recorded_audio, language="ko-KR")
        print(f"Recognized Text: {text}")
        
        # 화자 인증
        current_embedding = encoder.embed_utterance(audio_data)
        similarity = np.dot(registered_user_embedding, current_embedding) / (
            np.linalg.norm(registered_user_embedding) * np.linalg.norm(current_embedding)
        )
        print(f"Speaker Similarity: {similarity:.2f}")

        if similarity < 0.75:
            return None

        # 위험 상황 알림
        if "도와줘" in text:
            return JSONResponse(
                content={"code": 200101, "message": "play danger message"}, status_code=200
            )
        elif "괜찮아" in text:
            return JSONResponse(
                content={"code": 200102, "message": "play fine message"}, status_code=200
            )

    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
    except sr.RequestError as e:
        print(f"Google Speech Recognition 서비스에 접근할 수 없습니다. 오류: {e}")

    # 타임아웃 또는 인식 실패 시 무응답 처리
    if not recognized:
        print("10초 동안 응답이 없어 무응답으로 처리합니다.")
        send_line_notify("위험 예상 상황 발생 not recognized - 도움 요청")
        requests.post(f"{CAM_SERVER_URL}/start")
        return JSONResponse(content={"code": 200103, "message": "play no response message"}, status_code=200)
