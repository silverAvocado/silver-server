import json
from datetime import datetime, timedelta
import random

def generate_sample_data_with_intervals(intervals=12, missing_rate=0.2):
    """
    7일치 데이터 생성, 하루당 intervals (12개) 점수 생성.
    일부 데이터를 누락시키는 옵션 포함.
    
    Args:
        intervals (int): 하루에 생성할 점수의 개수 (기본값: 12).
        missing_rate (float): 데이터를 누락시킬 확률 (0.0 ~ 1.0, 기본값: 0.2).
    """
    today = datetime.now()
    data = {}

    for i in range(7):  # 최근 7일치 데이터
        day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        scores = {}

        for j in range(intervals):
            time_slot = f"{j * 2:02}:00"  # 2시간 간격의 시간대 생성
            if random.random() < missing_rate:
                scores[time_slot] = {"score": 0, "status": "insufficient_data"}  # 데이터 부족
            else:
                score = round(random.uniform(0.1, 1.0), 2)
                scores[time_slot] = {"score": score, "status": "ok"}  # 정상 데이터

        data[day] = scores

    with open("activity_data.json", "w") as file:
        json.dump(data, file, indent=4)

    print("7일치 샘플 데이터 생성 완료.")

# 샘플 데이터 생성 실행
generate_sample_data_with_intervals()
