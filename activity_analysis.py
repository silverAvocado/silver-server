import numpy as np
import json
from datetime import datetime

def calculate_activity_score(owner_pose_data):
    """
    관절 데이터를 기반으로 활동성 점수를 계산합니다.
    """
    owner_pose_data = np.array(owner_pose_data)
    if owner_pose_data.shape[0] < 2:
        # 데이터 부족 시 처리
        print("데이터가 부족하여 점수를 계산할 수 없습니다.")
        return None  # None 반환으로 명확히 구분

    movement = np.linalg.norm(owner_pose_data[1:] - owner_pose_data[:-1], axis=(1, 2))
    return movement.sum()

from datetime import datetime

def save_daily_score(date, time_slot, activity_score=None):
    """
    하루 활동성 점수를 저장합니다. 데이터 부족 시 '데이터 없음'으로 기록.
    
    Args:
        date (str): 날짜 (YYYY-MM-DD 형식).
        time_slot (str): 시간대 (예: "14:00").
        activity_score (float, optional): 저장할 활동성 점수. 없으면 데이터 부족으로 기록.
    """
    try:
        with open("activity_data.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # 날짜 초기화
    if date not in data:
        data[date] = {}

    # 시간대별 데이터 저장
    if activity_score is None:
        data[date][time_slot] = {"score": 0, "status": "insufficient_data"}
    else:
        data[date][time_slot] = {"score": activity_score, "status": "ok"}

    with open("activity_data.json", "w") as file:
        json.dump(data, file, indent=4)


def normalize_scores(scores):
    """
    입력된 점수를 0~1 범위로 정규화.
    """
    if not scores:  # 점수가 없을 경우
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:  # 모든 점수가 동일한 경우
        return [0.5 for _ in scores]  # 중간값 반환

    return [(score - min_score) / (max_score - min_score) for score in scores]

def analyze_periodic_data(daily_data):
    """
    하루, 3일, 7일 데이터를 분석하여 평균 및 정규화된 데이터를 반환.
    """
    current_day = datetime.now().strftime("%Y-%m-%d")
    current_day_data = daily_data.get(current_day, {})

    # 오늘의 데이터에서 "ok" 상태의 점수만 추출
    current_day_scores = [
        entry["score"] for entry in current_day_data.values() if entry["status"] == "ok"
    ]

    if not current_day_scores:
        return {"status": "error", "message": "오늘 데이터가 없습니다."}

    # 하루 평균 점수 계산
    current_day_avg = sum(current_day_scores) / len(current_day_scores)

    # 최근 3일 및 7일 데이터 처리
    last_3_days_data = list(daily_data.values())[-3:]
    last_7_days_data = list(daily_data.values())[-7:]

    def extract_scores(data):
        """데이터에서 점수를 추출하여 평균을 계산."""
        scores = [
            entry["score"]
            for day in data
            for entry in day.values()
            if entry["status"] == "ok"
        ]
        return scores

    # 3일 및 7일 점수 추출
    last_3_days_scores = extract_scores(last_3_days_data)
    last_7_days_scores = extract_scores(last_7_days_data)

    # 평균 점수 계산
    three_day_avg = (
        sum(last_3_days_scores) / len(last_3_days_scores)
        if last_3_days_scores
        else 0
    )
    seven_day_avg = (
        sum(last_7_days_scores) / len(last_7_days_scores)
        if last_7_days_scores
        else 0
    )

    # 정규화된 점수 계산 (7일치 데이터만 사용)
    normalized_today = normalize_scores(current_day_scores)
    normalized_weekly = normalize_scores(
        [
            sum(entry["score"] for entry in day.values() if entry["status"] == "ok")
            / len([entry for entry in day.values() if entry["status"] == "ok"])
            for day in last_7_days_data
            if len([entry for entry in day.values() if entry["status"] == "ok"]) > 0
        ]
    )

    # 현재 점수와 3일 평균 점수 비교
    comparison = compare_scores(current_day_avg, three_day_avg)

    return {
        "current_day_avg": current_day_avg,
        "three_day_avg": three_day_avg,  # 알림에 사용
        "seven_day_avg": seven_day_avg,
        "comparison": comparison,  # 알림에 사용
        "normalized_today": normalized_today,
        "normalized_weekly": normalized_weekly,  # 7일치 그래프에 사용
    }




def compare_scores(current_score, average_score):
    """
    현재 점수를 기준으로 증가/감소 여부를 판단.
    """
    if current_score < average_score * 0.8:  # 20% 이상 감소
        return {"status": "decreased", "message": "활동성이 감소했습니다. 활동을 추천합니다."}
    elif current_score > average_score * 1.2:  # 20% 이상 증가
        return {"status": "increased", "message": "활동성이 증가했습니다."}
    return {"status": "stable", "message": "활동성이 안정적입니다."}
