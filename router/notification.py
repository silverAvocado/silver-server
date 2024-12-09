from fastapi import APIRouter
from fastapi.responses import JSONResponse

import json
from activity_analysis import analyze_periodic_data

router = APIRouter()

@router.get("/notification", response_class=JSONResponse)
async def notify_activity_change():
    """
    3일치 평균과 오늘 데이터를 비교하여 급격한 변화를 알립니다.
    """
    try:
        # 활동 데이터 로드
        with open("activity_data.json", "r") as file:
            daily_data = json.load(file)

        # 데이터 분석
        analysis = analyze_periodic_data(daily_data)
        comparison = analysis["comparison"]

        if comparison["status"] in ["decreased", "increased"]:
            # 급격한 변화 알림 생성
            message = comparison["message"]
            print(f"알림: {message}")
            return JSONResponse(content={"status": "alert", "message": message})

        return JSONResponse(content={"status": "stable", "message": "활동성이 안정적입니다."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})