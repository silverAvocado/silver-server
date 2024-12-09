from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json
from jinja2 import Template
from activity_analysis import analyze_periodic_data, normalize_scores
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import numpy as np


router = APIRouter()

@router.get("/dashboard/daily", response_class=HTMLResponse)
async def daily_dashboard():
    """
    하루 활동성 점수를 시각화하고 정보 제공.
    """
    try:
        with open("activity_data.json", "r") as file:
            daily_data = json.load(file)

        current_day = datetime.now().strftime("%Y-%m-%d")
        daily_scores = daily_data.get(current_day, {})

        if not daily_scores:
            return HTMLResponse(content="<h1>오늘 데이터가 없습니다.</h1>", status_code=404)

        # 시간대별 점수와 상태 처리
        hours = []
        normalized_scores = []
        missing_data_hours = []
        no_movement_hours = []

        for time_slot, entry in daily_scores.items():
            hours.append(time_slot)
            if entry["status"] == "insufficient_data":
                normalized_scores.append(0)  # 데이터 부족 시 0으로 표시
                missing_data_hours.append(time_slot)
            elif entry["status"] == "ok" and entry["score"] == 0:
                no_movement_hours.append(time_slot)
                normalized_scores.append(entry["score"])
            else:
                normalized_scores.append(entry["score"])
        
        # x축 라벨 색상 설정
        xaxis_label_colors = [
            "red" if hour in no_movement_hours else
            "green" if hour in missing_data_hours else
            "black"
            for hour in hours
        ]

        # Plotly 그래프 생성
        fig = go.Figure()

        # 활동성 점수 막대 추가
        fig.add_trace(
            go.Bar(
                x=hours,
                y=normalized_scores,
                marker_color="skyblue",
                name="활동성 점수",
            )
        )

        fig.update_layout(
            title="하루 활동성 점수",
            xaxis_title="시간대",
            yaxis_title="점수",
            margin=dict(l=40, r=40, t=60, b=80),
            legend=dict(
                title="범례",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.3,
            ),
            plot_bgcolor="rgba(240,240,240,0.8)",
        )

        # x축 라벨 색상 업데이트
        fig.update_xaxes(
            tickvals=hours,
            ticktext=[
                f"<span style='color:{color}'>{hour}</span>"
                for hour, color in zip(hours, xaxis_label_colors)
            ]
        )
        
        # 툴바 숨기기
        config = {
            "displayModeBar": False,  # 툴바 비활성화
            "responsive": True        # 반응형 그래프
        }

        # HTML 템플릿 로드
        template_path = Path("templates/daily_dashboard.html")
        with open(template_path, "r", encoding="utf-8") as template_file:
            template = Template(template_file.read())

        # 평균 점수 계산
        avg_score = round(
            sum(score for score in normalized_scores if score > 0) / len(
                [score for score in normalized_scores if score > 0]
            ),
            2,
        ) if len([score for score in normalized_scores if score > 0]) > 0 else 0

        html_content = template.render(
            chart_html=fig.to_html(full_html=False, config=config),
            avg_score=avg_score,
            no_movement_hours=no_movement_hours,
            missing_data_hours=missing_data_hours,
            current_day=current_day,
        )
        return HTMLResponse(html_content)

    except Exception as e:
        return HTMLResponse(content=f"<h1>에러 발생</h1><p>{str(e)}</p>", status_code=500)
    
from datetime import datetime

@router.get("/dashboard/weekly", response_class=HTMLResponse)
async def weekly_dashboard():
    """
    최근 7일간 활동성 점수를 시각화하고 분석 결과를 제공.
    """
    try:
        # 데이터 로드
        with open("activity_data.json", "r") as file:
            daily_data = json.load(file)

        # 데이터 분석
        analysis = analyze_periodic_data(daily_data)
        weekly_scores = analysis["normalized_weekly"]

        # 숫자 변환 (문자열로 저장된 경우 처리)
        weekly_scores = [float(score) for score in weekly_scores if isinstance(score, (int, float, str))]

        days = list(daily_data.keys())[-7:]  # 최신 7일 가져오기
        days.sort()  # 날짜 순서 정렬
        avg_score = sum(weekly_scores) / len(weekly_scores) if weekly_scores else 0

        # 시작 날짜와 종료 날짜 계산
        start_date = datetime.strptime(days[0], "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(days[-1], "%Y-%m-%d").strftime("%Y-%m-%d")

        # 날짜를 x축에 숫자로 표시하기 위해 날짜 형식 변환
        x_labels = [datetime.strptime(day, "%Y-%m-%d").strftime("%m%d") for day in days]

        # 최고 점수의 날짜 계산
        max_index = weekly_scores.index(max(weekly_scores))
        highest_score_date = days[max_index]

        # 변동성 분석: 표준편차 계산
        if len(weekly_scores) > 1:
            std_dev = round(np.std(weekly_scores), 2)  # 표준편차
            volatility_message = f"이번 주 활동성의 표준편차는 {std_dev}로, 활동성이 {'안정적' if std_dev < 0.2 else '변동성이 큽니다'}."
        else:
            std_dev = 0
            volatility_message = "일주일치 데이터가 부족하여 변동성을 분석할 수 없습니다."

        # Plotly로 시각화 생성
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_labels,  # x축을 숫자 형식으로 표시
                    y=weekly_scores,
                    mode="lines+markers",
                    line=dict(color="green", width=2.5),
                    marker=dict(size=8, color="darkgreen"),
                )
            ]
        )
        fig.update_layout(
            title="최근 7일 활동성 점수",
            xaxis_title="날짜",
            yaxis_title="정규화된 점수 (0 ~ 1)",
            xaxis=dict(tickangle=0),
            plot_bgcolor="rgba(240,240,240,0.8)",
            font=dict(family="Arial", size=12),
            autosize=True,  # 그래프 크기를 자동으로 조정
            margin=dict(l=10, r=10, t=50, b=50),  # 여백 최소화
        )

        config = {
            "displayModeBar": False,  # 툴바 비활성화
            "responsive": True        # 반응형 그래프
        }

        # HTML 템플릿 로드
        template_path = Path("templates/weekly_dashboard.html")
        with open(template_path, "r", encoding="utf-8") as template_file:
            template = Template(template_file.read())

        # HTML 렌더링
        html_content = template.render(
            chart_html=fig.to_html(full_html=False, config=config),
            avg_score=round(avg_score, 2),
            highest_score_date=highest_score_date,
            volatility_message=volatility_message,
            start_date=start_date,
            end_date=end_date,
        )

        return HTMLResponse(html_content)

    except Exception as e:
        return HTMLResponse(content=f"<h1>에러 발생</h1><p>{str(e)}</p>", status_code=500)
