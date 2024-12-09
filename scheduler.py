from apscheduler.schedulers.background import BackgroundScheduler
from goCam_act import start_stream
from fastapi import BackgroundTasks
import logging

scheduler = BackgroundScheduler()

# 2시간마다 실행할 작업
def periodic_stream_task():
    print("2시간마다 활동성 분석을 시작합니다.")
    start_stream(background_tasks=BackgroundTasks(), source="scheduled")  # 활동성 분석 시작

# 스케줄러 초기화
def init_scheduler():
    try:
        if not scheduler.get_jobs():
            scheduler.add_job(periodic_stream_task, 'interval', hours=2)  # 2시간 간격
            scheduler.start()
            logging.info("스케줄러가 시작되었습니다.")
    except Exception as e:
        logging.error(f"스케줄러 초기화 실패: {e}")


# 스케줄러 중지
def shutdown_scheduler():
    try:
        scheduler.shutdown()
        logging.info("스케줄러가 중지되었습니다.")
    except Exception as e:
        logging.error(f"스케줄러 중지 실패: {e}")
