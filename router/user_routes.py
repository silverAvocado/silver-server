from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

router = APIRouter()

UPLOAD_DIR = Path("uploads")  # 업로드된 파일을 저장할 디렉토리
UPLOAD_DIR.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성

@router.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return JSONResponse(content={"message": "파일 업로드 성공", "file_path": str(file_path)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"파일 업로드 실패: {str(e)}"}, status_code=500)
