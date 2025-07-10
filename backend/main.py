from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 추후 필요시 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML/JS/CSS 정적 파일 경로 설정
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# --- API 부분은 기존 코드 유지 ---
@app.post("/api/recommend-platform")
async def recommend_platform(image: UploadFile = File(...)):
    # 이미지 분석 처리
    return {"result": "API 동작 확인 완료"}

# 로컬 실행용 (Cloud Run에는 필요 없음)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
