FROM python:3.9-slim

# 시스템 패키지 설치 (TensorFlow, Pillow, torch, transformers 실행을 위해 필수)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

# 코드 복사
COPY . .

# 포트 설정 (Cloud Run이 8080 포트를 자동 탐지)
ENV PORT=8080

# Gunicorn 실행 (멀티 워커 + uvicorn으로 안정적 운영)
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
