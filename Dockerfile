FROM python:3.9-slim

# 시스템 패키지 설치 (TensorFlow, Pillow 등 사용 시 필수)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

# 앱 소스 복사
COPY . .

# 포트 환경 변수 (Cloud Run 호환)
ENV PORT=8080

# 앱 실행 (Gunicorn + uvicorn 워커 방식)
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
