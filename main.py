from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import io, os, random

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="."), name="static")

# favicon.ico 요청 처리 (500 방지)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico", media_type="image/x-icon")
    return Response(content="", media_type="image/x-icon")

# 루트 경로 → index.html 또는 대체 메시지
@app.get("/", include_in_schema=False)
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return JSONResponse({"message": "서비스 준비 완료"}, status_code=200)

# Lazy-load 모델 설정
model = None
def get_model():
    global model
    if model is None:
        try:
            model = tf.keras.applications.MobileNetV2(weights="imagenet")
        except Exception as e:
            print("모델 로딩 실패:", str(e))
            raise RuntimeError("모델 로딩에 실패했습니다.")
    return model

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 파일을 불러올 수 없습니다.")
    img = img.resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    return np.expand_dims(arr, axis=0)

LABEL_INFO = {
    "coffee_mug": {"feature": "머그컵으로 일상에서 자주 사용하는 리빙 소품입니다.", "ko_name": "머그컵"},
    "cup": {"feature": "컵으로 다양한 음료를 담을 수 있는 실용적인 제품입니다.", "ko_name": "컵"},
    "laptop": {"feature": "노트북으로 휴대성과 성능을 모두 갖춘 전자기기입니다.", "ko_name": "노트북"},
    "cellular_telephone": {"feature": "스마트폰 등 휴대전화로 현대인의 필수품입니다.", "ko_name": "휴대전화"},
    "swimming_trunks": {"feature": "수영할 때 입는 남성용 수영복입니다.", "ko_name": "수영복(남성용)"}
}

def label_to_korean(label):
    return LABEL_INFO.get(label, {}).get("ko_name", label.replace('_', ' '))

def rgb_to_color_name(rgb):
    r, g, b = rgb
    if max(rgb) - min(rgb) < 20:
        mean_val = np.mean(rgb)
        return "하얀색" if mean_val > 200 else "검은색" if mean_val < 50 else "회색"
    if r > 200 and g > 200 and b < 100: return "노란색"
    if r > 200 and g < 100 and b < 100: return "빨간색"
    if r < 100 and g > 200 and b < 100: return "초록색"
    if r < 100 and g < 100 and b > 200: return "파란색"
    if r > 200 and g < 100 and b > 200: return "분홍색"
    if r > 200 and g > 100 and b > 100: return "베이지색"
    return "기타"

def guess_material(rgb):
    mean_val = np.mean(rgb)
    if mean_val > 200: return "플라스틱/세라믹"
    if mean_val < 60: return "금속/고무"
    if all(abs(rgb[i] - rgb[i + 1]) < 20 for i in range(2)) and 100 < mean_val < 200:
        return "천/패브릭"
    return "복합/기타"

def analyze_design(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((100, 100))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 분석 실패")
    avg_color = np.mean(np.array(img).reshape(-1, 3), axis=0)
    return {
        "main_color": rgb_to_color_name(avg_color),
        "material": guess_material(avg_color)
    }

@app.post("/api/recommend-platform")
async def recommend_platform(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        input_tensor = preprocess_image(image_bytes)
        preds = get_model().predict(input_tensor)
        label = decode_predictions(preds, top=1)[0][0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

    label_ko = label_to_korean(label)
    info = LABEL_INFO.get(label, {"feature": f"{label_ko}(으)로 분류된 제품입니다."})
    design_info = analyze_design(image_bytes)
    suitability = {k: random.randint(50, 100) for k in ["와디즈", "킥스타터", "마쿠아게", "젝젝"]}

    # 추천 로직
    if "mug" in label or "cup" in label:
        platform, category, reason = "와디즈", "리빙 소품", "머그컵 등 리빙 제품은 국내 플랫폼에 적합합니다."
    elif "laptop" in label or "cellular" in label:
        platform, category, reason = "킥스타터", "테크/디자인", "테크 제품은 글로벌 시장에 적합합니다."
    else:
        platform, category, reason = "젝젝", "라이프스타일", f"'{label_ko}' 관련 제품은 동남아 시장에 적합합니다."

    return {
        "platform": platform,
        "category": category,
        "reason": reason,
        "feature": info["feature"],
        "label_ko": label_ko,
        "design": design_info,
        "suitability": suitability
    }

# Cloud Run에서는 __main__ 실행 안 해도 됨
