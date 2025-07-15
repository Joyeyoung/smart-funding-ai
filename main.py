from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import io, os

# HuggingFace 번역기용
from transformers import MarianMTModel, MarianTokenizer

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (모든 출처 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 정적 파일 제공 (index.html, favicon 등)
app.mount("/static", StaticFiles(directory="."), name="static")

# favicon.ico 요청 처리
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico", media_type="image/x-icon")
    return Response(content="", media_type="image/x-icon")

# 루트 경로 처리
@app.get("/", include_in_schema=False)
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return JSONResponse({"message": "서비스 준비 완료"}, status_code=200)

# 모델 로딩 (MobileNetV2 사용)
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

# 이미지 전처리 함수
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 파일을 불러올 수 없습니다.")
    img = img.resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    return np.expand_dims(arr, axis=0)

# 분류 라벨에 대한 한글 정보
LABEL_INFO = {
    "coffee_mug": {"feature": "머그컵으로 일상에서 자주 사용하는 리빙 소품입니다.", "ko_name": "머그컵"},
    "cup": {"feature": "컵으로 다양한 음료를 담을 수 있는 실용적인 제품입니다.", "ko_name": "컵"},
    "laptop": {"feature": "노트북으로 휴대성과 성능을 모두 갖춘 전자기기입니다.", "ko_name": "노트북"},
    "cellular_telephone": {"feature": "스마트폰 등 휴대전화로 현대인의 필수품입니다.", "ko_name": "휴대전화"},
    "swimming_trunks": {"feature": "수영할 때 입는 남성용 수영복입니다.", "ko_name": "수영복(남성용)"}
}

# 라벨을 한글로 변환
def label_to_korean(label):
    return LABEL_INFO.get(label, {}).get("ko_name", label.replace('_', ' '))

# RGB 색상 → 색상 이름 추정
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

# 색상 평균으로 재질 추정
def guess_material(rgb):
    mean_val = np.mean(rgb)
    if mean_val > 200: return "플라스틱/세라믹"
    if mean_val < 60: return "금속/고무"
    if all(abs(rgb[i] - rgb[i + 1]) < 20 for i in range(2)) and 100 < mean_val < 200:
        return "천/패브릭"
    return "복합/기타"

# 이미지 디자인 정보 분석 (색상, 재질)
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

# HuggingFace 영어 → 한국어 번역기 초기화
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
model_translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")

# 텍스트 자동 번역 함수
def translate_to_korean(text):
    try:
        tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated = model_translator.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return f"[번역 오류] {str(e)}"

# 플랫폼 적합도 추정 함수 (간이 AI 룰 기반)
def infer_suitability(label, design):
    color = design["main_color"]
    material = design["material"]
    score = {}

    if "mug" in label or "cup" in label:
        score = {"와디즈": 90, "킥스타터": 60, "마쿠아게": 70, "젝젝": 65}
    elif "laptop" in label or "cellular" in label:
        score = {"와디즈": 55, "킥스타터": 95, "마쿠아게": 60, "젝젝": 75}
    else:
        score = {"와디즈": 60, "킥스타터": 60, "마쿠아게": 85, "젝젝": 90}

    # 색상/재질 추가 가중치
    if color == "검은색" and material == "금속/고무":
        score = {k: min(v + 5, 100) for k, v in score.items()}
    elif color == "노란색" or material == "플라스틱/세라믹":
        score["와디즈"] = min(score.get("와디즈", 60) + 10, 100)

    return score

# 주요 API: 이미지 업로드 후 플랫폼 추천
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
    suitability = infer_suitability(label, design_info)

    # 플랫폼 추천 로직
    if "mug" in label or "cup" in label:
        platform, category, reason = "와디즈", "리빙 소품", "머그컵 등 리빙 제품은 국내 플랫폼에 적합합니다."
    elif "laptop" in label or "cellular" in label:
        platform, category, reason = "킥스타터", "테크/디자인", "테크 제품은 글로벌 시장에 적합합니다."
    else:
        platform, category, reason = "젝젝", "라이프스타일", f"'{label_ko}' 관련 제품은 동남아 시장에 적합합니다."

    # 영어로도 번역한 reason 포함
    translated_reason = translate_to_korean(reason)

    return {
        "platform": platform,
        "category": category,
        "reason": reason,
        "reason_ko": translated_reason,
        "feature": info["feature"],
        "label_ko": label_ko,
        "design": design_info,
        "suitability": suitability
    }

# Cloud Run에서는 __main__ 실행 안 해도 됨
