from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smart-funding-a11tg02j1-joyeyoungs-projects.vercel.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = tf.keras.applications.MobileNetV2(weights="imagenet")
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

LABEL_INFO = {
    "coffee_mug": {
        "feature": "머그컵으로 일상에서 자주 사용하는 리빙 소품입니다.",
        "ko_name": "머그컵"
    },
    "cup": {
        "feature": "컵으로 다양한 음료를 담을 수 있는 실용적인 제품입니다.",
        "ko_name": "컵"
    },
    "laptop": {
        "feature": "노트북으로 휴대성과 성능을 모두 갖춘 전자기기입니다.",
        "ko_name": "노트북"
    },
    "cellular_telephone": {
        "feature": "스마트폰 등 휴대전화로 현대인의 필수품입니다.",
        "ko_name": "휴대전화"
    },
    "swimming_trunks": {
        "feature": "수영할 때 입는 남성용 수영복입니다.",
        "ko_name": "수영복(남성용)"
    }
}

def label_to_korean(label):
    info = LABEL_INFO.get(label)
    if info and "ko_name" in info:
        return info["ko_name"]
    return label.replace('_', ' ')

def rgb_to_color_name(rgb):
    r, g, b = rgb
    if max(rgb) - min(rgb) < 20:
        if np.mean(rgb) > 200:
            return "하얀색"
        elif np.mean(rgb) < 50:
            return "검은색"
        else:
            return "회색"
    if r > 200 and g > 200 and b < 100:
        return "노란색"
    if r > 200 and g < 100 and b < 100:
        return "빨간색"
    if r < 100 and g > 200 and b < 100:
        return "초록색"
    if r < 100 and g < 100 and b > 200:
        return "파란색"
    if r > 200 and g < 100 and b > 200:
        return "분홍색"
    if r > 200 and g > 100 and b > 100:
        return "베이지색"
    return "기타"

def guess_material(rgb):
    r, g, b = rgb
    if np.mean(rgb) > 200:
        return "플라스틱/세라믹"
    if np.mean(rgb) < 60:
        return "금속/고무"
    if abs(r - g) < 20 and abs(g - b) < 20 and np.mean(rgb) > 100 and np.mean(rgb) < 200:
        return "천/패브릭"
    return "복합/기타"

def analyze_design(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((100, 100))
    arr = np.array(img)
    main_color = np.mean(arr.reshape(-1, 3), axis=0)
    color_name = rgb_to_color_name(main_color)
    material = guess_material(main_color)
    return {
        "main_color": color_name,
        "material": material
    }

@app.post("/api/recommend-platform")
async def recommend_platform(image: UploadFile = File(...)):
    image_bytes = await image.read()
    input_tensor = preprocess_image(image_bytes)
    preds = model.predict(input_tensor)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    label_ko = label_to_korean(label)

    info = LABEL_INFO.get(label, {
        "feature": f"{label_ko}(으)로 분류된 제품입니다."
    })

    design_info = analyze_design(image_bytes)
    suitability = {
        "와디즈": random.randint(50, 100),
        "킥스타터": random.randint(50, 100),
        "마쿠아게": random.randint(50, 100),
        "젝젝": random.randint(50, 100)
    }

    if "mug" in label or "cup" in label:
        platform = "와디즈"
        category = "리빙 소품"
        reason = "머그컵 등 리빙 제품은 국내 플랫폼에 적합합니다."
    elif "laptop" in label or "cellular" in label:
        platform = "킥스타터"
        category = "테크/디자인"
        reason = "테크 제품은 글로벌 시장에 적합합니다."
    else:
        platform = "젝젝"
        category = "라이프스타일"
        reason = f"'{label_ko}' 관련 제품은 동남아 시장에 적합합니다."

    return {
        "platform": platform,
        "category": category,
        "reason": reason,
        "feature": info["feature"],
        "label_ko": label_ko,
        "design": design_info,
        "suitability": suitability
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
