# AI 모델 및 서비스 운영 비용 정리

## 1. 모델 및 라이브러리 비용
- MobileNetV2, TensorFlow, Pillow, FastAPI 등: **모두 오픈소스, 무료**
- 사전학습(pretrained) 모델 다운로드 및 사용: **무료**

## 2. 서버 인프라 비용
- **로컬 PC에서 테스트**: 별도 비용 없음
- **클라우드 서버(예: GCP, AWS, Azure)**
  - VM(가상머신) 사용 시: 시간당/월별 과금 (CPU, 메모리, GPU 사양에 따라 다름)
  - 예시: GCP e2-micro(무료 티어), n1-standard-1(약 $25/월), GPU 서버(수십~수백 달러/월)
  - Cloud Run, App Engine 등 서버리스: 호출/사용량 기반 과금

## 3. 네트워크/트래픽 비용
- 무료 트래픽 한도 초과 시 과금 (클라우드별 정책 상이)
- 이미지 업로드/다운로드가 많으면 추가 비용 발생 가능

## 4. 기타 비용
- **상업적 서비스**: 대규모 트래픽, SLA, 보안 등 추가 비용 고려
- **추가 AI API(예: Google Vision, Azure Cognitive 등) 사용 시**: 별도 과금

## 5. 요약
- 개발/테스트 단계: 대부분 무료(로컬 또는 무료 클라우드 티어)
- 실제 서비스 운영: 서버 사양, 트래픽, AI 연산량에 따라 월 수천~수십만 원까지 다양
- 오픈소스 모델 자체는 무료, 인프라(서버/클라우드) 비용이 주요 비용

---
**참고:**  
- [Google Cloud 가격표](https://cloud.google.com/pricing)
- [AWS EC2 가격표](https://aws.amazon.com/ko/ec2/pricing/)
- [Azure VM 가격표](https://azure.microsoft.com/ko-kr/pricing/details/virtual-machines/)
- [TensorFlow 라이선스](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
