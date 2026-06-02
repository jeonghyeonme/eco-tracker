# 🚀 Eco-Tracker Project Roadmap

## 📌 프로젝트 개요
AI-Hub 공공데이터를 활용한 생활 폐기물 실시간 탐지 및 추론 속도(FPS) 최적화 시스템 구축 프로젝트입니다. YOLOv8을 기반으로 다양한 모델 규모별 성능과 속도의 트레이드오프를 분석합니다.

---

## 📊 현재 프로젝트 상황 (2026-06-01 기준)

### ✅ 완료된 작업
1.  **프로젝트 구조 설계:** `eco_tracker_project_plan.md`를 기반으로 표준 디렉토리 구조 생성 완료.
2.  **환경 설정:** `requirements.txt` 생성 및 주요 라이브러리(`torch`, `ultralytics`, `albumentations` 등) 설치 완료.
3.  **데이터 구성:** YOLO 학습을 위한 `data/dataset.yaml` 클래스 정의(4종) 및 경로 설정 완료.
4.  **핵심 스크립트 구현:**
    *   `src/preprocess/json_to_yolo.py`: 데이터 포맷 변환 로직.
    *   `src/train.py`: 모델 학습 자동화 스크립트.
    *   `src/evaluate.py`: mAP 및 FPS 측정 로직.

### ⚠️ 현재 이슈 및 참고사항
*   **서버 용량 이슈:** AI-Hub 데이터셋(재활용 선별장 기준 약 70GB)이 EC2 기본 용량(100GB)에 비해 매우 큽니다. 분할 압축 해제 및 병합 시 용량 부족이 예상됩니다.
*   **클라우드 컴퓨팅:** 모델 학습 및 추론은 AWS EC2 G4dn(NVIDIA T4 GPU) 또는 G5 인스턴스 환경에서 수행하며, 원격 서버 기반의 실행 환경을 구축합니다.
*   **Android Studio 관련:** 현재 프로젝트는 Python 기반 딥러닝 학습/평가용 백엔드 코드로 구성되어 있습니다. Android Studio에서 직접 실행 가능한 앱 형태가 아니며, 추후 모델 최적화 후 TFLite 등으로 변환하여 모바일 앱에 통합하는 과정이 필요합니다.

---

## 📦 서버 용량 문제 및 해결 방안 (Storage Strategy)

서버 디스크 용량(100GB) 내에서 대용량 데이터(70GB+)를 처리하기 위한 전략입니다.

1.  **EBS 볼륨 일시 확장:** 데이터 처리 기간 동안만 EBS 볼륨을 250GB 이상으로 확장하여 물리적 공간 확보.
2.  **가상 병합 스트리밍(Virtual Streaming):** 분할 압축 파일을 물리적으로 병합하지 않고 파이썬 스트림으로 읽어 실시간 리사이징(640x640) 처리.
3.  **이미지 다이어트:** 원본 이미지를 학습 규격(640x640)으로 리사이징하여 데이터셋 전체 용량을 90% 이상 절감(최종 2~3GB 수준).

---

## 🗺️ 향후 작업 계획 (Milestones)

### 1단계: 데이터 확보 및 전처리 (1주차 목표)
- [ ] AI-Hub에서 '생활 폐기물 이미지 데이터' 확보.
- [ ] `json_to_yolo.py`를 실행하여 원본 JSON을 YOLO 포맷으로 변환.
- [ ] 데이터를 `data/train`, `data/val`, `data/test`로 적절히 분할 배치.
- [ ] `notebooks/eda_and_verification.ipynb`를 생성하여 데이터 분포 및 증강 효과 확인.

### 2단계: 모델 학습 및 실험 (2주차 목표)
- [ ] YOLOv8n (Nano) 모델로 베이스라인 학습 수행.
- [ ] Albumentations를 활용한 데이터 증강(Augmentation) 효과 검증.
- [ ] YOLOv8s (Small), YOLOv8m (Medium) 모델 순차 학습 및 로그 수집.

### 3단계: 성능 평가 및 최적화 분석 (3주차 목표)
- [ ] `evaluate.py`를 사용하여 모델별 추론 속도(ms) 및 FPS 정밀 측정.
- [ ] 정확도(mAP)와 속도 간의 상관관계 분석표 작성.
- [ ] **(선택 사항)** 모바일/엣지 디바이스 배포를 위한 모델 경량화(TensorRT, TFLite 변환).

### 4단계: 모바일 앱 통합 (추후 확장)
- [ ] 학습된 최적 모델을 모바일용 포맷(TFLite/ONNX)으로 내보내기.
- [ ] 안드로이드 환경에서 실시간 객체 탐지 기능을 가진 앱 UI 개발 및 연동.

---

## 📝 인수인계 노트
*   **실행 환경:** Python 3.12+ 환경에서 `pip install -r requirements.txt`가 선행되어야 함.
*   **모델 학습:** `python src/train.py --model yolov8n --epochs 50` 명령어로 시작 가능.
*   **성능 평가:** `weights/`에 학습된 가중치(`.pt`)를 넣고 `python src/evaluate.py` 실행.
