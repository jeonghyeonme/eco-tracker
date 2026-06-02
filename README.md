# ♻️ Eco-Tracker: 생활 폐기물 실시간 탐지 프로젝트

AI-Hub 공공데이터를 활용하여 생활 폐기물을 탐지하고, 추론 속도와 정확도 간의 최적점을 찾는 프로젝트입니다.

---

## 🛠️ 작업 가이드라인 (Workflow)

### 1. 데이터 준비 및 전처리
AI-Hub에서 다운로드한 원본 데이터(JSON, JPG)를 YOLO 학습용 포맷으로 변환하고 Train/Val/Test 세트로 나눕니다.

1.  **데이터 배치:** 원본 이미지를 `raw_images`, 라벨을 `raw_json` 폴더 등에 모읍니다.
2.  **전처리 실행:**
    ```powershell
    python src/preprocess/json_to_yolo.py --input_json_dir [라벨경로] --input_img_dir [이미지경로] --output_root data
    ```
    *   이 과정이 끝나면 `data/train`, `data/val`, `data/test` 폴더가 생성됩니다.

### 2. 모델 학습 (Training)
명령어 인자를 매번 수정할 필요 없이 설정 파일을 통해 학습을 제어합니다.

1.  **설정 수정:** `configs/train_config.yaml` 파일을 열어 `epochs`, `batch_size`, `model_variant` 등을 수정합니다.
2.  **학습 실행:**
    ```powershell
    python src/train.py
    ```
    *   학습 결과(가중치, 그래프 등)는 `runs/[실험이름]` 폴더에 저장됩니다.

### 3. 모델 평가 (Evaluation)
학습된 모델의 정확도(mAP)와 추론 속도(FPS)를 측정합니다.

1.  **평가 실행:**
    ```powershell
    python src/evaluate.py
    ```
    *   `src/evaluate.py` 내의 `model_weight` 경로를 학습된 결과물(`best.pt`)로 지정하여 실행합니다.

---

### 📂 프로젝트 구조
- `data/`: 전처리된 데이터셋 (Train/Val/Test)
- `src/`: 핵심 소스 코드 (전처리, 학습, 평가)
- `configs/`: 학습 하이퍼파라미터 설정 파일
- `notebooks/`: EDA 및 검증용 주피터 노트북
- `runs/`: 학습 결과 및 가중치 저장소

---

### ⚠️ 참고 사항
- **서버 환경:** AWS EC2 G4dn 등 GPU 인스턴스 사용 시 `configs/train_config.yaml`의 `device`를 `auto`로 두면 자동으로 GPU 학습이 진행됩니다.
- **클래스 확장:** 현재 10종의 폐기물 클래스를 지원하며, `data/dataset.yaml`에서 관리합니다.
