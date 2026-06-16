# ♻️ Eco-Tracker: 생활 폐기물 실시간 탐지 및 모델별 성능 분석 보고서

## 1. 문제 정의 (Problem Definition)
### 1.1 프로젝트 배경 및 사회적 가치
*   **재활용 효율화**: 급격한 생활 폐기물 증가 속에서 오분류로 인한 자원 낭비와 처리 비용 상승이 사회적 문제로 대두됨. 특히 비닐 등이 혼입된 종이류는 재활용이 불가능해 매립 비용을 발생시키므로, 정확한 사전 분류가 필수적임.
*   **목표**: 객체 탐지 모델 YOLOv8을 활용하여 10종의 폐기물을 실시간 탐지하고, 모델 규모별 성능 분석을 통해 최적의 솔루션을 제안함.

### 1.2 활용 시나리오
*   **스마트 분리수거함**: 무인 수거기 내부에서 투입되는 쓰레기를 실시간 판별하여 보상 체계와 연동.
*   **지자체 관리 시스템**: 선별장 CCTV 등에 적용하여 혼합 배출 여부 모니터링 및 자동 통계 산출.

## 2. 데이터셋 설명 (Dataset Description)
*   **출처**: AI-Hub '307. 생활폐기물 데이터 활용·환류' 정식 개방 데이터.
*   **데이터 구성 및 다양성**:
    *   **클래스 (10종)**: Paper, Paper_Pack, Paper_Cup, Can, Glass, PET, Plastic, Vinyl, Styrofoam, Battery.
    *   **실생활 데이터 반영**: 깨끗한 상태뿐만 아니라 찌그러지거나 훼손된 폐기물 이미지를 포함하여 실생활 탐지 정확도를 높임.
    *   **클래스 균형**: 각 클래스별 데이터 분포를 분석하여 특정 품목(예: PET)에 대한 편향 여부를 확인하고 학습에 반영함.

## 3. 사용 모델 설명 (Model Architecture)
*   **YOLOv8 핵심 기술**:
    *   **Anchor-free Detection**: 고정된 앵커 박스 없이 객체를 직접 탐지하여 작은 물체에 대한 민감도를 높이고 계산 복잡도를 줄임.
    *   **CSPDarknet & PAN-FPN**: 효율적인 특징 추출과 다중 스케일(크기) 객체 융합을 통한 높은 탐지 성능 유지.
*   **모델 스케일링 전략**:
    *   **YOLOv8n (Nano)**: 신경망의 깊이(Depth)와 폭(Width)을 최소화하여 모바일/엣지 기기에서 초당 100 FPS 이상의 속도 지향.
    *   **YOLOv8s/m**: 레이어 수와 채널 수를 점진적으로 확장하여 복잡한 배경에서의 탐지 정밀도 향상.

## 4. 성능 비교 방법 및 실험 설정 (Methodology)
*   **학습 전략**: Pretrained Weight 기반 Transfer Learning.
*   **데이터 증강(Augmentation) 전략**:
    *   **Mosaic**: 4장의 이미지를 합쳐 객체 크기 다양성을 확보, 작은 쓰레기 탐지 능력 강화.
    *   **Flip/HSV**: 좌우 반전 및 색상 변형을 통해 조명 변화와 투입 방향에 상관없는 강건한 모델 구축.
*   **실험 환경**: AWS EC2(Tesla T4) 및 Local(RTX 3000) 병렬 학습 진행.

## 5. 성능 지표 및 선정 근거 (Metrics & Rationale)
### 5.1 정확도 지표: mAP (mean Average Precision)
*   **선정 이유**: 쓰레기 탐지 시 크기와 각도가 다양하므로 위치 정밀도(IoU)를 반영한 mAP 지표가 단순 정확도보다 신뢰도가 높음.
*   **mAP@0.5**: 위치 인식의 기본 성능 측정.
*   **mAP@0.5:0.95**: 경계 상자의 정밀도를 평가하는 엄격한 지표.

### 5.2 효율성 지표: Inference Time & FPS
*   **선정 이유**: 실시간 시스템(30 FPS 이상) 적용 가능성을 판단하기 위한 필수 지표.

## 6. 성능 결과 (Performance Metrics)
*본 결과는 10 Epoch 학습 완료 후 Test 세트를 통해 측정된 수치입니다.*

| 모델 실험명 | mAP50 | mAP50-95 | Inference (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- |
| yolov8n_exp | **0.9600** | **0.9187** | **41.03** | **24.37** |
| yolov8s_exp | **0.9663** | **0.9325** | **103.82** | **9.63** |
| yolov8m_exp | **0.9664** | **0.9335** | **230.76** | **4.33** |


## 7. 시스템 구조 및 유지보수성 (System Architecture)

### 7.1 디렉토리 구조
프로젝트의 모듈화와 유지보수성을 극대화하기 위해 다음과 같은 계층적 구조를 설계하였습니다.

```text
eco-tracker/
├── configs/            # 모델별 학습/검증 하이퍼파라미터 설정 (YAML)
├── data/               # 데이터셋 관리
│   ├── raw_data/       # AI-Hub 원본 데이터 (ZIP/JSON)
│   └── yolo_dataset/   # 전처리 완료된 YOLO 포맷 데이터셋
├── src/                # 핵심 소스 코드
│   ├── preprocess/     # 데이터 정제 및 변환 엔진
│   ├── train.py        # 모델 학습 오케스트레이터
│   ├── evaluate.py     # 성능 및 속도 통합 평가 엔진
│   └── predict.py      # 실전 추론 및 테스트 스크립트
├── results/            # 모델별 실험 결과물 백업 (weights, plots, logs)
├── FINAL_REPORT.md     # 최종 분석 보고서
└── requirements.txt    # 환경 재현을 위한 의존성 명세
```

### 7.2 주요 컴포넌트 상세 설명
1.  **`src/preprocess/json_to_yolo.py`**:
    *   대용량 데이터(70GB+) 처리를 위한 **멀티프로세싱 스트리밍** 엔진.
    *   이중 압축 해제, 640x640 리사이징, YOLO 라벨 변환을 한 번에 수행하여 디스크 I/O 최적화.
2.  **`src/train.py`**:
    *   중앙 집중형 `configs/*.yaml` 파일을 파싱하여 실험을 제어.
    *   학습 중단 시 마지막 체크포인트부터 재개하는 **Resume 기능** 탑재.
3.  **`src/evaluate.py`**:
    *   학습된 모델의 정밀도(mAP)와 실시간성(FPS)을 동시에 측정.
    *   측정 결과를 `evaluation_results.txt`에 누적 기록하여 실험 이력 관리 자동화.
4.  **`configs/*.yaml`**:
    *   모델 종류(n, s, m), 데이터 증강 강도, 하이퍼파라미터를 코드 수정 없이 즉시 변경 가능한 구조.

### 7.3 형상 관리 및 재현성
*   **유지보수성**: 모든 실험 설정을 외부에 분리하여 향후 새로운 클래스 추가나 모델 업그레이드 시 유연한 대응 가능.
*   **재현성**: 가상 환경 설정 가이드와 자동화된 전처리 파이프라인을 통해 동일한 실험 결과 보장.
*   **GitHub Repository**: [jeonghyeonme/eco-tracker](https://github.com/jeonghyeonme/eco-tracker)

## 8. 심층 분석 및 통찰 (Insight)
*   **오답 분석 (Confusion Matrix)**: 종이와 종이팩, 플라스틱과 PET 등 질감이 유사한 클래스 간의 오분류 패턴을 분석하여 데이터 증강 및 학습 전략의 유효성 검증.
*   **한계점 및 개선 방향**: 겹쳐진 물체(Occlusion)나 투명한 용기(유리병 등)의 탐지 실패 사례를 분석하여 향후 보완 방향 제시.

## 9. 학습 로그 및 기술 증빙 (Learning Logs)

본 프로젝트의 모든 학습 과정은 투명하게 기록되었으며, 아래 로그 파일을 통해 각 에폭별 상세 지표(Loss, mAP, LR 등)를 확인할 수 있습니다.

### 9.1 모델별 상세 로그 링크 (GitHub)
*   **[YOLOv8n (Nano)](https://github.com/jeonghyeonme/eco-tracker/blob/ec2-results/results/yolov8n/train_final.txt)**: 최종 학습 완료 로그
*   **[YOLOv8s (Small)](https://github.com/jeonghyeonme/eco-tracker/blob/ec2-results/results/yolov8s/train_s.txt)**: 최종 학습 완료 로그
*   **[YOLOv8m (Medium)](https://github.com/jeonghyeonme/eco-tracker/blob/ec2-results/results/yolov8m/train_m.txt)**: 최종 학습 완료 로그


### 9.2 학습 완료 증빙 샘플
각 모델은 지정된 10 Epoch를 완주하였으며, 최종 가중치(`best.pt`)가 성공적으로 저장되었음을 로그를 통해 확인하였습니다.

```text
# 예시: YOLOv8m 최종 학습 완료 시점 로그
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/10      8.25G     0.2399     0.2670     0.8252        145        640
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       6044       8521     0.9192     0.9285     0.9533     0.9160
Training complete. Weights saved in results/yolov8m/yolov8m_exp-2/weights
```

---
*최종 업데이트: 2026-06-16*
