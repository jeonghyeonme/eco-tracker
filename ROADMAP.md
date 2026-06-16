# 🚀 Eco-Tracker Project Roadmap

## 📌 프로젝트 개요
AI-Hub 공공데이터를 활용한 생활 폐기물 실시간 탐지 및 추론 속도(FPS) 최적화 시스템 구축 프로젝트입니다. YOLOv8을 기반으로 다양한 모델 규모별 성능과 속도의 트레이드오프를 분석합니다.

---

## 📊 현재 프로젝트 상황 (2026-06-15 기준)

### ✅ 완료된 작업
1.  **프로젝트 구조 설계**: 표준 디렉토리 구조 및 `.gitignore` 설정 완료.
2.  **데이터 파이프라인 최적화**: 
    *   이중 압축(ZIP in ZIP) 구조를 직접 처리하는 멀티프로세싱 전처리 스크립트(`json_to_yolo.py`) 구현.
    *   학습 효율을 위해 원본 데이터를 640x640으로 자동 리사이징하는 스트리밍 전처리 완성.
3.  **학습 환경 이원화**:
    *   **Cloud**: AWS EC2 (Tesla T4) 환경 구축 및 Medium 모델 학습 진행.
    *   **Local**: RTX 3000 Series GPU 기반 가속 학습 환경(Python 3.12 + CUDA 12.1) 구축.
4.  **모델별 개별 실험 진행**:
    *   **YOLOv8n (Nano)**: 학습 완료 (EC2 & Local Lite 버전).
    *   **YOLOv8s (Small)**: 학습 완료 (Local).
5.  **문서화 기반 마련**:
    *   최종 보고서 가이드라인(`FINAL_REPORT.md`) 작성 완료.

### 🔄 진행 중인 작업 (WIP)
*   **YOLOv8m (Medium)**: EC2 환경에서 학습 진행 중 (안정성을 위해 10 Epoch / Batch 16 설정).
*   **성능 비교 분석**: 각 모델별 mAP, FPS, 추론 시간 지표 수집 중.

### ⚠️ 이슈 및 참고사항
*   **서버 안정성**: AWS Academy 세션 제한(4시간) 및 OOM 이슈로 인해 주기적인 `--resume` 학습 및 배치 사이즈 최적화 필요.
*   **인증 관리**: EC2에서 Git Push 시 Personal Access Token(PAT) 사용 필수.

---

## 🗺️ 향후 작업 계획 (Milestones)

### 1단계: 모델 학습 및 지표 수집 (진행 중)
- [x] YOLOv8n, yolov8s 학습 및 결과 백업 (`result/yolov8n` 브랜치).
- [ ] YOLOv8m 학습 완주 및 결과 도출.
- [ ] 데이터 증강 유무에 따른 Lite 버전 비교 실험 (`n_lite` vs `n_default`).

### 2단계: 최종 성능 분석 및 시각화
- [ ] `src/evaluate.py`를 활용한 전 모델 FPS/mAP 통합 측정.
- [ ] 모델 규모별 정확도-속도 트레이드오프 그래프 작성.
- [ ] 혼동 행렬(Confusion Matrix) 분석을 통한 오분류 특성 파악.

### 3단계: 최종 보고서 완성
- [ ] `FINAL_REPORT.md`의 수치 및 그래프 업데이트.
- [ ] 실제 테스트 이미지(실생활 사진) 추론 결과 첨부.
- [ ] 최종 모델(Best Weight) 선정 및 배포용 브랜치 정리.
