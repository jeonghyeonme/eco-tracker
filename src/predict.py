import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def run_prediction(model_path, input_dir, output_dir, max_images=5, conf=0.25):
    # 1. 모델 불러오기
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_path}")
        return

    model = YOLO(model_path)
    print(f"Model loaded from {model_path}")

    # 2. 이미지 파일 목록 가져오기
    input_path = Path(input_dir)
    img_formats = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for fmt in img_formats:
        image_files.extend(list(input_path.glob(fmt)))
    
    if not image_files:
        print(f"Error: {input_dir} 폴더에서 이미지를 찾을 수 없습니다.")
        return

    # 3. 최대 5장으로 제한
    test_images = image_files[:max_images]
    print(f"Processing {len(test_images)} images (Limited to {max_images})...")

    # 4. 추론 수행 및 결과 저장
    # save=True: 결과 이미지를 디스크에 저장
    # project/name: 저장될 폴더 경로 설정
    results = model.predict(
        source=[str(img) for img in test_images],
        conf=conf,        # 신뢰도 임계값
        save=True,        # 결과 이미지 저장 여부
        project=output_dir,
        name="prediction_results",
        exist_ok=True     # 폴더가 이미 있으면 덮어쓰기
    )

    print(f"\nDetection complete. Results saved in: {output_dir}/prediction_results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 모델을 사용한 이미지 추론 테스트")
    
    # 기본 경로는 가장 최근 학습 결과인 exp_augmented의 best.pt로 설정
    parser.add_argument("--model", type=str, default="runs/detect/runs/exp_augmented/weights/best.pt", help="학습된 모델 가중치 경로")
    parser.add_argument("--input", type=str, default="data/test_images", help="테스트할 이미지가 담긴 폴더")
    parser.add_argument("--output", type=str, default="inference_outputs", help="결과를 저장할 폴더")
    parser.add_argument("--limit", type=int, default=5, help="최대 처리 이미지 수")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값 (Confidence Threshold)")

    args = parser.parse_args()
    
    run_prediction(args.model, args.input, args.output, args.limit, args.conf)
