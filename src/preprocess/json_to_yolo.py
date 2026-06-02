import json
import os
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Mapping AI-Hub class codes to our consolidated class IDs
CLASS_MAP = {
    # Paper
    'c_1': 0, 'c_1_01': 0,
    # Paper Pack
    'c_2_01': 1,
    # Paper Cup
    'c_2_02': 2, 'c_2_02_01': 2,
    # Can
    'c_3': 3, 'c_3_01': 3,
    # Glass
    'c_4_01_01': 4, 'c_4_01_02': 4, 'c_4_02_01_01': 4, 'c_4_02_01_02': 4,
    'c_4_02_02_01': 4, 'c_4_02_02_02': 4, 'c_4_02_03_01': 4, 'c_4_02_03_02': 4,
    'c_4_03': 4, 'c_4_03_01': 4,
    # PET
    'c_5_01': 5, 'c_5_01_01': 5, 'c_5_02': 5, 'c_5_02_01': 5,
    # Plastic
    'c_6': 6, 'c_6_01': 6,
    # Vinyl
    'c_7': 7, 'c_7_01': 7,
    # Styrofoam
    'c_8_01': 8, 'c_8_01_01': 8, 'c_8_02': 8,
    # Battery
    'c_9': 9
}

def convert_to_yolo(json_path):
    """
    Converts AI-Hub JSON to YOLO format based on the sample structure.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract resolution
    res_str = data.get('Info', {}).get('RESOLUTION', '1920/1080')
    try:
        img_width, img_height = map(float, res_str.split('/'))
    except ValueError:
        img_width, img_height = 1920.0, 1080.0
        
    yolo_labels = []
    
    for obj in data.get('objects', []):
        class_code = obj.get('class_name')
        if class_code not in CLASS_MAP:
            continue
            
        class_id = CLASS_MAP[class_code]
        
        # Bbox coord: x, y are top-left
        coord = obj.get('annotation', {}).get('coord', {})
        x = coord.get('x')
        y = coord.get('y')
        w = coord.get('width')
        h = coord.get('height')
        
        if None in [x, y, w, h]:
            continue
            
        # Normalize to YOLO format [center_x, center_y, width, height]
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Clip values between 0 and 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    return yolo_labels

def main():
    parser = argparse.ArgumentParser(description="Convert AI-Hub JSON labels to YOLO format with splitting")
    parser.add_argument("--input_json_dir", type=str, required=True, help="Path to JSON directory")
    parser.add_argument("--input_img_dir", type=str, required=True, help="Path to Image directory")
    parser.add_argument("--output_root", type=str, default="data", help="Root path for YOLO dataset")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train/Val/Test split ratios")
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    for split in ['train', 'val', 'test']:
        os.makedirs(output_root / split / 'images', exist_ok=True)
        os.makedirs(output_root / split / 'labels', exist_ok=True)
    
    json_files = list(Path(args.input_json_dir).rglob('*.json'))
    random.shuffle(json_files)
    
    n_total = len(json_files)
    n_train = int(n_total * args.split[0])
    n_val = int(n_total * args.split[1])
    
    print(f"Total: {n_total} files. Splitting: Train {n_train}, Val {n_val}, Test {n_total - n_train - n_val}")
    
    for i, json_file in enumerate(tqdm(json_files, desc="Processing and Splitting")):
        # Determine split
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
            
        # 1. Convert and save label
        labels = convert_to_yolo(json_file)
        if not labels:
            continue
            
        label_out_path = output_root / split / 'labels' / (json_file.stem + '.txt')
        with open(label_out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
            
        # 2. Copy corresponding image
        img_name = json_file.stem + '.jpg'
        img_src_path = next(Path(args.input_img_dir).rglob(img_name), None)
        if img_src_path:
            shutil.copy(str(img_src_path), str(output_root / split / 'images' / img_name))

if __name__ == "__main__":
    main()

