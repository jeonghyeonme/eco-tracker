import json
import os
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Mapping AI-Hub class codes to our consolidated class IDs
CLASS_MAP = {
    'c_1': 0, 'c_1_01': 0, 'c_2_01': 1, 'c_2_02': 2, 'c_2_02_01': 2,
    'c_3': 3, 'c_3_01': 3, 'c_4_01_01': 4, 'c_4_01_02': 4, 'c_4_02_01_01': 4,
    'c_4_02_01_02': 4, 'c_4_02_02_01': 4, 'c_4_02_02_02': 4, 'c_4_02_03_01': 4,
    'c_4_02_03_02': 4, 'c_4_03': 4, 'c_4_03_01': 4, 'c_5_01': 5, 'c_5_01_01': 5,
    'c_5_02': 5, 'c_5_02_01': 5, 'c_6': 6, 'c_6_01': 6, 'c_7': 7, 'c_7_01': 7,
    'c_8_01': 8, 'c_8_01_01': 8, 'c_8_02': 8, 'c_9': 9
}

def process_single_file(json_path, img_dir, output_root, split_ratios, target_size=(640, 640)):
    """
    Processes a single JSON and its matching image: convert to YOLO, resize, and save.
    """
    # Determine split
    rand_val = random.random()
    if rand_val < split_ratios[0]:
        split = 'train'
    elif rand_val < split_ratios[0] + split_ratios[1]:
        split = 'val'
    else:
        split = 'test'
        
    # 1. Process Label
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return
        
    res_str = data.get('Info', {}).get('RESOLUTION', '1920/1080')
    try:
        img_w, img_h = map(float, res_str.split('/'))
    except:
        img_w, img_h = 1920.0, 1080.0
        
    yolo_labels = []
    for obj in data.get('objects', []):
        class_code = obj.get('class_name')
        if class_code not in CLASS_MAP: continue
        
        coord = obj.get('annotation', {}).get('coord', {})
        x, y, w, h = coord.get('x'), coord.get('y'), coord.get('width'), coord.get('height')
        if None in [x, y, w, h]: continue
        
        x_c = (x + w / 2) / img_w
        y_c = (y + h / 2) / img_h
        w_n, h_n = w / img_w, h / img_h
        yolo_labels.append(f"{CLASS_MAP[class_code]} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    
    if not yolo_labels: return
    
    # 2. Process Image (Find and Resize)
    img_name = json_path.stem + '.jpg'
    img_src_path = next(img_dir.rglob(img_name), None)
    if not img_src_path: return
    
    try:
        img = cv2.imread(str(img_src_path))
        if img is None: return
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    except:
        return
        
    # 3. Save
    stem = json_path.stem
    img_out = output_root / split / 'images' / f"{stem}.jpg"
    lbl_out = output_root / split / 'labels' / f"{stem}.txt"
    
    cv2.imwrite(str(img_out), img_resized)
    with open(lbl_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data/yolo_dataset")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    
    for s in ['train', 'val', 'test']:
        os.makedirs(output_root / s / 'images', exist_ok=True)
        os.makedirs(output_root / s / 'labels', exist_ok=True)
        
    print(f"Searching for JSON files in {json_dir}...")
    json_files = list(json_dir.rglob('*.json'))
    print(f"Found {len(json_files)} JSON files. Starting processing with {cpu_count()} cores...")
    
    worker = partial(process_single_file, img_dir=img_dir, output_root=output_root, split_ratios=args.split)
    
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(worker, json_files), total=len(json_files), desc="Processing"))

if __name__ == "__main__":
    main()
