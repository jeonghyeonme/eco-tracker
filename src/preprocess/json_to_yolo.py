import json
import os
import argparse
import random
import zipfile
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

def process_zip_pair(zip_paths, output_root, split_ratios, target_size=(640, 640)):
    """
    Processes a pair of (Image ZIP, Label ZIP) and converts to YOLO format with resizing.
    """
    img_zip_path, lbl_zip_path = zip_paths
    
    with zipfile.ZipFile(img_zip_path, 'r') as img_zip, zipfile.ZipFile(lbl_zip_path, 'r') as lbl_zip:
        # Get list of JSON files
        json_names = [n for n in lbl_zip.namelist() if n.endswith('.json')]
        
        for json_name in json_names:
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
                with lbl_zip.open(json_name) as f:
                    data = json.load(f)
            except:
                continue
                
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
            
            if not yolo_labels: continue
            
            # 2. Process Image (Resize)
            img_name = json_name.replace('.json', '.jpg').split('/')[-1]
            img_candidates = [n for n in img_zip.namelist() if n.endswith(img_name)]
            if not img_candidates: continue
            
            try:
                with img_zip.open(img_candidates[0]) as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is None: continue
                    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            except:
                continue
                
            # 3. Save
            stem = Path(json_name).stem
            img_out = output_root / split / 'images' / f"{stem}.jpg"
            lbl_out = output_root / split / 'labels' / f"{stem}.txt"
            
            cv2.imwrite(str(img_out), img_resized)
            with open(lbl_out, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_zip_dir", type=str, required=True)
    parser.add_argument("--lbl_zip_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data/yolo_dataset")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    for s in ['train', 'val', 'test']:
        os.makedirs(output_root / s / 'images', exist_ok=True)
        os.makedirs(output_root / s / 'labels', exist_ok=True)
        
    img_zips = sorted(list(Path(args.img_zip_dir).glob('*.zip')))
    lbl_zips = sorted(list(Path(args.lbl_zip_dir).glob('*.zip')))
    
    # Pair up zips (Assumes they have matching B1, B2... suffixes)
    zip_pairs = []
    for iz in img_zips:
        suffix = iz.stem.split('_')[-1]
        lz = next((l for l in lbl_zips if l.stem.endswith(suffix)), None)
        if lz: zip_pairs.append((iz, lz))
        
    print(f"Found {len(zip_pairs)} pairs of ZIP files. Starting processing with {cpu_count()} cores...")
    
    worker = partial(process_zip_pair, output_root=output_root, split_ratios=args.split)
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(worker, zip_pairs), total=len(zip_pairs), desc="Processing ZIP Pairs"))

if __name__ == "__main__":
    main()
