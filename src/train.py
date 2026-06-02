import torch
from ultralytics import YOLO
import argparse
import os
import yaml

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(config, resume=False):
    # Load a pretrained model or the last checkpoint if resuming
    if resume:
        # Path to the last checkpoint
        last_weights = os.path.join(config['project'], "detect", "runs", config['name'], "weights", "last.pt")
        if os.path.exists(last_weights):
            model = YOLO(last_weights)
            print(f"Resuming training from {last_weights}")
        else:
            print("Last weights not found. Starting from scratch.")
            model_name = f"{config['model_variant']}.pt"
            model = YOLO(model_name)
            resume = False
    else:
        model_name = f"{config['model_variant']}.pt"
        model = YOLO(model_name)

    # Determine device
    if config['device'] == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = config['device']
    
    print(f"--- Experiment: {config['name']} ---")
    print(f"Using device: {device}")

    # Train the model using values from config
    results = model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        lr0=config['lr0'],
        optimizer=config['optimizer'],
        patience=config['patience'],
        project=config['project'],
        name=config['name'],
        device=device,
        verbose=config['verbose'],
        resume=resume, # Add resume parameter
        # Augmentation Settings
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        shear=config.get('shear', 0.0),
        perspective=config.get('perspective', 0.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.0),
        copy_paste=config.get('copy_paste', 0.0)
    )
    
    print(f"Training complete. Weights saved in {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Eco-Tracker")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        config_data = load_config(args.config)
        train_model(config_data, resume=args.resume)
    else:
        print(f"Error: Config file {args.config} not found.")
