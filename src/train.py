import torch
from ultralytics import YOLO
import argparse
import os
import yaml

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(config):
    # Load a pretrained model
    model_name = f"{config['model_variant']}.pt"
    model = YOLO(model_name)

    # Determine device
    if config['device'] == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = config['device']
    
    print(f"--- Experiment: {config['name']} ---")
    print(f"Using device: {device}")
    print(f"Model: {model_name}, Epochs: {config['epochs']}, Batch: {config['batch_size']}")

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
        verbose=config['verbose']
    )
    
    print(f"Training complete. Weights saved in {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Eco-Tracker")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        config_data = load_config(args.config)
        train_model(config_data)
    else:
        print(f"Error: Config file {args.config} not found.")
