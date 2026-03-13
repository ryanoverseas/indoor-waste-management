import torch
from ultralytics import YOLO
import os
import yaml
import time
from rcnn_train import WasteDataset, collate_fn  # Reusing dataset logic
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def test_yolo():
    print("--- Evaluating YOLOv8 on Test Set ---")
    # Load your trained weights
    model_path = 'yolov8_waste_best.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first!")
        return

    model = YOLO(model_path)
    
    # Run validation on the 'test' split
    # This will automatically give you precision, recall, mAP, and speed
    results = model.val(data='data.yaml', split='test')
    
    print("\n--- YOLOv8 Test Results ---")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Inference Speed: {results.speed['inference']:.2f}ms per image")

def test_rcnn():
    print("--- Evaluating Faster R-CNN on Test Set ---")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load configuration
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Load Test Dataset
    test_dataset = WasteDataset(
        data_config['test'],
        data_config['test'].replace('images', 'labels')
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Initialize Model & Load Weights
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = data_config['nc'] + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    weights_path = 'rcnn_waste_best.pth'
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Train the model first!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print("Running inference on test images...")
    total_time = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            total_time += (end_time - start_time)

    avg_speed_ms = (total_time / len(test_loader)) * 1000
    print("\n--- Faster R-CNN Test Results ---")
    print(f"Avg Inference Speed: {avg_speed_ms:.2f}ms per image")
    print(f"Approx FPS: {1000/avg_speed_ms:.2f} FPS")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["yolo", "rcnn"], required=True)
    args = parser.parse_args()

    if args.model == "yolo":
        test_yolo()
    else:
        test_rcnn()
