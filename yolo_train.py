import os
from ultralytics import YOLO

def train_yolo():
    # Load a pretrained YOLOv8 model (n (nano) is fastest/lightest)
    model = YOLO('yolov8n.pt')

    # Path to the data configuration file
    data_yaml_path = 'C:/ennesse/softcomputing/data.yaml'

    print("--- Starting YOLOv8 Training ---")
    
    # Train the model
    # Parameters: 40 epochs, 640x640 imgsz
    results = model.train(
        data=data_yaml_path,
        epochs=1,
        imgsz=640,
        batch=4,
        device=0,
        name='waste_yolov8_run',
        project='waste_detection_results',
        exist_ok=True
    )

    print("--- YOLOv8 Training Completed ---")
    
    # The best weights are automatically saved by ultralytics in:
    # {project}/{name}/weights/best.pt
    # We can also save a copy to the root for easier access
    best_weights_path = os.path.join('waste_detection_results', 'waste_yolov8_run', 'weights', 'best.pt')
    if os.path.exists(best_weights_path):
        import shutil
        shutil.copy(best_weights_path, 'yolov8_waste_best.pt')
        print(f"Best weights saved to yolov8_waste_best.pt")

if __name__ == "__main__":
    train_yolo()
