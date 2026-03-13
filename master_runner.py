import argparse
import sys
from yolo_train import train_yolo
from rcnn_train import train_rcnn

def main():
    parser = argparse.ArgumentParser(description="Waste Detection Training Orchestrator")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["yolo", "rcnn", "both"], 
        required=True,
        help="Select the model to train: 'yolo', 'rcnn', or 'both'"
    )

    args = parser.parse_args()

    if args.model == "yolo":
        print(">>> Triggering YOLOv8 Implementation")
        train_yolo()
    elif args.model == "rcnn":
        print(">>> Triggering Faster R-CNN Implementation")
        train_rcnn()
    elif args.model == "both":
        print(">>> Triggering both YOLOv8 and Faster R-CNN Implementations")
        train_yolo()
        train_rcnn()

if __name__ == "__main__":
    main()
