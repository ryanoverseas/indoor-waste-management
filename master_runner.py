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
        help="Select the model to run: 'yolo', 'rcnn', or 'both'"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Select the operation mode: 'train' (learning) or 'test' (evaluation)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.model == "yolo":
            print(">>> Starting YOLOv8 Training")
            train_yolo()
        elif args.model == "rcnn":
            print(">>> Starting Faster R-CNN Training")
            train_rcnn()
        elif args.model == "both":
            print(">>> Starting Training for both models")
            train_yolo()
            train_rcnn()
    
    elif args.mode == "test":
        # Import test functions only when needed
        from test_models import test_yolo, test_rcnn
        if args.model == "yolo":
            test_yolo()
        elif args.model == "rcnn":
            test_rcnn()
        elif args.model == "both":
            test_yolo()
            test_rcnn()

if __name__ == "__main__":
    main()
