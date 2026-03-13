import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import yaml

class WasteDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def _parse_label_file(self, label_path, width, height):
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])
                        # Convert YOLO normalized to Pascal VOC [xmin, ymin, xmax, ymax]
                        xmin = (x_c - w / 2) * width
                        ymin = (y_c - h / 2) * height
                        xmax = (x_c + w / 2) * width
                        ymax = (y_c + h / 2) * height
                        boxes.append([xmin, ymin, xmax, ymax])
                        # Faster R-CNN labels are 1-indexed (0 is background)
                        labels.append(cls + 1)
        return boxes, labels

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_filename)
        
        boxes, labels = self._parse_label_file(label_path, width, height)
        
        if not boxes:
            # Handle images with no annotations (background only)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            img = self.transform(img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_rcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load data config
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # Initialize Dataset
    train_dataset = WasteDataset(
        data_config['train'],
        data_config['train'].replace('images', 'labels')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    # Initialize Model
    print("Loading pretrained Faster R-CNN weights (this may take a moment)...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = data_config['nc'] + 1  # classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    print("--- Starting Faster R-CNN Training ---")
    print("Note: On MX450, the first batch may take 1-2 minutes to load.")
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {losses.item():.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'rcnn_waste_best.pth')
    print("--- Faster R-CNN Training Completed. Model saved as rcnn_waste_best.pth ---")

if __name__ == "__main__":
    train_rcnn()
