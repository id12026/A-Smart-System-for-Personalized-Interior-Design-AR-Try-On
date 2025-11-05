import os
import torch
import torchvision
from torch.utils.data import random_split, WeightedRandomSampler
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import json
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# -----------------------------
# Data Preparation
# -----------------------------
data_dir = 'home_interior_dataset'  # Room classification dataset
yolo_dataset_root = 'C:/Users/Reliance Digital/Downloads/furniture_dataset'  # YOLO dataset
yolo_yaml_path = os.path.join(yolo_dataset_root, "data.yaml")
synthetic_dir = 'synthetic_kitchen_dataset'  # Directory for synthetic images
os.makedirs(synthetic_dir, exist_ok=True)

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Room dataset directory {data_dir} not found.")
if not os.path.exists(yolo_dataset_root):
    raise FileNotFoundError(f"YOLO dataset directory {yolo_dataset_root} not found.")

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = torchvision.datasets.ImageFolder(data_dir, transform=train_transforms)
val_test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=val_test_transforms)

# Stats
classes = dataset.classes
class_counts = Counter([dataset.targets[i] for i in range(len(dataset))])
print("Classes:", classes)
print("Total images:", len(dataset))
print("Class-wise counts:")
for cls, count in class_counts.items():
    print(f"{classes[cls]}: {count}")

# Split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}, Test size: {len(test_ds)}")

# Handle imbalance
class_weights = [1.0 / class_counts[i] for i in range(len(classes))]
weights = [class_weights[label] for _, label in dataset.samples]
train_weights = [weights[idx] for idx in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

# Loaders
batch_size = 32
num_workers = 0
train_dl = torch.utils.data.DataLoader(train_ds, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)

# -----------------------------
# ResNet Model
# -----------------------------
class ResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        for param in self.network.parameters():
            param.requires_grad = False
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch+1}: train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# -----------------------------
# Device
# -----------------------------
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

# -----------------------------
# Training
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, patience=5):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        scheduler.step(result['val_loss'])
        early_stopping(result['val_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load('checkpoint.pt'))
            break
    return history

# Train ResNet (2 epochs only)
model = ResNet(num_classes=len(classes), dropout_rate=0.5)
model = to_device(model, device)
history = fit(2, 1e-4, model, train_dl, val_dl, torch.optim.Adam, patience=5)

# -----------------------------
# YOLOv8 Training (2 epochs only)
# -----------------------------
def train_yolo(yaml_path, epochs=2, imgsz=640):
    yolo_model = YOLO("yolov8n.pt")
    results = yolo_model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device="cpu",
        project="indoor_detection",
        name="yolov8_indoor",
        exist_ok=True,
    )
    return yolo_model, results

yolo_model, yolo_results = train_yolo(yolo_yaml_path, epochs=2, imgsz=640)

# -----------------------------
# Synthetic Data Generation with Stable Diffusion
# -----------------------------
def generate_synthetic_kitchen_images(num_images=10):
    # Load Stable Diffusion model (using a pre-trained checkpoint)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to(device)
    
    # Prompts for kitchen interiors
    prompts = [
        "modern kitchen with cabinets, chair, and refrigerator, high detail",
        "farmhouse kitchen with kitchen island and dining table, realistic lighting",
        "minimalist kitchen with white cabinets and marble floor, photorealistic",
        "cozy kitchen with wooden cabinets and a sink, bright natural light"
    ]
    
    for i in range(num_images):
        prompt = random.choice(prompts)
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image_path = os.path.join(synthetic_dir, f"synthetic_kitchen_{i}.png")
        image.save(image_path)
        print(f"Generated and saved: {image_path}")
    
    # Update dataset (you'll need to manually label these or use an annotation tool)
    print(f"Generated {num_images} synthetic kitchen images. Label them and add to {data_dir}/kitchen or update {yolo_yaml_path}.")

# Generate synthetic images (run this before training if needed)
generate_synthetic_kitchen_images(num_images=10)

# -----------------------------
# Prediction
# -----------------------------
def predict_room_and_furniture(img_path, resnet_model, yolo_model, classes):
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, None
    
    try:
        # Room classification
        img = Image.open(img_path).convert('RGB')
        transform = val_test_transforms
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = to_device(img_tensor, device)
        resnet_model.eval()
        with torch.no_grad():
            out = resnet_model(img_tensor)
            _, pred = torch.max(out, dim=1)
            room_type = classes[pred.item()]
            print(f"Predicted room type: {room_type} (confidence not directly available)")
        
        # Furniture detection
        yolo_results = yolo_model.predict(img_path, conf=0.1, save=True, device="cpu")
        furniture = []
        for result in yolo_results:
            if result.boxes.cls.numel() > 0:
                furniture.extend([yolo_model.names[int(cls)] for cls in result.boxes.cls])
                print(f"Detected objects: {[yolo_model.names[int(cls)] for cls in result.boxes.cls]} with confidence > {0.1}")
            else:
                print(f"No objects detected above confidence threshold (conf={0.1}) in {img_path}")
        return room_type, furniture if furniture else ["None detected"]
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# -----------------------------
# Recommendations
# -----------------------------
def recommend_furniture_and_style(room_type, detected_furniture):
    recommendations = {
        'bedroom': {
            'furniture': ['bed', 'wardrobe', 'nightstand'],
            'style': 'Modern minimalist, Scandinavian, or Bohemian',
            'layout': 'Place bed against the longest wall, nightstands on either side, wardrobe near a corner.'
        },
        'livingroom': {
            'furniture': ['sofa', 'coffee table', 'bookshelf'],
            'style': 'Contemporary, Industrial, or Mid-century modern',
            'layout': 'Central sofa facing a TV, coffee table in front, bookshelf against a wall.'
        },
        'kitchen': {
            'furniture': ['kitchen island', 'dining table', 'cabinets', 'chair', 'refrigerator'],
            'style': 'Modern, Farmhouse, or Minimalist',
            'layout': 'Island in the center, dining table near a window, cabinets along walls.'
        },
    }
    
    rec = recommendations.get(room_type, {
        'furniture': ['table', 'chair'],
        'style': 'Generic modern',
        'layout': 'Flexible layout based on room size.'
    })
    
    missing_furniture = [f for f in rec['furniture'] if f not in detected_furniture]
    return {
        'room_type': room_type,
        'detected_furniture': detected_furniture,
        'recommended_furniture': missing_furniture,
        'style': rec['style'],
        'layout': rec['layout']
    }

# -----------------------------
# AR Try-On
# -----------------------------
def generate_ar_html(furniture_list):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AR Furniture Try-On</title>
    <script src="https://aframe.io/releases/1.4.2/aframe.min.js"></script>
    <script src="https://raw.githack.com/AR-js-org/AR.js/master/aframe/build/aframe-ar-nft.js"></script>
</head>
<body>
    <a-scene vr-mode-ui="enabled: true" embedded arjs="sourceType: webcam; debugUIEnabled: false;">
        <a-assets>
            <a-asset-item id="chair-model" src="https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/models/chair.glb"></a-asset-item>
            <a-asset-item id="table-model" src="https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/models/table.glb"></a-asset-item>
            <a-asset-item id="cabinet-model" src="https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/models/cabinet.glb"></a-asset-item>
            <a-asset-item id="refrigerator-model" src="https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/models/refrigerator.glb"></a-asset-item>
            <a-asset-item id="kitchen_island-model" src="https://raw.githubusercontent.com/AR-js-org/AR.js/master/data/models/island.glb"></a-asset-item>
        </a-assets>
        {''.join([f'<a-entity gltf-model="#{f.lower().replace(" ", "_")}-model" position="{i} 0 0" scale="0.5 0.5 0.5"></a-entity>' for i, f in enumerate(furniture_list)])}
        <a-camera position="0 1.6 0"></a-camera>
    </a-scene>
</body>
</html>
"""
    with open('ar_tryon.html', 'w') as f:
        f.write(html_content)
    return 'ar_tryon.html'

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Generate synthetic data before training
    generate_synthetic_kitchen_images(num_images=10)
    
    user_image = r"C:/Users/Reliance Digital/Downloads/indian-style-kitchen-interior-designs-for-your-home.jpg"
    room_type, furniture = predict_room_and_furniture(user_image, model, yolo_model, classes)
    
    if room_type and furniture:
        print(f"Room Type: {room_type}")
        print(f"Detected Furniture: {furniture}")
        
        recommendations = recommend_furniture_and_style(room_type, furniture)
        print("\nPersonalized Recommendations:")
        print(json.dumps(recommendations, indent=2))
        
        ar_html = generate_ar_html(recommendations['recommended_furniture'])
        print(f"AR visualization generated: {ar_tryon.html}")
    else:
        print("Prediction failed or no furniture detected.")