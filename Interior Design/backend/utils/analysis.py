import io
import os
import re
import torch
import torchvision.transforms as T
from PIL import Image

# Lazy globals
_places_model = None
_places_classes = None
_yolo_model = None


def _load_places365():
    global _places_model, _places_classes
    if _places_model is not None:
        return _places_model, _places_classes

    # Download labels if missing
    labels_path = os.path.join(os.path.dirname(__file__), "places365_categories.txt")
    if not os.path.exists(labels_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
        urllib.request.urlretrieve(url, labels_path)
    with open(labels_path, "r") as f:
        _places_classes = [line.strip().split(" ")[0][3:] for line in f]

    # Load resnet18 places365 weights
    weights_path = os.path.join(os.path.dirname(__file__), "resnet18_places365.pth.tar")
    if not os.path.exists(weights_path):
        import urllib.request
        url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
        urllib.request.urlretrieve(url, weights_path)

    import torchvision.models as models
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    _places_model = model
    return _places_model, _places_classes


def _classify_room(image: Image.Image) -> str:
    model, classes = _load_places365()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        top5 = torch.topk(prob, 5)
    labels = [classes[i] for i in top5.indices[0].tolist()]
    label_text = ",".join(labels)

    # Map to our room types by keyword
    lt = label_text.lower()
    if any(k in lt for k in ["bedroom", "bedchamber"]):
        return "bedroom"
    if any(k in lt for k in ["kitchen", "kitchenette"]):
        return "kitchen"
    if any(k in lt for k in ["bathroom", "toilet", "washroom"]):
        return "bathroom"
    if any(k in lt for k in ["living_room", "family_room", "lounge", "parlor"]):
        return "living"
    # Fallback
    return "living"


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def _detect_furniture(image: Image.Image):
    model = _load_yolo()
    results = model.predict(image, imgsz=960, conf=0.35, iou=0.5, verbose=False)
    detections = []
    if results and len(results) > 0:
        r0 = results[0]
        names = r0.names
        allowed = {"bed", "chair", "couch", "sofa", "dining table", "tv", "refrigerator", "oven", "toilet"}
        for box, cls, conf in zip(r0.boxes.xyxy.cpu().numpy(), r0.boxes.cls.cpu().numpy(), r0.boxes.conf.cpu().numpy()):
            name = names[int(cls)]
            if name in allowed:
                detections.append({
                    "label": "sofa" if name == "couch" else name,
                    "confidence": float(conf),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                })
    return detections


def analyze_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    room_type = _classify_room(image)
    detections = _detect_furniture(image)
    return {"room_type": room_type, "detections": detections}


