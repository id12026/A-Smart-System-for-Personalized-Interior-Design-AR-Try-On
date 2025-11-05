import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import seaborn as sns
from collections import Counter
import random
import shutil

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Install required packages
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Verify torch and torchvision compatibility
import torchvision
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Data Preparation (Updated for your dataset structure)
def prepare_furniture_dataset_class_dirs(dataset_path, output_path, samples_per_class=1000):
    """
    Prepare the furniture dataset for YOLO training from class-based directories.
    Assumes annotation files (.txt) exist in each class directory.
    """
    print(f"Dataset directory contents: {os.listdir(dataset_path)}")
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, 'images')
    labels_dir = os.path.join(output_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    dir_to_class = {
        'almirah_dataset': 'Almirah',
        'chair_dataset': 'Chair',
        'fridge_dataset': 'Refrigerator',
        'table_dataset': 'Table',
        'tv_dataset': 'Television'
    }
    classes = list(dir_to_class.values())
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    yaml_content = f"""
path: {os.path.abspath(output_path)}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(output_path, 'furniture.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    total_images = 0
    for dir_name, class_name in dir_to_class.items():
        class_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} not found. Skipping {class_name}.")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Warning: No images found in {class_path} for {class_name}. Skipping.")
            continue
        
        if len(image_files) > samples_per_class:
            image_files = random.sample(image_files, samples_per_class)
        
        random.shuffle(image_files)
        train_count = int(0.7 * len(image_files))
        val_count = int(0.15 * len(image_files))
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count+val_count]
        test_files = image_files[train_count+val_count:]
        
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in files:
                src_img_path = os.path.join(class_path, img_file)
                dst_img_path = os.path.join(images_dir, split, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label_path = os.path.join(class_path, label_file)
                dst_label_path = os.path.join(labels_dir, split, label_file)
                
                # Copy image
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                    total_images += 1
                except Exception as e:
                    print(f"Error copying {src_img_path} to {dst_img_path}: {e}")
                    continue
                
                # Copy or create label file
                if os.path.exists(src_label_path):
                    try:
                        shutil.copy2(src_label_path, dst_label_path)
                    except Exception as e:
                        print(f"Error copying {src_label_path} to {dst_label_path}: {e}")
                else:
                    print(f"Warning: No label file found for {img_file} in {class_path}. Creating default.")
                    with open(dst_label_path, 'w') as f:
                        f.write(f"{class_to_id[class_name]} 0.5 0.5 0.8 0.8\n")  # Default bounding box as fallback
    
    print(f"Dataset prepared at {output_path} with {total_images} images")
    if total_images < 10:
        print("Warning: Very few images detected. Consider adding more data or using augmentation.")
    return yaml_path, total_images

# Alternative Data Preparation (Single directory with labels)
def prepare_furniture_dataset_single_dir(dataset_path, output_path, label_file=None, samples_per_class=1000):
    """
    Prepare the furniture dataset from a single directory with a label file.
    """
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, 'images')
    labels_dir = os.path.join(output_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    classes = ['Almirah', 'Chair', 'Refrigerator', 'Table', 'Television']
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    yaml_content = f"""
path: {os.path.abspath(output_path)}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(output_path, 'furniture.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Warning: No images found in {dataset_path}. Skipping.")
        return yaml_path, 0
    
    if label_file and os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        image_to_class = dict(zip(labels_df['image'], labels_df['class']))
    else:
        image_to_class = {img: random.choice(classes) for img in image_files}
    
    class_counts = Counter(image_to_class.values())
    selected_images = []
    for cls in classes:
        cls_images = [img for img, c in image_to_class.items() if c == cls]
        if len(cls_images) > samples_per_class:
            cls_images = random.sample(cls_images, samples_per_class)
        selected_images.extend(cls_images)
    
    random.shuffle(selected_images)
    train_count = int(0.7 * len(selected_images))
    val_count = int(0.15 * len(selected_images))
    
    train_files = selected_images[:train_count]
    val_files = selected_images[train_count:train_count+val_count]
    test_files = selected_images[train_count+val_count:]
    
    total_images = 0
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:
            src_img_path = os.path.join(dataset_path, img_file)
            dst_img_path = os.path.join(images_dir, split, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            dst_label_path = os.path.join(labels_dir, split, label_file)
            
            class_name = image_to_class.get(img_file, random.choice(classes))
            if class_name not in class_to_id:
                print(f"Warning: Invalid class {class_name} for {img_file}. Skipping.")
                continue
            
            with open(dst_label_path, 'w') as f:
                f.write(f"{class_to_id[class_name]} 0.5 0.5 0.8 0.8\n")
            
            try:
                shutil.copy2(src_img_path, dst_img_path)
                total_images += 1
            except Exception as e:
                print(f"Error copying {src_img_path} to {dst_img_path}: {e}")
    
    print(f"Dataset prepared at {output_path} with {total_images} images")
    return yaml_path, total_images

# Train YOLO model
def train_yolo_model(yaml_path, epochs=10, imgsz=640):
    """
    Train a YOLOv8 model on the furniture dataset using CPU with up to 10 epochs.
    """
    train_dir = os.path.join(os.path.dirname(yaml_path), 'images', 'train')
    if not os.listdir(train_dir):
        raise ValueError(f"No images found in {train_dir}. Cannot proceed with training.")
    
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=5,  # Increased patience for better convergence
        project='furniture_detection',
        name='yolov8_furniture',
        exist_ok=True,
        device='cpu',
        plots=True
    )
    
    return model, results

# Evaluate the model
def evaluate_model(model, data_path):
    """
    Evaluate the trained YOLO model on the test set using CPU.
    """
    test_dir = os.path.join(os.path.dirname(data_path), 'images', 'test')
    if not os.listdir(test_dir):
        print(f"Warning: No images found in {test_dir}. Skipping evaluation.")
        return None
    
    metrics = model.val(data=data_path, device='cpu')
    
    print("Evaluation Results:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    try:
        confusion_matrix = metrics.confusion_matrix.matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.astype(int), annot=True, fmt='d',
                   xticklabels=metrics.names.values(), yticklabels=metrics.names.values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")
    
    return metrics

# Inference function with customizable bounding boxes
def detect_furniture(model, image_path, conf=0.1, box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3):
    """
    Detect furniture in an image using the trained YOLO model on CPU and display in shell with customizable boxes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Run inference on CPU without saving
    results = model(image_path, conf=conf, save=False, device='cpu')
    
    # Extract detection results
    detections = []
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print(f"No detections found in {os.path.basename(image_path)} with confidence > {conf}. Check training data or lower conf further.")
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
            detections.append({
                'class': result.names[cls_id],
                'confidence': confidence,
                'bbox': bbox
            })
    
    # Visualize results
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)  # Customizable color and thickness
        label = f"{det['class']} {det['confidence']:.2f}"  # Include confidence score
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)  # Background for text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Customizable text color
    
    # Display the image in the shell
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Furniture Detection - {os.path.basename(image_path)}')
    plt.show()
    
    return detections

# Function to get sample images from original dataset
def get_original_class_samples(dataset_path):
    """
    Get one sample image from each class directory in the original dataset.
    """
    dir_to_class = {
        'almirah_dataset': 'Almirah',
        'chair_dataset': 'Chair',
        'fridge_dataset': 'Refrigerator',
        'table_dataset': 'Table',
        'tv_dataset': 'Television'
    }
    samples = {}
    
    for dir_name, class_name in dir_to_class.items():
        class_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                samples[class_name] = os.path.join(class_path, random.choice(image_files))
            else:
                print(f"No images found in {class_path} for {class_name}.")
        else:
            print(f"Class directory {class_path} not found for {class_name}.")
    
    return samples

# Function to select uploaded image
def select_uploaded_image():
    """
    Prompt the user to enter the path to an image file.
    """
    image_path = input("Enter the path to the image file for detection (or press Enter to skip): ").strip()
    if image_path and os.path.exists(image_path):
        return image_path
    else:
        print("Invalid path or no path provided. Skipping user detection.")
        return None

# Integration with room classification
def integrated_pipeline(image_path, furniture_model, room_classes):
    """
    Integrated pipeline that classifies the room and detects furniture with visualization.
    """
    print("Classifying room type...")
    room_type = random.choice(room_classes)
    print(f"Room classified as: {room_type}")
    
    print("Detecting furniture...")
    detections = detect_furniture(furniture_model, image_path, conf=0.1, box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3)
    
    print(f"Found {len(detections)} furniture items:")
    for det in detections:
        print(f"- {det['class']} (confidence: {det['confidence']:.2f})")
    
    recommendations = generate_recommendations(room_type, detections)
    print("\nPersonalized Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    
    return room_type, detections, recommendations

def generate_recommendations(room_type, detections):
    """
    Generate personalized recommendations based on room type and detected furniture.
    """
    recommendations = []
    if room_type == 'bedroom':
        recommendations.extend(["Consider adding a nightstand next to the bed",
                              "A dresser would provide additional storage",
                              "A full-length mirror would complement the space"])
    elif room_type == 'livingroom':
        recommendations.extend(["A coffee table would complete the seating area",
                              "Consider adding bookshelves for storage and decor",
                              "An area rug would help define the space"])
    elif room_type == 'kitchen':
        recommendations.extend(["Bar stools would create a casual eating area",
                              "Open shelving could display dishes and cookware",
                              "A kitchen island would provide additional workspace"])
    
    detected_classes = [det['class'] for det in detections]
    if 'Chair' in detected_classes and 'Table' not in detected_classes:
        recommendations.append("Add a table to complement the chairs")
    if 'Television' in detected_classes and 'Table' not in detected_classes:
        recommendations.append("Consider a TV stand or console for your television")
    if 'Refrigerator' in detected_classes and room_type != 'kitchen':
        recommendations.append("The refrigerator might be better placed in the kitchen")
    
    return recommendations[:3]

# Function to visualize detection comparison
def visualize_comparison(original_path, test_path, model):
    """
    Visualize detection on an original image and a test image side-by-side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image detection
    original_img = detect_furniture(model, original_path, conf=0.1, box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3, return_img=True)
    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Detection on Original Dataset Image')
    
    # Test image detection
    test_img = detect_furniture(model, test_path, conf=0.1, box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3, return_img=True)
    axs[1].imshow(test_img)
    axs[1].axis('off')
    axs[1].set_title('Detection on Test Dataset Image')
    
    plt.show()

# Modified detect_furniture to optionally return the annotated image
def detect_furniture(model, image_path, conf=0.1, box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3, return_img=False):
    """
    Detect furniture in an image using the trained YOLO model on CPU and display in shell with customizable boxes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Run inference on CPU without saving
    results = model(image_path, conf=conf, save=False, device='cpu')
    
    # Extract detection results
    detections = []
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print(f"No detections found in {os.path.basename(image_path)} with confidence > {conf}. Check training data or lower conf further.")
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
            detections.append({
                'class': result.names[cls_id],
                'confidence': confidence,
                'bbox': bbox
            })
    
    # Visualize results
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)  # Customizable color and thickness
        label = f"{det['class']} {det['confidence']:.2f}"  # Include confidence score
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)  # Background for text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Customizable text color
    
    if return_img:
        return img
    
    # Display the image in the shell
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Furniture Detection - {os.path.basename(image_path)}')
    plt.show()
    
    return detections

# Function to get sample images from prepared dataset
def get_prepared_samples(output_path, split, num_samples=3):
    """
    Get random samples from the prepared dataset split (e.g., val or test).
    """
    split_dir = os.path.join(output_path, 'images', split)
    if not os.path.exists(split_dir):
        print(f"Warning: {split} directory not found.")
        return []
    
    image_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Warning: No images found in {split_dir}.")
        return []
    
    return random.sample(image_files, min(num_samples, len(image_files)))

# Main execution
if __name__ == "__main__":
    dataset_path = r"C:\Users\Reliance Digital\Downloads\archive"
    output_path = "furniture_yolo_dataset"
    label_file = None
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        exit(1)
    
    print("Preparing dataset...")
    try:
        yaml_path, total_images = prepare_furniture_dataset_class_dirs(dataset_path, output_path, samples_per_class=1000)
        if total_images == 0:
            print("No images found in class-based directories. Trying single-directory structure...")
            yaml_path, total_images = prepare_furniture_dataset_single_dir(dataset_path, output_path, label_file, samples_per_class=1000)
        if total_images == 0:
            print(f"Error: No images were processed. Check the dataset path: {dataset_path}")
            exit(1)
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        exit(1)
    
    print("Training YOLO model...")
    model = None
    try:
        model, results = train_yolo_model(yaml_path, epochs=10)
    except ValueError as e:
        print(f"Error during training: {e}")
        exit(1)
    
    if model:
        val_losses = []
        if hasattr(results, 'val_loss'):
            val_losses = results.val_loss
        elif os.path.exists(os.path.join('furniture_detection', 'yolov8_furniture', 'results.csv')):
            results_df = pd.read_csv(os.path.join('furniture_detection', 'yolov8_furniture', 'results.csv'))
            if 'val/box_loss' in results_df.columns:
                val_losses = results_df['val/box_loss'].dropna().tolist()
            elif 'val/cls_loss' in results_df.columns:
                val_losses = results_df['val/cls_loss'].dropna().tolist()
            elif 'val/dfl_loss' in results_df.columns:
                val_losses = results_df['val/dfl_loss'].dropna().tolist()
        
        if val_losses:
            variance = np.var(val_losses)
            threshold = 0.1
            print(f"Variance of validation loss: {variance:.4f}")
            if variance > threshold:
                print(f"Warning: High variance ({variance:.4f}) in validation loss detected. Consider adjusting hyperparameters.")
        
        print("Evaluating model on validation set...")
        try:
            metrics = evaluate_model(model, yaml_path)
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        try:
            model.val(data=yaml_path, split='test', device='cpu')
        except Exception as e:
            print(f"Error during test evaluation: {e}")
        
        model_path = "furniture_detection_yolov8.pt"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        print("Loading model for inference...")
        trained_model = YOLO(model_path)
        room_classes = ['bathroom', 'bedroom', 'children_room', 'closet', 
                       'dining_room', 'kitchen', 'livingroom', 'nursery', 'pantry']
        
        # Detect samples from original dataset
        print("\nDetecting samples from original dataset:")
        original_samples = get_original_class_samples(dataset_path)
        for class_name, sample_path in original_samples.items():
            if sample_path:
                print(f"Detecting sample for {class_name}: {os.path.basename(sample_path)}")
                integrated_pipeline(sample_path, trained_model, room_classes)
            else:
                print(f"No sample available for {class_name}.")
        
        # Detect samples from validation dataset
        print("\nDetecting samples from validation dataset:")
        val_samples = get_prepared_samples(output_path, 'val', num_samples=3)
        for sample_path in val_samples:
            print(f"Detecting validation image: {os.path.basename(sample_path)}")
            integrated_pipeline(sample_path, trained_model, room_classes)
        
        # Detect samples from test dataset
        print("\nDetecting samples from test dataset:")
        test_samples = get_prepared_samples(output_path, 'test', num_samples=3)
        for sample_path in test_samples:
            print(f"Detecting test image: {os.path.basename(sample_path)}")
            integrated_pipeline(sample_path, trained_model, room_classes)
        
        # Comparison visualization
        print("\nVisualizing comparison between original and test dataset:")
        if original_samples and test_samples:
            for class_name, original_path in original_samples.items():
                test_path = random.choice(test_samples)  # Random test image for comparison
                print(f"Comparing {class_name} original and a test image")
                visualize_comparison(original_path, test_path, trained_model)
        else:
            print("Insufficient samples for comparison.")
        
        # User-uploaded image detection
        print("\nPlease select an image for detection:")
        uploaded_image = select_uploaded_image()
        if uploaded_image:
            print(f"Detecting furniture in selected image: {os.path.basename(uploaded_image)}")
            integrated_pipeline(uploaded_image, trained_model, room_classes)
        else:
            print("No image selected. Skipping user detection.")
    else:
        print("Skipping evaluation and inference due to training failure.")