# A-Smart-System-for-Personalized-Interior-Design-AR-Try-On  🏠

## 📌 Introduction

Interior design often feels overwhelming — **90% of users struggle to visualize furniture in their rooms before purchasing**, leading to wrong choices and dissatisfaction.
This project builds a **smart interior design system** that combines **Computer Vision (CV), Generative AI, and Augmented Reality (AR)** to help users:

* Detect furniture & room layouts.
* Get **personalized design recommendations**.
* View 3D/AR  image.

---

## 🎯 Problem Statement

* Users cannot easily imagine how furniture will look or fit in their rooms.
* No **personalized recommendation system** adapts designs to actual room conditions.
* Lack of a proper **“try-before-you-buy” feature** in interior design/e-commerce.

---

## ✅ Objectives

1. Analyze room images using **CV models** (ResNet & YOLOv8).
2. Suggest **personalized furniture & design styles**.
3. Provide an **interactive AR Try-On feature** using WebXR.
4. Enable **feedback loops** to refine recommendations.

---

## 🧩 Methodology

1. Data Preparation – Collect room images, preprocess (resize, augment, normalize), and build ETL pipeline with annotations.
2. Computer Vision Models – Train ResNet for room classification and YOLOv8 for furniture detection.
3. Personalized Recommendations – Suggest styles, furniture, and optimized layouts.
4. AR Try-On – Allow users to virtually place furniture and see 3D/AR visualization using Web XR(Web Extended Reality-API).
5. Feedback & Improvement – Collect user feedback and refine future recommendations.

---

## ✨ Features

- **🧠 Smart Room Analysis**: Automatically detects room type (e.g., Bedroom, Kitchen) and existing furniture from a single upload using ResNet18 and YOLOv8 models.
- **🎨 AI-Powered Redesign**: Generates stunning, photorealistic redesigned images of your room based on your style preferences using Google's Gemini AI.
- **👁️ Interactive 3D Preview**: Visualize the AI-generated design in an immersive 3D space using Three.js. Rotate and zoom to explore the new layout.
- **📝 Personalized Recommendations**: Receives detailed, markdown-formatted suggestions for furniture, color schemes, and layout improvements.
- **📄 Exportable Reports**: Download a comprehensive PDF report of your design, complete with before/after comparisons and recommendations using jsPDF.
- **💾 Session History**: All your design sessions are automatically saved to SQLite database for future reference.

---


## 🔄 Workflow

<img width="1099" height="1024" alt="image" src="https://github.com/user-attachments/assets/dc82a3c9-0250-44a1-9a62-32d4b088fcc9" />



---

## 🛠️ Tech Stack

### **Frontend Stack**
- **React 18.2.0** - UI library
- **Vite 6.3.1** - Build tool and dev server
- **Ant Design 5.12.2** - UI component library
- **Three.js 0.181.0** - 3D graphics and visualization
- **Chart.js 4.4.7** - Data visualization (pie charts)
- **Axios 1.6.2** - HTTP client
- **React Markdown 10.1.0** - Markdown rendering
- **jsPDF 2.5.2** + **html2canvas 1.4.1** - PDF generation

### **Backend Stack**
- **Python 3.12+** - Backend language
- **FastAPI 0.115.12** - Web framework
- **Uvicorn 0.34.2** - ASGI server
- **Poetry** - Dependency management
- **SQLAlchemy 2.0.0** - ORM and database toolkit
- **SQLite** - Database engine

### **AI/ML Stack**
- **Google Gemini 2.0 Flash** - Image generation AI
- **PyTorch 2.3.0** - Deep learning framework
- **ResNet18 (Places365)** - Room classification model
- **YOLOv8** - Object detection model
- **Ultralytics 8.3.30** - YOLO implementation

### **Utilities & Tools**
- **python-dotenv** - Environment variable management
- **react-toastify** - Notifications
- **Pillow** - Image processing
- **torchvision** - Computer vision utilities

---
## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- Poetry (for Python dependency management)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/id12026/A-Smart-System-for-Personalized-Interior-Design-AR-Try-On.git
   cd your-repo-name
  
---

2. Backend Setup

bash
```
cd backend

# Install Python dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell

# Start the FastAPI server (runs on http://localhost:8000)
uvicorn main:app --reload
```
3. Frontend Setup
```
cd frontend

# Install Node.js dependencies
npm install

# Start the development server (runs on http://localhost:5173)
npm run dev

```

4. Access the Application

Frontend: http://localhost:5173

Backend API Docs: http://localhost:8000/docs   

## 🏗️ System Architecture
## 🏗️ Project Structure

```bash
GEN-AI-HOME-INTERIOR-DESIGNER/
├── backend/
│   ├── routers/
│   │   ├── auth.py          # Auth routes
│   │   ├── designs.py       # Design CRUD
│   │   └── tryon.py         # AI try-on endpoint
│   ├── utils/
│   │   ├── analysis.py
│   │   ├── base64_helpers.py
│   │   ├── places365_categories.txt
│   │   ├── resnet18_places365.pth.tar
│   │   └── yolov8n.pt
│   ├── .env
│   ├── database.py
│   ├── interior_designer.db
│   ├── main.py              # FastAPI entry
│   ├── poetry.lock
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── public/
│   ├── .env
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   └── vite.config.js
├── datasets/
│   ├── furniture_detection/yolov8_furniture/
│   ├── home_interior_dataset/        # 3,348 images
│   ├── furniture_yolo_dataset/
│   ├── indoor_detection/
│   ├── synthetic_dataset/
│   └── synthetic_kitchen_dataset/
├── models/
│   ├── best_classifier.pth
│   ├── checkpoint_resnet.pt
│   ├── checkpoint.pt
│   ├── furniture_detection_yolov8.pt
│   └── yolov8n.pt
├── notebooks/
│   ├── all.ipynb
│   └── interior.ipynb
├── evaluation/
│   ├── test_eval/
│   ├── test_evaluation/
│   ├── prediction_yolo8vl/
│   ├── runs/detect/
│   ├── comparison_output.png
│   ├── confusion_matrix_validation.png
│   └── test_detection_result.jpg
├── ar_tryon.html
├── furniture_detection.py
├── m.py
├── yolo_yaml
└── TECHNICAL_DOCUMENTATION.md
```
🎯 How It Works
1. Image Upload & Analysis
User uploads a room image.

Backend processes it with two models:

ResNet18: Classifies the room type (e.g., "Bedroom", "Kitchen").

YOLOv8: Detects and localizes existing furniture items.

2. Design Generation
User selects design preferences (style, colors, etc.).

Backend sends the original image and preferences to Google Gemini API.

AI generates a new interior design image and text recommendations.

3. Visualization & Export
Frontend displays the AI-generated result.

User can view the design in an interactive 3D space.

Download a detailed PDF report of the design.


## Request Flow
1. Image Upload & Analysis

```
User uploads image → Frontend sends to /api/analyze
→ Backend loads ResNet18 & YOLOv8 models
→ Room classification + Furniture detection
→ Returns room_type & detections array
→ Frontend auto-fills room type dropdown
```
2. Design Generation
``` 
User submits form → Frontend sends to /api/try-on
→ Backend validates image & parameters
→ Calls Google Gemini API with prompt + image
→ Receives generated image + markdown recommendations
→ Runs analysis on generated image (detections)
→ Returns JSON with image URL, text, success_rate, analyses
→ Frontend displays result, saves to database
```
3. Data Persistence
```
After generation → Frontend calls /api/designs/save
→ Backend validates & stores in SQLite
→ Returns saved design record
→ Frontend shows success toast
```

## 📊 Model Performance
Room Classification (ResNet18)
Dataset: 3,348 images across 9 room classes

Test Accuracy: 88%

Classes: Bathroom, Bedroom, Children Room, Closet, Dining Room, Kitchen, Livingroom, Nursery, Pantry

Furniture Detection (YOLOv8)
Model: YOLOv8l

Dataset: Indoor object detection with 10 furniture classes

mAP@0.5: 0.45

Classes: Door, Cabinet, Refrigerator, Window, Chair, Table, Couch, etc.

---

## 🖼️ Screenshots
<img width="701" height="335" alt="image" src="https://github.com/user-attachments/assets/88340c50-1ff5-46f0-9f07-5d64dc667011" />
<img width="439" height="309" alt="image" src="https://github.com/user-attachments/assets/32e12c54-cdad-44ae-b013-c6906068f5c0" />
<img width="750" height="638" alt="image" src="https://github.com/user-attachments/assets/5d3f4645-7846-4f09-a5e8-c07d0bd7754d" />
<img width="623" height="415" alt="image" src="https://github.com/user-attachments/assets/5125e091-06aa-483d-a97e-173eaf1e17b3" />
<img width="1184" height="683" alt="image" src="https://github.com/user-attachments/assets/52051f6f-e7e7-4b37-a359-069f6ce611cb" />
<img width="940" height="331" alt="image" src="https://github.com/user-attachments/assets/75927d71-7ec6-4381-b8dc-67f650c02069" />
<img width="489" height="400" alt="image" src="https://github.com/user-attachments/assets/76b56a84-82a1-4e2b-928e-37a03650ae03" />
<img width="673" height="886" alt="image" src="https://github.com/user-attachments/assets/2c392ba8-c4e0-43ae-abc3-c758a828481d" />
<img width="646" height="579" alt="image" src="https://github.com/user-attachments/assets/151155a4-6533-4058-abf4-632c38448ea3" />
<img width="646" height="864" alt="image" src="https://github.com/user-attachments/assets/b3b6c67f-14ea-4b98-aad1-25ded888d4b5" />
<img width="920" height="766" alt="image" src="https://github.com/user-attachments/assets/32489dc9-6d26-4164-841a-ffaea26adec2" />
<img width="548" height="887" alt="image" src="https://github.com/user-attachments/assets/661f77c9-c335-45e2-bfec-d4966b2b9f22" />

---
## 🔧 API Endpoints
Method	Endpoint	Description
POST	/api/analyze	Analyze room image (classification + detection)
POST	/api/try-on	Generate new interior design
POST	/api/designs/save	Save design session to database
GET	/api/designs	Retrieve saved design sessions

---
## 🚧 Challenges & Solutions
Challenge	Solution
Imbalanced Dataset	Used WeightedRandomSampler and data augmentation
Model Integration	Created a modular backend with separate model handlers
Large File Uploads	Implemented file validation and compression
3D Visualization	Used Three.js with optimized texture loading
PDF Generation	Client-side generation using jsPDF and html2canvas

---
## 🔮 Future Enhancements

Depth Estimation: Integrate MiDaS for better spatial understanding

User Authentication: Personal accounts with design history

E-commerce Integration: Direct links to purchase recommended furniture

Mobile App: React Native version for iOS and Android

AR Mode: True augmented reality using device camera

---

## 🎓 Project Details
This project was developed as a Capstone Project for the 7th Semester of B.Tech in Computer Science and Engineering - Data Science at Woxsen University.

## Team Members:

Mohitha Bandi (22WUO0105037)

Pailla Bhavya (22WUO0105020)

T. Harshavardhan Reddy (22WUO0105023)

Supervised by: Dr. Bhargav Prajwal Pathri, Assistant Professor, SOT, Woxsen University.
✨ *This project bridges AI + CV + AR +ML to revolutionize personalized interior design and reduce wrong furniture purchase decisions.*
