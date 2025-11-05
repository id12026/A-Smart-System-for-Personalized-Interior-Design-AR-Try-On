# 🏗️ Technical Documentation: Personalized Interior Design and AR Try-On

## 📋 Table of Contents
1. [Tech Stack Overview](#tech-stack-overview)
2. [Languages Used](#languages-used)
3. [Architecture & Workflow](#architecture--workflow)
4. [Implementation Steps](#implementation-steps)
5. [Technology Choices & Rationale](#technology-choices--rationale)

---

## 🎯 Tech Stack Overview

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

## 💻 Languages Used

### **1. JavaScript/JSX (Frontend)**
- **Why Used:**
  - React ecosystem standard
  - Rich ecosystem for UI development
  - Component-based architecture
  - Strong tooling (Vite, ESLint)
  - Real-time interactivity

### **2. Python (Backend)**
- **Why Used:**
  - Strong ML/AI libraries (PyTorch, Ultralytics)
  - FastAPI for async APIs
  - Large ecosystem for image processing
  - Easy integration with ML models
  - Good performance for AI workloads

### **3. SQL (Database)**
- **Why Used:**
  - SQLite for lightweight persistence
  - SQLAlchemy ORM for type safety
  - Standard relational data model

### **4. CSS**
- **Why Used:**
  - Styling and animations
  - Responsive design
  - Custom themes and gradients

### **5. Markdown**
- **Why Used:**
  - AI-generated recommendations formatting
  - Easy parsing and rendering
  - Human-readable report generation

---

## 🔄 Architecture & Workflow

### **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   UI Layer   │→ │  State Mgmt   │→ │  API Client   │   │
│  │ (Components) │  │  (React Hooks)│  │   (Axios)     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP/REST API
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (FastAPI + Python)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  API Routes  │→ │  ML Models    │→ │  Database     │   │
│  │   (Routers)  │  │ (ResNet/YOLO) │  │  (SQLAlchemy) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│           │                              │                 │
│           ▼                              ▼                 │
│  ┌──────────────────────────────────────────────┐          │
│  │      Google Gemini API (Image Generation)   │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### **Request Flow**

1. **Image Upload & Analysis**
   ```
   User uploads image → Frontend sends to /api/analyze
   → Backend loads ResNet18 & YOLOv8 models
   → Room classification + Furniture detection
   → Returns room_type & detections array
   → Frontend auto-fills room type dropdown
   ```

2. **Design Generation**
   ```
   User submits form → Frontend sends to /api/try-on
   → Backend validates image & parameters
   → Calls Google Gemini API with prompt + image
   → Receives generated image + markdown recommendations
   → Runs analysis on generated image (detections)
   → Returns JSON with image URL, text, success_rate, analyses
   → Frontend displays result, saves to database
   ```

3. **Data Persistence**
   ```
   After generation → Frontend calls /api/designs/save
   → Backend validates & stores in SQLite
   → Returns saved design record
   → Frontend shows success toast
   ```

---

## 🛠️ Implementation Steps

### **Phase 1: Project Setup & Infrastructure**

#### **Step 1.1: Backend Setup**
```bash
# Create backend directory structure
backend/
├── main.py              # FastAPI app entry point
├── routers/            # API route handlers
├── utils/              # Utility functions
├── pyproject.toml      # Poetry dependencies
└── .env                # Environment variables
```

**Actions:**
- Initialize Poetry project
- Install FastAPI, Uvicorn, dependencies
- Setup CORS middleware
- Configure environment variables

#### **Step 1.2: Frontend Setup**
```bash
# Create frontend with Vite
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

**Actions:**
- Install React, Vite, Ant Design
- Configure Vite for development
- Setup ESLint and code quality tools
- Create component structure

### **Phase 2: Core Backend Development**

#### **Step 2.1: API Endpoints**
- **`/api/analyze`** - Image analysis endpoint
  - Accepts image file
  - Runs ResNet18 for room classification
  - Runs YOLOv8 for furniture detection
  - Returns JSON with room_type and detections

- **`/api/try-on`** - Design generation endpoint
  - Accepts image + design parameters
  - Validates file size and type
  - Calls Google Gemini API
  - Processes response (image + text)
  - Analyzes generated image
  - Returns complete result

- **`/api/designs/save`** - Database persistence
  - Accepts design selection data
  - Stores in SQLite via SQLAlchemy
  - Returns saved record

#### **Step 2.2: ML Model Integration**

**ResNet18 (Places365) - Room Classification:**
```python
# Load pre-trained ResNet18 model
model = models.resnet18(num_classes=365)
checkpoint = torch.load("resnet18_places365.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Classify room type
transform = T.Compose([T.Resize(256), T.CenterCrop(224), ...])
image_tensor = transform(image)
predictions = model(image_tensor)
room_type = map_to_custom_categories(predictions)
```

**YOLOv8 - Furniture Detection:**
```python
# Load YOLOv8 model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Detect furniture
results = model.predict(image, imgsz=960, conf=0.35, iou=0.5)
detections = extract_furniture_boxes(results)
```

#### **Step 2.3: Database Schema**
```python
class DesignSelection(Base):
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    room_type = Column(String)
    design_style = Column(String)
    background_color = Column(String)
    foreground_color = Column(String)
    additional_instructions = Column(Text)
    interior_recommendations = Column(Text)
    plants_info = Column(Text)
    created_at = Column(DateTime)
```

### **Phase 3: Frontend Development**

#### **Step 3.1: Core Components**
- **ImageUpload.jsx** - File upload with preview
- **CompareSlider.jsx** - Before/after image comparison
- **DetectedImage.jsx** - Image with detection overlay
- **StepReport.jsx** - Step-by-step recommendations parser
- **MarkdownCard.jsx** - Markdown content renderer
- **ReportPrint.jsx** - PDF report template
- **ThreeShowcase.jsx** - 3D interactive image viewer

#### **Step 3.2: State Management**
```javascript
// React Hooks for state
const [homeImage, setHomeImage] = useState(null);
const [roomType, setRoomType] = useState("");
const [detections, setDetections] = useState([]);
const [result, setResult] = useState(null);
```

#### **Step 3.3: API Integration**
```javascript
// Auto-analyze on image upload
useEffect(() => {
  if (homeImage) {
    analyzeImage(homeImage);
  }
}, [homeImage]);

// Generate design
const handleSubmit = async () => {
  const response = await axios.post("/api/try-on", formData);
  setResult(response.data);
  saveToDatabase(response.data.text);
};
```

#### **Step 3.4: PDF Generation**
```javascript
// Capture HTML to canvas
const canvas = await html2canvas(target, {...});

// Convert to PDF with pagination
const pdf = new jsPDF("p", "mm", "a4");
// Slice canvas into pages
pdf.addImage(imgData, "PNG", margin, margin, width, height);
pdf.save("report.pdf");
```

### **Phase 4: UI/UX Enhancement**

#### **Step 4.1: Styling & Animations**
- CSS variables for theming
- Gradient animations (`gradientShift`, `spotlightMove`)
- Glassmorphism effects
- Glow effects on inputs/buttons
- Responsive grid layouts

#### **Step 4.2: Visualizations**
- Pie chart for success rate (Chart.js)
- 3D interactive image viewer (Three.js)
- Before/after comparison slider
- Detection overlay visualization

### **Phase 5: Testing & Refinement**

#### **Step 5.1: Error Handling**
- Try-catch blocks in API calls
- Toast notifications for errors
- Graceful fallbacks for ML failures
- Validation for file uploads

#### **Step 5.2: Performance Optimization**
- Lazy loading of ML models
- Caching model instances
- Image compression before API calls
- Debounced API requests

---

## 🤔 Technology Choices & Rationale

### **Why FastAPI?**
- ✅ **Async Support**: Handles concurrent requests efficiently
- ✅ **Type Safety**: Pydantic models for validation
- ✅ **Auto Documentation**: Swagger UI at `/docs`
- ✅ **Performance**: One of the fastest Python frameworks
- ✅ **Modern**: Built for Python 3.6+ with async/await

### **Why React?**
- ✅ **Component Reusability**: Modular UI components
- ✅ **State Management**: Built-in hooks for state
- ✅ **Ecosystem**: Rich library ecosystem
- ✅ **Performance**: Virtual DOM for efficient updates
- ✅ **Developer Experience**: Hot reload, debugging tools

### **Why Vite?**
- ✅ **Fast HMR**: Instant hot module replacement
- ✅ **Optimized Builds**: Rollup-based production builds
- ✅ **Modern**: Native ES modules support
- ✅ **Developer Experience**: Fast dev server startup

### **Why SQLite?**
- ✅ **Zero Configuration**: No server setup needed
- ✅ **Lightweight**: Single file database
- ✅ **ACID Compliant**: Reliable transactions
- ✅ **Sufficient**: Perfect for small-medium applications

### **Why Google Gemini?**
- ✅ **Image Generation**: Native image generation capability
- ✅ **Multimodal**: Handles image + text inputs
- ✅ **High Quality**: Photorealistic outputs
- ✅ **Integration**: Simple API with Python SDK

### **Why ResNet18 + YOLOv8?**
- ✅ **ResNet18**: Pre-trained on Places365 (365 scene categories)
- ✅ **Lightweight**: Fast inference, good accuracy
- ✅ **YOLOv8**: State-of-the-art object detection
- ✅ **Real-time**: Fast enough for interactive use
- ✅ **Offline**: No external API calls needed

### **Why Ant Design?**
- ✅ **Enterprise Ready**: Production-tested components
- ✅ **Consistent**: Design system with themes
- ✅ **Accessible**: WCAG compliant
- ✅ **Rich Components**: Forms, tables, layouts built-in

### **Why Three.js?**
- ✅ **3D Graphics**: WebGL-based 3D rendering
- ✅ **Interactive**: OrbitControls for user interaction
- ✅ **Cross-platform**: Works in all modern browsers
- ✅ **Performance**: Hardware-accelerated rendering

---

## 📊 Data Flow Diagram

```
┌─────────────┐
│   User      │
│  (Browser)  │
└──────┬──────┘
       │
       │ 1. Upload Image
       ▼
┌─────────────────────────────────────┐
│         Frontend (React)             │
│  ┌──────────────────────────────┐   │
│  │  ImageUpload Component       │   │
│  │  - File selection            │   │
│  │  - Preview display           │   │
│  └──────────────┬───────────────┘   │
│                 │                    │
│                 │ POST /api/analyze  │
│                 ▼                    │
│  ┌──────────────────────────────┐   │
│  │  Axios HTTP Client           │   │
│  └──────────────┬───────────────┘   │
└─────────────────┼───────────────────┘
                  │
                  │ HTTP Request
                  ▼
┌─────────────────────────────────────┐
│      Backend (FastAPI)              │
│  ┌──────────────────────────────┐  │
│  │  /api/analyze endpoint       │  │
│  │  - Receives image bytes      │  │
│  └──────────────┬───────────────┘  │
│                 │                    │
│                 ▼                    │
│  ┌──────────────────────────────┐  │
│  │  analyze_image() function    │  │
│  │  - Loads PIL Image           │  │
│  └──────────────┬───────────────┘  │
│                 │                    │
│       ┌─────────┴─────────┐        │
│       │                    │        │
│       ▼                    ▼        │
│  ┌──────────┐      ┌──────────┐  │
│  │ ResNet18 │      │  YOLOv8   │  │
│  │ (Room    │      │ (Furniture│  │
│  │  Class)  │      │ Detection)│  │
│  └────┬─────┘      └─────┬──────┘  │
│       │                  │          │
│       └────────┬─────────┘          │
│                │                    │
│                ▼                    │
│  ┌──────────────────────────────┐  │
│  │  Return JSON Response        │  │
│  │  {                           │  │
│  │    room_type: "bedroom",     │  │
│  │    detections: [...]         │  │
│  │  }                           │  │
│  └──────────────┬───────────────┘  │
└─────────────────┼───────────────────┘
                  │
                  │ HTTP Response
                  ▼
┌─────────────────────────────────────┐
│         Frontend (React)            │
│  ┌──────────────────────────────┐  │
│  │  Update State                 │  │
│  │  - setRoomType(result.type)   │  │
│  │  - setDetections(result.dets) │  │
│  └──────────────┬───────────────┘  │
│                 │                    │
│                 ▼                    │
│  ┌──────────────────────────────┐   │
│  │  Display Detection Overlay    │   │
│  │  - Draw bounding boxes       │   │
│  │  - Show labels              │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

## 🔑 Key Features Implementation

### **1. Automatic Room Classification**
- **Trigger**: On image upload
- **Process**: ResNet18 model inference
- **Output**: Auto-fills room type dropdown
- **Benefit**: Better design suggestions

### **2. Furniture Detection**
- **Trigger**: On image upload & after generation
- **Process**: YOLOv8 object detection
- **Output**: Bounding boxes with labels
- **Visualization**: Overlay on images

### **3. AI Image Generation**
- **Trigger**: On form submission
- **Process**: Google Gemini API call
- **Input**: Original image + design parameters
- **Output**: Redesigned image + recommendations

### **4. PDF Report Generation**
- **Trigger**: User clicks "Download PDF"
- **Process**: 
  1. Render hidden HTML component
  2. Capture with html2canvas
  3. Convert to PDF with jsPDF
  4. Paginate content
- **Output**: Multi-page PDF report

### **5. 3D Interactive Preview**
- **Technology**: Three.js with OrbitControls
- **Features**: Drag to rotate, scroll to zoom
- **Purpose**: Enhanced visualization

### **6. Database Persistence**
- **Purpose**: Store all design selections
- **Data**: Room type, style, colors, recommendations
- **Access**: No authentication required (public)

---

## 📈 Performance Considerations

### **Optimizations Implemented:**
1. **Lazy Model Loading**: ML models loaded only when needed
2. **Model Caching**: Models cached in memory after first load
3. **Image Compression**: Reduced file sizes before API calls
4. **Async Operations**: Non-blocking API calls
5. **Code Splitting**: Vite automatically splits code for production

### **Scalability Notes:**
- **Current**: Single-server deployment
- **Future**: Can scale horizontally with load balancer
- **Database**: Can migrate to PostgreSQL for production
- **Caching**: Can add Redis for session/data caching

---

## 🚀 Deployment Considerations

### **Frontend:**
- Build: `npm run build`
- Output: Static files in `dist/`
- Hosting: Any static host (Vercel, Netlify, GitHub Pages)

### **Backend:**
- Server: Uvicorn with Gunicorn workers
- Process Manager: systemd, PM2, or Docker
- Environment: Python 3.12+, Poetry installed

### **Database:**
- Current: SQLite (file-based)
- Production: PostgreSQL recommended
- Migration: Use Alembic for schema migrations

---

## 📝 Summary

This project demonstrates a **full-stack AI-powered web application** combining:
- **Modern web technologies** (React, FastAPI)
- **Deep learning** (ResNet18, YOLOv8)
- **Generative AI** (Google Gemini)
- **3D visualization** (Three.js)
- **Data persistence** (SQLite/SQLAlchemy)

The architecture follows **separation of concerns** with clear frontend/backend boundaries, enabling scalability and maintainability.

