# 💧 AquaVision — Water Body Segmentation
 
<div align="center">
 
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.10.0-D00000?style=flat&logo=keras&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat&logo=amazonaws&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=flat&logo=nginx&logoColor=white)
 
**A deep learning system for accurate water body segmentation from multispectral satellite imagery.**  
Built with U-Net and DeepLabV3+, deployed as microservices using Flask and Docker Compose on AWS EC2.
 
### 🌍 [Live Demo → https://aquavision.ddns.net](https://aquavision.ddns.net)
 
[Overview](#-overview) • [Dataset](#-dataset) • [Models](#-models) • [Results](#-results) • [Installation](#-installation) • [Deployment](#-deployment) • [Cloud](#-cloud-deployment)
 
</div>
---
 
## 📌 Overview
 
AquaVision segments water bodies from **Harmonized Sentinel-2/Landsat (HLS)** multispectral patches using two deep learning models. The system:
 
- Accepts 12-band GeoTIFF patches (128×128 pixels, 30m resolution)
- Runs both models **in parallel** and ensembles their predictions
- Displays RGB satellite imagery with water highlighted as a cyan overlay
- Reports IoU, F1, Precision, and Recall when a ground truth mask is provided
- Deployed as **3 independent microservices** orchestrated by Docker Compose
---
 
## 🛰️ Dataset
 
The dataset consists of **Harmonized Sentinel-2/Landsat** patches with the following 12 input bands:
 
| Index | Band | Description |
|-------|------|-------------|
| 0 | Coastal Aerosol | Short-wave atmospheric scattering |
| 1 | Blue | Visible blue |
| 2 | Green | Visible green |
| 3 | Red | Visible red |
| 4 | NIR | Near-infrared |
| 5 | SWIR1 | Shortwave infrared 1 |
| 6 | SWIR2 | Shortwave infrared 2 |
| 7 | QA Band | Quality assessment bitmask |
| 8 | Merit DEM | Digital elevation model (Merit) |
| 9 | Copernicus DEM | Digital elevation model (Copernicus) |
| 10 | ESA World Cover | Land cover classification |
| 11 | Water Occurrence | Historical water occurrence probability |
 
**Label:** Binary water mask (1 = water, 0 = background)  
**Patch size:** 128 × 128 pixels  
**Ground sampling distance:** 30 meters  
**Dataset split:** 80% train / 10% validation / 10% test (306 total samples)
 
### Channel Distribution
 
![Channel Distribution](The%20channel%20Distribution.jpeg)
 
The chart above shows the value distribution of each band across the dataset — used to guide normalization strategy and identify band-group-aware preprocessing.
 
### Engineered Features
 
Three additional water indices are computed during preprocessing and appended as extra channels:
 
| Index | Feature | Formula |
|-------|---------|---------|
| 12 | NDWI | (Green − NIR) / (Green + NIR) |
| 13 | MNDWI | (Green − SWIR1) / (Green + SWIR1) |
| 14 | AWEI | 4×(Green−SWIR1) − (0.25×NIR + 2.75×SWIR2) |
 
---
 
## 🧠 Models
 
### U-Net
A custom encoder-decoder architecture trained from scratch on the 14-channel input (12 bands + NDWI + MNDWI).
 
- **Framework:** TensorFlow / Keras 3.10
- **Input:** (128, 128, 14)
- **Loss:** Weighted BCE + Dice Loss
- **Optimizer:** Adam with ReduceLROnPlateau
### DeepLabV3+
A pretrained EfficientNet-B3 encoder with an ASPP decoder, fine-tuned on the 15-channel input (12 bands + NDWI + MNDWI + AWEI).
 
- **Framework:** PyTorch + segmentation-models-pytorch 0.5.0
- **Backbone:** EfficientNet-B3 (ImageNet pretrained)
- **Input:** (128, 128, 15)
- **Loss:** Weighted BCE + Dice Loss
- **Optimizer:** AdamW with CosineAnnealingLR
---
 
## 📊 Results
 
### Metrics at Threshold = 0.5
 
| Model | IoU | F1 | Precision | Recall | ROC-AUC |
|-------|-----|-----|-----------|--------|---------|
| U-Net | 0.7806 | 0.8768 | 0.8943 | 0.8599 | 0.9816 |
| DeepLabV3+ | 0.7169 | 0.8351 | 0.7706 | 0.9113 | 0.9748 |
| **Ensemble** | **0.7817** | **0.8775** | 0.8573 | 0.8987 | 0.9815 |
 
### Key Findings
 
- **U-Net outperforms DeepLabV3+** despite having fewer parameters (7.8M vs 11.7M) and no pretrained backbone — demonstrating that engineered water indices (NDWI, MNDWI) and careful preprocessing are highly effective for this task
- Both models achieve **ROC-AUC > 0.97** — excellent discriminative ability
- **Ensemble** marginally improves IoU over U-Net alone (+0.0011)
- DeepLabV3+ has higher **Recall** (0.91) but lower **Precision** (0.77) — it detects almost all water pixels but produces more false alarms
- U-Net is **faster and lighter** making it the better choice for deployment
---
 
## 🏗️ Architecture
 
```
Browser
   │
   │  HTTP POST /predict (.tif file)
   ▼
┌─────────────────────────────────┐
│         Flask App               │
│         port 5000               │
│  preprocess → call models       │
│  → ensemble → return overlay    │
└──────────┬──────────────────────┘
           │ parallel HTTP calls
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌────────────┐
│  U-Net  │  │ DeepLabV3+ │
│ service │  │  service   │
│ port    │  │  port      │
│ 5001    │  │  5002      │
│ TF 2.19 │  │ PyTorch    │
│         │  │ 2.10       │
└─────────┘  └────────────┘
     │            │
     └─────┬──────┘
           ▼
    ./models/ (shared volume)
    ├── best_unet.keras
    ├── best_deeplab.pth
    ├── fit_stats.pkl
    └── fit_stats_deeplab.pkl
```
 
All 3 services run as Docker containers on the same internal network (`aquavision-network`) and are orchestrated by Docker Compose.
 
---
 
## ⚙️ Installation
 
### Prerequisites
 
- Python 3.11+
- Docker Desktop
- Git LFS (for model files)
### Clone the Repository
 
```bash
git clone https://github.com/MahmoudOsama20/water_segmentation.git
cd water_segmentation
```
 
### Download Model Files
 
Model files are stored with Git LFS. Make sure Git LFS is installed:
 
```bash
git lfs install
git lfs pull
```
 
Or download manually from [Kaggle Notebook](https://kaggle.com/your-notebook) and place them in the `models/` folder:
 
```
models/
├── best_unet.keras
├── best_deeplab.pth
├── fit_stats.pkl
└── fit_stats_deeplab.pkl
```
 
---
 
## 🚀 Deployment
 
### Option 1 — Docker Compose (Recommended)
 
The easiest way to run everything with one command:
 
```bash
# Start all 3 services
docker compose up
 
# Or run in background
docker compose up -d
 
# Check all services are healthy
docker compose ps
 
# View logs
docker compose logs -f
 
# Stop everything
docker compose down
```
 
Then open your browser at:
```
http://localhost:5000
```
 
### Option 2 — Run Locally with Virtual Environments
 
**Step 1 — Create environments:**
```bash
python -m venv unet_env
python -m venv deeplab_env
python -m venv flask_env
```
 
**Step 2 — Install dependencies:**
```bash
# U-Net service
unet_env\Scripts\activate
pip install -r unet_service/requirements.txt
 
# DeepLab service
deeplab_env\Scripts\activate
pip install -r deeplab_service/requirements.txt
 
# Flask app
flask_env\Scripts\activate
pip install -r flask_app/requirements.txt
```
 
**Step 3 — Run each service in a separate terminal:**
 
Terminal 1:
```bash
cd unet_service
unet_env\Scripts\activate
python unet_service.py
# Wait for: U-Net ready ✓
```
 
Terminal 2:
```bash
cd deeplab_service
deeplab_env\Scripts\activate
python deeplab_service.py
# Wait for: DeepLab ready ✓
```
 
Terminal 3:
```bash
cd flask_app
flask_env\Scripts\activate
python app.py
# Open http://localhost:5000
```
 
---
 
## 🖥️ Usage
 
1. Open `http://localhost:5000` in your browser
2. Upload a **multispectral GeoTIFF** patch (`.tif`, 128×128×12 bands)
3. Optionally upload a **ground truth mask** (`.png` or `.tif`) to compute metrics
4. Click **Run Segmentation**
5. View the RGB satellite image with water highlighted in cyan for each model
6. Compare metrics across U-Net, DeepLabV3+, and Ensemble
### 🧪 Sample Test Files
 
Ready-to-use test samples are provided in the `Samples to test/` folder:
 
```
Samples to test/
├── satellite images/    ← .tif patches ready to upload
└── labels/              ← corresponding .png ground truth masks
```
 
You can upload any file from `satellite images/` directly into the app. Optionally pair it with the corresponding mask from `labels/` to see IoU, F1, Precision and Recall computed live.
 
> **Naming convention:** each satellite image and its label share the same filename — e.g. `35.tif` pairs with `35.png`.
 
### Preparing Input Data
 
Your input `.tif` file must have exactly **12 bands** in this order:
 
```
Band 1:  Coastal Aerosol
Band 2:  Blue
Band 3:  Green
Band 4:  Red
Band 5:  NIR
Band 6:  SWIR1
Band 7:  SWIR2
Band 8:  QA Band
Band 9:  Merit DEM
Band 10: Copernicus DEM
Band 11: ESA World Cover
Band 12: Water Occurrence Probability
```
 
To extract a test patch from your dataset:
 
```python
import numpy as np
import rasterio
from rasterio.transform import from_bounds
 
# Save a sample from your test set as GeoTIFF
sample = X_test_raw[0]   # (128, 128, 12)
 
with rasterio.open(
    'test_patch.tif', 'w',
    driver='GTiff',
    height=128, width=128,
    count=12,
    dtype=sample.dtype
) as dst:
    dst.write(sample.transpose(2, 0, 1))   # (C, H, W)
 
print("Saved test_patch.tif")
```
 
---
 
## 🐳 Docker Services
 
| Service | Image | Port | Framework |
|---------|-------|------|-----------|
| Flask App | `aquavision-flask` | 5000 | Flask 3.0 |
| U-Net | `aquavision-unet` | 5001 | TensorFlow 2.19 |
| DeepLabV3+ | `aquavision-deeplab` | 5002 | PyTorch 2.10 |
 
### Health Check Endpoints
 
```bash
curl http://localhost:5000/health   # Flask app + both models
curl http://localhost:5001/health   # U-Net only
curl http://localhost:5002/health   # DeepLab only
```
 
---
 
## 📁 Project Structure
 
```
water_segmentation/
│
├── 📓 Notebooks
│   ├── water_segmentation_unet.ipynb       # U-Net training & evaluation
│   ├── water_segmentation_deeplab.ipynb    # DeepLabV3+ training & evaluation
│   └── comparison.ipynb                   # Model comparison & ensemble
│
├── 🐳 Docker
│   ├── docker-compose.yml                 # Orchestrates all 3 services
│   │
│   ├── flask_app/
│   │   ├── Dockerfile
│   │   ├── app.py                         # Main Flask application
│   │   ├── requirements.txt
│   │   ├── templates/
│   │   │   └── index.html                 # Frontend HTML
│   │   └── static/
│   │       ├── style.css                  # Styles
│   │       └── app.js                     # Frontend JavaScript
│   │
│   ├── unet_service/
│   │   ├── Dockerfile
│   │   ├── unet_service.py                # U-Net inference microservice
│   │   └── requirements.txt
│   │
│   └── deeplab_service/
│       ├── Dockerfile
│       ├── deeplab_service.py             # DeepLabV3+ inference microservice
│       └── requirements.txt
│
├── 📦 models/                             # Stored with Git LFS
│   ├── best_unet.keras
│   ├── best_deeplab.pth
│   ├── fit_stats.pkl
│   └── fit_stats_deeplab.pkl
│
├── 🧪 Samples to test/                    # Ready-to-use test patches
│   ├── satellite images/                  # .tif input patches
│   └── labels/                            # .png ground truth masks
│
├── The channel Distribution.jpeg          # Band distribution visualization
├── .gitignore
├── .gitattributes                         # Git LFS tracking rules
└── README.md
```
 
---
 
## 🔧 Environment Variables
 
The Flask app reads these environment variables at runtime:
 
| Variable | Default (local) | Docker Compose value |
|----------|----------------|---------------------|
| `UNET_URL` | `http://localhost:5001/predict` | `http://unet-service:5001/predict` |
| `DEEPLAB_URL` | `http://localhost:5002/predict` | `http://deeplab-service:5002/predict` |
| `UNET_HEALTH` | `http://localhost:5001/health` | `http://unet-service:5001/health` |
| `DEEPLAB_HEALTH` | `http://localhost:5002/health` | `http://deeplab-service:5002/health` |
 
---
 
## 📦 Dependencies
 
| Service | Key Packages |
|---------|-------------|
| U-Net | TensorFlow 2.19, Keras 3.10, Rasterio |
| DeepLab | PyTorch 2.10, segmentation-models-pytorch 0.5.0, Rasterio |
| Flask App | Flask 3.0, NumPy, Pillow, Rasterio, Requests |
 
---
 
## ☁️ Cloud Deployment
 
The app is deployed on **AWS EC2** with a free domain and HTTPS.
 
### Infrastructure
 
| Component | Details |
|-----------|---------|
| Cloud provider | AWS EC2 |
| Instance type | t3.medium (2 vCPU, 4GB RAM) |
| OS | Ubuntu Server 22.04 LTS |
| Storage | 60 GB gp3 |
| IP | AWS Elastic IP (static) |
| Domain | No-IP free subdomain |
| Reverse proxy | Nginx |
| SSL | Let's Encrypt (free, auto-renew) |
 
### Architecture on AWS
 
```
Internet
    │
    │  https://aquavision.ddns.net
    ▼
┌─────────────────────────────────┐
│         Nginx (port 80/443)     │
│         SSL Termination         │
└──────────────┬──────────────────┘
               │ proxy_pass
               ▼
┌─────────────────────────────────┐
│      Docker Compose Network     │
│      (aquavision-network)       │
│                                 │
│  ┌──────────┐  ┌─────────────┐ │
│  │ U-Net    │  │ DeepLabV3+  │ │
│  │ :5001    │  │ :5002       │ │
│  └──────────┘  └─────────────┘ │
│         ▲            ▲         │
│         └────────────┘         │
│              ▲                  │
│  ┌───────────────────────────┐ │
│  │    Flask App :5000        │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```
 
### Deploy to Your Own Server
 
```bash
# 1. SSH into your EC2 instance
ssh -i "your-key.pem" ubuntu@YOUR_EC2_IP
 
# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
sudo apt-get install -y docker-compose-plugin
 
# 3. Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git lfs install
 
# 4. Clone repo and pull models
git clone https://github.com/MahmoudOsama20/water_segmentation.git
cd water_segmentation
git lfs pull
 
# 5. Build images
docker build -t aquavision-unet ./unet_service
docker build -t aquavision-deeplab ./deeplab_service
docker build -t aquavision-flask ./flask_app
 
# 6. Run
docker compose up -d
 
# 7. Install Nginx + SSL
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.ddns.net
```

## 🙏 Acknowledgements

- Dataset: [Water Segmentation Satellite Dataset](https://www.kaggle.com/datasets/mahmoudosamahassan/satellite-dataset)
- DeepLabV3+ implementation: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
