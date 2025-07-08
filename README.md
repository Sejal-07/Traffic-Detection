# 🚦 Traffic Detection System using YOLOv8

An end-to-end vehicle detection and counting system using YOLOv8, OpenCV, and Streamlit. It allows users to upload traffic images or videos, detects vehicles like cars, trucks, motorcycles, and buses, and provides annotated outputs with analytics reports.

---

## 📌 Features

- Detect and count vehicles in images and videos
- Uses YOLOv8 for high-speed, accurate detection
- Generates annotated images and videos with bounding boxes
- Provides downloadable CSV reports and bar chart visualizations
- Clean and intuitive Streamlit web interface

---

## 🧠 Tech Stack

| Component     | Technology         |
|---------------|--------------------|
| Object Detection | [YOLOv8 (Ultralytics)](https://docs.ultralytics.com) |
| Image Processing | OpenCV |
| Frontend Interface | Streamlit |
| Data Handling | Pandas, Matplotlib |
| Language | Python 3.8+ |

---

## 📁 Project Structure

```bash
├── main.ipynb               # Jupyter notebook for testing and prototyping
├── vehicle_app.py           # Streamlit UI for image upload and results
├── detector.py              # YOLOv8 detection logic for images & videos
├── utils.py                 # Reporting and visualization utilities
├── config.py                # Configuration (paths, labels, colors)
├── requirements.txt         # Required Python packages
├── data/
│   ├── test_images/         # Sample input images
│   └── test_videos/         # Sample input videos
└── output/
    ├── processed_images/    # YOLO-annotated output images
    ├── output_videos/       # Annotated video outputs
    └── reports/             # CSV summary and bar plots

```
---

## ⚙️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/Sejal-07/traffic-detection-yolov8.git
cd traffic-detection-yolov8
```
2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Download YOLOv8 model weights**
---
## 🚀 How to Run
### 🖼️ Image Detection via Streamlit
```bash
streamlit run vehicle_app.py
```
- Upload .jpg, .jpeg, or .png files
- View detection results with annotated bounding boxes
- Download output with one click

### 📹 Video Detection 
```bash
from detector import VehicleDetector
detector = VehicleDetector()
detector.process_video("data/test_videos/video.mp4", "output/output_videos/processed.mp4")
```
## 📸 Screenshots
![Screenshot 2025-07-08 104821](https://github.com/user-attachments/assets/6b1b1803-5e63-4515-b26b-37b2be61abc5)

---
## 📊 Sample Output

### 🗒️ CSV Report

| Image         | Car | Truck | Bus | Motorcycle |
|---------------|-----|-------|-----|------------|
| traffic1.jpg  |  3  |   2   |  1  |     2      |

### 📈 Bar Chart Example

The bar chart summarizing vehicle counts is automatically saved to:

![vehicle_counts_plot](https://github.com/user-attachments/assets/5ea95b05-0b7d-41f6-9cba-0f61af8fda7f)

---
## 🚀 Future Enhancements
- Real-time CCTV/stream support
- License plate recognition integration
- Vehicle speed estimation
- Responsive and mobile-friendly web interface
---
## 🏁 Credits
Developed by Sejal Dabre
Project • 2024-25


