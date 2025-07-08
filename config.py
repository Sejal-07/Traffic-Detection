import os
import cv2  
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME = "yolov8l.pt"
    CONFIDENCE_THRESHOLD = 0.5
    VEHICLE_CLASSES = {
        2: "car",  
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
   
    # Image paths
    INPUT_DIR = os.path.join(BASE_DIR, "data", "test_images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "processed_images")
    REPORT_DIR = os.path.join(BASE_DIR, "output", "reports")

    # Video paths
    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "data", "test_videos")
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "output_videos")
    
    
    COLORS = {
        "car": (0, 255, 0),       # Green
        "truck": (255, 0, 0),      # Blue
        "motorcycle": (0, 0, 255), # Red
        "bus": (255, 255, 0)       # Cyan
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2