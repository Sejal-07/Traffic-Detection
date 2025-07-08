import os
import csv
from config import Config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

def create_directories():
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.REPORT_DIR, exist_ok=True)

def save_report(counts_dict: dict, report_path: str):
   
    df = pd.DataFrame.from_dict(counts_dict, orient='index')
    df.index.name = 'image'
    
    df.to_csv(report_path)
    
    # Generate visualization
    def normalize_color(color_bgr):
        
        b, g, r = color_bgr
        return (r / 255, g / 255, b / 255)

    colors_rgb = [normalize_color(Config.COLORS.get(v, (128, 128, 128))) for v in df.columns]

    plt.figure(figsize=(10, 6))
    df.sum().plot(kind='bar', color=colors_rgb)
    plt.title('Total Vehicle Counts by Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    
    # Save visualization
    viz_path = os.path.join(Config.REPORT_DIR, "vehicle_counts_plot.png")
    plt.savefig(viz_path)
    plt.close()
    
    print(f"Report saved to {report_path}")
    print(f"Visualization saved to {viz_path}")


def display_image(image, title: str = "Processed Image"):
    """
    Display an OpenCV image (NumPy array) using matplotlib.
    
    Args:
        image: OpenCV image as a NumPy array
        title: Optional title for the plot
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()
