import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from config import Config
from utils import create_directories, save_report
import os

class VehicleDetector:
    def __init__(self):
        self.model = YOLO(Config.MODEL_NAME)
        self.classes = Config.VEHICLE_CLASSES
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        create_directories()

    def detect_vehicles(self, image_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        results = self.model(image, conf=self.conf_threshold, verbose=False)
        counts = {v: 0 for v in self.classes.values()}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                if class_id in self.classes:
                    vehicle_type = self.classes[class_id]
                    counts[vehicle_type] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = Config.COLORS.get(vehicle_type, (0, 255, 255))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    conf = float(box.conf)
                    label = f"{vehicle_type} {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                                Config.FONT, Config.FONT_SCALE, color, Config.FONT_THICKNESS)

        summary_y = 30
        for vehicle_type, count in counts.items():
            summary_text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(image, summary_text, (10, summary_y),
                        Config.FONT, Config.FONT_SCALE, (0, 0, 0), Config.FONT_THICKNESS + 1)
            cv2.putText(image, summary_text, (10, summary_y),
                        Config.FONT, Config.FONT_SCALE, (255, 255, 255), Config.FONT_THICKNESS)
            summary_y += 30

        return image, counts

    def process_images(self, input_dir: str = None, output_dir: str = None) -> Dict[str, Dict[str, int]]:
        input_dir = input_dir or Config.INPUT_DIR
        output_dir = output_dir or Config.OUTPUT_DIR

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        all_counts = {}

        for image_path in image_paths:
            try:
                annotated_image, counts = self.detect_vehicles(image_path)
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                cv2.imwrite(output_path, annotated_image)
                all_counts[filename] = counts
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        report_path = os.path.join(Config.REPORT_DIR, "vehicle_counts_report.csv")
        save_report(all_counts, report_path)

        return all_counts

    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ext = os.path.splitext(output_path)[-1].lower()
            if ext == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                raise ValueError("❌ Unsupported video format. Use .mp4 or .avi")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                 raise RuntimeError(f"❌ Failed to open VideoWriter for: {output_path}. Check codec or format.")
        

        total_counts = {v: 0 for v in self.classes.values()}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            frame_counts = {v: 0 for v in self.classes.values()}

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls)
                    if class_id in self.classes:
                        vehicle_type = self.classes[class_id]
                        frame_counts[vehicle_type] += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = Config.COLORS.get(vehicle_type, (0, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        conf = float(box.conf)
                        label = f"{vehicle_type} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    Config.FONT, Config.FONT_SCALE, color, Config.FONT_THICKNESS)

            for vehicle_type, count in frame_counts.items():
                total_counts[vehicle_type] += count

            summary_y = 30
            for vehicle_type, count in frame_counts.items():
                summary_text = f"{vehicle_type.capitalize()}: {count}"
                cv2.putText(frame, summary_text, (10, summary_y),
                            Config.FONT, Config.FONT_SCALE, (0, 0, 0), Config.FONT_THICKNESS + 1)
                cv2.putText(frame, summary_text, (10, summary_y),
                            Config.FONT, Config.FONT_SCALE, (255, 255, 255), Config.FONT_THICKNESS)
                summary_y += 30

            if out:
                frame = cv2.resize(frame, (frame_width, frame_height))
                out.write(frame)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        if out:
            out.release()

        return total_counts
