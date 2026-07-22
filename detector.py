"""
YOLO Detector Module
Handles car damage detection using YOLOv8 model

Covers FR2 (Object Detection), FR3 (Annotation Visualization)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

logger = logging.getLogger(__name__)

# Class mapping as per SRS Appendix A
CLASS_MAPPING = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat"
}

# Color mapping for visualization (BGR format for OpenCV)
COLOR_MAPPING = {
    "dent": (0, 165, 255),  # Orange
    "scratch": (0, 0, 255),  # Red
    "crack": (255, 0, 0),  # Blue
    "glass_shatter": (0, 255, 0),  # Green
    "lamp_broken": (255, 0, 255),  # Magenta
    "tire_flat": (255, 255, 0)  # Cyan
}


class YOLODetector:
    """
    YOLO-based detector for car damage.

    Handles:
    - Model loading and caching
    - Image inference
    - Bounding box annotation
    - Confidence filtering

    Production considerations:
    - GPU support (automatic via YOLO)
    - Batch processing ready
    - Error handling and logging
    """

    def __init__(self, model_path: str):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to best.pt model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.model_path = Path(model_path)
        self.model = None
        self.device = None

        self._load_model()

    def _load_model(self):
        """Load YOLO model with error handling."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(str(self.model_path))

            # Determine device (GPU if available, otherwise CPU)
            self.device = 0 if self._has_gpu() else "cpu"
            logger.info(f"YOLO model loaded successfully. Device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise

    @staticmethod
    def _has_gpu() -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def detect(
            self,
            image_path: str,
            confidence: float = 0.5,
            iou: float = 0.45
    ) -> List[Dict]:
        """
        Detect car damage in image.

        Args:
            image_path: Path to image file
            confidence: Confidence threshold (0.0-1.0)
            iou: IOU threshold for NMS

        Returns:
            List of detections with format:
            [
                {
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'x1': float, 'y1': float,
                    'x2': float, 'y2': float
                },
                ...
            ]

        Raises:
            FileNotFoundError: If image doesn't exist
            Exception: If detection fails
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            logger.info(f"Starting detection on: {image_path.name}")

            # Run inference
            results = self.model.predict(
                source=str(image_path),
                conf=confidence,
                iou=iou,
                device=self.device,
                verbose=False
            )

            detections = self._parse_results(results)
            logger.info(f"Detection complete: {len(detections)} objects found")

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise

    @staticmethod
    def _parse_results(results) -> List[Dict]:
        """
        Parse YOLO results into detection dictionaries.

        Args:
            results: YOLO prediction results

        Returns:
            List of detection dictionaries
        """
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for idx in range(len(boxes)):
                box = boxes[idx]

                detection = {
                    'class_id': int(box.cls[0].item()),
                    'class_name': CLASS_MAPPING.get(
                        int(box.cls[0].item()),
                        'unknown'
                    ),
                    'confidence': float(box.conf[0].item()),
                    'x1': float(box.xyxy[0][0].item()),
                    'y1': float(box.xyxy[0][1].item()),
                    'x2': float(box.xyxy[0][2].item()),
                    'y2': float(box.xyxy[0][3].item()),
                }

                detections.append(detection)

        return detections

    def annotate_image(
            self,
            image_path: str,
            detections: List[Dict],
            thickness: int = 2,
            font_scale: float = 0.6
    ) -> Image.Image:
        """
        Create annotated image with bounding boxes.

        Args:
            image_path: Path to original image
            detections: List of detections
            thickness: Bounding box line thickness
            font_scale: Font size scale

        Returns:
            PIL Image with annotations

        Raises:
            FileNotFoundError: If image doesn't exist
            Exception: If annotation fails
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Draw bounding boxes
            for detection in detections:
                x1, y1 = int(detection['x1']), int(detection['y1'])
                x2, y2 = int(detection['x2']), int(detection['y2'])

                class_name = detection['class_name']
                confidence = detection['confidence']
                color = COLOR_MAPPING.get(class_name, (200, 200, 200))

                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # Prepare label
                label = f"{class_name.replace('_', ' ').title()} {confidence:.1%}"

                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, font_scale, 1
                )

                # Draw background rectangle
                bg_y1 = max(0, y1 - text_height - 8)
                bg_y2 = y1
                cv2.rectangle(
                    image,
                    (x1, bg_y1),
                    (x1 + text_width + 4, bg_y2),
                    color,
                    -1
                )

                # Put text
                cv2.putText(
                    image,
                    label,
                    (x1 + 2, y1 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # Convert BGR to RGB and return as PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            logger.info(f"Image annotated successfully with {len(detections)} boxes")
            return pil_image

        except Exception as e:
            logger.error(f"Failed to annotate image: {e}", exc_info=True)
            raise

    def get_detection_summary(self, detections: List[Dict]) -> str:
        """
        Generate a text summary of detections for the AI agent.

        Args:
            detections: List of detection dictionaries

        Returns:
            Formatted string summary for LLM processing
        """
        if not detections:
            return "No damage detected."

        summary_parts = []

        # Group by class
        grouped = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(det['confidence'])

        # Build summary
        for class_name, confidences in grouped.items():
            count = len(confidences)
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)

            summary_parts.append(
                f"{class_name.replace('_', ' ').title()}: {count} found "
                f"(Avg confidence: {avg_conf:.1%}, Max: {max_conf:.1%})"
            )

        return "\n".join(summary_parts)

    def get_damage_assessment(self, detections: List[Dict]) -> Dict:
        """
        Get comprehensive damage assessment.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary with damage assessment data
        """
        if not detections:
            return {
                'total_damage_areas': 0,
                'damage_types': {},
                'severity_score': 0.0,
                'has_critical_damage': False
            }

        damage_types = {}
        total_confidence = 0

        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']

            if class_name not in damage_types:
                damage_types[class_name] = {
                    'count': 0,
                    'confidences': [],
                    'avg_confidence': 0.0
                }

            damage_types[class_name]['count'] += 1
            damage_types[class_name]['confidences'].append(confidence)
            total_confidence += confidence

        # Calculate averages
        for class_name in damage_types:
            confidences = damage_types[class_name]['confidences']
            damage_types[class_name]['avg_confidence'] = (
                    sum(confidences) / len(confidences)
            )

        # Calculate severity score (0-100)
        severity_score = min(100.0, (total_confidence / len(detections)) * 100)

        # Identify critical damage
        has_critical = any(
            det['class_name'] in ['crack', 'glass_shatter', 'tire_flat']
            and det['confidence'] > 0.8
            for det in detections
        )

        return {
            'total_damage_areas': len(detections),
            'damage_types': damage_types,
            'severity_score': severity_score,
            'has_critical_damage': has_critical,
            'summary': self.get_detection_summary(detections)
        }