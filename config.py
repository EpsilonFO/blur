"""
Configuration parameters for face blurring and pixelation using YOLOv8.
"""

from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")

# Parameters
pixel_size = 10  # Size of pixel blocks for pixelation
blur_size = 199    # Kernel size for Gaussian blur
band_size = 2000  # Size of frequency bands for FFT processing
shift_amount = 1050  # Frequency shift amount for FFT processing
model = YOLO("utils/yolov8n-face-lindevs.onnx", task="detect", verbose=False)  # Load YOLO model for face detection
image_exts = (".jpg", ".jpeg", ".png")  # Supported image formats
video_exts = (".mp4", ".mov", ".m4v")   # Supported video formats
margin = 0.4  # Margin around detected faces for anonymization

WHISPER_MODEL = "small"  # "small", "medium", "large" disponibles
TTS_MODEL_NAME = "tts_models/fr/mai/tacotron2-DDC"