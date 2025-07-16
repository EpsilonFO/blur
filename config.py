from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")


# Parameters
pixel_size = 10  # Size of pixel blocks for pixelation
blur_size = 99    # Kernel size for Gaussian blur
band_size = 2000  # Size of frequency bands for FFT processing
shift_amount = 800  # Frequency shift amount for FFT processing
model = YOLO("utils/yolov8n-face-lindevs.onnx", task="detect", verbose=False)  # Load YOLO model for face detection
image_exts = (".jpg", ".jpeg", ".png")  # Supported image formats
video_exts = (".mp4", ".mov", ".m4v")   # Supported video formats