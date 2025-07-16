import cv2
import numpy as np

from config import blur_size, pixel_size, model

# Fonction de pixellisation
def apply_pixelation(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = face
    return frame

# Fonction de flou
def apply_blur(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]
    face = cv2.GaussianBlur(face, (blur_size, blur_size), 30)
    frame[y:y+h, x:x+w] = face
    return frame

# Initialisation de la webcam
cap = cv2.VideoCapture(0)
mode = 'pixelate'  # Mode par défaut
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des visages avec YOLO
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if mode == 'pixelate':
                frame = apply_pixelation(frame, x1, y1, w, h)
            elif mode == 'blur':
                frame = apply_blur(frame, x1, y1, w, h)

    # Affichage
    cv2.putText(frame, f'Mode: {mode.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('YOLOv8 Face Filter', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        mode = 'pixelate'
    elif key == ord('b'):
        mode = 'blur'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
