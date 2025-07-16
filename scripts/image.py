import cv2
import os

from scripts.detect import detect_and_anonymize_faces

def process_image(image_path, method='pixelate'):
    """
    Traite une image pour flouter ou pixelliser les visages.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : impossible de lire l'image {image_path}")
        return

    img = detect_and_anonymize_faces(img, method)
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_anonymized{ext}"
    cv2.imwrite(output_path, img)
    print(f"Image sauvegardée : {output_path}")