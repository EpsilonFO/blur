import cv2
from collections import deque
import numpy as np
from config import blur_size, pixel_size, model, margin

# Mémoire des visages détectés sur les dernières frames (3 frames max)
recent_faces = deque(maxlen=30)

def anonymize_face(image, x, y, w, h, use_blur='pixelate'):
    """
    Applique un floutage ou une pixellisation sur une région du visage.

    Paramètres :
    - image : image d'entrée
    - x, y : coordonnées du coin supérieur gauche
    - w, h : largeur et hauteur de la région
    - use_blur : méthode d'anonymisation ('blur' ou 'pixelate')
    """
    face = image[y:y+h, x:x+w]

    if use_blur == 'blur':
        # Floutage classique avec une gaussienne
        face = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
    elif use_blur == 'pixelate':
        # Réduction de la résolution puis réagrandissement pour pixeliser
        temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Méthode inconnue : {use_blur} ; utilisez 'blur' ou 'pixelate'.")

    image[y:y+h, x:x+w] = face
    return image

def is_close(face1, face2, threshold=30):
    """
    Vérifie si deux visages sont spatialement proches.

    Utilisé pour détecter si un visage disparu est en fait réapparu très proche.

    Paramètres :
    - face1, face2 : tuples (x, y, w, h)
    - threshold : distance maximale pour considérer les visages comme proches
    """
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    return np.linalg.norm([x1 - x2, y1 - y2]) < threshold

def detect_and_anonymize_faces(img, use_blur='pixelate'):
    """
    Détecte les visages avec YOLO et applique l'anonymisation.
    Maintient le flou sur les visages même s'ils disparaissent temporairement.

    Paramètres :
    - img : image d'entrée
    - use_blur : méthode d'anonymisation ('blur' ou 'pixelate')
    """
    results = model(img)
    height, width = img.shape[:2]
    current_faces = []

    # Étape 1 : Détection des visages actuels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Calcul des coordonnées avec marge autour du visage
            x1_m = max(0, int(x1 - w * margin))
            y1_m = max(0, int(y1 - h * margin))
            x2_m = min(width, int(x2 + w * margin))
            y2_m = min(height, int(y2 + h * margin))

            new_w, new_h = x2_m - x1_m, y2_m - y1_m
            current_faces.append((x1_m, y1_m, new_w, new_h))

            # Application du flou sur le visage détecté
            img = anonymize_face(img, x1_m, y1_m, new_w, new_h, use_blur)

    # Étape 2 : Maintien du flou pour les visages récemment détectés
    for old_faces in recent_faces:
        for old_face in old_faces:
            # Si le visage n'est pas proche d'un visage actuel, on le floute quand même
            if not any(is_close(old_face, cf) for cf in current_faces):
                x, y, w, h = old_face
                img = anonymize_face(img, x, y, w, h, use_blur)

    # Étape 3 : Mise à jour de la mémoire des visages
    recent_faces.append(current_faces)

    return img
