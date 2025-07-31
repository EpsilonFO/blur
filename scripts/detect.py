import cv2

from config import blur_size, pixel_size, model, margin

def anonymize_face(image, x, y, w, h, use_blur='pixelate'):
    """
    Applique un floutage ou une pixellisation sur une région du visage.
    """
    face = image[y:y+h, x:x+w]

    if use_blur == 'blur':
        face = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
    elif use_blur == 'pixelate':
        temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Méthode inconnue : {use_blur} ; utilisez 'blur' ou 'pixelate'.")

    image[y:y+h, x:x+w] = face
    return image

def detect_and_anonymize_faces(img, use_blur='pixelate'):
    """
    Détecte les visages avec YOLO et applique l'anonymisation.
    """
    results = model(img)
    height, width = img.shape[:2]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Calcul des nouvelles coordonnées avec marge
            x1_m = max(0, int(x1 - w * margin))
            y1_m = max(0, int(y1 - h * margin))
            x2_m = min(width, int(x2 + w * margin))
            y2_m = min(height, int(y2 + h * margin))

            new_w, new_h = x2_m - x1_m, y2_m - y1_m
            img = anonymize_face(img, x1_m, y1_m, new_w, new_h, use_blur)

    return img
