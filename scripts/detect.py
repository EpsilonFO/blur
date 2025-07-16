import cv2

from config import blur_size, pixel_size, model

def anonymize_face(image, x, y, w, h, method='pixelate'):
    """
    Applique un floutage ou une pixellisation sur une région du visage.
    """
    face = image[y:y+h, x:x+w]

    if method == 'blur':
        face = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
    elif method == 'pixelate':
        temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Méthode inconnue : {method} ; utilisez 'blur' ou 'pixelate'.")

    image[y:y+h, x:x+w] = face
    return image

def detect_and_anonymize_faces(img, method='pixelate'):
    """
    Détecte les visages avec YOLO et applique l'anonymisation.
    """
    results = model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            img = anonymize_face(img, x1, y1, w, h, method)
    return img