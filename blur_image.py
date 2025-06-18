import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Blur or pixelate faces in a video.")
parser.add_argument("image_name", type=str, help="Name of the image file in the 'pics' folder")
parser.add_argument("--pixel", action="store_true", help="Apply pixelation instead of blur")
args = parser.parse_args()

# Construire le chemin complet vers l'image
image_path = os.path.join("pics", args.image_name)

# Lire l'image et la convertir en RGB
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Afficher l'image originale
print('Original Image')
plt.imshow(image)
plt.axis('off')
plt.show()

# Convertir en niveaux de gris pour la détection
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Charger le classificateur
cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")

# Détection des visages
face_data = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Flouter tous les visages détectés
for x, y, w, h in face_data:
    if args.pixel:
        face = image[y:y+h, x:x+w]
        pixel_size = 15 # Plus petit = plus pixelisé
        temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated_face
    else:
        image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, x:x+w], 35)

# Afficher l'image floutée
print('Blured Image')
plt.imshow(image)
plt.axis('off')
plt.show()
