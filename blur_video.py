# Importing libraries
import cv2
import argparse

parser = argparse.ArgumentParser(description="Blur or pixelate faces in a video.")
parser.add_argument("image_name", type=str, help="Name of the image file in the 'pics' folder")
parser.add_argument("--pixel", action="store_true", help="Apply pixelation instead of blur")
args = parser.parse_args()

# to detect the face of the human
cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
# Create a VideoCapture object and read from input file
video_capture = cv2.VideoCapture('pics/'+args.image_name)
# Read until video is completed
while(video_capture.isOpened()):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # convert the frame into grayscale(shades of black & white)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_data = cascade.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=5)

    for x, y, w, h in face_data:
        if args.pixel:
            # Extraire la région du visage
            face = frame[y:y+h, x:x+w]

            # Définir le niveau de pixelisation (plus petit = plus pixelisé)
            pixel_size = 15

            # Réduire puis agrandir l'image pour créer l'effet pixelisé
            temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

            # Remplacer la région du visage par la version pixelisée
            frame[y:y+h, x:x+w] = pixelated_face
        else:
            frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 35)

    # show the blurred face in the video
    cv2.imshow('face blurred', frame)
    key = cv2.waitKey(1)

# When everything done, release
# the video capture object
video_capture.release()
 
# Closes all the frames
cv2.destroyAllWindows()