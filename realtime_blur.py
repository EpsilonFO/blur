import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

mode = 'pixel'  # Mode par défaut

while True:
    success, img = cap.read()
    faces = faceCascade.detectMultiScale(img, 1.1, 4)

    if len(faces) == 0:
        cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        if mode == 'pixel':
            pixel_size = 10
            temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y:y+h, x:x+w] = pixelated
        else:
            blur = cv2.GaussianBlur(face, (91, 91), 0)
            img[y:y+h, x:x+w] = blur

    # Afficher le mode actuel
    cv2.putText(img, f'Mode: {mode.upper()}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Filter', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        mode = 'pixel'
    elif key == ord('b'):
        mode = 'blur'

cap.release()
cv2.destroyAllWindows()
