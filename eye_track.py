import cv2 as cv

# Initialize classifier
data = cv.data.haarcascades
eye_cascade = cv.CascadeClassifier(data + 'haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier(data + 'haarcascade_frontalface_alt.xml')

# Read images
chae = cv.imread('wiil.png')
grey_chae = cv.cvtColor(chae, cv.COLOR_BGR2GRAY)

# Detect eyes
eyes = eye_cascade.detectMultiScale(grey_chae, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Detect face
faces = face_cascade.detectMultiScale(grey_chae, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

def rectangle(detect, img, rgb):
    for (x, y, w, h) in detect:
        cv.rectangle(img, (x, y), (x+w, y+h), rgb, 2) 

rectangle(eyes, chae, (0, 255, 0))
rectangle(faces, chae, (255, 255, 0))

# Display the output
cv.imshow('Detected Eyes', chae)
cv.waitKey(0)
cv.destroyAllWindows()