import cv2 as cv

"""
Detects eye and face from an image
"""
# Initialize classifier
data = cv.data.haarcascades
eye_cascade = cv.CascadeClassifier(data + 'haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier(data + 'haarcascade_frontalface_alt.xml')

# Draw rectangles around eyes and face
def rectangle(detect, img, rgb):
    for (x, y, w, h) in detect:
        cv.rectangle(img, (x, y), (x+w, y+h), rgb, 2) 

# Read images
chae = cv.imread('blobchae.png')
grey_chae = cv.cvtColor(chae, cv.COLOR_BGR2GRAY)       

# Detect eyes and face
eyes = eye_cascade.detectMultiScale(grey_chae, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces = face_cascade.detectMultiScale(grey_chae, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

rectangle(eyes, chae, (0, 255, 0))
rectangle(faces, chae, (255, 255, 0))

# Display the output
cv.imshow('Detected Features', chae)
cv.waitKey(0)
cv.destroyAllWindows()
