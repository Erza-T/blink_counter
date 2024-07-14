"""
Detects eyes from a video camera
"""
import cv2 as cv
# Initialize classifier
data = cv.data.haarcascades
eye_cascade = cv.CascadeClassifier(data + 'haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier(data + 'haarcascade_frontalface_alt.xml')

# Draw rectangles around eyes and face
def rectangle(detect, img, rgb):
    for (x, y, w, h) in detect:
        cv.rectangle(img, (x, y), (x+w, y+h), rgb, 2) 

# Video capture
vid = cv.VideoCapture(0)
if not vid.isOpened():
    print('camera is not found')
    exit(100)

while True:
    # Capture video frame by frame
    ret, frame = vid.read()

    # Detect eyes and frame 
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rectangle(eyes, frame, (255, 255, 0))

    # Display frame
    cv.imshow("Eye Tracker Feed", frame)

    # Exiting window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture object and destroy windows
vid.release() 
cv.destroyAllWindows() 

