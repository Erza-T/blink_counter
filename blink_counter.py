import cv2 as cv
import dlib

# Initialize detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

blinks = 0

# Video capture
vid = cv.VideoCapture(0)
if not vid.isOpened():
    print('camera is not found')
    exit(100)

while True:
    # Capture video frame by frame and convert frame to greyscale
    ret, frame = vid.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  

    eyes = detector(grey)

    for eye in eyes:
        # Get eye facial landmarks
        landmarks = predictor(grey, eye)

        eye_coords = []

        for i in range(36,48):
             # Draw eye landmarks
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv.circle(frame, (x, y), 2, (128, 255, 0), -1)

            eye_coords.append(i)

    # Display frame
    cv.imshow("Eye Tracker Feed", frame)

    # Exiting window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture object and destroy windows
vid.release() 
cv.destroyAllWindows() 

