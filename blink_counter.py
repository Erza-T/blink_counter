"""
Blink counter. Given a webcam, the camera will detect a face and will check everytime the eyes are open or closed. 
A blink counter (blink_count) is added to count the amount of times the face blinked for the duration of the video capture.
"""
import cv2 as cv
import dlib
import math

# Initialize detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

blinks = 0

# Video capture
vid = cv.VideoCapture(0)
if not vid.isOpened():
    print('camera is not found')
    exit(100)

# Ear-Aspect-Ratio formula
def ear(a,b,c):
    return (a+b)/(2*c)

threshold = 0.17
blink_threshold = []
blink_count = 0
frame_count = 0

while True:
    # Capture video frame by frame and convert frame to greyscale
    ret, frame = vid.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
    eyes = detector(grey)

    for eye in eyes:
        # Get eye facial landmarks
        landmarks = predictor(grey, eye)
        coords = []

        for i in range(36, 48):
             # Draw eye landmarks and get eye coordinates
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv.circle(frame, (x, y), 2, (128, 255, 0), -1)
            text_coord = (x + 50, y + 50)
            coords.append((x,y))

        # Calculate EAR of each eye
        a = [math.dist(coords[1],coords[5]), math.dist(coords[7], coords[11])]
        b = [math.dist(coords[2],coords[4]), math.dist(coords[8], coords[10])]
        c = [math.dist(coords[0],coords[3]), math.dist(coords[6], coords[9])]

        ear1 = ear(a[0],b[0],c[0])
        ear2 = ear(a[1],b[1],c[1])
        
        # Change screen text to eyes either opened or closed
        if ear1 >= threshold or ear2 >= threshold:
            cv.putText(frame, "OPENED", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA) 
        else:
            cv.putText(frame, "CLOSED ", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA)  
            frame_count += 1 

    # Check if blinked and record
    if frame_count >= 2 and (ear1 >= threshold or ear2 >= threshold): # to check if the blinking (or eyes closed) stopped
        blink_count += 1
        blink_threshold.append((ear1,ear2))
        frame_count = 0
                 
    # Display frame
    cv.imshow("Eye Tracker Feed", frame)

    # Exiting window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture object and destroy windows
vid.release() 
cv.destroyAllWindows() 

print(blink_threshold)
print(f"Number of times blinked: {blink_count}")

exit(100)

