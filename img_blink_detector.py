"""
Detects eyes in an image. Writes "OPENED" if the eyes are opened or "CLOSED" if the eyes are closed.
"""
import cv2 as cv
import dlib
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv.imread('IMG_3472.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)     

# Detect faces
faces = detector(gray)

# Ear-Aspect-Ratio formula
def ear(a,b,c):
    return (a+b)/(2*c)

threshold = 0.2

for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)
    coords = []

    # Loop over all 68 landmarks and draw them on the image
    for n in range(36, 48):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv.circle(image, (x, y), 2, (128, 255, 0), -1)

        text_coord = (x + 50, y + 50)
        coords.append((x,y))

    # Calculate EAR of each eye
    print(coords)
    a = [math.dist(coords[1],coords[5]), math.dist(coords[7], coords[11])]
    b = [math.dist(coords[2],coords[4]), math.dist(coords[8], coords[10])]
    c = [math.dist(coords[0],coords[3]), math.dist(coords[6], coords[9])]

    ear1 = ear(a[0],b[0],c[0])
    ear2 = ear(a[1],b[1],c[1])
    
    print(ear1, ear2)


    # Check if blinked or not
    if ear1 >= threshold and ear2 >= threshold:
        cv.putText(image, "OPENED", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA) 
    elif ear1 >= threshold:
        cv.putText(image, "LEFT WINK", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA)
    elif ear2 >= threshold:
        cv.putText(image, "RIGHT WINK", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA)
    else:
        cv.putText(image, "CLOSED ", text_coord , cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2, cv.LINE_AA)        
    

# Display the output
cv.imshow("Output", image)
cv.waitKey(0)
cv.destroyAllWindows()

