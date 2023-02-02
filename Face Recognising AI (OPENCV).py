import cv2
import numpy as np

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("C:\\Users\\VINAYAK\\OneDrive\\Desktop\\PDF\\Academic\\Python Projects\\Haarcascade\\haarcascade_frontalface_default.xml")

# Start the default camera
cap = cv2.VideoCapture(0)


while True:
    # Read each frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)

    # Display the frame in a window
    cv2.imshow("Camera", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
