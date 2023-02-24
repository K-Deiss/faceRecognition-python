import cv2
import mediapipe as mp

# Load the neck cascade classifier
neck_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

# Initialize the Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize the Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the face landmarks using the Mediapipe Face Mesh model
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face landmarks on the frame
            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), circle_radius=1))

    # Detect the neck using the cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    necks = neck_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    for (x, y, w, h) in necks:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
