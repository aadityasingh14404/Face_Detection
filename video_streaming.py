import cv2 

# Load the pre-trained Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_video():
    """
    Detects faces in real-time video from the webcam and draws rectangles around them.
    """
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle with 2px thickness

        # Display the resulting frame with rectangles around the faces
        cv2.imshow('Real-time Face Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the face detection in video
detect_faces_in_video()
