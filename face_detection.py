import cv2 

# Load the pre-trained Haar cascade model for face detection
# This is a widely used face detection model provided by OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    """
    Detects faces in an image and draws rectangles around them.

    Args:
        image_path (str): The path to the image file.

    Steps:
        - Reads the image from the given path.
        - Converts the image to grayscale (required for face detection).
        - Uses Haar Cascade to detect faces.
        - Draws a rectangle around each detected face.
        - Displays the resulting image with rectangles around the faces.
    """
    # Read the image from the provided file path
    img = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if img is None:
        print("Error: Could not load the image. Check the file path.")
        return

    # Convert the image to grayscale as the face detection works better on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # scaleFactor: Specifies how much the image size is reduced at each image scale (for multi-scale detection)
    # minNeighbors: Specifies how many neighbors each rectangle should have to retain it (higher value results in fewer detections but better accuracy)
    # minSize: Minimum possible face size to detect
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle with thickness of 2 pixels

    # Display the image with the detected faces
    cv2.imshow('Faces Detected', img)
    
    # Wait for a key press before closing the displayed image
    cv2.waitKey(0)
    
    # Close the image window
    cv2.destroyAllWindows()

# Example usage: Provide the path to your image file
detect_faces('known_face2.jpg')
