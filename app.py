import streamlit as st 
import cv2
from PIL import Image
import numpy as np

# Load the pre-trained Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """
    Detects faces in the given image using Haar Cascade.

    Args:
        image (numpy array): The image where faces need to be detected.

    Returns:
        numpy array: The image with rectangles drawn around detected faces.
    """
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image

def main():
    """
    Main function for the Streamlit app.
    Handles the layout, file uploading, and face detection process.
    """
    # Set the page title
    st.title("Face Detection App")

    # Add custom styles for better layout
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #34495e;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .upload-section, .result-section {
            background-color: #f8f9fa;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Render the page title with a centered layout
    st.markdown('<div class="title">Face Detection App</div>', unsafe_allow_html=True)

    # Section for image upload
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.write("Upload an image to detect faces.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to a NumPy array and then to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Detect faces in the image
        st.write("Detecting faces...")
        result_image = detect_faces(image)
        
        # Convert the processed image back to PIL format for Streamlit display
        result_image = Image.fromarray(result_image)
        
        # Display the image with detected faces
        st.image(result_image, caption='Processed Image with Faces Detected.', use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
