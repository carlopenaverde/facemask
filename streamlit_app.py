import cv2
import streamlit as st
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_mask(image):
    return False

def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #E6F1F5;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .header {
            color: #333333;
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px;
        }
        .result {
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h1 class="header">Mask Detection on Images</h1>', unsafe_allow_html=True)
    

    st.markdown('</div>', unsafe_allow_html=True)

    # Upload the image
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region of interest
            face_roi = image[y:y + h, x:x + w]

            # Perform mask detection on the face ROI
            mask_detected = detect_mask(face_roi)

            if mask_detected:
                # Get the width and height of the text label
                (text_width, text_height) = cv2.getTextSize("Mask", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Calculate the position to place the text label in the middle of the bounding box
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2

                # Draw text label if mask is detected
                cv2.putText(image, "Mask", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Get the width and height of the text label
                (text_width, text_height) = cv2.getTextSize("No Mask", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Calculate the position to place the text label in the middle of the bounding box
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2

                # Draw text label if mask is not detected
                cv2.putText(image, "No Mask", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the image in Streamlit
        st.image(image, channels="BGR", use_column_width=True)

        # Show mask detection result
        if len(faces) > 0:
            if mask_detected:
                st.markdown('<p class="result" style="color: green;">Mask detected!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="result" style="color: red;">No mask detected.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result">No face detected.</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()