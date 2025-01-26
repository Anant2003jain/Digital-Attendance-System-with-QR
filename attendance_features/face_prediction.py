import cv2
import csv
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Load label mapping
label_dict = {}
with open(r'student_data\class.csv', mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        label_dict[int(row['classid'])] = row['classname']


def face_prediction():
    
    # Load the face recognition model
    face_model = load_model(r'Model\\new_face_recognition_model.h5')
    
    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
    
        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)
    
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        # List to store names corresponding to detected faces
        detected_names = []
    
        # Loop over the detected faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64))  # Resize the face to 64x64 pixels
            face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
            face_roi = face_roi / 255.0  # Normalize pixel values to be between 0 and 1
    
            # Reshape the image to match the input shape expected by the model
            face_roi = np.reshape(face_roi, (1, 64, 64, 1))
    
            prediction = face_model.predict(face_roi)
            predicted_label = np.argmax(prediction)
            print("Raw Predicted:", prediction) 
    
            # Initialize predicted_name with a default value
            predicted_name = "Unknown"
    
            # Check if predicted_label exists in label_dict
            if predicted_label in label_dict.keys():
                predicted_name = label_dict.get(predicted_label, "Unknown")
                print("Predicted Name:", predicted_name)  # Debug print statement
            else:
                print("Label not found in Student List.")
    
            detected_names.append(predicted_name)  # Add the detected name to the list
    
            # Draw a rectangle around the face
            flipped_frame = cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        # Display the names corresponding to detected faces
        for i, name in enumerate(detected_names):
            cv2.putText(flipped_frame, str(name), (faces[i][0], faces[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
        # Display the frame
        cv2.imshow("Face Prediction (Press Q to Exit)", flipped_frame)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

#face_prediction()
