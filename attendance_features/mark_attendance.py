import cv2
import csv
import os
import numpy as np
from pyzbar.pyzbar import decode
from tensorflow.keras.models import load_model  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array   # type: ignore

# Function to load and preprocess images for training
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

# Load label mapping
label_dict = {}
with open(r'StudentData//class.csv', mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        label_dict[int(row['classid'])] = row['classname']  # Use 'nameID' instead of 'classname'


train_data = []
labels = []
for folder_name in os.listdir(r'StudentData//images'):
        folder_path = os.path.join(r'StudentData//images', folder_name)
        images = load_images_from_folder(folder_path)
        train_data.extend(images)
        labels.extend([folder_name] * len(images))

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


def mark_attendance():
    # Load or create attendance file
    if os.path.exists(r'StudentData//attendancelist.csv'):
        attendance_df = pd.read_csv(r'StudentData//attendancelist.csv', index_col='nameID')
    else:
        attendance_df = pd.DataFrame(columns=['nameID'])
        attendance_df.set_index('nameID', inplace=True)

    # Get current date in the required format
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Initialize the face detector, video capture, and other components
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'Model\haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_model = load_model(r'Model//new_face_recognition_model.h5')

    while True:
        ret, frame = cap.read()
        flipped_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        decoded_objects = decode(flipped_frame)

        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            #print("QR: ", qr_data)
        
            predicted_nameID = None  # Initialize predicted_nameID
            match_found = False  # Flag to track if a match is found
        
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = face_roi / 255.0
                face_roi = np.reshape(face_roi, (1, 64, 64, 1))
                prediction = face_model.predict(face_roi)
                predicted_label = np.argmax(prediction)
                
                flipped_frame = cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
                if predicted_label in label_dict.keys():
                    predicted_nameID = label_encoder.classes_[predicted_label]  
                    print("Face Name: ", predicted_nameID)
                    print("Qr Name: ",qr_data)
                    
                    # Check if the QR code label matches the face recognition label
                    if qr_data == predicted_nameID:
                        match_found = True 
                        # Check if the current_date column exists, if not create it
                        if current_date not in attendance_df.columns:
                            attendance_df[current_date] = ''
        
                        # Mark attendance for the student with matching QR code and face label
                        attendance_df.loc[predicted_nameID, current_date] = "Present"
                        print("QR Code and Face Recognition labels matched.")
                        cv2.putText(flipped_frame, "Attendance Marked", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        break  
                    else:
                        print("QR Code and Face Recognition labels do not match.")
        
                else:
                    print("Label not found in Student List.")
        
                
        
            if not match_found:
                cv2.putText(flipped_frame, "Unmatched", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
            points = obj.polygon
            if len(points) == 4:
                pts = np.array([(pt.x, pt.y) for pt in points], dtype=int)
                cv2.polylines(flipped_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
            qr_code_text = f"QR Code: {qr_data}"
            face_recognition_text = f"Face Recognition: {predicted_nameID}"  # Use nameID instead of label number
            cv2.putText(flipped_frame, qr_code_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(flipped_frame, face_recognition_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Attendance System (Press Q to Exit)", flipped_frame)
        attendance_df.to_csv(r'StudentData\attendancelist.csv')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
    
def attendance_chart():
    df = pd.read_csv(r'Model/attendancelist.csv')

    df.set_index('nameID', inplace=True)

    # Count the number of present days for each student
    df['Student_Present'] = df.apply(lambda row: row.str.count('Present').sum(), axis=1)

    # Count the total number of days (excluding the 'nameID' column)
    total_days = len(df.columns) - 1

    # Calculate the percentage of attendance for each student
    df['Attendance_Percentage'] = ((df['Student_Present'] / total_days) * 100).round(2)

    # Print the DataFrame with total present days and attendance percentage
    text = "Total Classes: "+str(total_days)
    
    # Get the width of the console or terminal
    terminal_width = os.get_terminal_size().columns

    # Calculate the number of spaces needed to center the text
    left_padding = (terminal_width - len(text)) // 2

    # Printing Total Classes in Center
    print(" " * left_padding + text)
    print("\n")
    print(df[['Student_Present', 'Attendance_Percentage']])

    # Create a pie chart
    for student, attendance_percentage in df['Attendance_Percentage'].items():
        plt.figure(figsize=(6, 6))
        plt.pie([attendance_percentage, 100 - attendance_percentage], labels=['Present', 'Absent'], autopct='%1.1f%%', startangle=140)
        plt.title(f'Attendance Ratio for {student}')
        plt.savefig(f'StudentData/Attendance_chart/{student}_attendance_chart.png')
        plt.close()

    print("\n Pie charts generated and saved.")



def show_all_chart():
    chart_files = os.listdir('D:/Dl_Model/StudentData/Attendance_chart')

    # Iterate through the chart files and display them
    for chart_file in chart_files:
        if chart_file.endswith('.png'):
            # Create the full file path
            chart_path = os.path.join(r'StudentData/Attendance_chart', chart_file)
            
            # Load and display the chart
            img = plt.imread(chart_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(chart_file)  # Set the title as the file name
            plt.axis('off')  # Turn off axis
            plt.show()



def show_student_chart():
    firstname = str(input("Enter First Name: ")).title().strip()
    lastname = str(input("Enter Last Name: ")).title().strip()
    enroll = str(input("Enter Enrollment No: ")).upper().strip()
    nameID = firstname + lastname + "-" + enroll
    print(nameID)
    chart_path = os.path.join(r'StudentData/Attendance_chart',nameID+'_attendance_chart.png')
            
    # Load and display the chart
    img = plt.imread(chart_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.show()
    

#mark_attendance()
#attendance_chart()
#show_chart()
