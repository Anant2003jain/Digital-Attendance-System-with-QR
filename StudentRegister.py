import cv2
import os
import pandas as pd
from csv import writer
import qrcode as qc

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def register_user():
    firstname = str(input("Enter First Name: ")).title().strip()
    lastname = str(input("Enter Last Name: ")).title().strip()
    enroll = str(input("Enter Enrollment No: ")).upper().strip()
    
    nameID = firstname + lastname
    path = r'StudentData//images//' + nameID + "-" + enroll
    
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("Directory created successfully.")
        
        # Code to add user data to CSV file (class.csv)
        df = pd.read_csv(r"StudentData//class.csv")
        if df.empty:
            lst = [0, (nameID + "-" + enroll)]
        else:        
            df1 = df.tail(1)
            df3 = df1.iloc[0, 0]
            i = str(df3 + 1)
            lst = [i, (nameID + "-" + enroll)]
            
        with open(r'StudentData//class.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(lst)
            f_object.close()

        count = 0
        while True:
            ret, frame = video.read()
            faces = facedetect.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in faces:
                count = count + 1
                img_name = r'StudentData//images//' + nameID + '-' + enroll + '//' + str(count) + '.jpg'
                print("Creating Images........." + img_name)
                cv2.imwrite(img_name, frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imshow("Registration", frame)
            cv2.waitKey(1)
            if count > 500:
                break

        video.release()
        cv2.destroyAllWindows()

        # Generate QR code
        try:
            inp = firstname + lastname + "-" + enroll
            features = qc.QRCode(version=2, box_size=20, border=2)
            features.add_data(inp)
            features.make(fit=True)
            generate_image = features.make_image(fill_color="black", back_color="white")
            generate_image.save(r"StudentData//QR//{}.png".format(inp))
            print("Successfully Created Student QR")
        except Exception as e:
            print("Unable to generate QR: ", str(e))
    else:
        print("Directory already exists. Please enter a different Enrollment No.")
        
#register_user()
