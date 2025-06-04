import os
import send2trash
import pandas as pd

def remove_user():
    try:
        firstname = input("Enter First Name: ").strip().title()
        lastname = input("Enter Last Name: ").strip().title()
        enroll = input("Enter Enrollment No: ").strip().upper()
        
        nameID = f"{firstname}{lastname}-{enroll}"

        csv_file_path = r"student_data\class.csv"
        image_folder_path = os.path.join(r"student_data\images", nameID)
        qr_code_path = os.path.join(r"student_data\QR", f"{nameID}.png")

        # Read CSV file
        df = pd.read_csv(csv_file_path)

        # Find user index
        index_to_remove = df.index[df['classname'] == nameID].tolist()

        if index_to_remove:
            # Remove user details from CSV file
            df.drop(index_to_remove, inplace=True)
            df.to_csv(csv_file_path, index=False)
            print("\nUser details removed from CSV.")

            # Delete folder and QR code
            if os.path.exists(image_folder_path):
                try:
                    send2trash.send2trash(image_folder_path)  # Recursively remove directory and its contents
                    print("Image folder deleted.")
                except OSError as e:
                    print(f"Error: {e}")
            else:
                print("Image folder not found.")

            if os.path.exists(qr_code_path):
                os.remove(qr_code_path)
                print("QR code deleted.")
            else:
                print("QR code not found.")
        else:
            print("User not found in the CSV file.")

    except Exception as e:
        print(f"An error occurred: {e}")

#remove_user()
