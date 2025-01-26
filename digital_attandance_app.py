try:
    print("\n           Welcome")
    while True:
        print("-"*40)
        print("Press 1 to register user")
        print("Press 2 to Train Model")
        print("Press 3 for Face Prediction Testing")
        print("Press 4 to Mark Attendance")
        print("Press 5 for Attendace Info and update Graph")
        print("Press 6 for Attendance Chart")
        print("Press 7 to Delete User \n")
        
        choice = input("Enter your choice (-1 for exit the loop): ")
        
        if choice == '1':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.student_register import register_user
            register_user()
            
        elif choice == '2':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.face_train_model import face_train
            face_train()
    
        
        elif choice == '3':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.face_prediction import face_prediction
            face_prediction()
        
        elif choice == '4':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.mark_attendance import mark_attendance
            mark_attendance()
            
        elif choice == '5':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.mark_attendance import attendance_chart
            attendance_chart()
            
        elif choice == '6':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.mark_attendance import show_student_chart
            show_student_chart()
            
        elif choice == '7':
            print("Loading...")
            print("Please Wait for a while")
            from attendance_features.remove_data import remove_user
            remove_user()
    
        elif choice == '-1':
            break
        
        else:
            print("\n\n","-"*10,"Enter Correct Choice","-"*10,"\n")
            
    """except ImportError:
        print("Error: Unable to import a module.")
    except ValueError:
        print("Error: Invalid input. Please enter a valid number.")"""
    
except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    print("\n","-"*10,"Thankyou for using me. Have a nice day :)","-"*10)

















