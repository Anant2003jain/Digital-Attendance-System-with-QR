import os
import cv2
import csv
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout   # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Load label mapping
label_dict = {}
with open(r'student_data\class.csv', mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile) 
    for row in csvreader:
        label_dict[int(row['classid'])] = row['classname']

# Function to load and preprocess images for training
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            img = img / 255.0  # Normalize pixel values to be between 0 and 1
            images.append(img)
    return np.array(images)

def face_train():
    # Load training data
    train_data = []
    labels = []
    for folder_name in os.listdir(r'student_data\images'):
        folder_path = os.path.join(r'student_data\images', folder_name)
        images = load_images_from_folder(folder_path)
        train_data.extend(images)
        labels.extend([folder_name] * len(images))
    
    train_data = np.array(train_data)
    labels = np.array(labels)
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)
    
    # Reshape data to fit the model
    X_train = np.reshape(X_train, (X_train.shape[0], 64, 64, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 64, 64, 1))
    
    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), kernel_regularizer='l2'),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        
        Dense(len(np.unique(labels)), activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))
    
    # Evaluate the model
    model.evaluate(X_test, y_test)
    
    # Evaluate the model
    print("[INFO] Evaluating the model...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test, np.argmax(predictions, axis=1), target_names=label_encoder.classes_))
    
    # Save the model to a file
    model.save(r'Model//new_face_recognition_model.h5')
    
    print(model.summary())

if __name__ == "__main__":
    face_train()