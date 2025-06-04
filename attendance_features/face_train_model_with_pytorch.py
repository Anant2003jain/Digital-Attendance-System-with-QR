import os
import cv2
import csv
import numpy as np
import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
import torch.optim as optim             # type: ignore
import torch.utils.data as data         # type: ignore
from torchvision import transforms      # type: ignore
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

class FaceDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = FaceDataset(X_train, y_train, transform=transform)
    test_dataset = FaceDataset(X_test, y_test, transform=transforms.ToTensor())
    
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define the CNN model
    model = CNNModel(num_classes=len(np.unique(labels)))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.unsqueeze(1).float()  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.unsqueeze(1).float()  # Add channel dimension
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    print("[INFO] Evaluating the model...")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    
    # Save the model to a file
    torch.save(model.state_dict(), r'Model//new_face_recognition_model.pth')
    
    print(model)

#face_train()