## Load the saved model and classify
import torch
import streamlit as st
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Define the same Custom CNN model

class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))  
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x)))) 
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))

        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the state dic to the model

model = ConvNet()
model.load_state_dict(
    torch.load("brain_tumor_customCNN_model.pth")
)
model.eval()

st.markdown("<h1 style='text-align: center; color: green;'>🧠 Brain Tumor Image Classification</h1>", unsafe_allow_html=True)
st.write("# Upload a Brain MRI image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs.data, 1)

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    prediction = classes[predicted.item()]

    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Classify Image"):
        st.write(f"## Prediction: {prediction}")
        st.success(f"The model predicts: **{prediction}**")