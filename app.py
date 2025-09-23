from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import joblib
import cv2
from skimage.feature import hog
import numpy as np

app = Flask(__name__)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = LungCNN().to(device)
cnn_model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
cnn_model.eval()

# ResNet
resnet_model = models.resnet50(pretrained=False).to(device)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 2)
resnet_model.load_state_dict(torch.load('resnet_model.pth', map_location=device))
resnet_model.eval()

# NB
nb_model = joblib.load('nb_model.pkl')

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def hog_features(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64))
    features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features.reshape(1, -1)

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = file.read()
    
    if model_type == 'cnn' or model_type == 'resnet':
        img_tensor = preprocess_image(io.BytesIO(img_bytes))
        with torch.no_grad():
            output = cnn_model(img_tensor) if model_type == 'cnn' else resnet_model(img_tensor)
            pred = torch.argmax(output, 1).item()
    elif model_type == 'nb':
        features = hog_features(img_bytes)
        pred = nb_model.predict(features)[0]
    else:
        return jsonify({'error': 'Invalid model: cnn, nb, or resnet'}), 400
    
    label = 'Cancer' if pred == 0 else 'NonCancer'
    confidence = torch.softmax(output, 1).max().item() if model_type != 'nb' else nb_model.predict_proba(features).max()
    
    return jsonify({'prediction': label, 'confidence': f'{confidence:.4f}'})

if __name__ == '__main__':
    app.run(debug=True)