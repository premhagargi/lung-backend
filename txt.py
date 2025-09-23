import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import pickle

app = Flask(__name__)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_resnet_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    os.makedirs("model_weights", exist_ok=True)
    weight_os = {'state_dict': model.state_dict()}
    torch.save(weight_os, "model_weights/resnet50_lung_ct.pth")
    model.load_state_dict(torch.load("model_weights/resnet50_lung_ct.pth")['state_dict'])
    return model, feature_extractor

def simulate_training_data(num_samples=100):
    np.random.seed(42)
    X_train = np.random.rand(num_samples, 2048)
    y_train = np.random.randint(0, 2, num_samples)
    return X_train, y_train

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    os.makedirs("model_weights", exist_ok=True)
    with open("model_weights/knn_lung_ct.pkl", "wb") as f:
        pickle.dump(knn, f)
    with open("model_weights/knn_lung_ct.pkl", "rb") as f:
        knn = pickle.load(f)
    return knn

def train_naive_bayes(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    os.makedirs("model_weights", exist_ok=True)
    with open("model_weights/nb_lung_ct.pkl", "wb") as f:
        pickle.dump(nb, f)
    with open("model_weights/nb_lung_ct.pkl", "rb") as f:
        nb = pickle.load(f)
    return nb

def predict_lung_cancer(image_bytes, resnet_model, feature_extractor, knn_model, nb_model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        resnet_model.to(device)
        with torch.no_grad():
            outputs = resnet_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            resnet_confidence, resnet_predicted = torch.max(probabilities, 1)
        feature_extractor.to(device)
        with torch.no_grad():
            features = feature_extractor(image_tensor).cpu().numpy().flatten()
        knn_pred = knn_model.predict([features])[0]
        knn_probs = knn_model.predict_proba([features])[0]
        knn_confidence = np.max(knn_probs) * 100
        nb_pred = nb_model.predict([features])[0]
        nb_probs = nb_model.predict_proba([features])[0]
        nb_confidence = np.max(nb_probs) * 100
        class_names = ['Non-Cancerous', 'Cancerous']
        return {
            'status': 'success',
            'resnet50': {
                'prediction': class_names[resnet_predicted.item()],
                'confidence_score': round(resnet_confidence.item() * 100, 2)
            },
            'knn': {
                'prediction': class_names[knn_pred],
                'confidence_score': round(knn_confidence, 2)
            },
            'naive_bayes': {
                'prediction': class_names[nb_pred],
                'confidence_score': round(nb_confidence, 2)
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model, feature_extractor = load_resnet_model()
X_train, y_train = simulate_training_data()
knn_model = train_knn(X_train, y_train)
nb_model = train_naive_bayes(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'}), 400
    try:
        image_bytes = file.read()
        result = predict_lung_cancer(image_bytes, resnet_model, feature_extractor, knn_model, nb_model)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)