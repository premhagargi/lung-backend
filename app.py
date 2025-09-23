import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import kagglehub
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, jsonify
from PIL import Image
import io

# ------------------------------
# CONFIG
# ------------------------------
data_dir = './lung_ct_dataset'
MODEL_WEIGHTS_PATH = "lung_models"
os.makedirs(MODEL_WEIGHTS_PATH, exist_ok=True)
CNN_WEIGHTS = os.path.join(MODEL_WEIGHTS_PATH, "cnn_weights.pth")
RESNET_WEIGHTS = os.path.join(MODEL_WEIGHTS_PATH, "resnet_weights.pth")
PCA_MODEL = os.path.join(MODEL_WEIGHTS_PATH, "pca_model.pkl")
NB_MODEL = os.path.join(MODEL_WEIGHTS_PATH, "nb_model.pkl")
KNN_MODEL = os.path.join(MODEL_WEIGHTS_PATH, "knn_model.pkl")

num_classes = 4  # adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib, large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa, squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa, normal
class_names = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',
    'normal'
]
# Mapping from detailed train/valid names to test names
class_mapping = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'adenocarcinoma',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'large.cell.carcinoma',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous.cell.carcinoma',
    'normal': 'normal'
}

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

inference_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# 1. SIMPLE CNN
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*56*56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------------
# DATA LOADING
# ------------------------------
def load_data():
    train_dir = os.path.join(data_dir, next(f for f in os.listdir(data_dir) if f.lower() == "train"))
    val_dir   = os.path.join(data_dir, next(f for f in os.listdir(data_dir) if f.lower() == "valid"))
    test_dir  = os.path.join(data_dir, next(f for f in os.listdir(data_dir) if f.lower() == "test"))

    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory {train_dir} does not exist. Check dataset structure.")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory {val_dir} does not exist. Check dataset structure.")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} does not exist. Check dataset structure.")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=inference_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=inference_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Classes found in train:", train_dataset.classes)
    print("Classes found in valid:", val_dataset.classes)
    print("Classes found in test:", test_dataset.classes)
    
    if set(train_dataset.classes) != set(class_names):
        print("Warning: Detected classes in train/valid do not match expected class_names. Using detected classes.")
    
    return train_loader, val_loader, test_loader

# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    
    return model

# ------------------------------
# FEATURE EXTRACTION FOR ML MODELS
# ------------------------------
def extract_features(loader, feature_model):
    feature_model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = feature_model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# ------------------------------
# TRAIN AND SAVE MODELS
# ------------------------------
def train_and_save_models():
    if all(os.path.exists(path) for path in [CNN_WEIGHTS, RESNET_WEIGHTS, PCA_MODEL, NB_MODEL, KNN_MODEL]):
        print("Weights found, skipping training.")
        return
    print("\nTraining models...")
    # Rest of your training code...
    print("\nTraining models...")
    train_loader, val_loader, test_loader = load_data()
    
    # Train SimpleCNN
    cnn_model = SimpleCNN(num_classes)
    cnn_model = train_model(cnn_model, train_loader, val_loader)
    torch.save(cnn_model.state_dict(), CNN_WEIGHTS)
    print(f"Saved CNN weights to {CNN_WEIGHTS}")
    
    # Train ResNet
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = train_model(resnet, train_loader, val_loader)
    torch.save(resnet.state_dict(), RESNET_WEIGHTS)
    print(f"Saved ResNet weights to {RESNET_WEIGHTS}")
    
    # Extract features for Naive Bayes and kNN
    resnet_feature = models.resnet18(pretrained=True)
    resnet_feature.fc = nn.Identity()
    resnet_feature = resnet_feature.to(device)
    
    train_features, train_labels = extract_features(train_loader, resnet_feature)
    val_features, _ = extract_features(val_loader, resnet_feature)
    test_features, _ = extract_features(test_loader, resnet_feature)  # Optional: Use test features if needed
    
    # Train PCA
    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features)
    joblib.dump(pca, PCA_MODEL)
    print(f"Saved PCA model to {PCA_MODEL}")
    
    # Train Naive Bayes
    nb = GaussianNB()
    nb.fit(train_features_pca, train_labels)
    joblib.dump(nb, NB_MODEL)
    print(f"Saved Naive Bayes model to {NB_MODEL}")
    
    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features_pca, train_labels)
    joblib.dump(knn, KNN_MODEL)
    print(f"Saved kNN model to {KNN_MODEL}")

# ------------------------------
# LOAD MODELS
# ------------------------------
def load_models():
    print("\nLoading saved models...")
    cnn_model = SimpleCNN(num_classes).to(device)
    cnn_model.load_state_dict(torch.load(CNN_WEIGHTS))
    cnn_model.eval()
    
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)
    resnet.load_state_dict(torch.load(RESNET_WEIGHTS))
    resnet.eval()
    
    resnet_feature = models.resnet18(pretrained=True)
    resnet_feature.fc = nn.Identity()
    resnet_feature = resnet_feature.to(device)
    resnet_feature.eval()
    
    pca = joblib.load(PCA_MODEL)
    nb = joblib.load(NB_MODEL)
    knn = joblib.load(KNN_MODEL)
    
    return cnn_model, resnet, resnet_feature, pca, nb, knn

# ------------------------------
# FLASK API
# ------------------------------
app = Flask(__name__)


cnn_model, resnet, resnet_feature, pca, nb, knn = load_models()


def predict_image(image, model, transform, pca=None, is_cnn=True):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_cnn:
            output = model(image)
            probs = torch.softmax(output, dim=1)   # define probs here for CNN
            confidence, pred = torch.max(probs, 1)
        else:
            feats = resnet_feature(image)
            feats_np = feats.cpu().numpy()
            feats_pca = pca.transform(feats_np)
            proba = model.predict_proba(feats_pca)
            pred = np.argmax(proba, axis=1)[0]
            confidence = proba[0][pred] * 100

        if is_cnn:
            predicted_class = class_names[pred.item()]
        else:
            predicted_class = class_names[pred]
        mapped_class = class_mapping.get(predicted_class, predicted_class)
        return mapped_class, confidence


@app.route('/')
def index():
    return {"status": "API is running"}


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    try:
        image = Image.open(file).convert('RGB')
        
        cnn_pred, cnn_conf = predict_image(image, cnn_model, inference_transform, is_cnn=True)
        resnet_pred, resnet_conf = predict_image(image, resnet, inference_transform, is_cnn=True)
        nb_pred, nb_conf = predict_image(image, nb, inference_transform, pca=pca, is_cnn=False)
        knn_pred, knn_conf = predict_image(image, knn, inference_transform, pca=pca, is_cnn=False)
        
        return jsonify({
            'cnn': {'prediction': cnn_pred, 'confidence': f'{cnn_conf.item():.2f}%'},
            'resnet': {'prediction': resnet_pred, 'confidence': f'{resnet_conf.item():.2f}%'},
            'naive_bayes': {'prediction': nb_pred, 'confidence': f'{nb_conf:.2f}%'},
            'knn': {'prediction': knn_pred, 'confidence': f'{knn_conf:.2f}%'}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------
# TEST WITH SAMPLE IMAGES
# ------------------------------
def test_sample_images():
    print("\nTesting sample images...")
    sample_images = [
        os.path.join(data_dir, "test/normal/6 - Copy (2).png"),
        os.path.join(data_dir, "test/normal/7.png")
    ]
    for img_path in sample_images:
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                image = Image.open(f).convert('RGB')
                cnn_pred, cnn_conf = predict_image(image, cnn_model, inference_transform, is_cnn=True)
                resnet_pred, resnet_conf = predict_image(image, resnet, inference_transform, is_cnn=True)
                nb_pred, nb_conf = predict_image(image, nb, inference_transform, pca=pca, is_cnn=False)
                knn_pred, knn_conf = predict_image(image, knn, inference_transform, pca=pca, is_cnn=False)
                
                print(f"\nImage: {img_path}")
                print(f"CNN: {cnn_pred} ({cnn_conf.item():.2f}%)")
                print(f"ResNet: {resnet_pred} ({resnet_conf.item():.2f}%)")
                print(f"Naive Bayes: {nb_pred} ({nb_conf:.2f}%)")
                print(f"KNN: {knn_pred} ({knn_conf:.2f}%)")
        else:
            print(f"\nImage {img_path} not found")

# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == '__main__':
    # Check if all weights exist
    weights_exist = all(os.path.exists(path) for path in [CNN_WEIGHTS, RESNET_WEIGHTS, PCA_MODEL, NB_MODEL, KNN_MODEL])
    
    # Train and save models if weights don't exist
    if not weights_exist:
        train_and_save_models()
    
    # Load models
    
    # Test sample images
    test_sample_images()
    
    # Start Flask API
    print("\nStarting Flask API on http://localhost:5000/predict")
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000))) 