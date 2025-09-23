import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import os

# Load and extract features
def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Smaller for HOG
    features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# Collect data
train_dir = './data/train_processed'
test_dir = './data/test_processed'
classes = ['Cancer', 'NonCancer']

train_features = []
train_labels = []
for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        features = extract_hog_features(img_path)
        train_features.append(features)
        train_labels.append(0 if cls == 'Cancer' else 1)

test_features = []
test_labels = []
for cls in classes:
    cls_path = os.path.join(test_dir, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        features = extract_hog_features(img_path)
        test_features.append(features)
        test_labels.append(0 if cls == 'Cancer' else 1)

# Train NB
nb = GaussianNB()
nb.fit(train_features, train_labels)

# Predict
preds = nb.predict(test_features)
acc = accuracy_score(test_labels, preds)
print(f'Naive Bayes Accuracy: {acc:.4f}')
print(classification_report(test_labels, preds, target_names=['Cancer', 'NonCancer']))

# Save model
import joblib
joblib.dump(nb, 'nb_model.pkl')