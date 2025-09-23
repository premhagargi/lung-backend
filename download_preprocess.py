import os
from sklearn.model_selection import train_test_split
import cv2

data_dir = './data/ChestCTScan'
output_train_dir = './data/train_processed'
output_test_dir = './data/test_processed'

# Create output directories
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Helper to map folder names to class
def get_class(folder_name):
    return 'NonCancer' if 'normal' in folder_name.lower() else 'Cancer'

# Collect all image paths and labels
images, labels = [], []
for split in ['train', 'test', 'valid']:
    split_path = os.path.join(data_dir, split)
    for subfolder in os.listdir(split_path):
        sub_path = os.path.join(split_path, subfolder)
        if not os.path.isdir(sub_path):
            continue
        cls = get_class(subfolder)
        for img_name in os.listdir(sub_path):
            img_path = os.path.join(sub_path, img_name)
            if os.path.isfile(img_path):
                images.append(img_path)
                labels.append(cls)

# Encode classes
class_names = ['Cancer', 'NonCancer']
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
labels_idx = [class_to_idx[lbl] for lbl in labels]

# Split into train/test
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    images, labels_idx, test_size=0.2, stratify=labels_idx, random_state=42
)

# Resize and save
def resize_and_save(img_paths, labels, output_dir):
    for img_path, label in zip(img_paths, labels):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        cls_name = class_names[label]
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        cv2.imwrite(os.path.join(cls_dir, os.path.basename(img_path)), img)

resize_and_save(train_imgs, train_labels, output_train_dir)
resize_and_save(test_imgs, test_labels, output_test_dir)

print(f"Train: {len(train_imgs)} images, Test: {len(test_imgs)} images")
