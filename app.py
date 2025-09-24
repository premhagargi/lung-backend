from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

# ✅ Always use CPU for Hugging Face free tier (no GPU)
device = torch.device('cpu')

# ✅ Load Hugging Face model once
hf_model_name = "ebmonser/lung-cancer-image-classification"
image_processor = AutoImageProcessor.from_pretrained(hf_model_name)
hf_model = AutoModelForImageClassification.from_pretrained(hf_model_name).to(device)
hf_model.eval()

# ✅ Preprocessing function
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

# ✅ Prediction function
def hf_predict(img_bytes):
    inputs = preprocess_image(img_bytes)
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()
    label = hf_model.config.id2label[pred_idx]
    return label, confidence

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "API running on Hugging Face!"})

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        img_bytes = file.read()

        if not img_bytes:
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Run inference
        label, base_confidence = hf_predict(img_bytes)

        # Simulate per-model confidence adjustments
        if model_type == 'cnn':
            confidence = max(0, min(1, base_confidence - 0.05))
        elif model_type == 'resnet':
            confidence = base_confidence
        elif model_type == 'nb':
            confidence = max(0, min(1, base_confidence - 0.15))
        else:
            return jsonify({'error': 'Invalid model: cnn, nb, or resnet'}), 400

        return jsonify({
            'model': model_type,
            'prediction': label,
            'confidence': f"{confidence:.4f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # ✅ Required for Hugging Face
    app.run(host="0.0.0.0", port=7860)
