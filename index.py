# index.py
import io
import torch
import pathlib
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify

# 1. Configuration
MODEL_PATH = pathlib.Path("best_model/efficientvit_b0_oral_disease_classifier.pth")
CLASS_NAMES = ["calculus","caries","gingivitis",
               "hypodontia","toothDiscoloration","ulcers"]
IMG_SIZE = 224

# 2. Build the same model architecture
from model import EfficientvitB0Classifier  # adjust import to match Priyanshu’s code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientvitB0Classifier(num_classes=len(CLASS_NAMES))
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# 3. Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                   # scales [0–255]→[0.0–1.0]
    transforms.Normalize(                    # use ImageNet means/std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# 4. Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="no image file"), 400
    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # shape [1,3,224,224]

    with torch.no_grad():
        logits = model(input_tensor)            # shape [1,6]
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    result = {
        "supported_diseases": CLASS_NAMES,
        "prediction":          CLASS_NAMES[top_idx],
        "confidence":          float(np.round(probs[top_idx], 4)),
        "scores": {
            name: float(np.round(score, 4))
            for name, score in zip(CLASS_NAMES, probs)
        }
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
