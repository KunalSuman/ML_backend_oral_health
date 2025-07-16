from flask import Flask, request, jsonify
import io, base64
from PIL import Image
import torch, timm
from torchvision import transforms
import numpy as np
import requests

app = Flask(__name__)

# --- Config: Model paths and class names ---
MODEL1_PATH = "oral_disease_classifier.pth"   # EfficientNet-B0
MODEL2_PATH = "efficientvit_b0_oral_disease_classifier.pth"   # EfficientViT-B0
CLASS_NAMES = ["calculus","caries","gingivitis",
               "hypodontia","toothDiscoloration","ulcers"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# --- Load both models once at startup ---
model1 = timm.create_model("efficientnet_b0",
                          pretrained=False, num_classes=len(CLASS_NAMES))
model1.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
model1.to(DEVICE).eval()

model2 = timm.create_model("efficientvit_b0",
                          pretrained=False, num_classes=len(CLASS_NAMES))
model2.load_state_dict(torch.load(MODEL2_PATH, map_location=DEVICE))
model2.to(DEVICE).eval()


def predict(model, img_pil):
    """Run inference and return top label + all probs."""
    tensor = preprocess(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "label":      CLASS_NAMES[idx],
        "prob":       float(probs[idx]),
        "probs":      {k: float(p) for k,p in zip(CLASS_NAMES, probs)}
    }


@app.route("/predict", methods=["POST"])
def predict_api():
    # 1) Try multipart file upload
    if "file" in request.files:
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read()))

    # 2) Else try JSON with Base64 in 'image'
    elif request.is_json and "image" in request.json:
        b64 = request.json["image"]
        try:
            data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(data))
        except Exception as e:
            print(e)
            return jsonify({"error":"Invalid Base64 image"}), 400
    else:
        print("ERROR!")
        return jsonify({"error":"No image provided "
                                 "(use multipart 'file' or JSON 'image')."}), 400

    # Run predictions
    res1 = predict(model1, img)
    res2 = predict(model2, img)

    # Forward payload if needed
    payload = {
        "image": request.files.get("file", None) or "<base64>",
        "net_bo": res1,
        "vit_bo": res2
    }
    # you can uncomment to forward:
    # requests.post("http://192.168.160.126:8194", json=payload)

    print(res1, res2)
    return jsonify({
        "net-bo": res1,
        "vit-bo": res2
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
