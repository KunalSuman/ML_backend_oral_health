import pathlib, sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm

# ----- CONFIG -----
MODEL_PATH = pathlib.Path("oral_disease_classifier.pth")
CLASS_NAMES = ["calculus","caries","gingivitis","hypodontia","toothDiscoloration","ulcers"]
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------

# 1 Build EfficientNet‑B0 via timm
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

# 2 Preprocess
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    print(f"File  : {img_path}")
    print(f"Pred  : {CLASS_NAMES[idx]}  ({probs[idx]:.3f})")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:<17}: {p:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python infer_effnet.py image.jpg")
    predict(sys.argv[1])
