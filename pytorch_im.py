import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import timm  # If you want EfficientNet variants

# ------------ CONFIG ------------
DATA_DIR = Path("Oral_Dataset")
MODEL_OUT = "oral_disease_classifier_exp.pth"
BATCH_SIZE = 8
IMG_SIZE = 224
NUM_EPOCHS = 6
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 6  # Adjust if OOM or low on RAM

# ------------ DATASET -----------
# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Split dataset: 75% train, 15% val, 10% test (by indices)
num_total = len(dataset)
num_train = int(0.75 * num_total)
num_val = int(0.15 * num_total)
num_test = num_total - num_train - num_val
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, [num_train, num_val, num_test],
    generator=torch.Generator().manual_seed(42)
)

# Update transforms for val/test
val_ds.dataset.transform = val_transforms
test_ds.dataset.transform = val_transforms

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ---------- MODEL ---------------
# Use EfficientNet-B0 via timm or torchvision
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
model.to(DEVICE)

# ---------- TRAINING ------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    avg_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"✅ Saved best model to {MODEL_OUT}")

print("Training done.")

# Optional: Test accuracy
model.load_state_dict(torch.load(MODEL_OUT))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
