#!/usr/bin/env python
"""
train_oral_classifier.py
========================
Fine‑tune EfficientNet‑B0 on Oral_Dataset and export metrics matrix.
"""

# ---------- 0. Imports & GPU config ----------
import os, random, pathlib, datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

# Enable dynamic memory growth (avoids OOM on 4 GB GPUs)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Mixed precision → faster & leaner on Ampere GPUs (RTX 3050)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ---------- 1. Paths & Hyper‑params ----------
DATA_DIR   = pathlib.Path("Oral_Dataset")       # adjust if needed
IMG_SIZE   = (224, 224)                         # EfficientNet‑B0 input
BATCH_SIZE = 16                                 # safe for 4 GB w/ mixed precision
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.10                               # of (train+val) pool
SEED       = 42
EPOCHS_FROZEN = 8
EPOCHS_FT     = 4
NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ---------- 2. Build tf.data datasets ----------
train_val = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

class_names = train_val.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}):", class_names)

# carve out TEST from train_val
test_batches = int(len(train_val) * TEST_SPLIT)
test_ds  = train_val.take(test_batches)
train_ds = train_val.skip(test_batches)

# caching + prefetch
AUTOTUNE = tf.data.AUTOTUNE
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
])
train_ds = (train_ds
            .map(lambda x, y: (data_aug(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE))
val_ds  = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base.trainable = False  # stage‑1

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax", dtype="float32")  
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ---------- 4. Train (frozen) ----------
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN)

# ---------- 5. Fine‑tune last N layers ----------
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT)

model.save(f"oral_classifier_{NOW}.h5")
print("Model saved.")

# ---------- 6. Evaluation & Metrics Matrix ----------
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n--- Classification Report ---")
report = classification_report(
    y_true, y_pred, target_names=class_names, digits=3)
print(report)

# Confusion matrix heat‑map
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – Oral Disease Classifier")
plt.tight_layout()
plt.savefig(f"confusion_matrix_{NOW}.png", dpi=300)
plt.show()
print(f"Confusion matrix saved as confusion_matrix_{NOW}.png")
