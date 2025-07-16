#!/usr/bin/env python3
"""
download_weights.py

Fetches the outputs of a public Kaggle kernel and extracts any .pth/.h5 files
it produced into a local directory.
"""

import os
import zipfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

# ───── CONFIG ────────────────────────────────────────────────────────────────

# 1) Set these to the Kaggle kernel you want to pull from:
OWNER_SLUG  = "dharmaputra13"
KERNEL_SLUG = "oral-classification-inceptionresnetv2"

# 2) Where to put the downloaded output ZIP & extracted files:
DEST_DIR    = "kernel_output"
ZIP_PATH    = os.path.join(DEST_DIR, "kernel_output.zip")

# ───── HELPER FUNCTIONS ───────────────────────────────────────────────────────

def download_with_kaggle_api():
    """Uses the Kaggle Python API to pull the kernel output ZIP."""
    api = KaggleApi()
    api.authenticate()
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading kernel outputs for {OWNER_SLUG}/{KERNEL_SLUG}…")
    api.kernels_output(f"{OWNER_SLUG}/{KERNEL_SLUG}", path=DEST_DIR)
    print("✅ Download complete (kaggle API).")

def download_with_http():
    """
    Fallback: download via the Kaggle REST API using requests.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in env vars.
    """
    USER = os.environ.get("KAGGLE_USERNAME")
    KEY  = os.environ.get("KAGGLE_KEY")
    if not USER or not KEY:
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY for HTTP fallback.")
    url = (
        f"https://www.kaggle.com/api/v1/kernels/"
        f"{OWNER_SLUG}/{KERNEL_SLUG}/output"
    )
    print(f"Downloading kernel outputs via HTTP from {url} …")
    r = requests.get(url, auth=(USER, KEY), stream=True)
    r.raise_for_status()
    os.makedirs(DEST_DIR, exist_ok=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)
    print("✅ Download complete (HTTP).")
    print("Unzipping…")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DEST_DIR)
    os.remove(ZIP_PATH)
    print("✅ Unzipped to", DEST_DIR)

# ───── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try Kaggle API first
    try:
        download_with_kaggle_api()
    except Exception as e:
        print("Kaggle API download failed:", e)
        print("Falling back to HTTP download…")
        download_with_http()

    # List any .pth or .h5 files that showed up
    print("\nFiles in", DEST_DIR, ":")
    for root, _, files in os.walk(DEST_DIR):
        for fn in files:
            if fn.endswith((".pth", ".h5")):
                print("  ", os.path.join(root, fn))
