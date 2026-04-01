import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def decode_image(image_dict, resize_to=14, quantize_levels=32):
    #"""Decode PNG bytes to flattened pixel array with resizing and quantization."""
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        img = Image.open(BytesIO(image_dict['bytes']))
        if resize_to:
            img = img.resize((resize_to, resize_to), Image.Resampling.BILINEAR)
        pixels = np.array(img).flatten().astype(float)
        if quantize_levels:
            pixels = (pixels / 255.0 * (quantize_levels - 1)).astype(int)
        return pixels
    return np.array([])

# Load dataset
splits = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'}
df = pd.read_parquet(splits["train"])
fd = pd.read_parquet(splits["test"])

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

#dekodowanie obrazka 
print("\nDekodowanie obrazów (zmniejszanie do 14x14, kwantyzacja do 32-ki)...")
images = np.array([decode_image(img, resize_to=14, quantize_levels=32) for img in df['image']])
labels = df['label'].values
t_images = np.array([decode_image(img, resize_to=14, quantize_levels=32) for img in fd['image']])
t_labels = fd['label'].values

print(f"Train images shape: {images.shape}")
print(f"Train labels shape: {labels.shape}")
print(f"Test images shape: {t_images.shape}")
print(f"Test labels shape: {t_labels.shape}")

print("\nTraining decision tree with sklearn...")
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(images, labels)

y_pred = tree.predict(t_images)
acc = accuracy_score(t_labels, y_pred)
conf_matrix = confusion_matrix(t_labels, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")