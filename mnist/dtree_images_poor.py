import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     
        self.threshold = threshold  
        self.left = left          
        self.right = right          
        self.value = value         

def gini(y):
    c, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def split(X, Y):
    best_feature, best_threshold = None, None
    best_gini = float("inf")

    #n_samples, n_features = X.shape
    n_features = X.shape[1]
    n_features_to_try = min(50, n_features)  # 50 losowych featuresów
    features_to_try = np.random.choice(n_features, size=n_features_to_try, replace=False)
    
    #for feature in features_to_try:
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = X[:, feature] > t
            if len(Y[left_mask]) == 0 or len(Y[right_mask]) == 0:
                continue
            g = (
                len(Y[left_mask]) / len(Y) * gini(Y[left_mask]) +
                len(Y[right_mask]) / len(Y) * gini(Y[right_mask])
            )
            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

def build_tree(X, Y, max_depth=5, depth=0):
    if len(np.unique(Y)) == 1 or depth >= max_depth:
        classes, codes = np.unique(Y, return_inverse=True)
        value = classes[np.bincount(codes).argmax()]
        return Node(value=value)
    feature, threshold = split(X, Y)
    if feature is None:
        classes, codes = np.unique(Y, return_inverse=True)
        value = classes[np.bincount(codes).argmax()]
        return Node(value=value)

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left = build_tree(X[left_mask], Y[left_mask], max_depth=max_depth, depth=depth + 1)
    right = build_tree(X[right_mask], Y[right_mask], max_depth=max_depth, depth=depth + 1)

    return Node(feature, threshold, left, right)

def predict_one(node, x):
    if node.value is not None:
        return node.value
    elif x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i, j] = np.sum((y_true == c1) & (y_pred == c2))
    return matrix

def decode_image(image_dict, resize_to=14, quantize_levels=32):
    #"""Decode PNG bytes to flattened pixel array with resizing and quantization."""
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        img = Image.open(BytesIO(image_dict['bytes']))
        if resize_to:
            img = img.resize((resize_to, resize_to), Image.Resampling.BILINEAR)
        pixels = np.array(img).flatten().astype(float) # Convert to numpy array
        if quantize_levels:
            pixels = (pixels / 255.0 * (quantize_levels - 1)).astype(int)
        
        return pixels
    return np.array([])

#def train_test_split_manual(X, y, test_size=0.2):
#    n = len(X)
#    idx = np.random.permutation(n)
#    test_count = int(n * test_size)
#    test_idx = idx[:test_count]
#    train_idx = idx[test_count:]
#    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Ładowanie datasetu
splits = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'}
df = pd.read_parquet(splits["train"])
fd = pd.read_parquet(splits["test"])

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

# images -> pixel arrays, labels -> string + kwantyzacja
print("\nDekodowanie obrazów (zmniejszanie do 14x14, kwantyzacja do 32-ki)...")
images = np.array([decode_image(img, resize_to=14, quantize_levels=32) for img in df['image']])
labels = df['label'].values.astype(str)
t_images = np.array([decode_image(img, resize_to=14, quantize_levels=32) for img in fd['image']])
t_labels = fd['label'].values.astype(str)


print(f"images.shape: {images.shape}")
print(f"labels.shape: {labels.shape}")

SAMPLE_SIZE = None  # None for full dataset
if SAMPLE_SIZE:
    idx = np.random.choice(len(images), SAMPLE_SIZE, replace=False)
    images = images[idx]
    labels = labels[idx]
    print(f"Subsampled to {len(images)} images for faster training")

# Split train/test
#X_train, X_test, y_train, y_test = train_test_split_manual(images, labels, test_size=0.2)
X_train, y_train = images, labels
X_test, y_test = t_images, t_labels

print("\nTraining decision tree...")
tree = build_tree(X_train, y_train, max_depth=5) # mniejsza głębokość dla pixeli

y_pred = predict(tree, X_test)
print(f"\nAccuracy: {accuracy(y_test, y_pred):.4f}") #f
print(f"Confusion matrix:\n {confusion_matrix(y_test, y_pred)}")

