import pandas as pd
import numpy as np
import csv

splits = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet'}
df = pd.read_parquet( splits["train"]) #"hf://datasets/ylecun/mnist/" +

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

#def split_nlogn(X, y):
#    n_samples, n_features = X.shape
#    best_gini = float("inf")
#    best_feature, best_threshold = None, None
#
#    for feature in range(n_features):
#        sorted_idx = np.argsort(X[:, feature])
#        X_sorted = X[sorted_idx]
#        Y_sorted = y[sorted_idx]
#
#        num_left = {}
#        num_right = {}
#
#        for label in y:
#            num_right[label] = num_right.get(label, 0) + 1
#
#        left_size = 0
#        right_size = len(y)
#
#        for i in range(1, n_samples):
#            label = Y_sorted[i - 1]
#
#            num_left[label] = num_left.get(label, 0) + 1
#            num_right[label] -= 1
#
#            left_size += 1
#            right_size -= 1
#
#            if X_sorted[i] == X_sorted[i - 1]:
#                continue  
#
#            #if X_sorted[i] != X_sorted[i - 1]:
#             #   break  
#
#            g_left = gini(Y_sorted[:left_size])
#            g_right = gini(Y_sorted[left_size:])
#
#            g_total = (left_size / n_samples) * g_left + \
#                      (right_size / n_samples) * g_right
#
#            if g_total < best_gini:
#                best_gini = g_total
#                best_feature = feature
#                best_threshold = (X_sorted[i] + X_sorted[i - 1]) / 2
#
#    return best_feature, best_threshold

def split(X, Y):
    best_feature, best_threshold = None, None
    best_gini = float("inf")

    #n_samples, n_features = X.shape
    n_features = X.shape[1]
    #features = np.random.choice(n_features, size=y, replace=False)
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

    if x[node.feature] <= node.threshold:
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

df = pd.read_csv(
    "data.csv",
    sep=",",
    quotechar=None,
    quoting=csv.QUOTE_NONE,
    engine="python",
)
df.columns = df.columns.str.replace('"', '').str.strip()
df = df.map(lambda v: v.strip('"') if isinstance(v, str) else v)

#print(df.columns)
#print(df.head())    
#print(df.info()) 

X_df = df.drop("species", axis=1)
y_df = df["species"]
X = X_df.values
y = y_df.values
X_df = pd.get_dummies(X_df)
X = X_df.values


#####O(n^2)########
def train_test_split_manual(X, y, test_size=0.2):
    n = len(X)
    idx = np.random.permutation(n)
    test_count = int(n * test_size)
    test_idx = idx[:test_count]
    train_idx = idx[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
###################

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
X_df = X_df.astype(float)
X = X_df.to_numpy()
tree = build_tree(X_train, y_train, max_depth=5)

y_pred = predict(tree, X_test)

print("Accuracy:", accuracy(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#########
new_test = pd.DataFrame([{
    "sepal_length": 5.1,
    "sepal_width": 3.1,
    "petal_length": 1.42,
    "petal_width": 0.1,
}]) 
xy = pd.get_dummies(new_test)
xy = new_test.reindex(columns=X_df.columns, fill_value=0).astype(float).to_numpy()
prediction = predict(tree, xy)
print(prediction)
#########