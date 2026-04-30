import os
import sys
import time
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_mnist_data
from models import build_mlp_model

class PerClassKMeansClassifier:
    """K-means per class, then nearest centroid at prediction time."""
    def __init__(self, clusters_per_class: int, random_state: int = 42) -> None:
        self.clusters_per_class = clusters_per_class
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.centroids_ = None
        self.centroid_labels_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PerClassKMeansClassifier":
        x_scaled = self.scaler.fit_transform(x)
        centroids = []
        centroid_labels = []

        for label in sorted(np.unique(y)):
            class_x = x_scaled[y == label]
            k = min(self.clusters_per_class, len(class_x))
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(class_x)
            centroids.append(model.cluster_centers_)
            centroid_labels.extend([label] * k)

        self.centroids_ = np.vstack(centroids)
        self.centroid_labels_ = np.asarray(centroid_labels)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.centroid_labels_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        x_scaled = self.scaler.transform(x)
        centroids = self.centroids_
        centroid_sq_norms = np.sum(centroids * centroids, axis=1)
        nearest = np.empty(x_scaled.shape[0], dtype=np.int64)

        batch_size = 2048
        for start in range(0, x_scaled.shape[0], batch_size):
            end = min(start + batch_size, x_scaled.shape[0])
            batch = x_scaled[start:end]
            batch_sq_norms = np.sum(batch * batch, axis=1, keepdims=True)
            distances = batch_sq_norms + centroid_sq_norms[None, :] - 2.0 * (batch @ centroids.T)
            nearest[start:end] = np.argmin(distances, axis=1)

        return self.centroid_labels_[nearest]

def main():
    MNIST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reduced MNIST Data")
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_mnist_data(MNIST_DATA_DIR)
    
    input_size = x_train.shape[1] * x_train.shape[2]
    x_train_flat = x_train.reshape((-1, input_size)).astype("float32") / 255.0
    x_test_flat = x_test.reshape((-1, input_size)).astype("float32") / 255.0

    # 1. Build and Train AutoEncoder
    print("\n--- Training AutoEncoder ---")
    input_img = layers.Input(shape=(input_size,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    bottleneck = layers.Dense(32, activation='relu', name='bottleneck')(encoded)
    
    decoded = layers.Dense(64, activation='relu')(bottleneck)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_size, activation='sigmoid')(decoded)
    
    autoencoder = models.Model(input_img, decoded)
    encoder = models.Model(input_img, bottleneck)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    autoencoder.fit(
        x_train_flat, x_train_flat,
        epochs=30,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test_flat, x_test_flat),
        verbose=1
    )
    
    # Extract Features
    print("\n--- Extracting AutoEncoder Bottleneck Features ---")
    feat_time_start = time.perf_counter()
    x_train_feat = encoder.predict(x_train_flat, verbose=0)
    x_test_feat = encoder.predict(x_test_flat, verbose=0)
    feat_time_end = time.perf_counter()
    extraction_time_ms_per_sample = ((feat_time_end - feat_time_start) / (len(x_train_flat) + len(x_test_flat))) * 1000.0
    
    print(f"Feature extraction complete. Bottleneck size: {x_train_feat.shape[1]}")

    results = []

    def log_result(name, acc, t_ms):
        print(f"{name:30} | Acc: {acc*100:5.1f}% | Time: {t_ms:.1f} ms")
        results.append(f"{name}|{acc*100:.1f}|{t_ms:.1f}")

    # ==========================
    # ASSIGNMENT 1 CLASSIFIERS
    # ==========================
    print("\n--- Running K-Means Classifiers ---")
    for k in [1, 4, 16, 32]:
        start_t = time.perf_counter()
        clf = PerClassKMeansClassifier(clusters_per_class=k)
        clf.fit(x_train_feat, y_train)
        preds = clf.predict(x_test_feat)
        end_t = time.perf_counter()
        acc = accuracy_score(y_test, preds)
        t_ms = (end_t - start_t) * 1000.0
        log_result(f"K-Means K={k}", acc, t_ms)

    print("\n--- Running SVM Classifiers ---")
    for name, clf in [("SVM Linear", SVC(kernel="linear", random_state=42)),
                      ("SVM RBF", SVC(kernel="rbf", gamma="scale", random_state=42))]:
        start_t = time.perf_counter()
        clf.fit(x_train_feat, y_train)
        preds = clf.predict(x_test_feat)
        end_t = time.perf_counter()
        acc = accuracy_score(y_test, preds)
        t_ms = (end_t - start_t) * 1000.0
        log_result(name, acc, t_ms)

    # ==========================
    # ASSIGNMENT 2 (MLP) CLASSIFIERS
    # ==========================
    print("\n--- Running MLP Classifiers ---")
    
    experiments = [
        {"name": "MLP_1_Hidden", "layers": 1, "epochs": 50},
        {"name": "MLP_3_Hidden", "layers": 3, "epochs": 50},
        {"name": "MLP_4_Hidden", "layers": 4, "epochs": 50}
    ]

    for exp in experiments:
        start_t = time.perf_counter()
        
        # Build standard MLP from Problem 1 but using bottleneck dimension
        tf.keras.backend.clear_session()
        model = models.Sequential(name=exp["name"])
        model.add(layers.InputLayer(input_shape=(x_train_feat.shape[1],)))
        
        neurons_per_layer = [512, 256, 128, 64]
        for i in range(exp["layers"]):
            model.add(layers.Dense(units=neurons_per_layer[i], activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=0.2))
        
        model.add(layers.Dense(units=10, activation='softmax'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
                      
        # Train and test
        model.fit(x_train_feat, y_train, epochs=exp["epochs"], batch_size=256, verbose=0)
        _, acc = model.evaluate(x_test_feat, y_test, verbose=0)
        
        end_t = time.perf_counter()
        t_ms = (end_t - start_t) * 1000.0
        log_result(exp["name"], acc, t_ms)
        
    print("\n=================")
    print("FINAL RESULTS FOR TABLE")
    print("=================")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
