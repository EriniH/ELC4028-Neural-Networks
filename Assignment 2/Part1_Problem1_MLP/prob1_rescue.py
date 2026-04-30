"""
Assignment 2 - Problem 1.

This script compares PCA, DCT, and AutoEncoder feature extraction methods on a
reduced MNIST dataset, then writes classifier accuracy and timing results to a
text report.
"""

import os
import time
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, pairwise_distances

# ==============================================================================
# 0. PATHS AND DIRECTORIES
# ==============================================================================
# BASE_DIR is the folder containing this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RESULTS_DIR is where the generated report file will be saved.
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create the results folder if it does not already exist.
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
def load_reduced_mnist():
    """
    Load MNIST, normalize the images, and reduce the dataset size.

    The reduced training set contains 1000 images per digit.
    The reduced test set contains 200 images per digit.
    """
    print("[1] Loading and reducing MNIST dataset...")

    # x_train/x_test contain 28x28 digit images; y_train/y_test contain labels 0-9.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert each image to a 784-value vector and scale pixel values to [0, 1].
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    # X_train_red stores selected training images; Y_train_red stores their labels.
    X_train_red, Y_train_red = [], []

    # Select the first 1000 samples for each digit class from the training set.
    for i in range(10):
        # idx contains the row positions of the selected samples for digit i.
        idx = np.where(y_train == i)[0][:1000]
        X_train_red.append(x_train[idx])
        Y_train_red.append(y_train[idx])

    # X_test_red stores selected test images; Y_test_red stores their labels.
    X_test_red, Y_test_red = [], []

    # Select the first 200 samples for each digit class from the test set.
    for i in range(10):
        # idx contains the row positions of the selected samples for digit i.
        idx = np.where(y_test == i)[0][:200]
        X_test_red.append(x_test[idx])
        Y_test_red.append(y_test[idx])

    # Stack image batches and combine label batches into final NumPy arrays.
    return (np.vstack(X_train_red), np.concatenate(Y_train_red), 
            np.vstack(X_test_red), np.concatenate(Y_test_red))

# ==============================================================================
# 2. FEATURE EXTRACTION
# ==============================================================================
def get_pca_features(X_train, X_test, n_components=128):
    """
    Extract PCA features.

    X_train/X_test are normalized image vectors.
    n_components is the target number of PCA features.
    """
    print("[2] Extracting PCA Features...")

    # pca learns the strongest variance directions from the training data.
    pca = PCA(n_components=n_components, random_state=42)

    # Fit PCA on training data, then transform both train and test data consistently.
    return pca.fit_transform(X_train), pca.transform(X_test)

def get_dct_features(X_train, X_test, n_components=128):
    """
    Extract DCT features.

    X_train/X_test are normalized image vectors.
    n_components is the number of low-frequency coefficients kept.
    """
    print("[3] Extracting DCT Features...")

    # X_train_dct keeps the first DCT coefficients for each training image.
    X_train_dct = dct(X_train, axis=1, norm='ortho')[:, :n_components]

    # X_test_dct applies the same DCT feature selection to each test image.
    X_test_dct = dct(X_test, axis=1, norm='ortho')[:, :n_components]
    return X_train_dct, X_test_dct

def get_ae_features(X_train, X_test, encoding_dim=128):
    """
    Train a simple autoencoder and return encoded image features.

    X_train/X_test are normalized image vectors.
    encoding_dim is the compressed feature size produced by the encoder.
    """
    print("[4] Training AutoEncoder & Extracting Features...")

    # input_img defines the 784-value input layer for flattened MNIST images.
    input_img = layers.Input(shape=(784,))

    # encoded is the compact hidden representation used as extracted features.
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

    # decoded reconstructs the original 784-value image from the encoded features.
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # autoencoder maps image input to image reconstruction for training.
    autoencoder = models.Model(input_img, decoded)

    # encoder maps image input to only the compressed feature representation.
    encoder = models.Model(input_img, encoded)

    # Train the autoencoder to reconstruct its input images.
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, verbose=0)

    # Use the trained encoder to compress both training and test images.
    return encoder.predict(X_train, verbose=0), encoder.predict(X_test, verbose=0)

# ==============================================================================
# 3. CLASSIFIERS
# ==============================================================================
def evaluate_mlp(X_train, Y_train, X_test, Y_test, feature_name, num_hidden_layers):
    """ 
    Train and evaluate an MLP classifier.

    X_train/X_test are feature matrices.
    Y_train/Y_test are class labels.
    feature_name identifies the feature extraction method in console output.
    num_hidden_layers controls how many 128-neuron hidden layers are added.
    """
    print(f"--> Training MLP ({num_hidden_layers} Layers) on {feature_name}...")

    # start_time records when training and prediction begin.
    start_time = time.time()

    # model is the sequential MLP classifier.
    model = models.Sequential()

    # The input layer size matches the number of extracted features.
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Add the requested number of hidden layers with ReLU activation.
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(128, activation='relu'))

    # The output layer predicts probabilities for the 10 digit classes.
    model.add(layers.Dense(10, activation='softmax'))

    # Compile and train the classifier on the selected feature set.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=15, batch_size=32, verbose=0)

    # predictions stores the most likely digit class for each test sample.
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # end_time records when training and prediction finish.
    end_time = time.time()

    # acc is the classification accuracy as a percentage.
    acc = accuracy_score(Y_test, predictions) * 100

    # t_msec is elapsed training plus prediction time in milliseconds.
    t_msec = (end_time - start_time) * 1000
    return acc, t_msec

class KMeansPerClass:
    """Apply K-Means separately to each digit class and classify by nearest centroid."""
    def __init__(self, k):
        # k is the number of centroids learned for each digit class.
        self.k = k

        # centroids stores all class-specific cluster centers; labels maps them to digits.
        self.centroids, self.labels = [], []

    def fit(self, X, y):
        """Fit one K-Means model per class using feature matrix X and labels y."""
        for cls in np.unique(y):
            # kmeans clusters only the training samples that belong to class cls.
            kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=42).fit(X[y == cls])
            self.centroids.append(kmeans.cluster_centers_)
            self.labels.extend([cls] * self.k)

        # Convert centroid and label lists into arrays for vectorized prediction.
        self.centroids = np.vstack(self.centroids)
        self.labels = np.array(self.labels)

    def predict(self, X):
        """Assign each sample in X to the label of its nearest centroid."""
        # dists contains distances from each sample to every stored centroid.
        dists = pairwise_distances(X, self.centroids)
        return self.labels[np.argmin(dists, axis=1)]

def evaluate_classical(model, X_train, Y_train, X_test, Y_test):
    """
    Train and evaluate a classical classifier such as SVM or KMeansPerClass.

    model must provide fit and predict methods.
    X_train/X_test are feature matrices; Y_train/Y_test are class labels.
    """
    # start_time records when model fitting and prediction begin.
    start_time = time.time()

    # Fit the classifier and predict labels for the test data.
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    # end_time records when fitting and prediction finish.
    end_time = time.time()

    # Return accuracy percentage and elapsed time in milliseconds.
    return accuracy_score(Y_test, predictions) * 100, (end_time - start_time) * 1000

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Load the reduced MNIST dataset.
    X_train, Y_train, X_test, Y_test = load_reduced_mnist()

    # Extract 128-feature versions of the data using PCA, DCT, and AutoEncoder.
    X_train_pca, X_test_pca = get_pca_features(X_train, X_test)
    X_train_dct, X_test_dct = get_dct_features(X_train, X_test)
    X_train_ae, X_test_ae = get_ae_features(X_train, X_test)

    # report_path is the output text file inside the results directory.
    report_path = os.path.join(RESULTS_DIR, 'prob1_features_report.txt')

    # f is the writable report file handle used inside this block.
    with open(report_path, 'w') as f:
        f.write("=== PROB 1 FINAL RESULTS FOR REPORT ===\n\n")

        # First report section: MLP results for different hidden-layer counts.
        f.write("--- 1. MLP Classifier Results (Varying Hidden Layers) ---\n")

        # num_layers is the hidden-layer count tested in the current MLP run.
        for num_layers in [1, 3, 4]:
            f.write(f"\n--- MLP with {num_layers} Hidden Layers ---\n")

            # acc is accuracy percentage; t is elapsed time in milliseconds.
            # Evaluate the MLP on PCA features.
            acc, t = evaluate_mlp(X_train_pca, Y_train, X_test_pca, Y_test, "PCA", num_layers)
            f.write(f"PCA         -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")

            # Evaluate the MLP on DCT features.
            acc, t = evaluate_mlp(X_train_dct, Y_train, X_test_dct, Y_test, "DCT", num_layers)
            f.write(f"DCT         -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")

            # Evaluate the MLP on AutoEncoder features.
            acc, t = evaluate_mlp(X_train_ae, Y_train, X_test_ae, Y_test, "AutoEncoder", num_layers)
            f.write(f"AutoEncoder -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
            f.write("-" * 40 + "\n")

        # Second report section: classical models trained on AutoEncoder features.
        f.write("\n--- 2. Classical ML on AutoEncoder Features ---\n")

        # Train and evaluate class-wise K-Means for several centroid counts.
        # k is the number of centroids learned for each digit class.
        for k in [1, 4, 16, 32]:
            print(f"--> Training K-Means (K={k}) on AutoEncoder...")
            acc, t = evaluate_classical(KMeansPerClass(k), X_train_ae, Y_train, X_test_ae, Y_test)
            f.write(f"K-Means (K={k}) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")

        # Train and evaluate SVM classifiers with linear and RBF kernels.
        print("--> Training SVMs on AutoEncoder...")
        acc, t = evaluate_classical(SVC(kernel='linear'), X_train_ae, Y_train, X_test_ae, Y_test)
        f.write(f"SVM (Linear) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
        
        acc, t = evaluate_classical(SVC(kernel='rbf', gamma='scale'), X_train_ae, Y_train, X_test_ae, Y_test)
        f.write(f"SVM (RBF) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")

    print(f"\n[SUCCESS] Fixed and done! Open '{report_path}' inside the 'results' folder.")
