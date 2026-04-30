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
# 0. PATHS & DIRECTORIES (تحديد مسار الحفظ)
# ==============================================================================
# هنا بنحدد المسار الحالي للملف، وبنجبر الكود إنه يكريت فولدر results لو مش موجود
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True) # التأكد من وجود الفولدر لمنع أي إيرور

# ==============================================================================
# 1. DATA PREPARATION (تجهيز الداتا المصغرة)
# ==============================================================================
def load_reduced_mnist():
    """
    هذه الدالة تقوم بتحميل داتا MNIST الأصلية من مكتبة كيراس، 
    ثم تقوم بتقليل حجمها لتطابق متطلبات الـ Assignment:
    - 1000 صورة لكل رقم في التدريب.
    - 200 صورة لكل رقم في الاختبار.
    """
    print("[1] Loading and reducing MNIST dataset...")
    # تحميل الداتا الأصلية
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # فرد الصور من 28x28 إلى خط مستقيم 784 بيكسل، وعمل Normalization بالقسمة على 255
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    X_train_red, Y_train_red = [], []
    # تجميع 1000 صورة لكل كلاس (من 0 لـ 9) للتدريب
    for i in range(10):
        idx = np.where(y_train == i)[0][:1000]
        X_train_red.append(x_train[idx])
        Y_train_red.append(y_train[idx])
        
    X_test_red, Y_test_red = [], []
    # تجميع 200 صورة لكل كلاس (من 0 لـ 9) للاختبار
    for i in range(10):
        idx = np.where(y_test == i)[0][:200]
        X_test_red.append(x_test[idx])
        Y_test_red.append(y_test[idx])

    # تحويل القوائم إلى مصفوفات NumPy جاهزة للاستخدام
    return (np.vstack(X_train_red), np.concatenate(Y_train_red), 
            np.vstack(X_test_red), np.concatenate(Y_test_red))

# ==============================================================================
# 2. FEATURE EXTRACTION (استخراج الملامح بـ 3 طرق مختلفة)
# ==============================================================================
def get_pca_features(X_train, X_test, n_components=128):
    """ استخراج الملامح باستخدام PCA لتقليل الأبعاد إلى 128 """
    print("[2] Extracting PCA Features...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_train), pca.transform(X_test)

def get_dct_features(X_train, X_test, n_components=128):
    """ استخراج الملامح باستخدام DCT والاحتفاظ بأهم 128 تردد """
    print("[3] Extracting DCT Features...")
    X_train_dct = dct(X_train, axis=1, norm='ortho')[:, :n_components]
    X_test_dct = dct(X_test, axis=1, norm='ortho')[:, :n_components]
    return X_train_dct, X_test_dct

def get_ae_features(X_train, X_test, encoding_dim=128):
    """ بناء وتدريب AutoEncoder مبسط لضغط الصور إلى 128 ملمح """
    print("[4] Training AutoEncoder & Extracting Features...")
    # تعريف هيكل الـ AutoEncoder
    input_img = layers.Input(shape=(784,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = layers.Dense(784, activation='sigmoid')(encoded)
    
    autoencoder = models.Model(input_img, decoded)
    encoder = models.Model(input_img, encoded) # هذا الجزء هو ما نحتاجه لاستخراج الملامح فقط
    
    # تدريب سريع للـ AutoEncoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, verbose=0)
    
    # استخدام الـ Encoder لضغط صور التدريب والاختبار
    return encoder.predict(X_train, verbose=0), encoder.predict(X_test, verbose=0)

# ==============================================================================
# 3. CLASSIFIERS (موديلات التصنيف: MLP و SVM و KMeans)
# ==============================================================================
def evaluate_mlp(X_train, Y_train, X_test, Y_test, feature_name, num_hidden_layers):
    """ 
    دالة تقوم ببناء شبكة MLP ديناميكية تتغير طبقاتها المخفية بناءً على المدخل (1، 3، أو 4)
    وتحسب وقت المعالجة والدقة المطلوبة للجدول.
    """
    print(f"--> Training MLP ({num_hidden_layers} Layers) on {feature_name}...")
    start_time = time.time() # بدء تشغيل ساعة الإيقاف
    
    # بناء هيكل الموديل
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # إضافة عدد الطبقات المخفية المطلوب ديناميكياً
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(128, activation='relu'))
        
    model.add(layers.Dense(10, activation='softmax')) # طبقة الخرج
    
    # التدريب والاختبار
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=15, batch_size=32, verbose=0)
    
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    end_time = time.time() # إيقاف الساعة
    
    # حساب المقاييس
    acc = accuracy_score(Y_test, predictions) * 100
    t_msec = (end_time - start_time) * 1000 # تحويل الوقت لمللي ثانية
    return acc, t_msec

class KMeansPerClass:
    """ كلاس مخصص لتطبيق K-Means على كل رقم (كلاس) بشكل منفصل كما في Assignment 1 """
    def __init__(self, k):
        self.k = k
        self.centroids, self.labels = [], []
    def fit(self, X, y):
        for cls in np.unique(y):
            kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=42).fit(X[y == cls])
            self.centroids.append(kmeans.cluster_centers_)
            self.labels.extend([cls] * self.k)
        self.centroids = np.vstack(self.centroids)
        self.labels = np.array(self.labels)
    def predict(self, X):
        dists = pairwise_distances(X, self.centroids)
        return self.labels[np.argmin(dists, axis=1)]

def evaluate_classical(model, X_train, Y_train, X_test, Y_test):
    """ دالة موحدة لحساب دقة ووقت موديلات الكلاسيك (SVM و KMeans) """
    start_time = time.time()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    end_time = time.time()
    return accuracy_score(Y_test, predictions) * 100, (end_time - start_time) * 1000

# ==============================================================================
# 4. MAIN EXECUTION (نقطة انطلاق الكود وتوليد التقرير)
# ==============================================================================
if __name__ == "__main__":
    # 1. تحميل الداتا
    X_train, Y_train, X_test, Y_test = load_reduced_mnist()
    
    # 2. استخراج جميع الملامح
    X_train_pca, X_test_pca = get_pca_features(X_train, X_test)
    X_train_dct, X_test_dct = get_dct_features(X_train, X_test)
    X_train_ae, X_test_ae = get_ae_features(X_train, X_test)

    # 3. تحديد مسار ملف التقرير داخل فولدر results حصراً
    report_path = os.path.join(RESULTS_DIR, 'prob1_features_report.txt')
    
    # 4. بدء الكتابة في ملف التقرير
    with open(report_path, 'w') as f:
        f.write("=== PROB 1 FINAL RESULTS FOR REPORT ===\n\n")
        
        # --- القسم الأول: تجارب الـ MLP ---
        f.write("--- 1. MLP Classifier Results (Varying Hidden Layers) ---\n")
        for num_layers in [1, 3, 4]:
            f.write(f"\n--- MLP with {num_layers} Hidden Layers ---\n")
            
            # تجربة الـ PCA
            acc, t = evaluate_mlp(X_train_pca, Y_train, X_test_pca, Y_test, "PCA", num_layers)
            f.write(f"PCA         -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
            
            # تجربة الـ DCT
            acc, t = evaluate_mlp(X_train_dct, Y_train, X_test_dct, Y_test, "DCT", num_layers)
            f.write(f"DCT         -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
            
            # تجربة الـ AutoEncoder
            acc, t = evaluate_mlp(X_train_ae, Y_train, X_test_ae, Y_test, "AutoEncoder", num_layers)
            f.write(f"AutoEncoder -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
            f.write("-" * 40 + "\n")

        # --- القسم الثاني: تجارب الـ Classical ML على الـ AutoEncoder ---
        f.write("\n--- 2. Classical ML on AutoEncoder Features ---\n")
        
        # تجارب الـ KMeans
        for k in [1, 4, 16, 32]:
            print(f"--> Training K-Means (K={k}) on AutoEncoder...")
            acc, t = evaluate_classical(KMeansPerClass(k), X_train_ae, Y_train, X_test_ae, Y_test)
            f.write(f"K-Means (K={k}) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
            
        # تجارب الـ SVM
        print("--> Training SVMs on AutoEncoder...")
        acc, t = evaluate_classical(SVC(kernel='linear'), X_train_ae, Y_train, X_test_ae, Y_test)
        f.write(f"SVM (Linear) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")
        
        acc, t = evaluate_classical(SVC(kernel='rbf', gamma='scale'), X_train_ae, Y_train, X_test_ae, Y_test)
        f.write(f"SVM (RBF) -> Accuracy: {acc:.1f}%, Time: {t:.1f} msec\n")

    print(f"\n[SUCCESS] Fixed and done! Open '{report_path}' inside the 'results' folder.")