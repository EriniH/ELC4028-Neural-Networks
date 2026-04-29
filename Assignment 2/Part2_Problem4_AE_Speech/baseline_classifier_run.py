import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 1. Load Baseline Features (Average Frame)
x_train = np.load(os.path.join(RESULTS_DIR, 'X_baseline_train.npy'))
x_test  = np.load(os.path.join(RESULTS_DIR, 'X_baseline_test.npy'))
y_train = np.load(os.path.join(RESULTS_DIR, 'Y_baseline_train.npy'))
y_test  = np.load(os.path.join(RESULTS_DIR, 'Y_baseline_test.npy'))

print("Loaded Baseline Dataset shapes:", x_train.shape, x_test.shape)

# 2. Build Classifier Model on Baseline Features
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train Model
print("\n--- Training on Baseline (Average Frame) ---\n")
model.fit(x_train, y_train, epochs=40, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# 4. Evaluate Test Accuracy
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[FINAL RESULT] Baseline Classifier Test Accuracy: {acc*100:.2f}%")