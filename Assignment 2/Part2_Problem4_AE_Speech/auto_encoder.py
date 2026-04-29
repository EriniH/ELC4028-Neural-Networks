import os
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras import layers, models
import matplotlib.pyplot as plt

# Import our custom data loader from the first file
from input_preparation import load_and_preprocess_data


# ==========================================
# CONSTANTS & PATHS
# ==========================================
# The absolute path where all output files (.npy arrays, images, etc.) will be stored.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure the results directory exists. If it doesn't, Python will create it automatically.
# 'exist_ok=True' prevents the code from crashing if the folder is already there.
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import our custom data loader from the first file
from input_preparation import load_and_preprocess_data



# ==========================================
# 1. DATA LOADING3
# ==========================================
print("Loading Training Data...")
# Replace with your actual directory paths
train_path = os.path.join(BASE_DIR, "audio-dataset", "Train")
test_path = os.path.join(BASE_DIR, "audio-dataset", "Test")

# We assume max_frames=100 as defined in the preparation file
X_train, Y_train = load_and_preprocess_data(train_path, max_frames=100)
X_test, Y_test = load_and_preprocess_data(test_path, max_frames=100)

print(f"X_train shape: {X_train.shape}") # Should be (Num_Train_Samples, 25700)
print(f"X_test shape: {X_test.shape}")   # Should be (Num_Test_Samples, 25700)


# ==========================================
# 2. AUTOENCODER ARCHITECTURE DESIGN
# ==========================================
# Input Dimension: 100 frames * 257 bins = 25700
input_dim = X_train.shape[1]

# Encoding Dimension (The Bottleneck): 
# We want to compress 25700 features into a single vector of length 128
encoding_dim = 256

# --- Encoder Section ---
# Input layer takes the flattened spectrogram
input_audio = layers.Input(shape=(input_dim,), name="Encoder_Input")

# First compression layer (reduces dimensions from 25700 to 1024)
encoded_1 = layers.Dense(1024, activation='relu', name="Encoder_Hidden_1")(input_audio)

# The Bottleneck layer (reduces dimensions from 1024 to 128)
# This 'bottleneck' variable holds our target "Single Vector"
bottleneck = layers.Dense(encoding_dim, activation='relu', name="Bottleneck")(encoded_1)

# --- Decoder Section ---
# First decompression layer (expands dimensions from 128 back to 1024)
decoded_1 = layers.Dense(1024, activation='relu', name="Decoder_Hidden_1")(bottleneck)

# Output layer (expands from 1024 back to original 25700)
# We use 'sigmoid' because our input data was normalized between 0 and 1
output_reconstructed = layers.Dense(input_dim, activation='sigmoid', name="Decoder_Output")(decoded_1)

# ==========================================
# 3. MODEL COMPILATION
# ==========================================
# Combine Encoder and Decoder into one full model
autoencoder = models.Model(inputs=input_audio, outputs=output_reconstructed, name="Full_Autoencoder")

# Use Adam optimizer and Mean Squared Error (MSE) to compare input vs reconstructed output
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("\n--- Autoencoder Summary ---")
autoencoder.summary()

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("\nStarting Training...")
# Notice that input (X_train) and target (X_train) are the same!
# The model learns to recreate its own input.
history = autoencoder.fit(
    x=X_train, 
    y=X_train, 
    epochs=50,        # Number of times to loop through the data
    batch_size=32,      # Number of samples per gradient update
    shuffle=True,       # Shuffle data to prevent learning order patterns
    validation_data=(X_test, X_test) # Test reconstruction on unseen data
)


# ==========================================
# 5. VISUALIZING AND SAVING THE LOSS CURVE
# ==========================================
print("\nGenerating and saving the Training Loss Graph...")

# Create a plot showing how the Mean Squared Error (MSE) decreased over time
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='red')
plt.title('Autoencoder Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss / Error')
plt.legend()
plt.grid(True)

# Construct the full path to save the image inside the RESULTS_DIR
loss_curve_path = os.path.join(RESULTS_DIR, 'Autoencoder_Loss_Curve.png')

# Save the plot as an image file so you can open it easily and visually inspect learning progress
plt.savefig(loss_curve_path)
plt.close() # Close the plot to free up system memory
print(f"[*] Loss curve image saved successfully at: {loss_curve_path}")

# ==========================================
# 6. EXTRACTING THE ENCODER & TESTING
# ==========================================
# Now that the model is trained, we throw away the Decoder.
# We create a new model that starts at the input and ends at the Bottleneck.
encoder_only = models.Model(inputs=input_audio, outputs=bottleneck, name="Feature_Extractor")

print("\nExtracting feature vectors for the test set...")
# Pass the test audio through the Encoder to get the compressed vectors
# encoded_test_vectors will have the shape (Num_Test_Samples, 128)
encoded_test_vectors = encoder_only.predict(X_test)

# Also extract the training vectors so we can train our MLP classifier later
encoded_train_vectors = encoder_only.predict(X_train)

print(f"\nFinal Extracted Vectors Shape: {encoded_test_vectors.shape}")
print(f"Successfully compressed {input_dim} features into {encoding_dim} features per utterance!")

# (Optional Next Step): You can now take 'encoded_test_vectors' and 'Y_test', 
# and train a simple MLP or SVM Classifier to identify the spoken digits.

# ==========================================
# 7. SAVING THE COMPRESSED VECTORS TO DISK
# ==========================================
print("\nSaving the extracted Bottleneck features to the Results folder...")

# np.save() stores the extracted feature arrays as fast-loading binary .npy files
# We use os.path.join to ensure these arrays are saved directly inside the RESULTS_DIR
np.save(os.path.join(RESULTS_DIR, 'Bottleneck_Features_Train.npy'), encoded_train_vectors)
np.save(os.path.join(RESULTS_DIR, 'Bottleneck_Features_Test.npy'), encoded_test_vectors)

# We also need to save the Labels (Y_train and Y_test) to the results folder
# because the final MLP Classifier will need these labels to learn the spoken digits!
np.save(os.path.join(RESULTS_DIR, 'Y_train_ae.npy'), Y_train)
np.save(os.path.join(RESULTS_DIR, 'Y_test_ae.npy'), Y_test)

print(f"[SUCCESS] Bottleneck vectors and labels saved successfully in: {RESULTS_DIR}")

# (Optional Next Step): You can now take 'encoded_test_vectors' and 'Y_test', 
# and train a simple MLP or SVM Classifier to identify the spoken digits.