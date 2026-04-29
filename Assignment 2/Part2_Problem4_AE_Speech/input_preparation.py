import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTS & PATHS
# ==========================================
# The absolute path where all outputs (arrays, images, graphs) will be stored.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure the results directory exists. If it doesn't, Python will create it.
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_label_from_filename(filename):
    """
    Extracts the digit label from the audio filename.
    Input: filename (string) like 'C02_0.wav' or 'U10n_9.wav'
    Output: integer representing the digit (e.g., 0 or 9)
    Method: Takes the last character before '.wav' based on the dataset structure.
    """
    # Remove '.wav' and get the very last character, then convert to integer
    return int(filename.replace('.wav', '')[-1])

def load_and_preprocess_data(dataset_path, max_frames=100):
    """
    Loads .wav files, computes Spectrogram frames (15ms each), and pads/truncates them.
    
    Arguments:
        dataset_path: (string) Path to 'Train' or 'Test' folder.
        max_frames: (int) Fixed number of frames to unify utterance length. 
                    100 frames * 15ms = 1.5 seconds, which is enough for a single digit.
    
    Returns:
        X_normalized: (numpy array) 2D array of shape (Number_of_samples, Flattened_Features)
        Y_labels: (numpy array) 1D array of labels corresponding to each sample
    """
    X_data = []
    Y_labels = []
    
    # Check if directory exists to avoid errors
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Directory not found: {dataset_path}")
    
    # Flag to ensure we only save one visual spectrogram image (to avoid 1500 images)
    saved_visual = False


    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_path, file)
            
            # 1. Load Audio
            # sr=22050 (default Librosa sampling rate, meaning 22050 samples per second)
            y, sr = librosa.load(file_path, sr=22050)
            
            # 2. Calculate hop_length for 15 ms frames
            # 15 milliseconds = 0.015 seconds. 0.015 * 22050 = 330.75 samples.
            hop_length = int(sr * 0.015) # Roughly 330 samples per frame step
            
            # 3. Calculate STFT (Short-Time Fourier Transform)
            # n_fft=512 is a standard window size.
            # Number of frequency bins generated = (n_fft / 2) + 1 = (512 / 2) + 1 = 257 bins.
            stft = librosa.stft(y, n_fft=512, hop_length=hop_length)
            
            # Convert amplitude to Decibels (dB) for better scaling
            spectrogram = librosa.amplitude_to_db(np.abs(stft))

            # Visualizing the Spectrogram ---
            # Save the very first audio file as an image so you can visually inspect it
            if not saved_visual:
                plt.figure(figsize=(10, 4))
                # librosa.display.specshow formats the axes correctly for audio data
                librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram Visualization for file: {file}')
                
                # Save the image to your specific results folder
                img_path = os.path.join(RESULTS_DIR, 'Sample_Spectrogram_Visual.png')
                plt.savefig(img_path)
                plt.close() # Close plot to free memory
                print(f"[*] Visual spectrogram saved to: {img_path}")
                saved_visual = True # Prevent saving images for the remaining 1499 files
            # --------------------------------------------------
            
            # Transpose to get frames as rows: Shape becomes (Time_Frames, 257)
            frames = spectrogram.T 
            
            # 4. Zero Padding / Truncating
            current_frames = frames.shape[0]
            if current_frames < max_frames:
                # Add zeros to the end if the audio is too short
                pad_width = max_frames - current_frames
                # Pad only the rows (time axis), not the columns (frequency axis)
                frames = np.pad(frames, pad_width=((0, pad_width), (0, 0)), mode='constant')
            else:
                # Cut the audio if it's longer than max_frames
                frames = frames[:max_frames, :]
            
            # 5. Flatten the 2D Spectrogram into a 1D Vector for the Autoencoder
            # Size becomes: 100 frames * 257 frequency bins = 25700 features
            X_data.append(frames.flatten())
            
            # Extract and store the label
            Y_labels.append(get_label_from_filename(file))
            
    # Convert lists to NumPy arrays for machine learning processing
    X_array = np.array(X_data)
    Y_labels = np.array(Y_labels)
    
    # 6. Normalize Data to [0, 1] range
    # This is crucial because the Autoencoder's output layer uses a 'sigmoid' activation (0 to 1)
    X_min = X_array.min()
    X_max = X_array.max()
    
    # Avoid division by zero in case of an empty or silent array
    if X_max - X_min == 0:
        X_normalized = X_array
    else:
        X_normalized = (X_array - X_min) / (X_max - X_min)
    
    return X_normalized, Y_labels

# --- Standalone Testing Block ---
# This block runs only if you execute this file directly, useful for debugging
if __name__ == "__main__":
    print("--- Preparing and Saving Data for Autoencoder ---")
    train_folder = os.path.join(BASE_DIR, "audio-dataset", "Train")
    test_folder = os.path.join(BASE_DIR, "audio-dataset", "Test")
    
    try:
        # 1. Process and extract flattened spectrograms for Training set
        # max_frames=100 ensures fixed dimension of 25700 features (100 * 257)
        x_train_ae, y_train_ae = load_and_preprocess_data(train_folder, max_frames=100)
        
        # 2. Process and extract for Testing set
        x_test_ae, y_test_ae = load_and_preprocess_data(test_folder, max_frames=100)
        
        # 3. Saving to disk using NumPy binary format
        # These files contain the full normalized data ready for the Autoencoder input layer
        np.save(os.path.join(RESULTS_DIR, 'X_train_ae.npy'), x_train_ae) # Feature vector for AE training
        np.save(os.path.join(RESULTS_DIR, 'Y_train_ae.npy'), y_train_ae) # Labels for AE evaluation
        np.save(os.path.join(RESULTS_DIR, 'X_test_ae.npy'), x_test_ae)   # Feature vector for AE testing
        np.save(os.path.join(RESULTS_DIR, 'Y_test_ae.npy'), y_test_ae)   # Labels for AE testing
        
        print("\n[SUCCESS] Autoencoder input files saved successfully!")
        print(f"AE Train Shape: {x_train_ae.shape}, AE Test Shape: {x_test_ae.shape}")
        
    except Exception as e:
        print(f"Error while saving Autoencoder data: {e}")



## =========================================
# we will use the same get_label_from_filename function from the autoencoder file to maintain consistency in label extraction.
def get_baseline_vectors(dataset_path):
    """
    Calculates the Baseline representation of audio files by averaging their Spectrogram frames.
    Instead of using an Autoencoder to compress the data, this function uses a simple mathematical average.
    
    Arguments:
        dataset_path: (string) Path to the directory containing the .wav files (e.g., 'audio-dataset/Train').
        
    Returns:
        X_baseline: (numpy array) 2D array of shape (Number_of_samples, 257). 
                    Each row is a single vector representing the average frequency bins of an entire utterance.
        Y_labels: (numpy array) 1D array of labels corresponding to each audio file.
    """
    X_data = []
    Y_labels = []
    
    # Iterate through all audio files in the specified directory
    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_path, file)
            
            # 1. Load Audio
            # sr=22050 (Standard sample rate)
            y, sr = librosa.load(file_path, sr=22050)
            
            # 2. Calculate hop_length for 15 ms frames
            hop_length = int(sr * 0.015) 
            
            # 3. Calculate STFT (Short-Time Fourier Transform)
            # This generates a matrix of complex numbers representing frequencies over time
            stft = librosa.stft(y, n_fft=512, hop_length=hop_length)
            
            # Convert amplitude to Decibels (dB)
            spectrogram = librosa.amplitude_to_db(np.abs(stft))
            
            # Transpose to get frames as rows: Shape becomes (Time_Frames, Frequency_Bins)
            # Note: Frequency_Bins will be 257 based on n_fft=512
            frames = spectrogram.T 
            
            # 4. Calculate the Baseline (The Core Difference)
            # We average all the frames across the time axis (axis=0).
            # This crushes the (Time_Frames, 257) matrix into a single (257,) vector.
            average_frame = np.mean(frames, axis=0)
            
            # Append the calculated vector and its corresponding label
            X_data.append(average_frame)
            Y_labels.append(get_label_from_filename(file))
            
    # Convert lists to NumPy arrays
    X_baseline = np.array(X_data)
    Y_labels = np.array(Y_labels)
    
    # 5. Normalize Data to [0, 1] range for consistency and better classifier performance later
    X_min = X_baseline.min()
    X_max = X_baseline.max()
    
    if X_max - X_min == 0:
        X_normalized = X_baseline
    else:
        X_normalized = (X_baseline - X_min) / (X_max - X_min)
    
    return X_normalized, Y_labels