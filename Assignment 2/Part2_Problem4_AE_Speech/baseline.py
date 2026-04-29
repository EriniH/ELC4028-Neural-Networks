import os
import librosa
import numpy as np

# ==========================================
# CONSTANTS & PATHS
# ==========================================
# The absolute path where all output files (.npy arrays, images, etc.) will be stored.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure the results directory exists. If it doesn't, Python will create it automatically.
# 'exist_ok=True' prevents the code from crashing if the folder is already there.
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# HELPER FUNCTION: EXTRACT LABEL
# ==========================================
def get_label_from_filename(filename):
    """
    Extracts the digit label from the audio filename.
    
    Arguments:
        filename: (string) The name of the audio file. 
                  Example: 'C03_0.wav' or 'F11n_9.wav'.
                  
    Returns:
        label: (int) The integer representation of the spoken digit (e.g., 0 or 9).
        
    Description:
        Based on the dataset structure, the actual spoken digit is always 
        the last character right before the '.wav' extension.
    """
    # Remove the '.wav' extension, grab the last character, and convert it to an integer
    label_string = filename.replace('.wav', '')[-1]
    label = int(label_string)
    return label


# ==========================================
# MAIN FUNCTION: CALCULATE BASELINE VECTORS
# ==========================================
def extract_baseline_features(dataset_path):
    """
    Calculates the Baseline representation of audio files by averaging their Spectrogram frames.
    Instead of using a complex Autoencoder to compress the data, this function uses a 
    simple mathematical average across the time axis.
    
    Arguments:
        dataset_path: (string) Path to the directory containing the .wav files 
                      (e.g., 'audio-dataset/Train' or 'audio-dataset/Test').
                      
    Returns:
        X_normalized: (numpy.ndarray) A 2D array of shape (Number_of_samples, 257). 
                      Each row is a single vector representing the average frequency bins 
                      of an entire audio utterance.
        Y_labels:     (numpy.ndarray) A 1D array containing the integer labels corresponding 
                      to each audio file.
    """
    
    # Initialize empty lists to store the computed vectors and their labels
    X_data = []
    Y_labels = []
    
    print(f"Extracting baseline features from: {dataset_path}...")
    
    # Iterate through all files in the specified directory
    for file in os.listdir(dataset_path):
        
        # Process only audio files ending with '.wav'
        if file.endswith('.wav'):
            
            # Construct the full path to the audio file
            file_path = os.path.join(dataset_path, file)
            
            # 1. Load Audio
            # y: (numpy array) The audio time series (waveform)
            # sr: (int) The sampling rate, fixed at 22050 Hz for consistency
            y, sr = librosa.load(file_path, sr=22050)
            
            # 2. Frame Step Calculation
            # Calculate the number of samples that make up 15 milliseconds (0.015 seconds)
            # hop_length: (int) The number of audio samples between consecutive frames
            hop_length = int(sr * 0.015) 
            
            # 3. Calculate STFT (Short-Time Fourier Transform)
            # stft_matrix: (complex numpy array) Represents frequencies over time
            # n_fft=512 generates 257 frequency bins (calculated as: n_fft/2 + 1)
            stft_matrix = librosa.stft(y, n_fft=512, hop_length=hop_length)
            
            # Convert the amplitude (absolute value of STFT) to Decibels (dB)
            # spectrogram: (numpy array) A 2D representation of audio (Frequencies vs. Time)
            spectrogram = librosa.amplitude_to_db(np.abs(stft_matrix))
            
            # Transpose the matrix so that rows represent Time Frames and columns represent Frequency Bins
            # frames shape: (Number_of_Time_Frames, 257)
            frames = spectrogram.T 
            
            # 4. Calculate the Baseline (The Core Logic)
            # Average all the frames across the time axis (axis=0).
            # This mathematically compresses the (Time_Frames, 257) matrix into a single (257,) vector.
            # average_frame: (numpy array) A 1D vector of length 257
            average_frame = np.mean(frames, axis=0)
            
            # Add the computed vector and its true label to our lists
            X_data.append(average_frame)
            Y_labels.append(get_label_from_filename(file))
            
    # Convert Python lists to NumPy arrays for faster mathematical operations
    X_baseline = np.array(X_data)
    Y_labels = np.array(Y_labels)
    
    # 5. Data Normalization (Min-Max Scaling)
    # Neural Networks perform better when input features are scaled between 0 and 1
    X_min = X_baseline.min()
    X_max = X_baseline.max()
    
    # Prevent Division by Zero in case all values are exactly the same
    if X_max - X_min == 0:
        X_normalized = X_baseline
    else:
        X_normalized = (X_baseline - X_min) / (X_max - X_min)
        
    print(f"Successfully processed {len(X_normalized)} files.")
    
    return X_normalized, Y_labels


# ==========================================
# EXECUTION BLOCK (FOR TESTING)
# ==========================================
# This block only runs if you execute this specific file directly from the terminal.
# It is very useful to test the script independently before integrating it.
if __name__ == "__main__":
    # Define paths for Training and Testing sets
    train_folder = os.path.join(BASE_DIR, "audio-dataset", "Train")
    test_folder = os.path.join(BASE_DIR, "audio-dataset", "Test")
    
    try:
        # 1. Processing Training Data
        print("--- Processing Training Baseline ---")
        x_base_train, y_base_train = extract_baseline_features(train_folder)
        
        # 2. Processing Testing Data
        print("--- Processing Testing Baseline ---")
        x_base_test, y_base_test = extract_baseline_features(test_folder)

        # 3. Saving the extracted features to the disk
        # np.save(file_name, array) stores the data as binary .npy files
        # These files are extremely fast to load later using np.load()
        np.save(os.path.join(RESULTS_DIR, 'X_baseline_train.npy'), x_base_train) # Saves the feature matrix (Samples, 257)
        np.save(os.path.join(RESULTS_DIR, 'Y_baseline_train.npy'), y_base_train) # Saves the corresponding labels
        np.save(os.path.join(RESULTS_DIR, 'X_baseline_test.npy'), x_base_test)   # Saves the test features
        np.save(os.path.join(RESULTS_DIR, 'Y_baseline_test.npy'), y_base_test)   # Saves the test labels

        print("\n[SUCCESS] Baseline files saved successfully!")
        print(f"Train Shape: {x_base_train.shape}, Test Shape: {x_base_test.shape}")
        
    except Exception as e:
        print(f"An error occurred during saving: {e}")