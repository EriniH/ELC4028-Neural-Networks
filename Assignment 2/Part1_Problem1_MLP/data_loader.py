import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# ==============================================================================
# DATA LOADER FOR PROBLEM 1 (REDUCED MNIST)
# ==============================================================================

def load_mnist_data(mnist_folder_path):
    """
    Safely loads .jpg images from 'train' and 'test' subdirectories, skipping 
    any corrupted or non-image files to prevent system crashes.
    
    Arguments:
        mnist_folder_path (str): Absolute path to the "Reduced MNIST Data" folder.
            
    Returns:
        x_train (numpy.ndarray): Array of normalized training images.
        y_train (numpy.ndarray): Array of integer labels for training.
        x_test (numpy.ndarray): Array of normalized testing images.
        y_test (numpy.ndarray): Array of integer labels for testing.
    """
    
    def load_image_split(split_name):
        """ Internal helper function to load either the 'train' or 'test' split. """
        x_data = []    # List to store image matrices
        y_labels = []  # List to store integer labels
        
        # Build the path (e.g., /.../Reduced MNIST Data/train)
        split_path = os.path.join(mnist_folder_path, split_name)
        
        # Check if the path actually exists
        if not os.path.exists(split_path):
            print(f"[Error] Directory not found: {split_path}")
            return np.array([]), np.array([])

        print(f"Scanning directory: {split_path} ...")
        
        # Iterate over the folders inside the split (these should be named '0', '1', ..., '9')
        for folder_name in sorted(os.listdir(split_path)):
            folder_path = os.path.join(split_path, folder_name)
            
            # Check if it is actually a directory
            if os.path.isdir(folder_path):
                # The folder name (e.g., '5') is the integer label for all images inside it
                current_label = int(folder_name)
                
                # Iterate over all files inside this specific digit folder
                for image_filename in os.listdir(folder_path):
                    
                    # Strictly allow only standard image extensions to avoid OSError
                    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, image_filename)
                        
                        # --- ROBUST ERROR HANDLING ---
                        # If a file is corrupted, the 'try' block will fail safely,
                        # the 'except' block will catch it, print a warning, and continue.
                        try:
                            # Load the image in grayscale (1 channel) and resize to 28x28
                            img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
                            
                            # Convert the image object to a NumPy matrix
                            img_array = img_to_array(img)
                            
                            # Add to our main lists
                            x_data.append(img_array)
                            y_labels.append(current_label)
                        except Exception as e:
                            print(f"[Warning] Skipped unreadable file {image_filename}: {e}")
                            continue # Move to the next image without crashing
        
        # Convert lists to NumPy arrays for Deep Learning compatibility
        x_data_np = np.array(x_data, dtype='float32')
        y_labels_np = np.array(y_labels, dtype='int32')
        
        # Min-Max Normalization: Scale pixel values from [0, 255] down to [0.0, 1.0]
        x_data_np = x_data_np / 255.0
        
        return x_data_np, y_labels_np

    # Execute the internal helper function for both training and testing datasets
    print("\n[Loader] Loading Training Images...")
    x_train, y_train = load_image_split('train')
    
    print("[Loader] Loading Testing Images...")
    x_test, y_test = load_image_split('test')
    
    print(f"\n[SUCCESS] Loaded {len(x_train)} Train samples and {len(x_test)} Test samples.")
    return x_train, y_train, x_test, y_test