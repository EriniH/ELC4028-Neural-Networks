import os
import cv2
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

# Set random seeds to ensure we get exactly the same 350 images every time we run the code.
# This is crucial for reproducibility, ensuring our baseline never changes between runs.
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class GANDataLoader:
    """
    The main class responsible for handling the Reduced MNIST dataset for the GAN problem.
    It performs data loading, sampling, modern Data Augmentation, and saves reference
    Baselines, Arrays, and visual samples in highly organized directories.
    """
    
    def __init__(self):
        """
        The constructor of the class.
        Initializes the fundamental absolute paths for inputs and outputs tailored for Prob2_GAN.
        
        Variables:
            self.dataset_path (str): The absolute path to the original dataset.
            self.out_base (str): The specific output directory for Prob2_GAN.
            self.baseline_350_dir (str): Path to save the 350 reference images per digit.
            self.baseline_1000_dir (str): Path to save the 1000 reference images per digit.
            self.aug_samples_dir (str): Path to save augmentation visual samples for the report.
            self.arrays_dir (str): Path to save the final Numpy Arrays (.npy) for later use.
        """
        # Absolute path for the original dataset
        self.dataset_path = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Reduced MNIST Data/train"
        
        # Specific absolute path for Prob2_GAN Outputs
        self.out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob2_GAN/Outputs"
        
        # Define output sub-directories
        self.baseline_350_dir = os.path.join(self.out_base, "0_Baselines", "350_real")
        self.baseline_1000_dir = os.path.join(self.out_base, "0_Baselines", "1000_real")
        self.aug_samples_dir = os.path.join(self.out_base, "1_Augmented_Images")
        self.arrays_dir = os.path.join(self.out_base, "2_Data_Arrays")
        
        # Trigger directory creation immediately upon initialization
        self._create_directories()

    def _create_directories(self):
        """
        Creates the necessary directory structure for the outputs.
        
        Function Logic:
            Uses os.makedirs with exist_ok=True. This ensures that if the folder 
            already exists from a previous run, it will be skipped without throwing an error.
            If it does not exist, it will be created.
            
        Returns:
            None (Modifies the File System only).
        """
        dirs_to_make = [self.baseline_350_dir, self.baseline_1000_dir, self.aug_samples_dir, self.arrays_dir]
        for d in dirs_to_make:
            os.makedirs(d, exist_ok=True)
            # Create subfolders for digits 0-9 inside the baseline directories
            for i in range(10):
                os.makedirs(os.path.join(self.baseline_350_dir, str(i)), exist_ok=True)
                os.makedirs(os.path.join(self.baseline_1000_dir, str(i)), exist_ok=True)

    def _add_light_noise(self, image_array):
        """
        A helper function to add light Gaussian Noise to an image array.
        
        Args:
            image_array (numpy.ndarray): The input image array. Expected shape is (N, 28, 28, 1) or (28, 28, 1).
                                         Pixel values must be normalized between 0.0 and 1.0.
            
        Logic & Math:
            We generate a noise matrix of the exact same shape as the input 
            using a Normal Distribution with a Mean of 0.0 and a Standard Deviation of 0.05.
            We add this noise to the original array. Finally, np.clip is used to guarantee 
            that any pixel exceeding 1.0 or dropping below 0.0 is clamped to safe boundaries.
            
        Returns:
            noisy_image (numpy.ndarray): The corrupted image, preserving the original shape.
        """
        # loc is mean, scale is standard deviation. 0.05 is chosen for "light" noise.
        noise = np.random.normal(loc=0.0, scale=0.05, size=image_array.shape)
        noisy_image = image_array + noise
        
        # Clip ensures pixels don't go out of the [0.0, 1.0] valid range
        return np.clip(noisy_image, 0.0, 1.0)

    def prepare_baselines_and_load(self):
        """
        The core engine to read data, split baselines, load the 350-set, and save its arrays.
        
        Function Logic:
            1. Loops through each digit folder from 0 to 9.
            2. Reads all image filenames and performs a seeded random shuffle.
            3. Selects the first 1000 images as the 1000-Baseline, and the first 350 
               of those 1000 as the 350-Baseline.
            4. Copies the images to the new directories (shutil.copy overwrites automatically).
            5. Reads the 350 images as Grayscale, normalizes them, and adjusts their shapes.
            6. Saves the final Numpy arrays to the disk.
            
        Returns:
            x_train_350 (numpy.ndarray): The 350 images for each digit (Total 3500 images). 
                                         Final shape is (3500, 28, 28, 1).
            y_train_350 (numpy.ndarray): The corresponding labels. Final shape is (3500,).
        """
        print(">>> Starting to sample and copy baselines for GAN...")
        x_train_350, y_train_350 = [], []

        for digit in range(10):
            digit_path = os.path.join(self.dataset_path, str(digit))
            images = [f for f in os.listdir(digit_path) if f.endswith(('.jpg', '.png'))]
            
            # Shuffle the list to pick a random subset
            random.shuffle(images)
            
            # Select baselines
            sampled_1000 = images[:1000]
            sampled_350 = sampled_1000[:350] # Subset ensures 350 is part of the 1000

            for img_name in sampled_1000:
                src_path = os.path.join(digit_path, img_name)
                
                # 1. Save to 1000_real baseline folder (will overwrite if exists)
                dest_1000 = os.path.join(self.baseline_1000_dir, str(digit), img_name)
                shutil.copy(src_path, dest_1000)
                
                # 2. Save to 350_real baseline folder and load into RAM
                if img_name in sampled_350:
                    dest_350 = os.path.join(self.baseline_350_dir, str(digit), img_name)
                    shutil.copy(src_path, dest_350)
                    
                    # Read image as Grayscale. Current shape: (28, 28)
                    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Normalize pixels from [0, 255] to [0.0, 1.0] for Neural Network constraints
                    img = img.astype(np.float32) / 255.0
                    
                    # Expand dimensions to add the channel axis required by CNNs. Shape becomes: (28, 28, 1)
                    img = np.expand_dims(img, axis=-1)
                    
                    x_train_350.append(img)
                    y_train_350.append(digit)

        # Convert Python lists to NumPy arrays for Deep Learning compatibility
        x_train_350 = np.array(x_train_350)
        y_train_350 = np.array(y_train_350)
        
        # Save the 350 baseline arrays to disk
        np.save(os.path.join(self.arrays_dir, "x_train_350.npy"), x_train_350)
        np.save(os.path.join(self.arrays_dir, "y_train_350.npy"), y_train_350)
        
        print(f">>> Baselines prepared and saved to disk! Training data shape: {x_train_350.shape}")
        return x_train_350, y_train_350

    def generate_augmented_data(self, x_train_350, y_train_350, multiplier=15):
        """
        Uses Modern Keras Augmentation Layers (TF 2.16+) to create synthetic modifications,
        and saves the resulting array to disk.
        
        Args:
            x_train_350 (numpy.ndarray): The input original images (3500, 28, 28, 1).
            y_train_350 (numpy.ndarray): The original labels (3500,).
            multiplier (int): The factor by which to multiply the dataset. Default 15 
                              will generate 15 * 3500 = 52500 new images.
            
        Logic & Iterations:
            Builds a tf.keras.Sequential model containing RandomRotation, RandomTranslation, 
            and RandomZoom. Iterates 'multiplier' times. In each iteration, it passes the 
            entire dataset through the augmentation pipeline (training=True ensures layers apply randomness),
            adds light noise, and stores the results. Finally, saves it as .npy files.
            
        Returns:
            x_train_aug (numpy.ndarray): The generated augmented images. Shape: (52500, 28, 28, 1).
            y_train_aug (numpy.ndarray): The labels matching the augmented images.
        """
        print(f">>> Augmenting data by {multiplier} times using Modern Keras Layers...")
        
        # Build the modern augmentation pipeline
        # factor=0.0416 is approximately 15 degrees (15 / 360)
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(factor=0.0416, fill_mode='nearest'),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='nearest')
        ])

        x_aug = []
        y_aug = []
        
        # Convert numpy array to tensor for faster GPU/CPU layer processing
        x_tensor = tf.convert_to_tensor(x_train_350, dtype=tf.float32)
        
        for i in range(multiplier):
            print(f"    -> Generating augmentation batch {i+1}/{multiplier}...")
            # training=True is strictly required to activate Random layers during inference time
            augmented_tensor = data_augmentation(x_tensor, training=True)
            
            # Convert back to numpy and apply our custom noise function
            noisy_augmented_batch = self._add_light_noise(augmented_tensor.numpy())
            
            x_aug.append(noisy_augmented_batch)
            y_aug.append(y_train_350)
            
        # Concatenate lists of arrays into one unified NumPy Array vertically (axis=0)
        x_train_aug = np.concatenate(x_aug, axis=0)
        y_train_aug = np.concatenate(y_aug, axis=0)
        
        # Save the Augmented arrays to disk to avoid re-generating them later
        np.save(os.path.join(self.arrays_dir, "x_train_aug.npy"), x_train_aug)
        np.save(os.path.join(self.arrays_dir, "y_train_aug.npy"), y_train_aug)
        
        print(f">>> Augmentation complete and saved to disk! Augmented data shape: {x_train_aug.shape}")
        return x_train_aug, y_train_aug

    def save_augmentation_samples(self, x_data, y_data):
        """
        Generates a grid plot for EACH digit (0 to 9) showing the isolated effect 
        of each augmentation type. This creates 10 separate images for the final report.
        
        Args:
            x_data (numpy.ndarray): The dataset containing images.
            y_data (numpy.ndarray): The labels to filter out exactly one image per digit.
            
        Logic:
            Loops from digit 0 to 9. Finds the first occurrence of that digit in the dataset.
            Defines 4 isolated Keras Sequential pipelines.
            Draws 5 subplots per digit and saves the figure as digit_X_aug_samples.png.
            
        Returns:
            None (Saves 10 .png files to the specified directory).
        """
        print(">>> Generating Augmentation Samples for EACH digit for the report...")
        
        # Define isolated layers to show specific effects individually
        # factor=0.083 is approximately 30 degrees (30 / 360)
        aug_rot = tf.keras.Sequential([layers.RandomRotation(factor=0.083, fill_mode='nearest')])
        aug_shift = tf.keras.Sequential([layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='nearest')])
        aug_zoom = tf.keras.Sequential([layers.RandomZoom(height_factor=0.3, width_factor=0.3, fill_mode='nearest')])

        for target_digit in range(10):
            # Find the index of the first image that belongs to the current target_digit
            index = np.where(y_data == target_digit)[0][0]
            sample_image = x_data[index]
            
            # Expand dims to act as a batch of size 1 and convert to tensor
            img_batch = np.expand_dims(sample_image, axis=0)
            img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

            # Generate isolated samples
            images_to_plot = {
                'Original': img_batch,
                'Rotated (30 deg)': aug_rot(img_tensor, training=True).numpy(),
                'Shifted (20%)': aug_shift(img_tensor, training=True).numpy(),
                'Zoomed (30%)': aug_zoom(img_tensor, training=True).numpy(),
                'With Light Noise': self._add_light_noise(img_batch)
            }

            # Matplotlib setup: 1 row, 5 columns
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            
            # Set the main title for the figure
            fig.suptitle(f'Augmentation Samples for Digit {target_digit}', fontsize=16)

            for ax, (title, img_result) in zip(axes, images_to_plot.items()):
                # Squeeze removes the batch and channel dims: (1, 28, 28, 1) -> (28, 28)
                ax.imshow(np.squeeze(img_result), cmap='gray')
                ax.set_title(title)
                ax.axis('off')

            plot_path = os.path.join(self.aug_samples_dir, f"digit_{target_digit}_aug_samples.png")
            plt.tight_layout()
            # Savefig automatically overwrites if the filename is exactly the same
            plt.savefig(plot_path)
            plt.close()
            
        print(f">>> 10 Sample plots saved successfully to: {self.aug_samples_dir}")

# ==========================================
# Test Execution Block (If script is run directly)
# ==========================================
if __name__ == "__main__":
    loader = GANDataLoader()
    
    # 1. Load data, setup baselines, and save the 350-arrays to disk
    x_train, y_train = loader.prepare_baselines_and_load()
    
    # 2. Save 10 visual samples (one for each digit) for the report
    loader.save_augmentation_samples(x_train, y_train)
    
    # 3. Generate augmented data and save the augmented arrays to disk
    x_aug, y_aug = loader.generate_augmented_data(x_train, y_train, multiplier=15)