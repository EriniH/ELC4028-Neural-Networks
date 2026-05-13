import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models import build_lenet5 # Importing our LeNet-5 architecture

# Set random seeds for strict reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class Prob1Evaluator:
    """
    The final evaluation pipeline. It judges VAE-generated data, filters them by
    confidence, saves visual proofs for the report, retrains classification models, 
    and benchmarks the results.
    """
    def __init__(self):
        """
        Initializes paths and creates necessary directories.
        """
        # Define absolute paths
        self.test_dataset_path = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Reduced MNIST Data/test"
        self.out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob1_VAE/Outputs"
        
        self.arrays_dir = os.path.join(self.out_base, "2_Data_Arrays")
        self.filtered_dir = os.path.join(self.out_base, "3_Filtered_Datasets")
        self.models_dir = os.path.join(self.out_base, "4_Saved_Models")
        self.plots_dir = os.path.join(self.out_base, "5_Evaluation_Plots")
        self.baseline_1000_dir = os.path.join(self.out_base, "0_Baselines", "1000_real")
        
        # Create final directories if they do not exist
        os.makedirs(self.filtered_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _load_images_from_directory(self, dir_path):
        """
        Helper function to load images from a structured directory (subfolders 0-9).
        
        Args:
            dir_path (str): The absolute path to the directory containing 0-9 folders.
            
        Returns:
            x_data (numpy.ndarray): Images array of shape (N, 28, 28, 1).
            y_data (numpy.ndarray): Labels array of shape (N,).
        """
        x_list, y_list = [], []
        for digit in range(10):
            folder = os.path.join(dir_path, str(digit))
            if not os.path.exists(folder): continue
            
            for img_name in os.listdir(folder):
                if not img_name.endswith(('.jpg', '.png')): continue
                img_path = os.path.join(folder, img_name)
                
                # Read, normalize, and reshape
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                
                x_list.append(img)
                y_list.append(digit)
                
        return np.array(x_list), np.array(y_list)

    def load_all_data(self):
        """
        Loads all necessary datasets: 350-real, 50k-synthetic, 1000-real, and the Test set.
        """
        print("\n>>> Loading all data into memory...")
        
        # 1. Load Pre-saved Numpy Arrays
        self.x_350 = np.load(os.path.join(self.arrays_dir, "x_train_350.npy"))
        self.y_350 = np.load(os.path.join(self.arrays_dir, "y_train_350.npy"))
        
        self.x_fake_50k = np.load(os.path.join(self.arrays_dir, "x_generated_50k.npy"))
        self.y_fake_50k = np.load(os.path.join(self.arrays_dir, "y_generated_50k.npy"))
        
        # 2. Load 1000-real Baseline directly from the 0_Baselines folder
        self.x_1000, self.y_1000 = self._load_images_from_directory(self.baseline_1000_dir)
        
        # 3. Load the completely unseen Test Set
        self.x_test, self.y_test = self._load_images_from_directory(self.test_dataset_path)
        
        print(f"    -> 350-Real Shape: {self.x_350.shape}")
        print(f"    -> 1000-Real Shape: {self.x_1000.shape}")
        print(f"    -> Fake 50k Shape: {self.x_fake_50k.shape}")
        print(f"    -> Test Set Shape: {self.x_test.shape}")

    def train_judge_and_filter(self):
        """
        Trains a LeNet-5 on the 350-real dataset to act as the pure "Judge".
        Passes the 50k fake images through the Judge to calculate Softmax Confidence.
        Filters the fake data into Set A (All), Set B (>=0.9), and Set C (0.6 to 0.9).
        Saves visual proofs of Set B and Set C for the report, then saves the arrays.
        """
        print("\n>>> Training the LeNet-5 'Judge' purely on the 350-real baseline...")
        tf.keras.backend.clear_session()
        
        judge_model = build_lenet5()
        judge_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train quickly (10 epochs is enough for LeNet to memorize 3500 images)
        judge_model.fit(self.x_350, self.y_350, epochs=10, batch_size=64, verbose=0)
        
        print(">>> Judge is ready! Evaluating the 50,000 synthetic images...")
        # Predict returns probability distributions (Shape: 50000, 10)
        predictions = judge_model.predict(self.x_fake_50k, batch_size=512)
        
        # Calculate Confidence (The maximum probability across the 10 classes for each image)
        # np.max extracts the highest float value in axis 1
        confidences = np.max(predictions, axis=1)
        
        print(">>> Filtering datasets based on confidence rules...")
        
        # --- Create Set A: All generated samples ---
        self.x_set_a = self.x_fake_50k
        self.y_set_a = self.y_fake_50k
        
        # --- Create Set B: High Confidence (>= 0.9) ---
        idx_b = np.where(confidences >= 0.9)[0]
        self.x_set_b = self.x_fake_50k[idx_b]
        self.y_set_b = self.y_fake_50k[idx_b]
        conf_set_b = confidences[idx_b] # Keep track of specific confidences for the plot
        
        # --- Create Set C: Mid Confidence (0.6 <= x <= 0.9) ---
        idx_c = np.where((confidences >= 0.6) & (confidences <= 0.9))[0]
        self.x_set_c = self.x_fake_50k[idx_c]
        self.y_set_c = self.y_fake_50k[idx_c]
        conf_set_c = confidences[idx_c] # Keep track of specific confidences for the plot
        
        print(f"    -> Set A (All): {len(self.x_set_a)} images.")
        print(f"    -> Set B (High Conf): {len(self.x_set_b)} images.")
        print(f"    -> Set C (Mid Conf): {len(self.x_set_c)} images.")
        
        # --- NEW: Generate Visual Proofs for the Report ---
        self._plot_confidence_samples(self.x_set_b, self.y_set_b, conf_set_b, "Set_B_High_Confidence", "Set B Examples (Confidence >= 0.9)")
        self._plot_confidence_samples(self.x_set_c, self.y_set_c, conf_set_c, "Set_C_Mid_Confidence", "Set C Examples (0.6 <= Confidence <= 0.9)")
        
        # Save datasets to disk
        np.save(os.path.join(self.filtered_dir, "x_set_a.npy"), self.x_set_a)
        np.save(os.path.join(self.filtered_dir, "y_set_a.npy"), self.y_set_a)
        np.save(os.path.join(self.filtered_dir, "x_set_b.npy"), self.x_set_b)
        np.save(os.path.join(self.filtered_dir, "y_set_b.npy"), self.y_set_b)
        np.save(os.path.join(self.filtered_dir, "x_set_c.npy"), self.x_set_c)
        np.save(os.path.join(self.filtered_dir, "y_set_c.npy"), self.y_set_c)

    def _plot_confidence_samples(self, x_data, y_data, conf_data, filename, title):
        """
        Helper function to plot a random grid of 10 images from a specific set,
        displaying the predicted label and the exact confidence score above each image.
        
        Args:
            x_data (numpy.ndarray): The filtered images array.
            y_data (numpy.ndarray): The filtered labels array.
            conf_data (numpy.ndarray): The specific confidence scores for these images.
            filename (str): The output filename.
            title (str): The main title for the matplotlib figure.
            
        Returns:
            None (Saves a .png file to the filtered_dir).
        """
        print(f"    -> Drawing visual proofs for {title}...")
        
        # If the set is empty (very rare), skip plotting
        if len(x_data) == 0:
            print(f"       Warning: {title} is empty, skipping plot.")
            return

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)
        axes = axes.flatten()
        
        # Pick up to 10 random indices from this set
        sample_size = min(10, len(x_data))
        random_indices = np.random.choice(len(x_data), sample_size, replace=False)
        
        for i, idx in enumerate(random_indices):
            img = x_data[idx]
            label = y_data[idx]
            conf = conf_data[idx]
            
            axes[i].imshow(np.squeeze(img), cmap='gray')
            # Display both the label and the confidence percentage
            axes[i].set_title(f"Label: {label}\nConf: {conf*100:.1f}%", fontsize=10)
            axes[i].axis('off')
            
        # Turn off any unused axes if we had less than 10 images
        for j in range(sample_size, 10):
            axes[j].axis('off')
            
        plot_path = os.path.join(self.filtered_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def train_and_benchmark(self):
        """
        The Final Benchmark. Trains LeNet-5 models on different dataset combinations
        and tests them against the unseen Test Set to answer the Assignment Question.
        
        Returns:
            dict: The final accuracy results.
        """
        print("\n>>> Commencing the Final Benchmark Training...")
        
        # Datasets configurations dictionary
        # Concatenate 350-real with each generated set as instructed in Step 7
        datasets = {
            "350-Real Baseline": (self.x_350, self.y_350),
            "1000-Real Baseline": (self.x_1000, self.y_1000),
            "350-Real + Set A": (np.concatenate([self.x_350, self.x_set_a]), np.concatenate([self.y_350, self.y_set_a])),
            "350-Real + Set B": (np.concatenate([self.x_350, self.x_set_b]), np.concatenate([self.y_350, self.y_set_b])),
            "350-Real + Set C": (np.concatenate([self.x_350, self.x_set_c]), np.concatenate([self.y_350, self.y_set_c]))
        }
        
        results = {}
        
        for name, (x_train, y_train) in datasets.items():
            print(f"\n--- Training LeNet-5 on: {name} | Data Size: {len(x_train)} ---")
            tf.keras.backend.clear_session()
            
            model = build_lenet5()
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train the model
            model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=0)
            
            # Evaluate strictly on the Test Set
            loss, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
            
            # Save the accuracy percentage
            results[name] = acc * 100.0
            print(f"    -> Test Accuracy: {results[name]:.2f}%")
            
            # Save weights
            safe_name = name.replace(" ", "_").replace("+", "plus")
            model.save(os.path.join(self.models_dir, f"lenet5_{safe_name}.keras"))
            
        self._generate_final_report(results)
        return results

    def _generate_final_report(self, results):
        """
        Helper function to plot the bar chart and print the Assignment Table.
        
        Args:
            results (dict): Dictionary containing the dataset names and their accuracies.
        """
        # 1. Plotting the Bar Chart
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        accs = list(results.values())
        
        bars = plt.bar(names, accs, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0'])
        plt.ylim(min(accs) - 5, 100) # Adjust Y-axis to highlight differences
        plt.ylabel('Test Accuracy (%)')
        plt.title('Performance Comparison of VAE Synthetic Data Selection Strategies')
        plt.xticks(rotation=15)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
            
        plot_path = os.path.join(self.plots_dir, "vae_benchmarking_results.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\n>>> Final Bar Chart saved to: {plot_path}")
        
        # 2. Printing the Text Table for the assignment
        print("\n=========================================================================")
        print("                        FINAL ASSIGNMENT TABLE (VAE)                     ")
        print("=========================================================================")
        print(f"| Model | No. of Examples | all generated data | CL >= 0.9 only | 0.6 <= CL <= 0.9 |")
        print("-------------------------------------------------------------------------")
        
        # Getting the lengths of Set A, B, C for the table
        len_a = len(self.x_set_a)
        len_b = len(self.x_set_b)
        len_c = len(self.x_set_c)
        
        acc_a = results["350-Real + Set A"]
        acc_b = results["350-Real + Set B"]
        acc_c = results["350-Real + Set C"]
        
        # Format the table row
        print(f"|  VAE  |  Set Size       | {len_a:<18} | {len_b:<14} | {len_c:<16} |")
        print(f"|       |  Test Accuracy  | {acc_a:>6.2f}%           | {acc_b:>6.2f}%         | {acc_c:>6.2f}%           |")
        print("=========================================================================\n")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    evaluator = Prob1Evaluator()
    
    # Execute Pipeline
    evaluator.load_all_data()
    evaluator.train_judge_and_filter()
    evaluator.train_and_benchmark()