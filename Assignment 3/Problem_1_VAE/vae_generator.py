import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import build_cvae_encoder, build_cvae_decoder # Importing our architectures

# Set random seed for reproducibility in generation
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. Custom VAE Trainer Class
# ==========================================
class CVAETrainer(tf.keras.Model):
    """
    A Custom Keras Model subclass strictly used for TRAINING the VAE.
    
    Function Logic (The "Why"):
        The VAE cannot be trained using a simple 'model.compile(loss="mse")' because its total loss
        is a combination of two separate mathematical equations:
        1. Reconstruction Loss: How well the decoder recreates the input image.
        2. KL Divergence: How closely the encoder's Latent Distribution matches a Standard Normal Distribution.
        By overriding `train_step`, we manually calculate these losses using `tf.GradientTape`.
    """
    def __init__(self, encoder, decoder, **kwargs):
        """
        Constructor mapping the isolated encoder and decoder to the trainer.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Trackers to log the loss metrics during training
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """Returns the list of metrics to be updated per epoch."""
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        """
        The custom training logic executed for every batch of data.
        
        Args:
            data (tuple): Contains (x, y) where x is the image and y is the label.
        
        Returns:
            dict: A dictionary containing the updated loss values.
        """
        x, y = data
        
        with tf.GradientTape() as tape:
            # 1. Forward pass through the Encoder
            z_mean, z_log_var, z = self.encoder([x, y])
            
            # 2. Forward pass through the Decoder
            reconstruction = self.decoder([z, y])
            
            # 3. Calculate Reconstruction Loss (MSE scaled by image size 28x28)
            # We scale by 784 to sum the error across all pixels rather than averaging them,
            # keeping the scale balanced with the KL loss.
            mse_loss = tf.keras.losses.MeanSquaredError()(x, reconstruction)
            reconstruction_loss = tf.reduce_mean(mse_loss) * 28.0 * 28.0
            
            # 4. Calculate KL Divergence Loss
            # Math: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # 5. Total Loss
            total_loss = reconstruction_loss + kl_loss
            
        # Compute gradients and apply them to update the Neural Network weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metric trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# ==========================================
# 2. VAE Generator Class
# ==========================================
class VAEGenerator:
    """
    Main class to handle the 5 independent VAE training runs and generate the 50,000 synthetic images.
    """
    def __init__(self, latent_dim=16):
        """
        Initializes paths and essential parameters.
        """
        self.latent_dim = latent_dim
        
        # Base Output Path
        self.out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob1_VAE/Outputs"
        
        # Define exact paths for outputs
        self.arrays_dir = os.path.join(self.out_base, "2_Data_Arrays")
        self.gen_images_dir = os.path.join(self.out_base, "2_Generated_Images")
        self.models_dir = os.path.join(self.out_base, "4_Saved_Models")
        
        # Ensure directories exist
        os.makedirs(self.gen_images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_training_data(self):
        """
        Loads the .npy arrays saved previously by data_loader.py and concatenates them.
        
        Returns:
            x_train_full (numpy.ndarray): Combined Real (350/digit) + Augmented data.
            y_train_full (numpy.ndarray): Combined labels.
        """
        print(">>> Loading Data Arrays from disk...")
        x_real = np.load(os.path.join(self.arrays_dir, "x_train_350.npy"))
        y_real = np.load(os.path.join(self.arrays_dir, "y_train_350.npy"))
        
        x_aug = np.load(os.path.join(self.arrays_dir, "x_train_aug.npy"))
        y_aug = np.load(os.path.join(self.arrays_dir, "y_train_aug.npy"))
        
        # Combine real and augmented data to form the final training set
        x_train_full = np.concatenate([x_real, x_aug], axis=0)
        y_train_full = np.concatenate([y_real, y_aug], axis=0)
        
        print(f">>> Full Training Data Shape: {x_train_full.shape}")
        return x_train_full, y_train_full

    def execute_runs_and_generate(self, x_train, y_train, runs=5, samples_per_digit=1000, epochs=15):
        """
        Executes the independent training runs and generates synthetic images.
        
        Function Logic (The "Why"):
            The assignment explicitly requires "Generate 5x... Each run should start with 
            different random values of the weights". By instantiating a new encoder and decoder
            inside the loop, TensorFlow automatically assigns completely new random weights.
            We use 'tf.keras.backend.clear_session()' at the start of each run to clear the Memory Graph,
            preventing Layer Name collisions (e.g., 'flatten_1 used 2 times' error).
            After training, we sample random noise from N(0, 1), pair it with desired labels,
            and push it through the trained decoder to generate purely fake images.
            
        Args:
            x_train, y_train: The combined dataset.
            runs (int): Number of independent training cycles. Default is 5.
            samples_per_digit (int): How many fake images to generate per digit in ONE run.
            epochs (int): Number of epochs per run. 15 is generally sufficient for 56k images.
            
        Returns:
            x_gen_all (numpy.ndarray): 50,000 synthetic images. Shape: (50000, 28, 28, 1).
            y_gen_all (numpy.ndarray): The labels for the synthetic images. Shape: (50000,).
        """
        print(f"\n>>> Starting {runs} independent VAE runs...")
        
        all_generated_x = []
        all_generated_y = []
        
        # Prepare the target labels for generation (1000 zeros, 1000 ones, etc.)
        # Shape: (10000,) representing 10 digits * 1000 samples each
        target_labels = np.repeat(np.arange(10), samples_per_digit)

        for run in range(1, runs + 1):
            print(f"\n--- [RUN {run}/{runs}] ---")
            
            # --- CRITICAL FIX ---
            # Clear the Keras Session Memory to destroy old layer names from previous runs.
            # This prevents ValueError: The name "layer_name" is used 2 times.
            tf.keras.backend.clear_session()
            
            # 1. Instantiate FRESH Models (This guarantees new random weight initialization)
            encoder = build_cvae_encoder(latent_dim=self.latent_dim)
            decoder = build_cvae_decoder(latent_dim=self.latent_dim)
            
            # 2. Instantiate and Compile Trainer
            vae = CVAETrainer(encoder, decoder)
            vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
            
            # 3. Train the Model
            print(f"Training VAE for {epochs} epochs...")
            vae.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1)
            
            # 4. Generate Synthetic Data
            print(f"Generating {samples_per_digit * 10} synthetic images...")
            
            # Sample random noise z ~ N(0, 1) of shape (10000, latent_dim)
            random_latent_vectors = np.random.normal(size=(samples_per_digit * 10, self.latent_dim))
            
            # Pass the random noise AND the target labels through the trained Decoder
            # This generates the fake images based on the requested conditions
            generated_images = decoder.predict([random_latent_vectors, target_labels], batch_size=256)
            
            # Append generated batch to the main lists
            all_generated_x.append(generated_images)
            all_generated_y.append(target_labels)
            
            # 5. Save a visual sample grid of this specific run
            self._save_run_visual_sample(generated_images, target_labels, run)
            
            # Save the models of the very last run for future backup
            if run == runs:
                print(">>> Saving the weights of the final trained Decoder...")
                decoder.save(os.path.join(self.models_dir, "final_cvae_decoder.keras"))
                encoder.save(os.path.join(self.models_dir, "final_cvae_encoder.keras"))

        # Consolidate all 5 runs into massive Numpy arrays
        x_gen_all = np.concatenate(all_generated_x, axis=0)
        y_gen_all = np.concatenate(all_generated_y, axis=0)
        
        # Save the 50k generated dataset arrays
        print("\n>>> Saving the 50,000 generated synthetic images to disk...")
        np.save(os.path.join(self.arrays_dir, "x_generated_50k.npy"), x_gen_all)
        np.save(os.path.join(self.arrays_dir, "y_generated_50k.npy"), y_gen_all)
        
        print(f">>> Generation Pipeline Complete! Generated shape: {x_gen_all.shape}")
        return x_gen_all, y_gen_all

    def _save_run_visual_sample(self, generated_images, target_labels, run_number):
        """
        Internal helper function to plot a 2x5 grid showing one generated fake image
        per digit (0-9) for a given Run, saving it to the report folder.
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(f'Synthetic Images Generated by VAE - Run {run_number}', fontsize=16)
        axes = axes.flatten()
        
        for digit in range(10):
            # Find the index of the first generated image for this specific digit
            idx = np.where(target_labels == digit)[0][0]
            img = generated_images[idx]
            
            axes[digit].imshow(np.squeeze(img), cmap='gray')
            axes[digit].set_title(f"Fake Digit: {digit}")
            axes[digit].axis('off')
            
        plot_path = os.path.join(self.gen_images_dir, f"vae_generated_samples_run_{run_number}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    generator_pipeline = VAEGenerator(latent_dim=16)
    
    # 1. Load the real and augmented data prepared earlier
    x_train_full, y_train_full = generator_pipeline.load_training_data()
    
    # 2. Train VAE 5 times and generate the 50,000 synthetic images
    # Note: epochs are set to 15 to balance training quality and computational time.
    x_fake, y_fake = generator_pipeline.execute_runs_and_generate(x_train_full, y_train_full, runs=5, samples_per_digit=1000, epochs=15)