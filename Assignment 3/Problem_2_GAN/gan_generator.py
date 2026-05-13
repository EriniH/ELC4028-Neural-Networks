import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models_gan import build_gan_generator, build_gan_discriminator

# Set random seed for reproducibility in noise generation
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. Custom cDCGAN Trainer Class
# ==========================================
class cDCGANTrainer(tf.keras.Model):
    """
    A Custom Keras Model subclass strictly used for coordinating the Adversarial Training.
    
    Function Logic (The "Why"):
        GANs require a two-step adversarial training process per batch:
        Step 1 (Train Discriminator): Feed real images labeled as 1. Feed fake images labeled as 0. 
                                      Calculate loss and update Discriminator weights.
        Step 2 (Train Generator): Feed noise to Generator to create fakes. Pass fakes to Discriminator.
                                  Calculate Generator loss by seeing how many fakes the Discriminator 
                                  wrongly classified as 1 (Real). Update Generator weights.
    """
    def __init__(self, generator, discriminator, latent_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        # Binary Crossentropy is the standard loss for binary classification (Real vs Fake)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Trackers to log the metrics during training
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        """Returns the list of metrics to be updated per epoch."""
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        """
        The custom adversarial training logic executed for every batch of data.
        
        Args:
            data (tuple): Contains (real_images, target_labels).
        """
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise for the Generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Decode the noise into fake images
        fake_images = self.generator([random_latent_vectors, labels])
        
        # Combine real and fake images into a single batch for the Discriminator
        combined_images = tf.concat([fake_images, real_images], axis=0)
        # Combine labels identically so the conditional discriminator knows what it's looking at
        combined_labels = tf.concat([labels, labels], axis=0)
        
        # Create ground truth labels for the Discriminator: 0 for Fake, 1 for Real
        # We use a slight trick called "Label Smoothing" (adding noise to labels) 
        # to prevent the Discriminator from overpowering the Generator too quickly.
        labels_fake = tf.zeros((batch_size, 1)) + 0.05 * tf.random.uniform((batch_size, 1))
        labels_real = tf.ones((batch_size, 1)) - 0.05 * tf.random.uniform((batch_size, 1))
        combined_truth = tf.concat([labels_fake, labels_real], axis=0)
        
        # --- Step 1: Train the Discriminator ---
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels])
            d_loss = self.loss_fn(combined_truth, predictions)
            
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # --- Step 2: Train the Generator ---
        # Generate new random noise
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # The Generator wants to trick the Discriminator into thinking fakes are Real (label 1)
        misleading_truth = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, labels])
            predictions = self.discriminator([fake_images, labels])
            g_loss = self.loss_fn(misleading_truth, predictions)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metric trackers
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

# ==========================================
# 2. GAN Generator Pipeline Class
# ==========================================
class GANGeneratorPipeline:
    """
    Main class to handle the 5 independent cDCGAN training runs and generate 50k synthetic images.
    """
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        
        # Absolute Output Path strictly assigned to Prob2_GAN
        self.out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob2_GAN/Outputs"
        
        # Target Directories
        self.arrays_dir = os.path.join(self.out_base, "2_Data_Arrays")
        self.gen_images_dir = os.path.join(self.out_base, "3_Generated_Images")
        self.models_dir = os.path.join(self.out_base, "5_Saved_Models")
        
        # Create directories if missing
        os.makedirs(self.gen_images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_training_data(self):
        """
        Loads the .npy arrays (Real + Augmented) saved previously by the data_loader.
        """
        print(">>> Loading Data Arrays from disk...")
        x_real = np.load(os.path.join(self.arrays_dir, "x_train_350.npy"))
        y_real = np.load(os.path.join(self.arrays_dir, "y_train_350.npy"))
        
        x_aug = np.load(os.path.join(self.arrays_dir, "x_train_aug.npy"))
        y_aug = np.load(os.path.join(self.arrays_dir, "y_train_aug.npy"))
        
        # Combine real and augmented data to form the final training set
        x_train_full = np.concatenate([x_real, x_aug], axis=0)
        y_train_full = np.concatenate([y_real, y_aug], axis=0)
        
        print(f">>> Full GAN Training Data Shape: {x_train_full.shape}")
        return x_train_full, y_train_full

    def execute_runs_and_generate(self, x_train, y_train, runs=5, samples_per_digit=1000, epochs=15):
        """
        Executes the independent adversarial training runs and generates synthetic images.
        
        Function Logic (The "Why"):
            To create diverse "variants" of fake images, we must clear the session and build 
            a completely new Generator/Discriminator with fresh random weights per run.
            After adversarial training, we sample standard normal noise, pass it along with
            the target labels to the trained Generator, and extract the generated pixels.
        """
        print(f"\n>>> Starting {runs} independent cDCGAN runs...")
        
        all_generated_x = []
        all_generated_y = []
        
        # Target labels for generation: 0 to 9, each repeated 1000 times (Shape: 10000,)
        target_labels = np.repeat(np.arange(10), samples_per_digit)

        for run in range(1, runs + 1):
            print(f"\n--- [RUN {run}/{runs}] ---")
            
            # Clear Memory Session to allow fresh model building
            tf.keras.backend.clear_session()
            
            # 1. Instantiate FRESH Models
            generator = build_gan_generator(latent_dim=self.latent_dim)
            discriminator = build_gan_discriminator()
            
            # 2. Instantiate and Compile Trainer
            # Note: We use Adam with a smaller learning rate and beta_1=0.5 for GAN stability
            gan = cDCGANTrainer(generator, discriminator, latent_dim=self.latent_dim)
            gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            )
            
            # 3. Train the Model
            print(f"Training GAN for {epochs} epochs...")
            gan.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1)
            
            # 4. Generate Synthetic Data
            print(f"Generating {samples_per_digit * 10} synthetic images...")
            random_latent_vectors = np.random.normal(size=(samples_per_digit * 10, self.latent_dim))
            
            # Pass noise + labels through the trained Generator
            generated_images = generator.predict([random_latent_vectors, target_labels], batch_size=256)
            
            all_generated_x.append(generated_images)
            all_generated_y.append(target_labels)
            
            # 5. Save a general visual overview of this run
            self._save_run_visual_sample(generated_images, target_labels, run)
            
            # Save final models from the last run for backup
            if run == runs:
                print(">>> Saving the weights of the final trained Generator...")
                generator.save(os.path.join(self.models_dir, "final_cdcgan_generator.keras"))
                discriminator.save(os.path.join(self.models_dir, "final_cdcgan_discriminator.keras"))

        # Consolidate arrays
        x_gen_all = np.concatenate(all_generated_x, axis=0)
        y_gen_all = np.concatenate(all_generated_y, axis=0)
        
        # --- Create Dense Visual Samples for the Report ---
        self._save_all_digits_variants(x_gen_all, y_gen_all)
        
        print("\n>>> Saving the 50,000 generated synthetic images to disk...")
        np.save(os.path.join(self.arrays_dir, "x_generated_50k_gan.npy"), x_gen_all)
        np.save(os.path.join(self.arrays_dir, "y_generated_50k_gan.npy"), y_gen_all)
        
        print(f">>> GAN Generation Pipeline Complete! Generated shape: {x_gen_all.shape}")
        return x_gen_all, y_gen_all

    def _save_run_visual_sample(self, generated_images, target_labels, run_number):
        """
        Saves a basic 2x5 visual grid of generated images (one per digit) for a specific run.
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(f'Synthetic Images Generated by cDCGAN - Run {run_number}', fontsize=16)
        axes = axes.flatten()
        
        for digit in range(10):
            idx = np.where(target_labels == digit)[0][0]
            axes[digit].imshow(np.squeeze(generated_images[idx]), cmap='gray')
            axes[digit].set_title(f"GAN Fake: {digit}")
            axes[digit].axis('off')
            
        plot_path = os.path.join(self.gen_images_dir, f"gan_run_{run_number}_overview.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def _save_all_digits_variants(self, all_x, all_y):
        """
        Generates a specific plot for EACH digit (0 to 9). 
        Each plot contains a 1x10 grid showing 10 different fake variants 
        generated by the GAN, serving as a strong visual proof for the report.
        """
        print("\n>>> Extracting detailed visual proofs for EACH digit...")
        for digit in range(10):
            # Get all indices belonging to the current digit
            indices = np.where(all_y == digit)[0]
            # Select 10 random indices to show variants
            selected_indices = np.random.choice(indices, 10, replace=False)
            
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            fig.suptitle(f'10 GAN Generated Variants for Digit {digit}', fontsize=14)
            
            for idx, ax in zip(selected_indices, axes):
                img = all_x[idx]
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.axis('off')
                
            plot_path = os.path.join(self.gen_images_dir, f"digit_{digit}_gan_variants.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        
        print(f">>> 10 Detailed variant plots saved successfully to: {self.gen_images_dir}")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    pipeline = GANGeneratorPipeline(latent_dim=100)
    
    # 1. Load Data
    x_train_full, y_train_full = pipeline.load_training_data()
    
    # 2. Execute the 5 independent training runs and save outputs
    # Using epochs=15 to balance GAN stabilization time with computational limits
    pipeline.execute_runs_and_generate(x_train_full, y_train_full, runs=5, samples_per_digit=1000, epochs=15)