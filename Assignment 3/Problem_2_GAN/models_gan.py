import os
import tensorflow as tf
from keras import layers, Model

# ==========================================
# 1. cDCGAN Generator (The Forger)
# ==========================================
def build_gan_generator(latent_dim=100):
    """
    Builds the Generator network for the Conditional Deep Convolutional GAN (cDCGAN).
    
    Function Logic (The "Why"):
        The Generator's goal is to create a realistic 28x28 grayscale image from pure random noise.
        Since it is "Conditional", it must know WHICH digit to draw. 
        1. We take a random noise vector of size `latent_dim`.
        2. We take a discrete label (0-9), embed it into a continuous vector, and flatten it.
        3. We concatenate the noise and the label embedding.
        4. We map this combined vector to a low-resolution spatial feature map (7x7).
        5. We use Conv2DTranspose (Deconvolution) layers to progressively upsample the image 
           to 14x14, and finally to 28x28.
        
    Args:
        latent_dim (int): The size of the random noise vector (z). Standard practice in GANs is 100.
        
    Returns:
        generator (tf.keras.Model): The compiled Generator model.
            Inputs: [noise_input, label_input]
            Outputs: generated_image of shape (28, 28, 1)
    """
    # --- 1. Input Definition ---
    noise_input = layers.Input(shape=(latent_dim,), name="gen_noise_input")
    label_input = layers.Input(shape=(1,), name="gen_label_input")
    
    # --- 2. Label Conditioning ---
    # Convert the discrete label (0-9) into a 50-dimensional dense vector
    label_embedding = layers.Embedding(input_dim=10, output_dim=50, name="gen_label_embedding")(label_input)
    label_flatten = layers.Flatten(name="gen_label_flatten")(label_embedding)
    
    # Combine the Noise Vector with the Label Embedding
    # Total shape becomes: latent_dim + 50 (e.g., 100 + 50 = 150)
    model_input = layers.Concatenate(name="gen_concat")([noise_input, label_flatten])
    
    # --- 3. Initial Spatial Expansion ---
    # We want our first spatial resolution to be 7x7 with 128 channels.
    # 7 * 7 * 128 = 6272 neurons.
    x = layers.Dense(7 * 7 * 128, use_bias=False, name="gen_dense_base")(model_input)
    # BatchNormalization stabilizes GAN training significantly by maintaining activations mean/variance
    x = layers.BatchNormalization(name="gen_bn_1")(x)
    # LeakyReLU is preferred in GANs over standard ReLU to prevent "dying gradients"
    x = layers.LeakyReLU(alpha=0.2, name="gen_leaky_1")(x)
    # Reshape the 1D vector into a 3D Tensor: (7, 7, 128)
    x = layers.Reshape((7, 7, 128), name="gen_reshape")(x)
    
    # --- 4. Upsampling (Deconvolution) Blocks ---
    # Block 1: Upsample from (7, 7, 128) to (14, 14, 64)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False, name="gen_deconv_1")(x)
    x = layers.BatchNormalization(name="gen_bn_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="gen_leaky_2")(x)
    
    # Block 2: Upsample from (14, 14, 64) to (28, 28, 1)
    # We use 'sigmoid' to ensure the output pixels match normalized original images [0.0, 1.0].
    # Sometimes 'tanh' is used [-1, 1], but our data_loader normalizes to [0, 1], so sigmoid is correct.
    generated_image = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="sigmoid", name="gen_output_image")(x)
    
    generator = Model([noise_input, label_input], generated_image, name="cDCGAN_Generator")
    return generator

# ==========================================
# 2. cDCGAN Discriminator (The Detective)
# ==========================================
def build_gan_discriminator():
    """
    Builds the Discriminator network for the Conditional Deep Convolutional GAN (cDCGAN).
    
    Function Logic (The "Why"):
        The Discriminator acts as a binary classifier. It looks at an image and decides if it is Real (1) or Fake (0).
        Since it is "Conditional", it must judge if the image matches the specific label provided.
        1. We take the label, embed it, and scale it up to a 28x28 spatial map.
        2. We concatenate this label map as a second channel to the input image (Real or Fake).
        3. We use Conv2D layers with strides=2 to spatially downsample and extract deep features.
        4. We flatten the result and use a single neuron with a Sigmoid activation to output a probability (0.0 to 1.0).
        
    Returns:
        discriminator (tf.keras.Model): The compiled Discriminator model.
            Inputs: [image_input, label_input]
            Outputs: validity_score (1 for Real, 0 for Fake)
    """
    # --- 1. Input Definition ---
    image_input = layers.Input(shape=(28, 28, 1), name="disc_image_input")
    label_input = layers.Input(shape=(1,), name="disc_label_input")
    
    # --- 2. Label Conditioning (Spatial Mapping) ---
    # Convert discrete label to a dense vector
    label_embedding = layers.Embedding(input_dim=10, output_dim=50, name="disc_label_embedding")(label_input)
    label_flatten = layers.Flatten(name="disc_label_flatten")(label_embedding)
    # Scale up to exactly 784 neurons to match a 28x28 image grid
    label_dense = layers.Dense(28 * 28, name="disc_label_dense")(label_flatten)
    # Reshape to act as an additional image channel: (28, 28, 1)
    label_reshape = layers.Reshape((28, 28, 1), name="disc_reshape")(label_dense)
    
    # Concatenate the image and the label map. Shape becomes (28, 28, 2)
    merged_input = layers.Concatenate(name="disc_concat")([image_input, label_reshape])
    
    # --- 3. Feature Extraction (Downsampling Blocks) ---
    # Block 1: Downsample from (28, 28, 2) to (14, 14, 64)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", name="disc_conv_1")(merged_input)
    x = layers.LeakyReLU(alpha=0.2, name="disc_leaky_1")(x)
    # Dropout is used in the Discriminator to prevent it from overpowering the Generator too quickly
    x = layers.Dropout(0.3, name="disc_drop_1")(x)
    
    # Block 2: Downsample from (14, 14, 64) to (7, 7, 128)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", name="disc_conv_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="disc_leaky_2")(x)
    x = layers.Dropout(0.3, name="disc_drop_2")(x)
    
    # --- 4. Final Classification ---
    x = layers.Flatten(name="disc_flatten_final")(x)
    
    # Output a single probability scalar [0.0 = Fake, 1.0 = Real]
    validity = layers.Dense(1, activation="sigmoid", name="disc_validity_output")(x)
    
    discriminator = Model([image_input, label_input], validity, name="cDCGAN_Discriminator")
    return discriminator

# ==========================================
# 3. LeNet-5 Classifier (The Judge)
# ==========================================
def build_lenet5():
    """
    Builds a modified LeNet-5 architecture specifically adapted for 28x28 images.
    Identical to Prob 1, imported here to maintain absolute structural independence for Prob 2.
    
    Function Logic (The "Why"):
        This acts as the final "Judge" to evaluate the Confidence of our GAN generated images. 
        It uses ZeroPadding2D to pad 28x28 images to 32x32, preserving original LeNet spatial logic.
        
    Returns:
        model (tf.keras.Sequential): The compiled LeNet-5 classifier model.
    """
    model = tf.keras.Sequential(name="lenet5_classifier_gan")
    
    # Input padding: (28, 28, 1) -> (32, 32, 1)
    model.add(layers.InputLayer(input_shape=(28, 28, 1), name="lenet_input"))
    model.add(layers.ZeroPadding2D(padding=(2, 2), name="lenet_padding"))
    
    # Block 1
    model.add(layers.Conv2D(filters=6, kernel_size=5, activation="relu", padding="valid", name="lenet_conv1"))
    model.add(layers.AveragePooling2D(pool_size=2, strides=2, name="lenet_pool1"))
    
    # Block 2
    model.add(layers.Conv2D(filters=16, kernel_size=5, activation="relu", padding="valid", name="lenet_conv2"))
    model.add(layers.AveragePooling2D(pool_size=2, strides=2, name="lenet_pool2"))
    
    # Fully Connected
    model.add(layers.Flatten(name="lenet_flatten"))
    model.add(layers.Dense(units=120, activation="relu", name="lenet_dense1"))
    model.add(layers.Dense(units=84, activation="relu", name="lenet_dense2"))
    
    # Output Layer for Confidence max(Softmax)
    model.add(layers.Dense(units=10, activation="softmax", name="lenet_confidence_output"))
    
    return model

# ==========================================
# Automated Testing & Saving Block
# ==========================================
if __name__ == "__main__":
    """
    If the script is run directly, it will instantiate the GAN models and 
    both print their architecture summaries to the terminal AND save them 
    into a text file in the specific Prob2_GAN Outputs folder.
    """
    print(">>> Instantiating GAN Models...")
    
    # Instantiate the Architectures
    # We use Latent Dimension = 100 for GANs as per standard DCGAN papers
    generator = build_gan_generator(latent_dim=100)
    discriminator = build_gan_discriminator()
    lenet = build_lenet5()
    
    # Define absolute path for the GAN summaries output
    out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob2_GAN/Outputs"
    summary_dir = os.path.join(out_base, "3_Model_Summaries")
    
    # Create the directory if it does not exist (skip if it does)
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_filepath = os.path.join(summary_dir, "gan_models_architecture.txt")
    
    print(f">>> Saving and Printing GAN Model Summaries to: {summary_filepath}\n")
    
    # Open the file in write mode ('w') which automatically overwrites previous content
    with open(summary_filepath, 'w') as f:
        
        def log_output(text):
            """
            Helper function to write text to the console AND to the opened file simultaneously.
            """
            print(text)
            f.write(text + '\n')
            
        log_output("==============================================================")
        log_output("               GAN MODELS ARCHITECTURE SUMMARY                ")
        log_output("==============================================================\n")
        
        log_output("1. cDCGAN GENERATOR (THE FORGER)")
        log_output("--------------------------------------------------------------")
        generator.summary(print_fn=log_output)
        log_output("\n")
        
        log_output("2. cDCGAN DISCRIMINATOR (THE DETECTIVE)")
        log_output("--------------------------------------------------------------")
        discriminator.summary(print_fn=log_output)
        log_output("\n")
        
        log_output("3. LeNet-5 CLASSIFIER (THE JUDGE)")
        log_output("--------------------------------------------------------------")
        lenet.summary(print_fn=log_output)
        log_output("\n")
        
    print("\n>>> Summaries processed successfully! Check your Terminal and the 3_Model_Summaries folder.")