import os
import tensorflow as tf
from keras import layers, Model

# ==========================================
# 1. Custom Sampling Layer (The Math Core of VAE)
# ==========================================
class Sampling(layers.Layer):
    """
    Custom Keras Layer to perform the "Reparameterization Trick".
    
    Function Logic (The "Why"):
        In a standard Autoencoder, the Encoder outputs a fixed deterministic vector. 
        In a VAE, it outputs parameters of a probability distribution (Mean and Log-Variance). 
        Backpropagation (gradient descent) cannot flow through a purely random/stochastic node. 
        Therefore, we sample a random Noise (epsilon) from a Standard Normal Distribution N(0, I), 
        and shift/scale it using our predicted Mean and Variance.
        Math: z = Mean + exp(0.5 * LogVariance) * Epsilon
    """
    def call(self, inputs):
        """
        Executes the sampling logic during the forward pass of the network.
        
        Args:
            inputs (tuple): A tuple containing exactly two Tensors: (z_mean, z_log_var).
                - z_mean (Tensor): The predicted mean, shape: (Batch_Size, Latent_Dim)
                - z_log_var (Tensor): The predicted log-variance, shape: (Batch_Size, Latent_Dim)
            
        Returns:
            z (Tensor): The sampled Latent Vector. Shape: (Batch_Size, Latent_Dim).
        """
        z_mean, z_log_var = inputs
        
        # Extract dynamic batch size and latent dimension size
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Generate random Epsilon from Standard Normal Distribution N(0, 1)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Calculate final z. 
        # Note: tf.exp(0.5 * z_log_var) mathematically yields the standard deviation (sigma).
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ==========================================
# 2. Conditional VAE Components
# ==========================================
def build_cvae_encoder(latent_dim=16):
    """
    Builds the Encoder network for the Conditional VAE.
    
    Function Logic (The "Why"):
        Since it's a "Conditional" VAE, the Encoder must see BOTH the image and the label it represents.
        We use an Embedding layer to convert the discrete digit label (0-9) into a dense vector,
        scale it up via a Dense layer, and reshape it to match the spatial dimensions of the image (28x28).
        We then concatenate this label representation as a second "channel" to the image.
        Finally, Conv2D layers extract features and compress the dimensions down to the Latent Space.
        
    Args:
        latent_dim (int): The predefined size of the bottleneck/latent space vector. Default is 16.
        
    Returns:
        encoder (tf.keras.Model): The compiled Encoder model object.
            Model Inputs: [image_input, label_input]
            Model Outputs: [z_mean, z_log_var, z]
    """
    # --- 1. Input Definition ---
    image_input = layers.Input(shape=(28, 28, 1), name="encoder_image_input")
    label_input = layers.Input(shape=(1,), name="encoder_label_input")
    
    # --- 2. Label Conditioning (Turning a digit into a 28x28 spatial map) ---
    # Convert label integer (0-9) to a 50-dimensional dense vector
    x_label = layers.Embedding(input_dim=10, output_dim=50, name="encoder_embedding")(label_input)
    x_label = layers.Flatten()(x_label)
    
    # Scale it up to 784 neurons to mathematically match a 28x28 grid (28 * 28 = 784)
    x_label = layers.Dense(28 * 28, name="encoder_label_dense")(x_label)
    
    # Reshape to (28, 28, 1) so it acts exactly like an additional image channel
    x_label = layers.Reshape((28, 28, 1))(x_label)
    
    # Concatenate image and label spatial map. Tensor shape transforms from (28, 28, 1) to (28, 28, 2)
    x = layers.Concatenate(name="encoder_concat")([image_input, x_label])
    
    # --- 3. Feature Extraction (Spatial Downsampling) ---
    # Shape transforms to (14, 14, 32) because strides=2 halves the spatial dimensions
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name="encoder_conv1")(x)
    # Shape transforms to (7, 7, 64) because strides=2 halves the spatial dimensions again
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="encoder_conv2")(x)
    
    # Flatten the 3D tensor to a 1D vector. Shape becomes (7 * 7 * 64) = (3136,)
    x = layers.Flatten(name="flatten_1")(x)
    x = layers.Dense(128, activation="relu", name="encoder_dense1")(x)
    
    # --- 4. Latent Space Outputs (The Distribution Parameters) ---
    # Output 1: The Mean (mu) of the probability distribution
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    # Output 2: The Log-Variance of the probability distribution
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    # Pass mean and log-variance to our custom Sampling layer to get the final latent vector (z)
    z = Sampling(name="encoder_sampler")([z_mean, z_log_var])
    
    # Create the Keras Model
    encoder = Model([image_input, label_input], [z_mean, z_log_var, z], name="cvae_encoder")
    return encoder

def build_cvae_decoder(latent_dim=16):
    """
    Builds the Decoder network for the Conditional VAE.
    
    Function Logic (The "Why"):
        The Decoder's job is Data Generation. It takes the sampled latent vector (z) AND the targeted label.
        It concatenates them, expands them via a large Dense layer, reshapes them back into a small 3D Tensor,
        and uses Conv2DTranspose (Deconvolution) to spatially upsample back to a (28, 28, 1) image.
        
    Args:
        latent_dim (int): The size of the incoming latent vector (z). Must match the Encoder. Default is 16.
        
    Returns:
        decoder (tf.keras.Model): The compiled Decoder model object.
            Model Inputs: [latent_input, label_input]
            Model Outputs: reconstructed_image
    """
    # --- 1. Input Definition ---
    latent_input = layers.Input(shape=(latent_dim,), name="decoder_latent_input")
    label_input = layers.Input(shape=(1,), name="decoder_label_input")
    
    # --- 2. Label Conditioning ---
    # Embed the label to a 50-dimensional vector and flatten it
    x_label = layers.Embedding(input_dim=10, output_dim=50, name="decoder_embedding")(label_input)
    x_label = layers.Flatten(name="flatten_2")(x_label)
    
    # Concatenate the Latent Vector (z) with the Label Embedding
    # Shape becomes (latent_dim + 50). E.g., (16 + 50) = (66,)
    x = layers.Concatenate(name="decoder_concat")([latent_input, x_label])
    
    # --- 3. Initial Expansion ---
    # Map the 1D vector to a large Dense layer enough to form a 7x7 spatial map with 64 channels
    x = layers.Dense(7 * 7 * 64, activation="relu", name="decoder_dense_expand")(x)
    # Reshape back to 3D Tensor: shape becomes (7, 7, 64)
    x = layers.Reshape((7, 7, 64), name="reshape_1")(x)
    
    # --- 4. Spatial Upsampling (Deconvolution) ---
    # Upsample to (14, 14, 64) due to strides=2
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="decoder_deconv1")(x)
    # Upsample to (28, 28, 32) due to strides=2
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="decoder_deconv2")(x)
    
    # Final Output Layer: Conv2DTranspose without strides to collapse channels to 1.
    # Shape becomes (28, 28, 1).
    # We use 'sigmoid' activation to ensure output pixel values are bounded strictly between [0.0, 1.0]
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name="decoder_output_image")(x)
    
    # Create the Keras Model
    decoder = Model([latent_input, label_input], decoder_outputs, name="cvae_decoder")
    return decoder

def build_cvae(encoder, decoder):
    """
    Combines the Encoder and Decoder into a single end-to-end Conditional VAE Model.
    
    Function Logic (The "Why"):
        While we train the VAE as a single integrated unit, we define it using the previously 
        built encoder and decoder. This modularity allows us to pass inputs entirely through the pipeline,
        and later use only the Decoder for generation.
        
    Args:
        encoder (tf.keras.Model): The instantiated encoder model.
        decoder (tf.keras.Model): The instantiated decoder model.
        
    Returns:
        cvae_model (tf.keras.Model): The full end-to-end model object.
            Model Inputs: [image_input, label_input]
            Model Outputs: reconstructed_image
    """
    # Define inputs for the whole integrated model
    image_input = layers.Input(shape=(28, 28, 1), name="cvae_image_input")
    label_input = layers.Input(shape=(1,), name="cvae_label_input")
    
    # Pass inputs through the encoder to get the latent vector (z)
    # The encoder returns [z_mean, z_log_var, z]. We only extract z (index 2) for the decoder.
    _, _, z = encoder([image_input, label_input])
    
    # Pass z and the label through the decoder to get the final reconstructed image
    reconstructed_image = decoder([z, label_input])
    
    # Build the full combined model
    cvae_model = Model(inputs=[image_input, label_input], outputs=reconstructed_image, name="full_cvae")
    return cvae_model

# ==========================================
# 3. LeNet-5 Classifier (The Evaluator)
# ==========================================
def build_lenet5():
    """
    Builds a modified LeNet-5 architecture specifically adapted for 28x28 images.
    
    Function Logic (The "Why"):
        This acts as the "Judge" to evaluate the Confidence of our VAE generated images. 
        It uses standard Convolution and Average Pooling to extract spatial features, 
        followed by Fully Connected layers to classify the digit (0-9).
        Since the original LeNet-5 was historically designed for 32x32 images, we apply 
        ZeroPadding2D to our 28x28 images to pad them to 32x32 before the first Convolution. 
        This preserves the exact mathematical dimensions of the original LeNet architecture.
        
    Returns:
        model (tf.keras.Sequential): The compiled LeNet-5 classifier model.
            Input shape: (28, 28, 1)
            Output shape: (10,) representing softmax probabilities.
    """
    model = tf.keras.Sequential(name="lenet5_classifier")
    
    # Step 1: Input & Padding
    # Input image is (28, 28, 1). ZeroPadding adds 2 pixels to each side, converting it to (32, 32, 1)
    model.add(layers.InputLayer(input_shape=(28, 28, 1), name="lenet_input"))
    model.add(layers.ZeroPadding2D(padding=(2, 2), name="lenet_padding"))
    
    # Step 2: First Block (Conv + Pool)
    # Shape goes from (32, 32, 1) -> (28, 28, 6)
    model.add(layers.Conv2D(filters=6, kernel_size=5, activation="relu", padding="valid", name="lenet_conv1"))
    # Shape goes from (28, 28, 6) -> (14, 14, 6)
    model.add(layers.AveragePooling2D(pool_size=2, strides=2, name="lenet_pool1"))
    
    # Step 3: Second Block (Conv + Pool)
    # Shape goes from (14, 14, 6) -> (10, 10, 16)
    model.add(layers.Conv2D(filters=16, kernel_size=5, activation="relu", padding="valid", name="lenet_conv2"))
    # Shape goes from (10, 10, 16) -> (5, 5, 16)
    model.add(layers.AveragePooling2D(pool_size=2, strides=2, name="lenet_pool2"))
    
    # Step 4: Fully Connected Layers
    # Flatten (5, 5, 16) -> (400,)
    model.add(layers.Flatten(name="lenet_flatten"))
    model.add(layers.Dense(units=120, activation="relu", name="lenet_dense1"))
    model.add(layers.Dense(units=84, activation="relu", name="lenet_dense2"))
    
    # Step 5: Output Layer
    # Output vector of length 10. Softmax ensures the sum of all probabilities equals exactly 1.0.
    # The max value of this output will be extracted later as our "Confidence" score.
    model.add(layers.Dense(units=10, activation="softmax", name="lenet_confidence_output"))
    
    return model

# ==========================================
# Automated Testing & Saving Block
# ==========================================
if __name__ == "__main__":
    """
    If the script is run directly, it will instantiate all models and 
    both print their architecture summaries to the terminal AND save them 
    into a text file in the Outputs folder simultaneously.
    """
    print(">>> Instantiating Models...")
    encoder = build_cvae_encoder(latent_dim=16)
    decoder = build_cvae_decoder(latent_dim=16)
    cvae = build_cvae(encoder, decoder)
    lenet = build_lenet5()
    
    # Define absolute path for the summaries output
    out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob1_VAE/Outputs"
    summary_dir = os.path.join(out_base, "3_Model_Summaries")
    
    # Create the directory if it does not exist (skip if it does)
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_filepath = os.path.join(summary_dir, "models_architecture_output.txt")
    
    print(f">>> Saving and Printing Model Summaries to: {summary_filepath}\n")
    
    # Open the file in write mode ('w') which automatically overwrites previous content
    with open(summary_filepath, 'w') as f:
        
        def log_output(text):
            """
            Helper function to write text to the console AND to the opened file.
            """
            print(text)
            f.write(text + '\n')
            
        log_output("==============================================================")
        log_output("                  MODELS ARCHITECTURE SUMMARY                 ")
        log_output("==============================================================\n")
        
        log_output("1. CVAE ENCODER")
        log_output("--------------------------------------------------------------")
        # Pass the dual-logging function to the summary method
        encoder.summary(print_fn=log_output)
        log_output("\n")
        
        log_output("2. CVAE DECODER")
        log_output("--------------------------------------------------------------")
        decoder.summary(print_fn=log_output)
        log_output("\n")
        
        log_output("3. LeNet-5 CLASSIFIER (THE JUDGE)")
        log_output("--------------------------------------------------------------")
        lenet.summary(print_fn=log_output)
        log_output("\n")
        
    print(">>> Summaries processed successfully! Check your Terminal and the 3_Model_Summaries folder.")