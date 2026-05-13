import os
import tensorflow as tf
from keras import layers, Model

# ==========================================
# 1. Custom Spatial Attention Layer
# ==========================================
class SpatialAttention(layers.Layer):
    """
    A Custom Keras Layer that implements Spatial Attention.
    
    Function Logic (The "Why"):
        Instead of treating all regions of an image equally, this layer learns to create 
        a spatial "Mask" (values between 0 and 1). It multiplies this mask with the 
        incoming feature maps to emphasize important features and suppress irrelevant background noise.
        This is heavily inspired by the CBAM (Convolutional Block Attention Module) paper.
    """
    def __init__(self, kernel_size=7, **kwargs):
        """
        Constructor for the Spatial Attention layer.
        
        Args:
            kernel_size (int): The size of the Convolutional filter used to learn the spatial mask.
                               7x7 is the standard size recommended by research to capture a broad spatial context.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
        # Define the Conv2D layer that will learn the attention mask.
        # We output exactly 1 channel, and use 'sigmoid' to ensure mask values are in [0, 1].
        self.conv = layers.Conv2D(
            filters=1, 
            kernel_size=self.kernel_size, 
            strides=1, 
            padding='same', 
            activation='sigmoid', 
            use_bias=False,
            name="spatial_attention_conv"
        )

    def call(self, inputs):
        """
        The forward pass of the attention layer.
        
        Args:
            inputs (Tensor): The incoming feature map of shape (Batch, Height, Width, Channels).
            
        Returns:
            Tensor: The attention-scaled feature map of the exact same shape.
        """
        # Step 1: Compress the channel information using Average Pooling and Max Pooling
        # We compute the mean and max across the channel axis (axis=-1)
        # keepdims=True ensures the shape stays (Batch, Height, Width, 1) instead of dropping the axis.
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Step 2: Concatenate the two pooled maps together.
        # Shape becomes (Batch, Height, Width, 2)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Step 3: Pass through the Conv2D layer to generate the attention Mask
        # The Conv layer reduces the 2 channels back to 1, and Sigmoid squashes values to [0, 1]
        # Shape is now (Batch, Height, Width, 1)
        attention_mask = self.conv(concat)
        
        # Step 4: Multiply the original inputs by the mask
        # TensorFlow automatically broadcasts the (H, W, 1) mask across all original Channels (C)
        return inputs * attention_mask

    def get_config(self):
        """
        Required by Keras to properly save and load models containing this custom layer.
        """
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


# ==========================================
# Automated Testing & Verification Block
# ==========================================
if __name__ == "__main__":
    """
    If the script is run directly, it will build a dummy model to verify 
    that the SpatialAttention layer compiles correctly and shapes are consistent.
    It automatically overwrites the output text file for documentation.
    """
    print(">>> Testing the Custom Spatial Attention Layer...")
    
    # Define absolute paths dynamically matching the agreed structure
    out_base = "/home/mohamedkhalid/Desktop/Neural Assignments/Neural Assignment3/Prob3_Attention/Outputs"
    os.makedirs(out_base, exist_ok=True)
    log_file = os.path.join(out_base, "attention_layer_test.txt")
    
    # Build a Dummy Model
    # Let's assume an intermediate feature map of 14x14 with 64 channels
    dummy_input = layers.Input(shape=(14, 14, 64), name="dummy_feature_map")
    attended_output = SpatialAttention(kernel_size=7, name="custom_spatial_attention")(dummy_input)
    
    dummy_model = Model(inputs=dummy_input, outputs=attended_output, name="Attention_Test_Model")
    
    # Save the output to a text file and print it
    print(f">>> Writing Architecture Test to: {log_file}\n")
    with open(log_file, 'w') as f:
        def log_output(text):
            print(text)
            f.write(text + '\n')
            
        log_output("==============================================================")
        log_output("          SPATIAL ATTENTION LAYER ARCHITECTURE TEST           ")
        log_output("==============================================================\n")
        log_output("The following summary proves that the custom layer accepts an")
        log_output("input Tensor, processes the spatial mask, and returns a Tensor")
        log_output("of the exact same dimensions, making it perfectly plug-and-play.\n")
        
        dummy_model.summary(print_fn=log_output)
        
    print(">>> Spatial Attention module is verified and ready for deployment!")