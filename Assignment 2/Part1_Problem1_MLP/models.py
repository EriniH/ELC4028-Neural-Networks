import tensorflow as tf
from keras.optimizers import Adam
from keras import layers, models

# ==============================================================================
# DYNAMIC MULTI-LAYER PERCEPTRON (MLP) ARCHITECTURE
# ==============================================================================

def build_mlp_model(input_dim, num_classes=10, num_layers=3, use_regularization=True, learning_rate=0.001):
    """
    Constructs a dynamic MLP where you can control the number of layers, 
    regularization, and learning rate. This directly fulfills the assignment requirements.
    
    Arguments:
        input_dim (int): Features dimension (784 for flattened MNIST).
        num_classes (int): Number of digits (10).
        num_layers (int): Number of hidden layers (1, 3, or 4).
        use_regularization (bool): Toggles BatchNormalization and Dropout ON/OFF.
        learning_rate (float): The step size for the Adam optimizer.
    """
    
    model = models.Sequential(name=f"MLP_{num_layers}_Layers")
    
    # We define a predefined list of neuron counts to maintain a tapering architecture
    neurons_per_layer = [512, 256, 128, 64]
    
    # Input definition for the first layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Dynamically build the hidden layers based on 'num_layers'
    for i in range(num_layers):
        model.add(layers.Dense(units=neurons_per_layer[i], name=f"Hidden_{i+1}"))
        
        # Apply Regularization if requested (Batch Norm + Dropout)
        if use_regularization:
            model.add(layers.BatchNormalization(name=f"BatchNorm_{i+1}"))
            
        model.add(layers.Activation('relu', name=f"Relu_{i+1}"))
        
        if use_regularization:
            model.add(layers.Dropout(rate=0.2, name=f"Dropout_{i+1}"))
            
    # Final Output Layer
    model.add(layers.Dense(units=num_classes, activation='softmax', name="Output_Layer"))
    
    # Compile with the dynamic learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model