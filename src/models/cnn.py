# CNN architecture for brain tumor MRI classification

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src.utils import config as cfg


def _conv_block(x, filters: int, dropout: float, block_id: int):
    # Each block has two convolutional layers followed by pooling and dropout
    
    # First conv layer with batch normalization and ReLU activation
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=f"block{block_id}_conv1")(x)
    x = layers.BatchNormalization(name=f"block{block_id}_bn1")(x)
    x = layers.Activation("relu", name=f"block{block_id}_relu1")(x)

    # Second conv layer with batch normalization and ReLU activation
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=f"block{block_id}_conv2")(x)
    x = layers.BatchNormalization(name=f"block{block_id}_bn2")(x)
    x = layers.Activation("relu", name=f"block{block_id}_relu2")(x)

    # Max pooling reduces the spatial size by half
    x = layers.MaxPooling2D(pool_size=2, name=f"block{block_id}_pool")(x)
    
    # Dropout randomly switches off neurons to prevent overfitting
    x = layers.Dropout(dropout, name=f"block{block_id}_drop")(x)
    return x


def build_cnn(
    input_shape=(*cfg.IMG_SIZE, cfg.IMG_CHANNELS),
    num_classes: int = cfg.NUM_CLASSES,
    l2_reg: float = cfg.L2_REG,
    dropout_block: float = cfg.DROPOUT_BLOCK,
    dropout_head: float = cfg.DROPOUT_HEAD,
) -> tf.keras.Model:
    # Build the full CNN model
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # Five convolutional blocks with increasing filter sizes
    # More filters in deeper layers allows the model to learn complex features
    x = _conv_block(inputs, 32,  dropout_block, 1)
    x = _conv_block(x,      64,  dropout_block, 2)
    x = _conv_block(x,      128, dropout_block, 3)
    x = _conv_block(x,      256, dropout_block, 4)
    x = _conv_block(x,      512, dropout_block, 5)

    # Global average pooling converts feature maps into a single vector
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Fully connected layer with L2 regularization to reduce overfitting
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(l2_reg),
                     kernel_initializer="he_normal", name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout_head, name="fc1_drop")(x)

    # Output layer with softmax gives a probability for each of the 4 classes
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = models.Model(inputs, outputs, name="BrainTumorCNN")
    return model


# Name of the last convolutional layer - used by Grad-CAM to generate heatmaps
LAST_CONV_LAYER_NAME = "block5_relu2"