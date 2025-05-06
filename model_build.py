#!/usr/bin/env python3
"""
Model builder for sign language recognition using transfer learning.
Provides various model architectures optimized for hand gesture recognition.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, concatenate

class ModelBuilder:
    """
    Builds and configures neural network models for sign language recognition.
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=29):
        """
        Initialize model builder.
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            num_classes: Number of output classes
        """
        self.logger = logging.getLogger(__name__)
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_mobilenet_model(self, weights='imagenet', trainable_base=False):
        """
        Build a MobileNetV2-based model that's efficient for mobile/edge deployment.
        
        Args:
            weights: Pre-trained weights ('imagenet' or None)
            trainable_base: Whether to make base model layers trainable
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.logger.info("Building MobileNetV2 model")
        
        # Create base model
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights
        )
        base_model.trainable = trainable_base
        
        # Create the model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self._compile_model(model)
        return model
    
    def build_efficientnet_model(self, weights='imagenet', trainable_base=False):
        """
        Build an EfficientNet-based model with excellent accuracy/efficiency tradeoff.
        
        Args:
            weights: Pre-trained weights ('imagenet' or None)
            trainable_base: Whether to make base model layers trainable
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.logger.info("Building EfficientNet model")
        
        # Create base model
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights
        )
        base_model.trainable = trainable_base
        
        # Create the model
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        self._compile_model(model)
        return model
    
    def build_resnet_model(self, weights='imagenet', trainable_base=False):
        """
        Build a ResNet-based model for higher accuracy.
        
        Args:
            weights: Pre-trained weights ('imagenet' or None)
            trainable_base: Whether to make base model layers trainable
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.logger.info("Building ResNet model")
        
        # Create base model
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights
        )
        base_model.trainable = trainable_base
        
        # Create the model
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        self._compile_model(model)
        return model
    
    def build_hybrid_model(self, image_input_shape=(128, 128, 3), 
                          landmark_input_shape=(63,), weights='imagenet'):
        """
        Build a hybrid model that combines image features with hand landmarks.
        
        Args:
            image_input_shape: Shape of the image input
            landmark_input_shape: Shape of the landmark input
            weights: Pre-trained weights for the image model
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.logger.info("Building hybrid image + landmark model")
        
        # Image branch
        img_input = Input(shape=image_input_shape)
        base_model = MobileNetV2(
            input_shape=image_input_shape,
            include_top=False,
            weights=weights
        )
        base_model.trainable = False
        
        x1 = base_model(img_input)
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        # Landmark branch
        landmark_input = Input(shape=landmark_input_shape)
        x2 = Dense(128, activation='relu')(landmark_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(128, activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        
        # Merge branches
        merged = concatenate([x1, x2])
        
        # Output layers
        x = Dense(512, activation='relu')(merged)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=[img_input, landmark_input], outputs=outputs)
        
        self._compile_model(model)
        return model
    
    def build_custom_cnn(self):
        """
        Build a custom CNN model without transfer learning.
        Useful as a baseline or when pre-trained models aren't suitable.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        self.logger.info("Building custom CNN model")
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self._compile_model(model)
        return model
    
    def _compile_model(self, model, optimizer='adam', loss='categorical_crossentropy'):
        """
        Compile the model with appropriate optimizer and metrics.
        
        Args:
            model: The Keras model to compile
            optimizer: Optimizer to use ('adam' or custom)
            loss: Loss function
        """
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            opt = optimizer
            
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Print model summary
        model.summary(print_fn=self.logger.info)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model builder
    builder = ModelBuilder(input_shape=(128, 128, 3), num_classes=29)
    
    # Build and test different models
    model1 = builder.build_mobilenet_model()
    model2 = builder.build_efficientnet_model()
    model3 = builder.build_custom_cnn()
    
    # To use the hybrid model, you would need landmark input
    # hybrid_model = builder.build_hybrid_model()
    
    print("Model creation test complete")