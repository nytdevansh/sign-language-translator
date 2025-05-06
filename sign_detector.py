#!/usr/bin/env python3
"""
Improved SignDetector class with better preprocessing, model architecture,
and training capabilities for sign language recognition.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import mediapipe as mp
import logging
import time

class SignDetector:
    """
    Improved detector for sign language recognition with:
    1. MediaPipe hand detection preprocessing
    2. Transfer learning with MobileNetV2
    3. Data augmentation
    4. Advanced training procedures
    """
    
    def __init__(self, config):
        """
        Initialize the improved sign detector.
        
        Args:
            config: Configuration object with settings
        """
        self.config = config
        self.model = None
        self.input_shape = (128, 128, 3)
        self.class_names = []
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Set to False for video streams
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load model if in inference mode
        if config.mode == 'inference':
            self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if os.path.exists(self.config.model_path):
                logging.info(f"Loading model from {self.config.model_path}")
                self.model = load_model(self.config.model_path)
                
                # Get class names if included in the model
                if hasattr(self.model, 'class_names'):
                    self.class_names = self.model.class_names
                elif os.path.exists(self.config.model_path + ".classes"):
                    with open(self.config.model_path + ".classes", "r") as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                else:
                    logging.warning("No class names found, using alphabetical classes")
                    self.class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
                
                logging.info(f"Loaded model with {len(self.class_names)} classes")
            else:
                logging.error(f"Model not found at {self.config.model_path}")
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def create_model(self, num_classes):
        """
        Create an improved model using transfer learning with MobileNetV2.
        
        Args:
            num_classes: Number of sign language classes
            
        Returns:
            The compiled model
        """
        # Create base model from MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # First train with base model frozen
        base_model.trainable = False
        
        # Create new model on top
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_hand_image(self, image, normalize=True, detect_hands=True):
        """
        Preprocess an image to extract and normalize hand regions.
        
        Args:
            image: Input image
            normalize: Whether to normalize pixel values
            detect_hands: Whether to perform hand detection
            
        Returns:
            Preprocessed image or None if no hand detected
        """
        if image is None or image.size == 0:
            return None
        
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        if detect_hands:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_image)
            
            # If no hands found, return the center-cropped and resized original image
            if not results.multi_hand_landmarks:
                # Center crop as fallback
                h, w = img.shape[:2]
                size = min(h, w)
                x_center, y_center = w // 2, h // 2
                x_start = max(0, x_center - size // 2)
                y_start = max(0, y_center - size // 2)
                cropped = img[y_start:y_start + size, x_start:x_start + size]
                resized = cv2.resize(cropped, (self.input_shape[0], self.input_shape[1]))
                
                if normalize:
                    resized = resized.astype(np.float32) / 255.0
                
                return resized
            
            # Extract hand region
            h, w = img.shape[:2]
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # Get bounding box
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Add padding around the hand (20% of the hand size)
            padding_x = int((x_max - x_min) * 0.2)
            padding_y = int((y_max - y_min) * 0.2)
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(w, x_max + padding_x)
            y_max = min(h, y_max + padding_y)
            
            # Ensure square crop (important for sign language gestures)
            crop_size = max(x_max - x_min, y_max - y_min)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            
            x_min = max(0, x_center - crop_size // 2)
            y_min = max(0, y_center - crop_size // 2)
            x_max = min(w, x_min + crop_size)
            y_max = min(h, y_min + crop_size)
            
            # Extract the hand region
            hand_region = img[y_min:y_max, x_min:x_max]
            
            # Debug info
            if self.config.debug:
                logging.debug(f"Hand detected at: ({x_min}, {y_min}), ({x_max}, {y_max})")
            
            # Resize to model input size
            if hand_region.size > 0:
                resized = cv2.resize(hand_region, (self.input_shape[0], self.input_shape[1]))
                
                # Normalize if requested
                if normalize:
                    resized = resized.astype(np.float32) / 255.0
                
                return resized
            
            return None
        else:
            # Simple resize without hand detection
            resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            
            if normalize:
                resized = resized.astype(np.float32) / 255.0
            
            return resized
    
    def predict_sign(self, image):
        """
        Predict the sign from an image.
        
        Args:
            image: Input image
            
        Returns:
            Predicted class label and confidence
        """
        if self.model is None:
            logging.error("Model not loaded")
            return None, 0
        
        # Preprocess the image
        processed_image = self._preprocess_hand_image(image)
        
        if processed_image is None:
            return "nothing", 0.0
        
        # Ensure correct shape for the model
        input_data = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Map to class name
        if predicted_class_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = str(predicted_class_idx)
        
        return predicted_class, float(confidence)
    
    def process_frame(self, frame, visualize=False):
        """
        Process a video frame for sign detection.
        
        Args:
            frame: Input video frame
            visualize: Whether to add visualization overlays
            
        Returns:
            Processed frame, predicted letter, and confidence
        """
        if frame is None:
            return None, "error", 0.0
        
        # Make a copy for visualization
        output_frame = frame.copy()
        
        # Predict sign
        letter, confidence = self.predict_sign(frame)
        
        if visualize:
            # Draw hand landmarks if available
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Draw bounding box and landmarks
                h, w = frame.shape[:2]
                
                # Draw landmarks
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    output_frame, 
                    results.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS
                )
                
                # Create a bounding box
                landmarks = results.multi_hand_landmarks[0].landmark
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding
                padding_x = int((x_max - x_min) * 0.2)
                padding_y = int((y_max - y_min) * 0.2)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                
                # Draw rectangle
                cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add prediction and confidence
            text = f"{letter}: {confidence:.2f}"
            cv2.putText(output_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output_frame, letter, confidence
    
    def create_data_generators(self, dataset_dir, batch_size=32, validation_split=0.2):
        """
        Create data generators with augmentation for training.
        
        Args:
            dataset_dir: Directory containing class subfolders with images
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training generator, validation generator, class names
        """
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Get class names
        class_names = list(train_generator.class_indices.keys())
        logging.info(f"Found {len(class_names)} classes: {class_names}")
        
        return train_generator, val_generator, class_names
    
    def train_model(self, dataset_dir, model_save_path, epochs=50, batch_size=32, fine_tune_epochs=20):
        """
        Train the sign language model using the improved approach.
        
        Args:
            dataset_dir: Directory containing class subfolders with images
            model_save_path: Path to save the trained model
            epochs: Number of initial training epochs
            batch_size: Batch size for training
            fine_tune_epochs: Number of fine-tuning epochs
            
        Returns:
            Training history
        """
        logging.info(f"Training model with dataset from {dataset_dir}")
        logging.info(f"Will save model to {model_save_path}")
        
        # Create data generators
        train_generator, val_generator, class_names = self.create_data_generators(
            dataset_dir, batch_size=batch_size
        )
        
        self.class_names = class_names
        
        # Calculate class weights to handle imbalanced data
        try:
            # Get all class indices from the generator
            labels = np.array(train_generator.classes)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels),
                y=labels
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            logging.info(f"Using class weights: {class_weight_dict}")
        except Exception as e:
            logging.warning(f"Could not compute class weights: {e}")
            class_weight_dict = None
        
        # Create the model
        num_classes = len(class_names)
        self.model = self.create_model(num_classes)
        logging.info(f"Created model for {num_classes} classes")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                model_save_path + '.checkpoint',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train the model with frozen base layers
        logging.info("Starting initial training with frozen base model...")
        history1 = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        # Fine-tune the model by unfreezing some top layers
        logging.info("Fine-tuning the model...")
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the top 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train again with fine-tuning
        history2 = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        # Save the final model
        logging.info(f"Saving model to {model_save_path}")
        
        # Save class names with the model
        setattr(self.model, 'class_names', class_names)
        self.model.save(model_save_path)
        
        # Also save class names to a text file for easier access
        with open(model_save_path + ".classes", "w") as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        # Plot and save training history
        self._plot_training_history({
            'Initial Training': history1,
            'Fine Tuning': history2
        }, model_save_path)
        
        return history1, history2
    
    def _plot_training_history(self, histories, save_path):
        """
        Plot and save training history.
        
        Args:
            histories: Dictionary of history objects from model.fit
            save_path: Base path to save the plots
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Colors for different phases
        colors = ['blue', 'green', 'red', 'purple']
        
        # Plot accuracy
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        
        # Plot loss
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        
        for i, (name, history) in enumerate(histories.items()):
            # Get history data
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            # Get epochs range
            epochs = range(1, len(acc) + 1)
            
            # Add offset for continuation phases
            if i > 0:
                # Get last epoch from previous phase
                last_epoch = next_epoch
            else:
                last_epoch = 0
            
            next_epoch = last_epoch + len(epochs)
            
            # Plot with offset
            epochs_offset = range(last_epoch + 1, next_epoch + 1)
            
            # Plot accuracy
            ax1.plot(epochs_offset, acc, color=colors[i], label=f'{name} - Training')
            ax1.plot(epochs_offset, val_acc, color=colors[i], linestyle='dashed', 
                    label=f'{name} - Validation')
            
            # Plot loss
            ax2.plot(epochs_offset, loss, color=colors[i], label=f'{name} - Training')
            ax2.plot(epochs_offset, val_loss, color=colors[i], linestyle='dashed',
                   label=f'{name} - Validation')
        
        ax1.legend()
        ax2.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{save_path}_training_history.png")
        plt.close()
    
    def collect_data(self, dataset_dir, max_samples_per_class=1000):
        """
        Collect sign language data from webcam.
        
        Args:
            dataset_dir: Directory to save collected data
            max_samples_per_class: Maximum samples per class
        """
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Define classes to collect
        classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open webcam")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        current_class_idx = 0
        current_class = classes[current_class_idx]
        
        samples_collected = 0
        collecting = False
        delay_counter = 0
        delay_between_captures = 5  # Frames between captures when collecting
        
        logging.info(f"Starting data collection. Press 'c' to start/stop collection, " 
                    "SPACE to skip to next class, ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame")
                continue
            
            # Create directory for current class if needed
            class_dir = os.path.join(dataset_dir, current_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Count existing samples
            existing_samples = len([f for f in os.listdir(class_dir) 
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])
            
            # Process the frame for visualization
            processed_frame, _, _ = self.process_frame(frame, visualize=True)
            
            # Add overlay information
            status_text = "COLLECTING" if collecting else "PAUSED"
            cv2.putText(processed_frame, f"Class: {current_class} ({existing_samples}/{max_samples_per_class})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Status: {status_text}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                      (0, 255, 0) if collecting else (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Data Collection", processed_frame)
            
            # Collect samples if active
            if collecting:
                delay_counter += 1
                if delay_counter >= delay_between_captures:
                    delay_counter = 0
                    
                    # Preprocess and save the frame
                    processed_image = self._preprocess_hand_image(frame, normalize=False)
                    
                    if processed_image is not None:
                        # Save the image
                        timestamp = int(time.time() * 1000)
                        filename = os.path.join(class_dir, f"{current_class}_{timestamp}.jpg")
                        cv2.imwrite(filename, processed_image)
                        samples_collected += 1
                        existing_samples += 1
                        
                        logging.info(f"Saved {filename}, {existing_samples}/{max_samples_per_class}")
                        
                        # Check if we've collected enough samples for this class
                        if existing_samples >= max_samples_per_class:
                            logging.info(f"Completed collection for {current_class}")
                            collecting = False
                            current_class_idx = (current_class_idx + 1) % len(classes)
                            current_class = classes[current_class_idx]
                            samples_collected = 0
            
            # Check for key presses
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                # Toggle collection
                collecting = not collecting
                logging.info(f"Collection {'started' if collecting else 'paused'}")
            elif key == 32:  # SPACE
                # Skip to next class
                collecting = False
                current_class_idx = (current_class_idx + 1) % len(classes)
                current_class = classes[current_class_idx]
                samples_collected = 0
                logging.info(f"Switched to class {current_class}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Data collection completed")


# Example configuration class (simplified)
class Config:
    def __init__(self):
        self.model_path = "models/improved_sign_language_model.h5"
        self.mode = "inference"
        self.confidence_threshold = 0.7
        self.stable_duration = 0.5
        self.debug = False
    
    def load_from_args(self, args):
        """Load configuration from command line arguments."""
        if hasattr(args, 'model'):
            self.model_path = args.model
        if hasattr(args, 'mode'):
            self.mode = args.mode
        if hasattr(args, 'debug'):
            self.debug = args.debug