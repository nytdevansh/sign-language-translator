#!/usr/bin/env python3
"""
Simplified test script for the Sign Language Translator model.
This script evaluates model performance on a test dataset.
"""

import os
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import sys

# Import SignDetector with error handling
try:
    from sign_detector import SignDetector
    from utils.config import Config
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project directory.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description='Test Sign Language Translator Model')
    parser.add_argument('--model', type=str, default='models/sign_language_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--output', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to test per class (for faster testing)')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Fraction of images to sample for testing (0.0-1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--no_metal', action='store_true',
                        help='Disable Metal/GPU acceleration')
    return parser.parse_args()

def load_test_data(test_dir, max_images=None, sample_rate=1.0):
    """
    Load test data from directory structure.
    Each subdirectory is a class with images inside.
    
    Args:
        test_dir: Directory containing class folders with images
        max_images: Maximum number of images to load per class
        sample_rate: Fraction of images to sample (0.0-1.0)
        
    Returns:
        images: List of loaded images
        labels: List of numeric labels
        class_names: List of class names
    """
    print(f"Loading test data from {test_dir}")
    
    images = []
    labels = []
    class_names = []
    
    # Get all class directories
    for class_name in sorted(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Found class: {class_name}")
        class_names.append(class_name)
        class_idx = len(class_names) - 1
        
        # Get all images for this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Apply sampling if needed
        if sample_rate < 1.0:
            import random
            num_to_sample = int(len(image_files) * sample_rate)
            image_files = random.sample(image_files, num_to_sample)
        
        # Apply max_images limit if specified
        if max_images is not None:
            image_files = image_files[:max_images]
            
        print(f"  Loading {len(image_files)} images for class {class_name}")
        
        # Load each image
        for img_file in tqdm(image_files, desc=f"Class {class_name}"):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} images across {len(class_names)} classes")
    return images, labels, class_names

def process_images(detector, images, class_names, batch_size=32):
    """
    Process images through the detector and get predictions.
    
    Returns:
        predictions: List of predicted class indices
        confidences: List of confidence values
    """
    predictions = []
    confidences = []
    
    print(f"Processing {len(images)} images...")
    
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        
        for image in batch:
            # Process through detector
            _, letter, confidence = detector.process_frame(image, visualize=False)
            
            # Map letter to class index
            if letter in class_names:
                pred_idx = class_names.index(letter)
            else:
                pred_idx = -1  # Unknown class
                
            predictions.append(pred_idx)
            confidences.append(confidence)
    
    return predictions, confidences

def save_confusion_matrix(labels, predictions, class_names, output_dir):
    """Generate and save confusion matrix visualization."""
    # Filter out invalid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    if not valid_indices:
        print("No valid predictions for confusion matrix")
        return
        
    valid_preds = [predictions[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    # Compute confusion matrix
    cm = confusion_matrix(valid_labels, valid_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    print(f"Saved confusion matrix to {output_path}")
    
    # Save CSV version
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

def save_sample_images(images, labels, predictions, confidences, class_names, output_dir, num_samples=10):
    """Save sample images with predictions for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose random samples (some correct, some incorrect)
    correct_indices = [i for i, (y, p) in enumerate(zip(labels, predictions)) 
                      if y == p and p >= 0]
    incorrect_indices = [i for i, (y, p) in enumerate(zip(labels, predictions)) 
                        if y != p and p >= 0]
    
    # Try to get a mix of correct and incorrect samples
    sample_indices = []
    
    # Add some correct predictions
    if correct_indices:
        num_correct = min(num_samples // 2, len(correct_indices))
        sample_indices.extend(np.random.choice(correct_indices, num_correct, replace=False))
    
    # Add some incorrect predictions
    if incorrect_indices:
        num_incorrect = min(num_samples - len(sample_indices), len(incorrect_indices))
        sample_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))
    
    # Fill remaining slots with random samples if needed
    remaining = num_samples - len(sample_indices)
    if remaining > 0:
        all_indices = list(range(len(images)))
        remaining_indices = [i for i in all_indices if i not in sample_indices]
        if remaining_indices:
            sample_indices.extend(np.random.choice(remaining_indices, 
                                                 min(remaining, len(remaining_indices)), 
                                                 replace=False))
    
    # Save sample images
    for i, idx in enumerate(sample_indices):
        image = images[idx]
        label = labels[idx]
        pred = predictions[idx]
        conf = confidences[idx]
        
        # Get class names
        true_class = class_names[label] if 0 <= label < len(class_names) else "Unknown"
        pred_class = class_names[pred] if 0 <= pred < len(class_names) else "Unknown"
        
        # Create visualization
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Add info bar at bottom
        info_height = 60
        info_img = np.ones((info_height, w, 3), dtype=np.uint8) * 240
        
        # Add text
        is_correct = label == pred
        result_color = (0, 150, 0) if is_correct else (0, 0, 200)  # Green for correct, red for wrong
        
        cv2.putText(info_img, f"True: {true_class}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(info_img, f"Pred: {pred_class} ({conf:.2f})", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
        
        # Combine images
        result = np.vstack([vis_image, info_img])
        
        # Save
        result_filename = f"sample_{i:02d}_{true_class}_as_{pred_class}.jpg"
        output_path = os.path.join(output_dir, result_filename)
        cv2.imwrite(output_path, result)
    
    print(f"Saved {len(sample_indices)} sample images to {output_dir}")

def main():
    """Main function for testing the model."""
    args = parse_arguments()
    
    # Disable GPU if requested
    if args.no_metal:
        os.environ['DISABLE_METAL'] = '1'
        print("Disabled Metal/GPU acceleration")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector with config
    print("Initializing sign detector...")
    config = Config()
    detector = SignDetector(config)
    
    # Load model
    print(f"Loading model from {args.model}")
    detector.load_model(args.model)
    
    # Load test data
    images, labels, class_names = load_test_data(
        args.test_dir, 
        max_images=args.max_images,
        sample_rate=args.sample_rate
    )
    
    if not images:
        print("No images loaded. Check the test directory.")
        return
    
    # Process images
    predictions, confidences = process_images(detector, images, class_names, args.batch_size)
    
    # Calculate accuracy
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    if valid_indices:
        valid_preds = [predictions[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        accuracy = accuracy_score(valid_labels, valid_preds)
        print(f"Overall accuracy: {accuracy:.4f}")
    else:
        accuracy = 0
        print("No valid predictions")
    
    # Generate classification report
    report = classification_report(
        [labels[i] for i in valid_indices],
        [predictions[i] for i in valid_indices],
        target_names=class_names,
        output_dict=True
    )
    
    # Save results
    results = {
        "accuracy": accuracy,
        "num_images": len(images),
        "num_classes": len(class_names),
        "classes": class_names
    }
    
    # Save as JSON
    with open(os.path.join(args.output, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(args.output, "classification_report.csv")
    )
    
    # Generate confusion matrix
    save_confusion_matrix(labels, predictions, class_names, args.output)
    
    # Save sample images
    save_sample_images(
        images, labels, predictions, confidences, 
        class_names, os.path.join(args.output, "samples")
    )
    
    print(f"Testing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()