#!/usr/bin/env python3
"""
Main entry point for the Sign Language Translator application.
"""

import cv2
import time
import argparse
import os
import numpy as np
from sign_detector import SignDetector
from utils.config import Config
from utils.logger import setup_logger
from mediapipe import solutions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sign Language Translator')
    parser.add_argument('--model', type=str, default='models/sign_language_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index to use')
    parser.add_argument('--mode', type=str, choices=['inference', 'training', 'data_collection'],
                        default='inference', help='Operation mode')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Dataset directory for training or collection')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with visualization')
    parser.add_argument('--subtitle_size', type=float, default=1.0,
                        help='Size multiplier for subtitles')
    parser.add_argument('--word_timeout', type=float, default=2.0,
                        help='Timeout in seconds to form words')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (default: console only)')
    parser.add_argument('--hand_padding', type=float, default=0.2,
                        help='Padding ratio around detected hand (0.2 = 20%)')
    return parser.parse_args()

def detect_hands(frame):
    """
    Detect hand landmarks in the frame using MediaPipe.
    Returns hand landmarks and hand rectangle coordinates.
    """
    # Initialize MediaPipe Hands solution
    mp_hands = solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        # Convert the BGR image to RGB and process it with MediaPipe
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate bounding box for the hand
        h, w, _ = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        return hand_landmarks, (x_min, y_min, x_max, y_max)

def crop_hand(frame, bbox, padding=0.2):
    """
    Crop hand region from the frame with padding.
    
    Args:
        frame: Original camera frame
        bbox: Hand bounding box (x_min, y_min, x_max, y_max)
        padding: Padding ratio around the hand (0.2 = 20%)
        
    Returns:
        Cropped hand image
    """
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    h, w, _ = frame.shape
    
    # Calculate padding
    width = x_max - x_min
    height = y_max - y_min
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    # Apply padding with bounds checking
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)
    
    # Make the crop square (required by many models)
    crop_width = x2 - x1
    crop_height = y2 - y1
    max_dim = max(crop_width, crop_height)
    
    # Center the square crop
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Calculate new square bounds
    x1_new = max(0, center_x - max_dim // 2)
    y1_new = max(0, center_y - max_dim // 2)
    x2_new = min(w, x1_new + max_dim)
    y2_new = min(h, y1_new + max_dim)
    
    # If we hit the image boundary, shift the square to fit within the image
    if x2_new == w:
        x1_new = max(0, w - max_dim)
    if y2_new == h:
        y1_new = max(0, h - max_dim)
    
    # Crop the image
    hand_crop = frame[y1_new:y2_new, x1_new:x2_new]
    
    return hand_crop, (x1_new, y1_new, x2_new, y2_new)

def draw_hand_box(frame, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box around the hand on the frame."""
    if bbox is None:
        return frame
    
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def main():
    """Main application function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(log_file=args.log_file)
    logger.info("Starting Sign Language Translator")
    
    # Load configuration
    config = Config()
    config.load_from_args(args)
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    # Initialize detector
    try:
        detector = SignDetector(config)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    if args.mode == 'data_collection':
        logger.info(f"Starting data collection mode with dataset: {args.dataset}")
        detector.collect_data(args.dataset)
        return
    elif args.mode == 'training':
        logger.info(f"Starting training mode with dataset: {args.dataset}")
        detector.train_model(args.dataset, args.model)
        return
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {args.camera}")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Variables for sentence building
    current_text = ""
    current_word = []
    last_letter = None
    letter_start_time = 0
    letter_confidence = 0
    stable_duration = config.stable_duration  # Time a letter needs to be stable to be added
    
    logger.info("Starting inference loop")
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            # Detect hand in the frame
            hand_landmarks, hand_bbox = detect_hands(frame)
            
            # Process hand if detected
            letter = None
            confidence = 0
            
            if hand_bbox:
                # Crop hand region from the frame
                hand_crop, square_bbox = crop_hand(frame, hand_bbox, padding=args.hand_padding)
                
                if hand_crop is not None and hand_crop.size > 0:
                    # Draw hand bounding box on display frame
                    draw_hand_box(display_frame, square_bbox)
                    
                    # Process the cropped hand with the model
                    _, letter, confidence = detector.process_frame(hand_crop, visualize=False)
                    
                    # If debug is enabled, show the hand crop in a separate window
                    if args.debug:
                        cv2.imshow('Hand Crop', hand_crop)
            
            # Work with letter predictions
            current_time = time.time()
            
            # Only consider predictions with decent confidence
            if letter and confidence > config.confidence_threshold:
                if letter == last_letter:
                    # Same letter is being detected
                    if current_time - letter_start_time > stable_duration and letter != "HOLD":
                        # Letter has been stable long enough and not already added
                        if letter == "SPACE":
                            # Add the current word to the sentence
                            if current_word:
                                word = ''.join(current_word)
                                current_text += word + " "
                                current_word = []
                                logger.debug(f"Added word: {word}")
                        elif letter == "DELETE":
                            # Delete last character or word
                            if current_word:
                                if current_word:
                                    deleted = current_word.pop()
                                    logger.debug(f"Deleted character: {deleted}")
                            elif current_text:
                                current_text = current_text[:-1]
                                logger.debug("Deleted last character from text")
                        elif letter == "NOTHING":
                            # Ignore "NOTHING" gesture
                            pass
                        else:
                            # Add letter to current word
                            current_word.append(letter)
                            logger.debug(f"Added letter: {letter}")
                        
                        # Mark this letter as added
                        last_letter = "HOLD"
                        letter_confidence = confidence
                else:
                    # New letter detected
                    last_letter = letter
                    letter_start_time = current_time
                    letter_confidence = confidence
            
            # Handle timeout for word completion
            if current_word and last_letter != "HOLD" and (current_time - letter_start_time > args.word_timeout):
                # Complete the word if there's been no activity
                word = ''.join(current_word)
                current_text += word + " "
                current_word = []
                logger.debug(f"Word timeout - added: {word}")
            
            # Add subtitle area (dark background strip)
            subtitle_height = int(100 * args.subtitle_size)
            subtitle_bg = display_frame.copy()
            subtitle_bg[-subtitle_height:, :] = (0, 0, 0)
            cv2.addWeighted(subtitle_bg, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Add subtitle text
            display_text = current_text
            if current_word:
                display_text += ''.join(current_word)
            
            if last_letter and last_letter != "HOLD":
                # Show the currently detected letter
                letter_text = f"Detecting: {last_letter} ({int(letter_confidence*100)}%)"
                cv2.putText(display_frame, letter_text, 
                          (20, display_frame.shape[0] - subtitle_height + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6 * args.subtitle_size, 
                          (200, 200, 200), 1)

            # Display the recognized text
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       1.0 * args.subtitle_size, 2)[0]
            text_x = max(10, (display_frame.shape[1] - text_size[0]) // 2)
            cv2.putText(display_frame, display_text, 
                      (text_x, display_frame.shape[0] - subtitle_height + 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0 * args.subtitle_size, 
                      (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Sign Language Translator', display_frame)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                # Clear current text
                current_text = ""
                current_word = []
                logger.debug("Text cleared by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application closed")

if __name__ == "__main__":
    main()